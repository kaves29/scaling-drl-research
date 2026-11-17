"""One L4, persistent parent data, and resumable replay pilot workers.

Launch: modal run --detach main/infra/modal/exp2_modal.py --mode train
Verify uploads without GPU: modal run main/infra/modal/exp2_modal.py --mode verify
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import tempfile
import time

import modal

app = modal.App("echocritic-exp2-replay-pilot")
parents = modal.Volume.from_name("exp1-pilot-states", create_if_missing=True)
results = modal.Volume.from_name("exp2-pilot-results", create_if_missing=True)

# Modal re-imports this module inside the container at /root/exp2_modal.py.
# Bundle creation and local file mounts exist only in the client checkout.
if modal.is_local():
    ROOT = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise RuntimeError("commit reviewed source before building the Modal image")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    bundle = Path(tempfile.gettempdir()) / f"exp2-modal-source-{source_commit}.bundle"
    if not bundle.exists():
        subprocess.run(["git", "bundle", "create", str(bundle), "HEAD"], cwd=ROOT, check=True)
    image = (
        modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.12")
        .apt_install("git", "libgl1", "libegl1", "libosmesa6", "build-essential")
        .pip_install_from_requirements(str(ROOT / "main/requirements.txt"))
        .pip_install("jax[cuda12]==0.4.34")
        .add_local_file(str(bundle), remote_path="/tmp/source.bundle", copy=True)
        .run_commands("git clone /tmp/source.bundle /source",
                      f"git -C /source checkout --detach {source_commit}",
                      "rm /tmp/source.bundle")
    )
    verify_image = modal.Image.debian_slim(python_version="3.12").add_local_file(
        str(ROOT / "main/infra/modal/parent_inventory.py"), "/opt/parent_inventory.py", copy=True
    )
else:
    image = verify_image = modal.Image.debian_slim(python_version="3.12")

MOUNTS = {"/parents": parents, "/results": results}
def resource_rate(memory_gib, gpu="L4"):
    if type(memory_gib) is not int or memory_gib not in (20, 48, 75):
        raise ValueError("unsupported host-memory reservation")
    if gpu not in ("L4", "A100-80GB"):
        raise ValueError("unsupported GPU reservation")
    gpu_rate = 0.000222 if gpu == "L4" else 0.000694
    return gpu_rate + 8 * 0.0000131 + memory_gib * 0.00000222


PER_SECOND = resource_rate(48)
# Modal published per-second A100-80GB, CPU and RAM rates, checked 2026-11-17.
A100_SMOKE_RATE = 0.000694 + 8 * 0.0000131 + 20 * 0.00000222
A100_SMOKE_RESERVATION = 600 * A100_SMOKE_RATE


def reservation_seconds(reserved, ceiling, session_limit, memory_gib=48, gpu="L4"):
    if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in (reserved, ceiling, session_limit))
            or reserved < 0 or not 0 < ceiling <= 40 or session_limit <= 0):
        raise ValueError("invalid spending limit or ledger")
    seconds = min(22 * 3600, int(min(ceiling - reserved, session_limit) / resource_rate(memory_gib, gpu)))
    if seconds < 300:
        raise ValueError("remaining reserved budget below five minutes")
    return seconds


def has_start_headroom(gpu_free_mib, host_free_mib):
    # One prior L4 worker sampled 8.28 GiB GPU / 6.99 GiB RSS before branches.
    # Leave a 2-GiB margin beyond rounded-up per-worker estimates.
    return gpu_free_mib >= 11 * 1024 and host_free_mib >= 9 * 1024


def memory_headroom():
    """Use device free memory and the container's actual cgroup memory budget."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    ).strip().splitlines()
    if len(output) != 1:
        raise ValueError("expected exactly one visible GPU")
    gpu_free_mib = int(output[0].strip())
    cgroup = Path("/sys/fs/cgroup")
    candidates = ((cgroup / "memory.max", cgroup / "memory.current"),
                  (cgroup / "memory/memory.limit_in_bytes", cgroup / "memory/memory.usage_in_bytes"),
                  (cgroup / "memory.limit_in_bytes", cgroup / "memory.usage_in_bytes"))
    for limit_path, used_path in candidates:
        if limit_path.is_file() and used_path.is_file():
            limit = int(limit_path.read_text().strip())
            used = int(used_path.read_text().strip())
            break
    else:
        raise ValueError("cgroup memory limit and usage are unavailable")
    # Pilot containers have a hard 75-GiB limit; reject unlimited/host cgroups.
    if limit <= 0 or limit > 75 * 1024 ** 3 or used < 0:
        raise ValueError("invalid cgroup memory measurements")
    return gpu_free_mib, max(0, (limit - used) // (1024 * 1024))


def qualification_context(manifest):
    import yaml
    row = json.loads(manifest.read_text())["runs"][0]
    if row["architecture"] not in ("D2W512", "D4W1536"):
        raise ValueError("unknown qualification architecture")
    identity = {"environment": row["environment"], "seed": row["seed"],
                "reference_regime": "reference_small" if row["architecture"] == "D2W512" else "reference_large"}
    config = Path("/source/main/configs") / f"online_exp2_pilot_{identity['reference_regime']}.yaml"
    horizon = yaml.safe_load(config.read_text())["online_exp2"]["branch_raw_steps"]
    return identity, horizon


def validate_qualification(receipt, source_commit, manifest_hash, identity, expected_horizon):
    workers = receipt.get("qualified_workers")
    horizon = receipt.get("branch_horizon")
    counters = receipt.get("branch_counters")
    interruption = receipt.get("interruption")
    conditions = {f"{arch}_{mode}" for arch in ("small", "large")
                  for mode in ("frozen", "learning")}
    if (receipt.get("identity") != identity or horizon != expected_horizon
            or type(horizon) is not int or horizon <= 0
            or not isinstance(counters, dict) or set(counters) != conditions
            or any(not isinstance(value, dict) or value.get("raw_env_steps") != horizon
                   for value in counters.values())
            or sorted(receipt.get("probe_architectures", [])) != ["large", "small"]
            or not isinstance(interruption, dict) or interruption.get("checkpoint") != 5000
            or type(interruption.get("raw_env_steps")) is not int
            or not 0 < interruption["raw_env_steps"] < horizon
            or receipt.get("source_commit") != source_commit
            or receipt.get("manifest_sha256") != manifest_hash
            or receipt.get("gpu_model") != "NVIDIA L4"
            or receipt.get("qualification_protocol") != "replay-qualification-5000-v1"
            or receipt.get("qualification_updates") != 5000
            or receipt.get("checkpoint") != 5000
            or not all(receipt.get(key) is True for key in
                       ("parent_restore", "full_branch_horizon", "plasticity_probe", "restart"))
            or type(workers) is not int or not 1 <= workers <= 7):
        raise ValueError("incomplete or stale L4 qualification")
    return workers


def validate_a100_smoke(receipt, smoke_commit, manifest_hash, identity):
    conditions = {f"{arch}_{mode}" for arch in ("small", "large")
                  for mode in ("frozen", "learning")}
    counters = receipt.get("branch_counters")
    interrupted = receipt.get("interruption")
    probe = {"rounds": 1, "steps": 5, "pool_size": 256, "batch_size": 32, "eval_chunk": 64}
    if (receipt.get("source_commit") != smoke_commit or receipt.get("manifest_sha256") != manifest_hash
            or receipt.get("identity") != identity or receipt.get("smoke_only") is not True
            or "A100" not in receipt.get("gpu_model", "")
            or type(receipt.get("gpu_memory_mib")) is not int or receipt["gpu_memory_mib"] < 75000
            or receipt.get("qualification_protocol") != "a100-smoke-100-20-v1"
            or receipt.get("qualification_updates") != 100 or receipt.get("checkpoint") != 100
            or receipt.get("qualified_workers") != 0 or receipt.get("parent_restore") is not True
            or receipt.get("restart") is not True or receipt.get("branch_horizon") != 20
            or receipt.get("full_branch_horizon") is not False or receipt.get("plasticity_probe") is not False
            or receipt.get("smoke_probe") != probe
            or sorted(receipt.get("probe_architectures", [])) != ["large", "small"]
            or not isinstance(counters, dict) or set(counters) != conditions
            or any(not isinstance(v, dict) or v.get("raw_env_steps") != 20 for v in counters.values())
            or not isinstance(interrupted, dict) or interrupted.get("checkpoint") != 100
            or type(interrupted.get("raw_env_steps")) is not int
            or not 0 < interrupted["raw_env_steps"] < 20):
        raise ValueError("invalid A100 smoke evidence")


def worker_environment():
    return dict(os.environ, PYTHONPATH="/source/main", JAX_PLATFORMS="cuda,cpu",
                MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID="0",
                WANDB_MODE="online", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


@app.function(
    image=verify_image,
    volumes=MOUNTS,
    timeout=3600,
)
def verify():
    import sys
    sys.path.insert(0, "/opt")
    from parent_inventory import verify_transfer

    root = Path("/parents/exp1-pilot-parents")
    source = Path("/parents/exp1-pilot-transfer-receipts/echocritic-exp1-pilot-checksums-20261116.json")
    uploaded_receipt = source.with_name("exp1-pilot-modal-verification-20261116.json")
    verification = root / "verification.json"
    verification.unlink(missing_ok=True)
    parents.commit()
    rows = verify_transfer(root, source, uploaded_receipt)
    manifest = root / "manifest.json"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps({"runs": rows}, indent=2) + "\n")
    temporary.replace(manifest)
    receipt = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "source_inventory_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "all_files_ok": True, "files": json.loads(source.read_text())["file_count"],
               "verified_unix": time.time()}
    temporary = verification.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2))
    temporary.replace(verification)
    parents.commit()
    return receipt


@app.function(image=image, gpu="L4", cpu=8, memory=49152, volumes=MOUNTS,
              secrets=[modal.Secret.from_name("exp2-wandb")], timeout=23 * 3600,
              max_containers=1, retries=0)
def qualify(budget_usd: float = 10.0, session_usd: float = 5.0, memory_gib: int = 48):
    """Run a single-worker, separately restarted qualification under a reservation."""
    source_commit = subprocess.check_output(["git", "-C", "/source", "rev-parse", "HEAD"], text=True).strip()
    if not 0 < budget_usd <= 40:
        raise ValueError("spending ceiling exceeded")
    manifest = Path("/parents/exp1-pilot-parents/manifest.json")
    verification = json.loads(manifest.with_name("verification.json").read_text())
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if verification.get("all_files_ok") is not True or verification.get("manifest_sha256") != digest:
        raise ValueError("verify uploaded parents first")
    output = Path("/results/replay-pilot")
    output.mkdir(parents=True, exist_ok=True)
    qualification = output / f"qualification-5000-{source_commit}"
    qualification.mkdir(exist_ok=True)
    final = output / f"l4-qualification-{source_commit}.json"
    if final.exists():
        raise ValueError("qualification receipt already exists; review it before replacing")
    ledger = output / "budget.json"
    old = json.loads(ledger.read_text()) if ledger.exists() else {"reserved_usd": 2.0}
    seconds = reservation_seconds(old["reserved_usd"], budget_usd, session_usd, memory_gib)
    old["reserved_usd"] += seconds * resource_rate(memory_gib)
    ledger.write_text(json.dumps(old, indent=2))
    results.commit()
    identity, expected_horizon = qualification_context(manifest)
    env = worker_environment()
    deadline = time.monotonic() + seconds
    try:
        for phase in ("interrupt", "resume"):
            if phase == "interrupt" and (qualification / "qualification-interruption.json").exists():
                continue
            command = ["python", "-m", "experiments.exp12.replay_exp2_qualification",
                       "--parent-manifest", str(manifest), "--output", str(qualification),
                       "--identity", json.dumps(identity), "--phase", phase]
            with (qualification / "process.log").open("a") as log:
                proc = subprocess.Popen(command, cwd="/source/main", env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
                try:
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("qualification reservation expired")
                        try:
                            code = proc.wait(timeout=min(60, remaining))
                            if code:
                                raise subprocess.CalledProcessError(code, command)
                            break
                        except subprocess.TimeoutExpired:
                            results.commit()
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
            results.commit()
        receipt = json.loads((qualification / "qualification-result.json").read_text())
        validate_qualification(receipt, source_commit, digest, identity, expected_horizon)
        final.write_text(json.dumps(receipt, indent=2))
    finally:
        results.commit()
    return receipt


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=20480, volumes=MOUNTS,
              timeout=600, max_containers=1, retries=0)
def a100_smoke(budget_usd: float = 40.0):
    """A 100-update, shortened engineering smoke; never admits pilot training."""
    source_commit = subprocess.check_output(
        ["git", "-C", "/source", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = Path("/parents/exp1-pilot-parents/manifest.json")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    verification = json.loads(manifest.with_name("verification.json").read_text())
    if verification.get("all_files_ok") is not True or verification.get("manifest_sha256") != digest:
        raise ValueError("verify uploaded parents first")
    output = Path("/results/replay-pilot")
    directory = output / f"a100-smoke-{source_commit}"
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "a100-smoke-result.json"
    if result_path.exists():
        raise ValueError("A100 smoke result already exists for this source")
    ledger = output / "budget.json"
    old = json.loads(ledger.read_text()) if ledger.exists() else {"reserved_usd": 2.0}
    reserved = old["reserved_usd"]
    if (not isinstance(reserved, (int, float)) or isinstance(reserved, bool)
            or not math.isfinite(reserved) or reserved < 0
            or not isinstance(budget_usd, (int, float)) or isinstance(budget_usd, bool)
            or not math.isfinite(budget_usd) or not 0 < budget_usd <= 40
            or reserved + A100_SMOKE_RESERVATION > budget_usd):
        raise ValueError("A100 smoke exceeds the cumulative spending ceiling")
    old["reserved_usd"] += A100_SMOKE_RESERVATION
    ledger.write_text(json.dumps(old, indent=2))
    results.commit()
    identity, _ = qualification_context(manifest)
    env = worker_environment()
    env["WANDB_MODE"] = "disabled"
    deadline = time.monotonic() + 540
    try:
        for phase in ("interrupt", "resume"):
            if phase == "interrupt" and (directory / "qualification-interruption.json").exists():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("A100 smoke time limit reached")
            command = ["python", "-m", "experiments.exp12.replay_exp2_qualification",
                       "--parent-manifest", str(manifest), "--output", str(directory),
                       "--identity", json.dumps(identity), "--phase", phase, "--a100-smoke"]
            with (directory / "process.log").open("a") as log:
                subprocess.run(command, cwd="/source/main", env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=remaining)
            results.commit()
        receipt = json.loads(result_path.read_text())
        if (receipt.get("source_commit") != source_commit or receipt.get("manifest_sha256") != digest
                or receipt.get("identity") != identity or receipt.get("smoke_only") is not True
                or receipt.get("qualification_protocol") != "a100-smoke-100-20-v1"
                or receipt.get("checkpoint") != 100 or receipt.get("branch_horizon") != 20
                or receipt.get("restart") is not True or "A100" not in receipt.get("gpu_model", "")):
            raise ValueError("A100 smoke evidence mismatch")
        return receipt
    finally:
        results.commit()


@app.function(image=image, gpu="L4", cpu=8, memory=76800, volumes=MOUNTS,
              secrets=[modal.Secret.from_name("exp2-wandb")], timeout=23 * 3600,
              max_containers=1, retries=0)
def train(max_workers: int = 7, budget_usd: float = 40.0,
          max_seconds: int = 22 * 3600):
    """A session reserves its maximum estimated charge before starting work."""
    source_commit = subprocess.check_output(["git", "-C", "/source", "rev-parse", "HEAD"], text=True).strip()
    if (type(max_workers) is not int or not 1 <= max_workers <= 7
            or type(max_seconds) is not int or not 300 <= max_seconds <= 22 * 3600
            or not 0 < budget_usd <= 40):
        raise ValueError("worker, runtime or spending ceiling exceeded")
    manifest = Path("/parents/exp1-pilot-parents/manifest.json")
    verification = manifest.with_name("verification.json")
    if not verification.exists():
        raise ValueError("verify uploaded parents first")
    verified = json.loads(verification.read_text())
    if (verified.get("all_files_ok") is not True or
            verified.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest()):
        raise ValueError("verification receipt does not match manifest")
    output = Path("/results/replay-pilot")
    output.mkdir(parents=True, exist_ok=True)
    qualification_path = output / f"l4-qualification-{source_commit}.json"
    if not qualification_path.exists():
        raise ValueError("actual L4 restore/branch/probe/restart qualification is required")
    qualification = json.loads(qualification_path.read_text())
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    identity, expected_horizon = qualification_context(manifest)
    qualified_workers = validate_qualification(qualification, source_commit, manifest_hash,
                                               identity, expected_horizon)
    max_workers = min(max_workers, qualified_workers)
    # Published per-second GPU + requested CPU + requested host-memory rates.
    # Deliberately bill all requested resources for the full session, even idle.
    reserve = 2.0  # CPU-only verification/build/storage and shutdown allowance.
    ledger = output / "budget.json"
    old = json.loads(ledger.read_text()) if ledger.exists() else {"reserved_usd": reserve}
    rate = resource_rate(75)
    seconds = reservation_seconds(old["reserved_usd"], budget_usd,
                                  min(budget_usd, max_seconds * rate), 75)
    old["reserved_usd"] += seconds * rate
    ledger.write_text(json.dumps(old, indent=2))
    results.commit()
    return run_pilot_workers(manifest, output, max_workers, source_commit, seconds, old)


def validate_completed_run(directory, source_commit, identity):
    path = directory / "analysis.json"
    if not path.is_file():
        raise ValueError(f"worker exited without analysis: {directory}")
    completed = json.loads(path.read_text())
    if (completed.get("source_commit") != source_commit
            or completed.get("protocol") != "exp1-final-replay-v1"
            or completed.get("environment") != identity["environment"]
            or completed.get("training_seed") != identity["seed"]
            or completed.get("reference_regime") != identity["reference_regime"]):
        raise ValueError("completed run identity mismatch; use new output root")


def run_pilot_workers(manifest, output, max_workers, source_commit, seconds, old,
                      wandb_tags=None):
    deadline = time.monotonic() + seconds
    rows = json.loads(manifest.read_text())["runs"]
    pending = [dict(environment=r["environment"], seed=r["seed"],
                    reference_regime="reference_small" if r["architecture"] == "D2W512" else "reference_large")
               for r in rows]
    active = []
    env = worker_environment()
    if wandb_tags:
        env["WANDB_TAGS"] = ",".join(filter(None, (env.get("WANDB_TAGS"), wandb_tags)))
    last_commit = time.monotonic()
    last_launch = 0.0
    try:
        while (pending or active) and time.monotonic() < deadline - 120:
            while pending and len(active) < max_workers:
                if active and time.monotonic() - last_launch < 30:
                    break
                gpu_free, host_free = memory_headroom()
                if not has_start_headroom(gpu_free, host_free):
                    if not active:
                        raise RuntimeError("insufficient GPU or host memory headroom")
                    break
                identity = pending.pop(0)
                name = f"{identity['environment']}_seed{identity['seed']}_{identity['reference_regime']}"
                directory = output / name
                directory.mkdir(exist_ok=True)
                if (directory / "analysis.json").exists():
                    validate_completed_run(directory, source_commit, identity)
                    continue
                log = (directory / "process.log").open("a")
                command = ["python", "-m", "experiments.exp12.replay_exp2_pilot",
                           "--parent-manifest", str(manifest), "--output", str(directory),
                           "--identity", json.dumps(identity)]
                proc = subprocess.Popen(command, env=env, cwd="/source/main", stdout=log,
                                        stderr=subprocess.STDOUT)
                active.append((proc, log, identity, directory))
                last_launch = time.monotonic()
                print("START", name, "gpu_free_mib", gpu_free,
                      "host_free_mib", host_free, flush=True)
            for item in list(active):
                proc, log, identity, directory = item
                code = proc.poll()
                if code is not None:
                    log.close()
                    active.remove(item)
                    if code:
                        tail = (directory / "process.log").read_text(errors="replace")[-10000:]
                        if "RESOURCE_EXHAUSTED" in tail or "out of memory" in tail.lower():
                            max_workers = max(1, max_workers - 1)
                            if max_workers == 1 and not active:
                                raise RuntimeError(f"worker failed; inspect {directory}")
                            pending.append(identity)
                        else:
                            raise RuntimeError(f"worker exit {code}; inspect {directory}")
                    else:
                        validate_completed_run(directory, source_commit, identity)
                    print("EXIT", identity, code, flush=True)
            if time.monotonic() - last_commit >= 60:
                results.commit()
                last_commit = time.monotonic()
            time.sleep(2)
    finally:
        for proc, log, _, _ in active:
            proc.terminate()
        shutdown_deadline = time.monotonic() + 90
        for proc, log, _, _ in active:
            try:
                proc.wait(timeout=max(0, shutdown_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
        results.commit()
    return {"pending": len(pending), "interrupted": len(active), "budget": old}


A100_SMOKE_SOURCE = "f1b0aeb20859f2bd3d6e07c62d0360212be664cb"


def verify_a100_smoke_source(smoke_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", smoke_commit):
        raise ValueError("invalid smoke source commit")
    subprocess.run(["git", "-C", "/source", "merge-base", "--is-ancestor",
                    smoke_commit, "HEAD"], check=True)
    # Launcher and tests may change; executable experiment, configs and utilities may not.
    subprocess.run(["git", "-C", "/source", "diff", "--quiet", smoke_commit, "HEAD", "--",
                    "main", ":(exclude)main/infra/**", ":(exclude)main/tests/**",
                    ":(exclude)main/docs/**", ":(exclude)main/notebooks/**"], check=True)


@app.function(image=image, gpu="A100-80GB", cpu=(8, 8), memory=(76800, 76800), volumes=MOUNTS,
              secrets=[modal.Secret.from_name("exp2-wandb")], timeout=23 * 3600,
              max_containers=1, retries=0)
def a100_pilot(max_workers: int = 8, budget_usd: float = 40.0):
    """Run full replay pilot after a short, explicitly nonqualifying A100 smoke."""
    if type(max_workers) is not int or not 1 <= max_workers <= 8:
        raise ValueError("A100 worker ceiling exceeded")
    source_commit = subprocess.check_output(
        ["git", "-C", "/source", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = Path("/parents/exp1-pilot-parents/manifest.json")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    verification = json.loads(manifest.with_name("verification.json").read_text())
    if verification.get("all_files_ok") is not True or verification.get("manifest_sha256") != manifest_hash:
        raise ValueError("verify uploaded parents first")
    model_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True
    ).strip().splitlines()
    if len(model_lines) != 1:
        raise ValueError("expected exactly one visible GPU")
    model, total = (part.strip() for part in model_lines[0].rsplit(",", 1))
    if "A100" not in model or int(total) < 75000:
        raise ValueError("A100 80 GB required")
    identity, _ = qualification_context(manifest)
    smoke_path = (Path("/results/replay-pilot") / f"a100-smoke-{A100_SMOKE_SOURCE}" /
                  "a100-smoke-result.json")
    smoke_bytes = smoke_path.read_bytes()
    smoke = json.loads(smoke_bytes)
    validate_a100_smoke(smoke, A100_SMOKE_SOURCE, manifest_hash, identity)
    verify_a100_smoke_source(A100_SMOKE_SOURCE)
    # Do not reserve an entire session if telemetry or initial headroom fails.
    gpu_free, host_free = memory_headroom()
    if not has_start_headroom(gpu_free, host_free):
        raise RuntimeError("insufficient GPU or host memory headroom")
    output = Path("/results/replay-pilot-a100") / source_commit
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"source_commit": source_commit, "manifest_sha256": manifest_hash,
                "smoke_source_commit": A100_SMOKE_SOURCE,
                "smoke_receipt_sha256": hashlib.sha256(smoke_bytes).hexdigest(),
                "smoke_scope": "short engineering smoke; no concurrency or scientific qualification",
                "gpu_model": model, "gpu_memory_mib": int(total),
                "max_workers": max_workers, "host_memory_gib": 75}
    provenance = output / "execution.json"
    if provenance.exists() and json.loads(provenance.read_text()) != metadata:
        raise ValueError("A100 output provenance mismatch")
    provenance.write_text(json.dumps(metadata, indent=2))
    ledger = Path("/results/replay-pilot/budget.json")
    old = json.loads(ledger.read_text()) if ledger.exists() else {"reserved_usd": 2.0}
    rate = resource_rate(75, "A100-80GB")
    seconds = reservation_seconds(old["reserved_usd"], budget_usd, budget_usd, 75, "A100-80GB")
    old["reserved_usd"] += seconds * rate
    ledger.write_text(json.dumps(old, indent=2))
    results.commit()
    print("A100 engineering smoke passed; concurrency and scientific qualification remain unverified", flush=True)
    return run_pilot_workers(manifest, output, max_workers, source_commit, seconds, old,
                             wandb_tags="a100-pilot,short-smoke-only,concurrency-unqualified")


@app.function(image=image, gpu="L4", cpu=8, memory=76800, volumes=MOUNTS,
              secrets=[modal.Secret.from_name("exp2-wandb")], timeout=23 * 3600,
              max_containers=1, retries=0)
def qualify_then_train(budget_usd: float = 40.0, qualification_usd: float = 5.0):
    """Qualify then train sequentially in one GPU container."""
    session_start = time.monotonic()
    source_commit = subprocess.check_output(
        ["git", "-C", "/source", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = Path("/parents/exp1-pilot-parents/manifest.json")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    identity, expected_horizon = qualification_context(manifest)
    results.reload()
    path = Path("/results/replay-pilot") / f"l4-qualification-{source_commit}.json"
    if path.exists():
        receipt = json.loads(path.read_text())
    else:
        receipt = qualify.local(budget_usd, qualification_usd, memory_gib=75)
        validate_qualification(receipt, source_commit, digest, identity, expected_horizon)
        results.reload()
        if json.loads(path.read_text()) != receipt:
            raise ValueError("persisted qualification differs from returned evidence")
    validate_qualification(receipt, source_commit, digest, identity, expected_horizon)
    remaining = int(22 * 3600 - (time.monotonic() - session_start))
    return train.local(max_workers=7, budget_usd=budget_usd, max_seconds=remaining)


@app.local_entrypoint()
def main(mode: str = "verify", budget_usd: float = 40.0,
         qualification_usd: float = 5.0):
    if mode == "verify":
        print(verify.remote())
    elif mode == "qualify":
        print(qualify.remote(budget_usd, qualification_usd))
    elif mode == "a100_smoke":
        print(a100_smoke.remote(budget_usd))
    elif mode == "a100_pilot":
        print(a100_pilot.remote(max_workers=8, budget_usd=budget_usd))
    elif mode == "qualify_then_train":
        print(qualify_then_train.spawn(budget_usd, qualification_usd).object_id)
    elif mode == "train":
        print(train.remote(budget_usd=budget_usd))
    else:
        raise ValueError("mode must be verify, qualify, a100_smoke, a100_pilot, qualify_then_train or train")
