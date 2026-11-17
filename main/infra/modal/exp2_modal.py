"""One L4, persistent parent data, and resumable replay pilot workers.

Launch: modal run --detach main/infra/modal/exp2_modal.py --mode train
Verify uploads without GPU: modal run main/infra/modal/exp2_modal.py --mode verify
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import modal

ROOT = Path(__file__).resolve().parents[3]
# Bundle the committed source instead of copying a worktree's external .git link.
# Creating this local source snapshot does not contact Modal or allocate a GPU.
if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
    raise RuntimeError("commit reviewed source before building the Modal image")
SOURCE_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
bundle = Path(tempfile.gettempdir()) / f"exp2-modal-source-{SOURCE_COMMIT}.bundle"
if not bundle.exists():
    subprocess.run(["git", "bundle", "create", str(bundle), "HEAD"], cwd=ROOT, check=True)
app = modal.App("echocritic-exp2-replay-pilot")
parents = modal.Volume.from_name("exp1-pilot-states", create_if_missing=True)
results = modal.Volume.from_name("exp2-pilot-results", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libgl1", "libegl1", "libosmesa6", "build-essential")
    .pip_install_from_requirements(str(ROOT / "main/requirements.txt"))
    .pip_install("jax[cuda12]==0.4.34")
    .add_local_file(str(bundle), remote_path="/tmp/source.bundle", copy=True)
    .run_commands("git clone /tmp/source.bundle /source",
                  f"git -C /source checkout --detach {SOURCE_COMMIT}",
                  "rm /tmp/source.bundle")
)
MOUNTS = {"/parents": parents, "/results": results}


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes=MOUNTS, timeout=3600)
def verify():
    root = Path("/parents/exp1-pilot-parents")
    manifest = root / "manifest.json"
    count = 0
    inventory = set()
    for line in (root / "sha256.txt").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if relative in inventory:
            raise ValueError("duplicate checksum path")
        inventory.add(relative)
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("checksum path escapes parent directory")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {relative}")
        count += 1
    if count == 0:
        raise ValueError("empty checksum inventory")
    rows = json.loads(manifest.read_text())["runs"]
    if len(rows) != 20:
        raise ValueError("expected 20 parent identities")
    for row in rows:
        run = Path(row["architecture"]) / row["environment"] / f"seed_{row['seed']}"
        state = run / "state" / row["state"]
        required = [run / "DONE", run / "run_metadata.json", run / "state/LATEST",
                    *(state / name for name in ("meta.pkl", "buffer.npz", "buffer_meta.pkl", "obs_rms.pkl"))]
        if any(path.as_posix() not in inventory for path in required):
            raise ValueError("checksum inventory omits required parent file")
        if not any(name.startswith((state / "agent_ckpt").as_posix() + "/") for name in inventory):
            raise ValueError("checksum inventory omits agent checkpoint")
    receipt = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "all_files_ok": True, "files": count, "verified_unix": time.time()}
    (root / "verification.json").write_text(json.dumps(receipt, indent=2))
    parents.commit()
    return receipt


@app.function(image=image, gpu="L4", cpu=8, memory=49152, volumes=MOUNTS,
              secrets=[modal.Secret.from_name("exp2-wandb")], timeout=23 * 3600,
              max_containers=1, retries=0)
def qualify(budget_usd: float = 10.0):
    """Run a single-worker, separately restarted qualification under a reservation."""
    if not 0 < budget_usd <= 40:
        raise ValueError("spending ceiling exceeded")
    manifest = Path("/parents/exp1-pilot-parents/manifest.json")
    verification = json.loads(manifest.with_name("verification.json").read_text())
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if verification.get("all_files_ok") is not True or verification.get("manifest_sha256") != digest:
        raise ValueError("verify uploaded parents first")
    output = Path("/results/replay-pilot")
    output.mkdir(parents=True, exist_ok=True)
    qualification = output / "qualification"
    qualification.mkdir(exist_ok=True)
    final = output / "l4-qualification.json"
    if final.exists():
        raise ValueError("qualification receipt already exists; review it before replacing")
    ledger = output / "budget.json"
    old = json.loads(ledger.read_text()) if ledger.exists() else {"reserved_usd": 2.0}
    per_second = 0.000222 + 8 * 0.0000131 + 48 * 0.00000222
    seconds = min(22 * 3600, int((budget_usd - old["reserved_usd"]) / per_second))
    if seconds < 300:
        raise ValueError("remaining reserved budget below five minutes")
    old["reserved_usd"] += seconds * per_second
    ledger.write_text(json.dumps(old, indent=2))
    results.commit()
    row = json.loads(manifest.read_text())["runs"][0]
    identity = {"environment": row["environment"], "seed": row["seed"],
                "reference_regime": "reference_small" if row["architecture"] == "D2W512" else "reference_large"}
    env = dict(os.environ, PYTHONPATH="/source/main", JAX_PLATFORMS="cuda,cpu",
               MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="0", WANDB_MODE="online",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    deadline = time.monotonic() + seconds
    try:
        for phase in ("interrupt", "resume"):
            if phase == "interrupt" and (qualification / "qualification-interruption.json").exists():
                continue
            command = ["python", "-m", "experiments.exp12.replay_exp2_qualification",
                       "--parent-manifest", str(manifest), "--output", str(qualification),
                       "--identity", json.dumps(identity), "--phase", phase]
            with (qualification / "process.log").open("a") as log:
                subprocess.run(command, cwd="/source/main", env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True,
                               timeout=max(1, deadline - time.monotonic()))
            results.commit()
        receipt = json.loads((qualification / "qualification-result.json").read_text())
        if receipt.get("source_commit") != SOURCE_COMMIT or receipt.get("manifest_sha256") != digest:
            raise ValueError("qualification evidence does not match source and manifest")
        final.write_text(json.dumps(receipt, indent=2))
    finally:
        results.commit()
    return receipt


@app.function(image=image, gpu="L4", cpu=8, memory=49152, volumes=MOUNTS,
              secrets=[modal.Secret.from_name("exp2-wandb")], timeout=23 * 3600,
              max_containers=1, retries=0)
def train(max_workers: int = 7, budget_usd: float = 40.0):
    """A session reserves its maximum estimated charge before starting work."""
    if not 1 <= max_workers <= 7 or not 0 < budget_usd <= 40:
        raise ValueError("worker or spending ceiling exceeded")
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
    qualification_path = output / "l4-qualification.json"
    if not qualification_path.exists():
        raise ValueError("actual L4 restore/branch/probe/restart qualification is required")
    qualification = json.loads(qualification_path.read_text())
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if (qualification.get("source_commit") != SOURCE_COMMIT
            or qualification.get("manifest_sha256") != manifest_hash
            or qualification.get("gpu_model") != "NVIDIA L4"
            or not all(qualification.get(key) is True for key in
                       ("parent_restore", "full_branch_horizon", "plasticity_probe", "restart"))):
        raise ValueError("incomplete or stale L4 qualification")
    qualified_workers = qualification.get("qualified_workers", 0)
    if not isinstance(qualified_workers, int) or not 1 <= qualified_workers <= 7:
        raise ValueError("invalid qualified worker count")
    max_workers = min(max_workers, qualified_workers)
    # Published per-second GPU + requested CPU + requested host-memory rates.
    # Deliberately bill all requested resources for the full session, even idle.
    per_second = 0.000222 + 8 * 0.0000131 + 48 * 0.00000222
    reserve = 2.0  # CPU-only verification/build/storage and shutdown allowance.
    ledger = output / "budget.json"
    old = json.loads(ledger.read_text()) if ledger.exists() else {"reserved_usd": reserve}
    seconds = min(22 * 3600, int((budget_usd - old["reserved_usd"]) / per_second))
    if seconds < 300:
        raise ValueError("remaining reserved budget below five minutes")
    old["reserved_usd"] += seconds * per_second
    ledger.write_text(json.dumps(old, indent=2))
    results.commit()
    deadline = time.monotonic() + seconds
    rows = json.loads(manifest.read_text())["runs"]
    pending = [dict(environment=r["environment"], seed=r["seed"],
                    reference_regime="reference_small" if r["architecture"] == "D2W512" else "reference_large")
               for r in rows]
    active = []
    env = dict(os.environ, PYTHONPATH="/source/main", JAX_PLATFORMS="cuda,cpu",
               MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="0", WANDB_MODE="online",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    last_commit = time.monotonic()
    try:
        while (pending or active) and time.monotonic() < deadline - 120:
            while pending and len(active) < max_workers:
                identity = pending.pop(0)
                name = f"{identity['environment']}_seed{identity['seed']}_{identity['reference_regime']}"
                directory = output / name
                directory.mkdir(exist_ok=True)
                if (directory / "analysis.json").exists():
                    completed = json.loads((directory / "analysis.json").read_text())
                    if (completed.get("source_commit") != SOURCE_COMMIT
                            or completed.get("protocol") != "exp1-final-replay-v1"
                            or completed.get("environment") != identity["environment"]
                            or completed.get("training_seed") != identity["seed"]
                            or completed.get("reference_regime") != identity["reference_regime"]):
                        raise ValueError("completed run identity mismatch; use new output root")
                    continue
                log = (directory / "process.log").open("a")
                command = ["python", "-m", "experiments.exp12.replay_exp2_pilot",
                           "--parent-manifest", str(manifest), "--output", str(directory),
                           "--identity", json.dumps(identity)]
                proc = subprocess.Popen(command, env=env, cwd="/source/main", stdout=log,
                                        stderr=subprocess.STDOUT)
                active.append((proc, log, identity, directory))
                print("START", name, flush=True)
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
                    print("EXIT", identity, code, flush=True)
            if time.monotonic() - last_commit >= 60:
                results.commit()
                last_commit = time.monotonic()
            time.sleep(2)
    finally:
        for proc, log, _, _ in active:
            proc.terminate()
        for proc, log, _, _ in active:
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
        results.commit()
    return {"pending": len(pending), "interrupted": len(active), "budget": old}


@app.local_entrypoint()
def main(mode: str = "verify"):
    if mode == "verify":
        print(verify.remote())
    elif mode == "qualify":
        print(qualify.remote())
    elif mode == "train":
        print(train.remote())
    else:
        raise ValueError("mode must be verify, qualify or train")
