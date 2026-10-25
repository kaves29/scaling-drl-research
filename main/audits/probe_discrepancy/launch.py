"""Fail-closed two-GPU pilot controller. No subcommand submits a Slurm job."""

import argparse
import datetime as dt
import hashlib
import json
import os
import importlib.metadata
import platform
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

from audits.probe_discrepancy.plan import plan_hash, validate_design

GATES = (
    "adapter_review",
    "scientific_review",
    "cpu_restoration_equivalence",
    "cuda_smoke",
    "five_process_per_gpu",
    "storage_profile",
    "live_delta_layout",
    "wandb_identities",
    "allocation_review",
    "owner_launch",
)
WRAPPER = "main/scripts/sbatch_echocritic_diagnostic_pilot.sh"
LAUNCHER = "main/audits/probe_discrepancy/launch.py"
SOURCE_FILES = (
    WRAPPER,
    LAUNCHER,
    "main/audits/probe_discrepancy/plan.py",
    "main/audits/probe_discrepancy/source_config_bindings.json",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def require(ok, message):
    if not ok:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and 0 < value < float("inf")


def bound_evidence(binding):
    path = Path(binding["path"])
    require(
        path.is_absolute() and path.is_file() and sha(path) == binding["sha256"],
        "missing or changed gate evidence",
    )
    return read(path)


def preflight(manifest_path, approval_path, checkout, *, check_git=True):
    """Validate approvals/evidence before command printing and again in allocation."""
    m, a, checkout = read(manifest_path), read(approval_path), Path(checkout).resolve()
    validate_design(m)
    source = a.get("expected_commit", "")
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", source)),
        "exact executable source SHA required",
    )
    require(
        m["diagnostic_implementation_source"] == source,
        "diagnostic adapter/source is not frozen",
    )
    require(
        a.get("approved") is True and a.get("decision_reference"),
        "explicit owner approval required",
    )
    require(
        a.get("manifest_sha256") == sha(manifest_path), "approval/manifest mismatch"
    )
    require(a.get("plan_sha256") == plan_hash(m), "approval/scientific-plan mismatch")
    for path in SOURCE_FILES:
        require(
            a.get("source_files", {}).get(path) == sha(checkout / path),
            "launcher source hash mismatch",
        )
    if check_git:
        require(
            subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
            == source,
            "wrong checkout source",
        )
        require(
            not subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ],
                text=True,
            ).strip(),
            "checkout must be clean",
        )
        require(
            Path(__file__).resolve() == checkout / LAUNCHER,
            "run the launcher from the approved checkout",
        )
    adapter = a.get("adapter", {})
    entry = Path(adapter.get("entrypoint", ""))
    require(
        not entry.is_absolute() and ".." not in entry.parts and entry.suffix == ".py",
        "reviewed adapter entrypoint required",
    )
    require(
        (checkout / entry).resolve() == checkout / entry,
        "adapter must be inside the reviewed checkout without symlink redirection",
    )
    require(
        adapter.get("protocol") == "echocritic-observation-only-v1"
        and adapter.get("sha256") == sha(checkout / entry),
        "unimplemented or unreviewed diagnostic adapter",
    )
    runtime = a.get("runtime", {})
    require(
        Path(runtime.get("python", "")).is_absolute()
        and Path(runtime["python"]).is_file(),
        "pinned Python executable required",
    )
    require(
        set(runtime.get("versions", {}))
        >= {"python", "jax", "jaxlib", "optax", "cuda", "driver"},
        "runtime fingerprint incomplete",
    )
    allocation = m["allocation_proposal"]
    require(
        a.get("allocation") == allocation, "allocation differs from reviewed manifest"
    )
    require(
        (allocation["nodes"], allocation["gpus"], allocation["ntasks"]) == (1, 2, 1),
        "one-node two-GPU controller layout required",
    )
    require(
        all(
            type(allocation[k]) is int and allocation[k] > 0
            for k in ("cpus", "host_memory_GiB", "wall_minutes")
        ),
        "allocation resources incomplete",
    )
    require(
        allocation["cpus"] >= 32, "thirty child CPUs plus controller headroom required"
    )
    for value in (
        allocation["account"],
        allocation["partition"],
        a.get("node_name", ""),
        a.get("wandb_namespace", ""),
        a.get("wandb_project", ""),
    ):
        require(
            bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value)),
            "safe node/account/project/namespace identity required",
        )
    root = Path(m["new_campaign_root"] or "")
    require(
        root.is_absolute()
        and root != Path("/")
        and root.resolve() == root
        and "campaign_owner_f597ee9" not in str(root),
        "new distinct absolute output root required",
    )
    require(
        a.get("new_campaign_root") == str(root) and root.is_dir(),
        "fresh output root must exist and be approval-bound",
    )
    limits = a.get("limits", {})
    for key in (
        "gpu_headroom_MiB",
        "host_headroom_GiB",
        "storage_headroom_GiB",
        "barrier_timeout_seconds",
        "max_exec_skew_seconds",
        "stop_margin_seconds",
        "telemetry_interval_seconds",
        "idle_gpu_memory_limit_MiB",
    ):
        require(positive(limits.get(key)), "missing reviewed operational limit: " + key)
    require(
        limits["stop_margin_seconds"] < allocation["wall_minutes"] * 60,
        "stop margin consumes allocation",
    )
    gates = {}
    for name in GATES:
        require(name in a.get("gates", {}), "missing gate: " + name)
        r = bound_evidence(a["gates"][name])
        require(
            r.get("verdict") == "PASS" and r.get("decision_reference"),
            "gate not reviewed PASS: " + name,
        )
        require(
            r.get("expected_commit") == source and r.get("plan_sha256") == plan_hash(m),
            "stale source/plan gate: " + name,
        )
        require(
            r.get("runtime") == runtime and r.get("allocation") == allocation,
            "runtime/layout gate mismatch: " + name,
        )
        require(r.get("raw_evidence"), "gate lacks underlying evidence: " + name)
        for item in r["raw_evidence"]:
            require(
                Path(item["path"]).is_absolute()
                and sha(item["path"]) == item["sha256"],
                "changed underlying evidence: " + name,
            )
        gates[name] = r
    packing = gates["five_process_per_gpu"]
    require(
        packing.get("simultaneous_all_ten") is True
        and packing.get("max_concurrent_processes_per_gpu") == [5, 5]
        and packing.get("waves") == 1,
        "representative simultaneous five/five qualification missing",
    )
    require(
        packing.get("run_config_hashes")
        == sorted(r["source_scientific_config_sha256"] for r in m["runs"]),
        "packing qualified a different mixture/configuration",
    )
    for phase in (
        "cold_compile",
        "steady_training",
        "paired_full_probe",
        "checkpoint_io",
        "binding_verified",
        "on_off_equivalence",
    ):
        require(packing.get(phase) is True, "packing phase not qualified: " + phase)
    require(packing.get("node_name") == a["node_name"], "different qualification node")
    uuids = packing.get("gpu_uuids", [])
    require(
        len(uuids) == len(set(uuids)) == 2
        and all(str(x).startswith("GPU-") for x in uuids),
        "qualified physical GPU identities required",
    )
    for key in ("peak_gpu_memory_MiB", "gpu_total_memory_MiB"):
        require(
            len(packing.get(key, [])) == 2 and all(positive(v) for v in packing[key]),
            "missing measured VRAM envelope",
        )
    require(
        all(
            used + limits["gpu_headroom_MiB"] < total
            for used, total in zip(
                packing["peak_gpu_memory_MiB"], packing["gpu_total_memory_MiB"]
            )
        ),
        "five-process VRAM envelope fails",
    )
    require(
        positive(packing.get("peak_host_memory_GiB"))
        and packing["peak_host_memory_GiB"] + limits["host_headroom_GiB"]
        < allocation["host_memory_GiB"],
        "host RAM envelope fails",
    )
    require(
        positive(packing.get("projected_endpoint_seconds"))
        and packing["projected_endpoint_seconds"] + limits["stop_margin_seconds"]
        < allocation["wall_minutes"] * 60,
        "projected full endpoints do not fit walltime",
    )
    require(
        positive(packing.get("minimum_interactions_per_second"))
        and positive(packing.get("paired_probe_seconds"))
        and positive(packing.get("cold_compile_seconds")),
        "compile/steady-state throughput absent",
    )
    live = gates["live_delta_layout"]
    expiry = dt.datetime.fromisoformat(live["valid_until_utc"].replace("Z", "+00:00"))
    require(
        expiry > dt.datetime.now(dt.timezone.utc)
        and live.get("node_name") == a["node_name"],
        "live Delta layout evidence expired/mismatched",
    )
    require(
        live.get("scheduler_layout_verified") is True
        and live.get("cpu_memory_cgroup_verified") is True
        and live.get("storage_verified") is True,
        "actual Delta layout/resources unverified",
    )
    require(
        m["launch_approved"] is True
        and m["executable"] is True
        and m["resource_estimate"]["packing_qualified"] is True,
        "draft flags remain closed; no command or launch",
    )
    return m, a, gates


def sbatch_argv(manifest, approval, checkout):
    m, a, _ = preflight(manifest, approval, checkout)
    r = m["allocation_proposal"]
    return [
        "sbatch",
        "--parsable",
        f"--account={r['account']}",
        f"--partition={r['partition']}",
        f"--nodelist={a['node_name']}",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:2",
        f"--cpus-per-task={r['cpus']}",
        f"--mem={r['host_memory_GiB']}G",
        f"--time={r['wall_minutes']}",
        "--job-name=echocritic-diagnostic-10",
        f"--chdir={Path(checkout).resolve()}/main",
        f"--output={m['new_campaign_root']}/slurm-%j.out",
        f"--error={m['new_campaign_root']}/slurm-%j.err",
        str(Path(checkout).resolve() / WRAPPER),
        a["runtime"]["python"],
        str(Path(manifest).resolve()),
        str(Path(approval).resolve()),
        sha(manifest),
        sha(approval),
    ]


def capture(command):
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {
            "argv": command,
            "returncode": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": command, "error": str(exc)}


def inspect():
    """Read-only inventory, useful on Delta login and again inside the allocation."""
    return {
        "hostname": os.uname().nodename,
        "cpu_topology": capture(["lscpu", "--json"]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "meminfo": Path("/proc/meminfo").read_text(),
        "cgroup": {
            p: Path(p).read_text()
            for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpu.max")
            if Path(p).exists()
        },
        "scheduler": [
            capture(["sinfo", "-N", "-p", "gpuA100x4", "-o", "%N %G %c %m %T"]),
            capture(["scontrol", "show", "partition", "gpuA100x4"]),
        ],
        "gpus": capture(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def gpu_info(token):
    result = capture(
        [
            "nvidia-smi",
            "-i",
            token,
            "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    require(result.get("returncode") == 0, "cannot verify allocated GPU")
    fields = [s.strip() for s in result["stdout"].strip().split(",")]
    require(len(fields) == 5, "unexpected GPU topology")
    return {
        "uuid": fields[0],
        "name": fields[1],
        "total_MiB": float(fields[2]),
        "used_MiB": float(fields[3]),
        "utilization": float(fields[4]),
    }


def allocated_tokens(environ):
    tokens = environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    require(
        len(tokens) == 2
        and len(set(tokens)) == 2
        and all(re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", t) for t in tokens),
        "allocation must expose exactly two unique GPU tokens; no guessed global ordinals",
    )
    return tokens


def child_environment(m, a, row, tokens):
    env = os.environ.copy()
    for name in (
        "JAX_PLATFORM_NAME",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "NVIDIA_TF32_OVERRIDE",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
    ):
        env.pop(name, None)
    env.update(m["process_environment"])
    env.update(
        CUDA_VISIBLE_DEVICES=tokens[row["gpu"]],
        WANDB_MODE="online",
        WANDB_RUN_ID=a["wandb_namespace"] + "-" + row["wandb_id"],
        WANDB_PROJECT=a["wandb_project"],
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        WANDB_DIR=str(Path(m["new_campaign_root"]) / row["output_relative"]),
        WANDB_RESUME="never",
    )
    return env


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def child(args):
    m, a, _ = preflight(args.manifest, args.approval, args.checkout)
    row = next(r for r in m["runs"] if r["pilot_id"] == args.run_id)
    os.sched_setaffinity(0, set(int(c) for c in args.cpu_list.split(",")))
    require(
        len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1,
        "child must see one GPU before JAX import",
    )
    import jax  # Deliberately after per-child GPU binding and approval checks.

    require(len(jax.devices("gpu")) == 1, "child JAX backend must expose one GPU")
    require(
        jax.devices("gpu")[0].device_kind == m["allocation_proposal"]["gpu_model"],
        "JAX GPU model mismatch",
    )
    versions = a["runtime"]["versions"]
    require(platform.python_version() == versions["python"], "Python runtime changed")
    for package in ("jax", "jaxlib", "optax"):
        require(
            importlib.metadata.version(package) == versions[package],
            package + " runtime changed",
        )
    require(
        jax.devices("gpu")[0].client.platform_version == versions["cuda"],
        "CUDA backend changed",
    )
    driver = capture(
        [
            "nvidia-smi",
            "-i",
            os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    require(
        driver.get("returncode") == 0
        and driver["stdout"].strip() == versions["driver"],
        "GPU driver changed",
    )
    barrier = Path(args.barrier)
    atomic(
        barrier / f"{args.run_id}.ready.json",
        {
            "pid": os.getpid(),
            "gpu": row["gpu"],
            "visible_token": os.environ["CUDA_VISIBLE_DEVICES"],
            "time": time.time(),
            "jax": jax.__version__,
        },
    )
    deadline = time.monotonic() + a["limits"]["barrier_timeout_seconds"]
    while not (barrier / "GO").exists():
        require(
            time.monotonic() < deadline and not (barrier / "ABORT").exists(),
            "barrier aborted or timed out",
        )
        time.sleep(0.05)
    atomic(
        barrier / f"{args.run_id}.exec.json", {"pid": os.getpid(), "time": time.time()}
    )
    command = [
        a["runtime"]["python"],
        str(Path(args.checkout) / a["adapter"]["entrypoint"]),
        "--pilot-manifest",
        args.manifest,
        "--pilot-run-id",
        args.run_id,
        "--run-dir",
        str(Path(m["new_campaign_root"]) / row["output_relative"]),
        "--wandb-id",
        os.environ["WANDB_RUN_ID"],
        "--expected-commit",
        a["expected_commit"],
    ]
    os.execv(command[0], command)


def process_counters(pid):
    counters = {}
    for name in ("status", "stat", "io"):
        try:
            counters[name] = Path(f"/proc/{pid}/{name}").read_text()
        except (FileNotFoundError, ProcessLookupError):
            pass
    return counters


def cgroup_memory():
    """Finite job/parent cgroup headroom, not just node-wide MemAvailable."""
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    choices = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        v2 = parts[0] == "0" and parts[1] == ""
        if not v2 and "memory" not in parts[1].split(","):
            continue
        mount = Path("/sys/fs/cgroup") if v2 else Path("/sys/fs/cgroup/memory")
        candidate = mount / parts[2].lstrip("/")
        if (
            not candidate.is_dir()
            or mount not in candidate.resolve().parents
            and candidate != mount
        ):
            candidate = mount
        limit_name = "memory.max" if v2 else "memory.limit_in_bytes"
        usage_name = "memory.current" if v2 else "memory.usage_in_bytes"
        for folder in (candidate, *candidate.parents):
            if folder != mount and mount not in folder.parents:
                continue
            limit, usage = folder / limit_name, folder / usage_name
            if limit.exists() and usage.exists():
                text = limit.read_text().strip()
                if text.isdigit():
                    choices.append(
                        {
                            "path": str(folder),
                            "limit_bytes": int(text),
                            "usage_bytes": int(usage.read_text()),
                        }
                    )
    require(bool(choices), "finite memory-cgroup envelope unavailable; fail closed")
    return min(choices, key=lambda x: x["limit_bytes"] - x["usage_bytes"])


def worker(args):
    m, a, gates = preflight(args.manifest, args.approval, args.checkout)
    require(
        args.manifest_sha == sha(args.manifest)
        and args.approval_sha == sha(args.approval),
        "submission/startup files changed",
    )
    require(
        os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_NUM_NODES") == "1",
        "must run inside one-node Slurm allocation",
    )
    require(os.environ.get("SLURMD_NODENAME") == a["node_name"], "wrong allocated node")
    require(
        int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        >= m["allocation_proposal"]["cpus"],
        "wrong CPU allocation",
    )
    require(
        int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
        >= m["allocation_proposal"]["host_memory_GiB"] * 1024,
        "wrong host-memory allocation",
    )
    tokens = allocated_tokens(os.environ)
    gpus = [gpu_info(t) for t in tokens]
    require(
        [g["uuid"] for g in gpus] == gates["five_process_per_gpu"]["gpu_uuids"],
        "allocated GPU identities/order differ from qualification",
    )
    require(
        all(g["name"] == m["allocation_proposal"]["gpu_model"] for g in gpus),
        "wrong GPU model",
    )
    require(
        all(g["used_MiB"] < a["limits"]["idle_gpu_memory_limit_MiB"] for g in gpus),
        "unexpected pre-existing GPU memory use",
    )
    cg = cgroup_memory()
    require(
        cg["limit_bytes"] >= m["allocation_proposal"]["host_memory_GiB"] * 1024**3,
        "actual memory cgroup too small",
    )
    cpus = sorted(os.sched_getaffinity(0))
    require(len(cpus) >= 32, "insufficient actual CPU affinity")
    require(
        cpus == gates["five_process_per_gpu"].get("cpu_affinity"),
        "CPU placement differs from qualification",
    )
    root = Path(m["new_campaign_root"])
    require(
        root.exists()
        and not (root / "pilot_started.json").exists()
        and not (root / "runs").exists(),
        "fresh durable output root required; never resume/relaunch automatically",
    )
    barrier = root / "barrier"
    barrier.mkdir(exist_ok=False)
    atomic(
        root / "pilot_started.json",
        {
            "source": a["expected_commit"],
            "manifest_sha256": sha(args.manifest),
            "approval_sha256": sha(args.approval),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "gpu_tokens": tokens,
            "gpus": gpus,
            "inventory": inspect(),
        },
    )
    processes, files = {}, []
    os.sched_setaffinity(0, set(cpus[30:32]))
    start = time.monotonic()

    def stop(signum, frame):
        raise RuntimeError(f"pilot interrupted by signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        # Spawn all ten once. No completion-driven admission, waves or retries.
        for i, row in enumerate(m["runs"]):
            run_dir = root / row["output_relative"]
            run_dir.mkdir(parents=True, exist_ok=False)
            log = (run_dir / "process.log").open("x")
            files.append(log)
            argv = [
                a["runtime"]["python"],
                "-m",
                "audits.probe_discrepancy.launch",
                "child",
                "--manifest",
                args.manifest,
                "--approval",
                args.approval,
                "--checkout",
                args.checkout,
                "--run-id",
                row["pilot_id"],
                "--barrier",
                str(barrier),
                "--cpu-list",
                ",".join(map(str, cpus[i * 3 : i * 3 + 3])),
            ]
            processes[row["pilot_id"]] = subprocess.Popen(
                argv,
                cwd=Path(args.checkout) / "main",
                env=child_environment(m, a, row, tokens),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + a["limits"]["barrier_timeout_seconds"]
        while len(list(barrier.glob("*.ready.json"))) != 10:
            require(
                all(p.poll() is None for p in processes.values()),
                "child failed before all-ten barrier",
            )
            require(
                time.monotonic() < deadline,
                "all-ten readiness timed out; no wave fallback",
            )
            time.sleep(0.1)
        require(
            all(p.poll() is None for p in processes.values()),
            "child died at all-ten readiness barrier",
        )
        (barrier / "GO").write_text("all ten ready\n")
        exec_deadline = time.monotonic() + a["limits"]["max_exec_skew_seconds"]
        while len(list(barrier.glob("*.exec.json"))) != 10:
            require(
                time.monotonic() < exec_deadline
                and all(p.poll() is None for p in processes.values()),
                "simultaneous exec failed",
            )
            time.sleep(0.02)
        times = [read(p)["time"] for p in barrier.glob("*.exec.json")]
        require(
            max(times) - min(times) <= a["limits"]["max_exec_skew_seconds"],
            "adapter release skew exceeded reviewed bound",
        )
        with (root / "telemetry.jsonl").open("x") as log:
            while True:
                codes = {rid: p.poll() for rid, p in processes.items()}
                require(
                    all(c in (None, 0) for c in codes.values()),
                    "run failed; retain all partial evidence, no replacements",
                )
                if all(c == 0 for c in codes.values()):
                    break
                stats = [gpu_info(t) for t in tokens]
                mem = dict(
                    (line.split(":")[0], int(line.split()[1]))
                    for line in Path("/proc/meminfo").read_text().splitlines()
                )
                free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize / 1024**3
                require(
                    all(
                        g["used_MiB"] + a["limits"]["gpu_headroom_MiB"] < g["total_MiB"]
                        for g in stats
                    ),
                    "GPU headroom exhausted",
                )
                cg = cgroup_memory()
                require(
                    (cg["limit_bytes"] - cg["usage_bytes"]) / 1024**3
                    > a["limits"]["host_headroom_GiB"],
                    "job-cgroup memory headroom exhausted",
                )
                require(
                    mem["MemAvailable"] / 1024**2 > a["limits"]["host_headroom_GiB"]
                    and free > a["limits"]["storage_headroom_GiB"],
                    "host/storage headroom exhausted",
                )
                require(
                    time.monotonic() - start
                    < m["allocation_proposal"]["wall_minutes"] * 60
                    - a["limits"]["stop_margin_seconds"],
                    "allocation walltime guard",
                )
                per_pid = {
                    rid: process_counters(p.pid)
                    for rid, p in processes.items()
                    if p.poll() is None
                }
                log.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "gpu": stats,
                            "MemAvailable_kB_nodewide": mem["MemAvailable"],
                            "filesystem_free_GiB": free,
                            "process_status": per_pid,
                            "memory_cgroup": cg,
                            "diskstats_nodewide": Path("/proc/diskstats").read_text(),
                            "returncodes": codes,
                        }
                    )
                    + "\n"
                )
                log.flush()
                time.sleep(a["limits"]["telemetry_interval_seconds"])
        for row in m["runs"]:
            done = read(root / row["output_relative"] / "diagnostic_completion.json")
            require(
                done.get("expected_commit") == a["expected_commit"]
                and done.get("plan_sha256") == plan_hash(m)
                and done.get("pilot_id") == row["pilot_id"]
                and done.get("config_sha256") == row["source_scientific_config_sha256"]
                and done.get("diagnostics_complete") is True
                and done.get("scientific_endpoint_complete") is True,
                "missing or incomplete per-run diagnostic/endpoint receipt",
            )
        atomic(
            root / "pilot_processes.json",
            {
                "verdict": "ENGINEERING_COMPLETION_ONLY_NOT_SCIENTIFIC_CERTIFICATION",
                "returncodes": codes,
            },
        )
    except BaseException as exc:
        (barrier / "ABORT").write_text(str(exc) + "\n")
        for p in processes.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        deadline = time.monotonic() + 20
        while (
            any(p.poll() is None for p in processes.values())
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        for p in processes.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        atomic(
            root / "pilot_processes.json",
            {
                "verdict": "FAILED_OR_PARTIAL",
                "error": str(exc),
                "returncodes": {k: p.returncode for k, p in processes.items()},
            },
        )
        raise
    finally:
        for log in files:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("preflight", "command", "inspect", "worker", "child")
    )
    parser.add_argument("--manifest")
    parser.add_argument("--approval")
    parser.add_argument("--checkout")
    for name in (
        "manifest-sha",
        "approval-sha",
        "run-id",
        "barrier",
        "cpu-list",
        "out",
    ):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    try:
        if args.action == "inspect":
            with Path(args.out).open("x") as out:
                json.dump(inspect(), out, indent=2)
        elif args.action == "command":
            print(shlex.join(sbatch_argv(args.manifest, args.approval, args.checkout)))
        elif args.action == "preflight":
            preflight(args.manifest, args.approval, args.checkout)
            print("PASS: source-bound launch gates; no job submitted")
        elif args.action == "child":
            child(args)
        else:
            worker(args)
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print("BLOCKED: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
