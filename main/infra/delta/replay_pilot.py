"""Run the verified replay pilot within one Delta Slurm allocation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def gpu_inventory():
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free",
         "--format=csv,noheader,nounits"], text=True,
    ).splitlines()
    return {uuid: (name, int(total), int(free))
            for uuid, name, total, free in
            (tuple(part.strip() for part in row.split(",")) for row in rows)}


def run(args, identities, source_commit):
    if not identities:
        return 0
    inventory = gpu_inventory()
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    gpus = [gpu if gpu in inventory else list(inventory)[int(gpu)] for gpu in visible]
    if len(set(gpus)) != 2 or any(
        "A100" not in inventory[gpu][0] or not 38000 <= inventory[gpu][1] <= 45000
        for gpu in gpus
    ):
        raise RuntimeError("expected two allocated A100 40 GB devices")
    active = {}
    log_start = {}
    pending = list(identities)
    ceilings = dict.fromkeys(gpus, args.workers_per_gpu)
    next_start = dict.fromkeys(gpus, 0)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGINT, stop)
    env = dict(os.environ, JAX_PLATFORMS="cuda", MUJOCO_GL="egl",
               PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID="0",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    try:
        while (active or pending) and not stopping:
            for proc, (gpu, identity, log, root) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                del active[proc]
                if code:
                    with (root / "process.log").open("rb") as stream:
                        end = stream.seek(0, 2)
                        stream.seek(max(log_start[root], end - 65536))
                        tail = stream.read().lower()
                    if ceilings[gpu] > 1 and (
                        b"resource_exhausted" in tail or b"out of memory" in tail
                    ):
                        ceilings[gpu] -= 1
                        pending.insert(0, identity)
                        print(f"{gpu}: reducing worker limit to {ceilings[gpu]}", flush=True)
                    else:
                        raise RuntimeError(f"{root.name} exited {code}. See {root}/process.log")
                else:
                    validate_completed(root, identity, source_commit)
            if not pending:
                if active:
                    time.sleep(2)
                continue
            inventory = gpu_inventory()
            for gpu in gpus:
                if stopping or not pending:
                    break
                workers = sum(entry[0] == gpu for entry in active.values())
                if workers >= ceilings[gpu] or time.monotonic() < next_start[gpu]:
                    continue
                if inventory[gpu][2] < args.min_free_mib:
                    if workers == 0:
                        raise RuntimeError(f"insufficient GPU memory on idle device {gpu}")
                    continue
                identity = pending.pop(0)
                root = args.results / run_name(identity)
                root.mkdir(exist_ok=True)
                log = (root / "process.log").open("ab")
                log_start[root] = log.tell()
                proc = subprocess.Popen(
                    [sys.executable, "-m", "experiments.exp12.replay_exp2_pilot",
                     "--parent-manifest", str(args.manifest), "--output", str(root),
                     "--identity", json.dumps(identity)],
                    stdout=log, stderr=subprocess.STDOUT,
                    env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                )
                active[proc] = gpu, identity, log, root
                next_start[gpu] = time.monotonic() + 30
                print(f"Started {root.name} on {gpu}", flush=True)
            time.sleep(2)
    finally:
        for proc in active:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 45
        for proc, (_, _, log, _) in active.items():
            try:
                proc.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
    return 75 if stopping else 0


def run_name(identity):
    return "{environment}_seed{seed}_{reference_regime}".format(**identity)


def validate_completed(root, identity, source_commit):
    completed = json.loads((root / "analysis.json").read_text())
    expected = dict(source_commit=source_commit, protocol="exp1-final-replay-v1",
                    environment=identity["environment"], training_seed=identity["seed"],
                    reference_regime=identity["reference_regime"])
    if any(completed.get(k) != v for k, v in expected.items()):
        raise ValueError(f"completed run identity mismatch: {root}")


def main():
    from experiments.exp12.replay_exp2_contract import admit_parent
    from experiments.exp12.state import latest_state_dir

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--min-free-mib", type=int, default=11264)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.manifest, args.results = args.manifest.resolve(), args.results.resolve()
    identities = []
    for row in json.loads(args.manifest.read_text())["runs"]:
        regime = {"D2W512": "reference_small", "D4W1536": "reference_large"}[row["architecture"]]
        identity = dict(environment=row["environment"], seed=row["seed"], reference_regime=regime)
        admit_parent(args.manifest, identity)
        identities.append(identity)
    if len(identities) != 20 or len({run_name(i) for i in identities}) != 20:
        raise ValueError("expected 20 unique admitted pilot identities")
    if args.preflight:
        print("Admitted 20 verified parents")
        return 0
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]).strip():
        raise ValueError("launch requires a clean pinned source checkout")
    args.results.mkdir(parents=True, exist_ok=True)
    with (args.results / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for identity in identities:
            root = args.results / run_name(identity)
            analysis = root / "analysis.json"
            state = latest_state_dir(root / "state")
            if state is not None:
                saved = json.loads((state / "online_exp2_manifest.json").read_text())
                if saved["source_commit"] != source_commit:
                    raise ValueError(f"{root}: checkpoint requires verified source migration")
            elif not analysis.exists() and identity["reference_regime"] == "reference_small" and (
                identity["environment"] in {"dog-run", "humanoid-run"} and identity["seed"] in {1, 2, 3}
                or identity["environment"] == "myo-key-turn" and identity["seed"] in {1, 2}
            ):
                raise ValueError(f"{root}: expected migrated Modal checkpoint")
            if not analysis.exists():
                pending.append(identity)
            else:
                validate_completed(analysis.parent, identity, source_commit)
        return run(args, pending, source_commit)


if __name__ == "__main__":
    sys.exit(main())
