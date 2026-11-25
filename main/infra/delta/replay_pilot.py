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


PILOT_ENVIRONMENTS = {
    "dog-run": (1, 2, 3),
    "humanoid-run": (1, 2, 3),
    "myo-key-turn": (1, 2),
    "myo-pen-twirl": (1, 2),
}
REGIMES = {"D2W512": "reference_small", "D4W1536": "reference_large"}


def identities_from_manifest(manifest):
    """Require ten paired Exp1 parents across the four established pilot tasks."""
    rows = manifest.get("runs")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("expected exactly 20 Exp1 parent rows (ten matched pairs)")
    paired, seen = {}, set()
    for row in rows:
        environment, seed = row.get("environment"), row.get("seed")
        architecture = row.get("architecture")
        if environment not in PILOT_ENVIRONMENTS or architecture not in REGIMES:
            raise ValueError("parent manifest contains an unexpected environment or architecture")
        if type(seed) is not int or seed not in PILOT_ENVIRONMENTS[environment]:
            raise ValueError("parent manifest contains an unexpected seed")
        key, regime = (environment, seed), REGIMES[architecture]
        if (key, regime) in seen:
            raise ValueError("duplicate parent identity in manifest")
        seen.add((key, regime))
        paired.setdefault(key, set()).add(regime)
    expected = {(env, seed) for env, seeds in PILOT_ENVIRONMENTS.items() for seed in seeds}
    if set(paired) != expected or any(
        regimes != {"reference_small", "reference_large"} for regimes in paired.values()
    ):
        raise ValueError("manifest must contain both reference regimes for all ten matched pairs")
    return [
        {"environment": env, "seed": seed, "reference_regime": regime}
        for env, seeds in PILOT_ENVIRONMENTS.items()
        for seed in seeds
        for regime in ("reference_small", "reference_large")
    ]


def identities_for_shard(identities, shard_index, shard_count):
    """Split matched environment/seed pairs deterministically across job shards."""
    if type(shard_count) is not int or shard_count < 1:
        raise ValueError("shard count must be positive")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be between zero and shard_count - 1")
    pairs = {}
    for identity in identities:
        pairs.setdefault((identity["environment"], identity["seed"]), []).append(identity)
    ordered = list(pairs.values())
    if any(len(pair) != 2 or {row["reference_regime"] for row in pair} !=
           {"reference_small", "reference_large"} for pair in ordered):
        raise ValueError("sharding requires complete small/large parent pairs")
    return [identity for pair_index, pair in enumerate(ordered)
            if pair_index % shard_count == shard_index for identity in pair]


def gpu_inventory():
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free",
         "--format=csv,noheader,nounits"], text=True,
    ).splitlines()
    return {uuid: (name, int(total), int(free))
            for uuid, name, total, free in
            (tuple(part.strip() for part in row.split(",")) for row in rows)}


def shard_status(results, identities):
    """Summarize complete, resumable, and not-yet-started identities."""
    from experiments.exp12.state import latest_state_dir

    records = []
    for identity in identities:
        root = results / run_name(identity)
        if (root / "analysis.json").is_file():
            state = "complete"
        elif latest_state_dir(root / "state") is not None:
            state = "resumable"
        else:
            state = "not_started"
        records.append({**identity, "status": state})
    return records


def run(args, identities, source_commit):
    if not identities:
        return 0
    inventory = gpu_inventory()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not visible or visible == [""]:
        visible = [str(i) for i in range(len(inventory))]
    gpus = [gpu if gpu in inventory else list(inventory)[int(gpu)] for gpu in visible]
    if len(set(gpus)) != args.gpu_count:
        raise RuntimeError(f"expected {args.gpu_count} allocated GPUs, found {len(set(gpus))}")
    for gpu in gpus:
        name, memory = inventory[gpu][0], inventory[gpu][1]
        if args.gpu_model == "A100" and ("A100" not in name or not 38000 <= memory <= 45000):
            raise RuntimeError(f"expected an A100 40 GB, found {name} ({memory} MiB)")
        if args.gpu_model == "L40" and ("L40" not in name or "L40S" in name
                                         or not 44000 <= memory <= 50000):
            raise RuntimeError(f"expected an L40 48 GB, found {name} ({memory} MiB)")
    active = {}
    log_start = {}
    pending = list(identities)
    ceilings = dict.fromkeys(gpus, args.workers_per_gpu)
    next_start = dict.fromkeys(gpus, 0)
    stopping = False
    stop_reason = None
    started = time.monotonic()

    def stop(*_, reason="signal"):
        nonlocal stopping, stop_reason
        stopping = True
        stop_reason = reason

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGINT, stop)
    env = dict(os.environ, JAX_PLATFORMS="cuda", MUJOCO_GL="egl",
               PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID="0",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    try:
        while (active or pending) and not stopping:
            if args.max_wall_seconds and time.monotonic() - started >= args.max_wall_seconds:
                stop(reason="runtime_limit")
                print("Runtime slice reached; stopping dispatch and saving active workers", flush=True)
                break
            for proc, (gpu, identity, log, root) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                del active[proc]
                if code == 75 and stopping:
                    print(f"Saved resumable state for {root.name} at the wall-time stop", flush=True)
                elif code:
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
                    validate_completed(root, identity, source_commit, args.heldout)
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
                module = ("experiments.exp12.replay_exp2_heldout" if args.heldout
                          else "experiments.exp12.replay_exp2_pilot")
                command = [sys.executable, "-m", module,
                           "--parent-manifest", str(args.manifest), "--output", str(root),
                           "--identity", json.dumps(identity)]
                if args.heldout:
                    command.extend(("--checkpoint-count", str(args.checkpoint_count)))
                proc = subprocess.Popen(
                    command,
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
                proc.send_signal(signal.SIGUSR1)
        deadline = time.monotonic() + 90
        for proc, (_, _, log, _) in active.items():
            try:
                proc.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            log.close()
    records = shard_status(args.results, identities)
    from utils.atomic_io import atomic_write_text
    atomic_write_text(args.results / "slice_status.json", json.dumps({
        "source_commit": source_commit,
        "protocol": "exp2-heldout-v1" if args.heldout else "exp1-final-replay-v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "requested_runtime_seconds": args.max_wall_seconds,
        "stop_reason": stop_reason,
        "status": "complete" if all(row["status"] == "complete" for row in records)
                  else "partial",
        "runs": records,
    }, indent=2))
    return 0 if stop_reason == "runtime_limit" else (75 if stopping else 0)


def run_name(identity):
    return "{environment}_seed{seed}_{reference_regime}".format(**identity)


def validate_completed(root, identity, source_commit, heldout=False):
    completed = json.loads((root / "analysis.json").read_text())
    expected = dict(source_commit=source_commit,
                    protocol="exp2-heldout-v1" if heldout else "exp1-final-replay-v1",
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
    parser.add_argument("--workers-per-gpu", type=int, choices=range(1, 8), default=6)
    parser.add_argument("--gpu-model", choices=("A100", "L40"), default="A100")
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--max-wall-seconds", type=int, default=0,
                        help="stop cleanly after this runtime; zero means no internal limit")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--checkpoint-count", type=int, choices=(15, 20), default=15)
    parser.add_argument("--min-free-mib", type=int, default=11264)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--heldout", action="store_true",
                        help="run the separate held-out diagnostic protocol")
    parser.add_argument("--fresh", action="store_true", help="start a new pilot in an unused results directory")
    args = parser.parse_args()
    args.manifest, args.results = args.manifest.resolve(), args.results.resolve()
    all_identities = identities_from_manifest(json.loads(args.manifest.read_text()))
    for identity in all_identities:
        admit_parent(args.manifest, identity)
    identities = identities_for_shard(all_identities, args.shard_index, args.shard_count)
    if args.preflight:
        print(f"Admitted 20 parents in ten matched pairs; shard {args.shard_index + 1}/"
              f"{args.shard_count} contains {len(identities)} runs")
        return 0
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]).strip():
        raise ValueError("launch requires a clean pinned source checkout")
    if args.fresh:
        args.results.mkdir(parents=True, exist_ok=False)
        marker_data = {"source_commit": source_commit, "manifest": str(args.manifest)}
        if args.heldout:
            marker_data["protocol"] = "exp2-heldout-v1"
            if args.shard_count > 1:
                marker_data.update(shard_count=args.shard_count, shard_index=args.shard_index)
        (args.results / ".fresh-pilot.json").write_text(json.dumps(marker_data, sort_keys=True))
    else:
        args.results.mkdir(parents=True, exist_ok=True)
    with (args.results / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = args.results / ".fresh-pilot.json"
        expected_marker = {"source_commit": source_commit, "manifest": str(args.manifest)}
        if args.heldout:
            expected_marker["protocol"] = "exp2-heldout-v1"
            if args.shard_count > 1:
                expected_marker.update(shard_count=args.shard_count, shard_index=args.shard_index)
        fresh_pilot = marker.is_file() and json.loads(marker.read_text()) == expected_marker
        if marker.exists() and not fresh_pilot:
            raise ValueError("fresh pilot marker does not match source and manifest")
        if args.heldout and not fresh_pilot:
            raise ValueError("held-out pilot requires a new output directory started with --fresh")
        pending = []
        for identity in identities:
            root = args.results / run_name(identity)
            analysis = root / "analysis.json"
            state = latest_state_dir(root / "state")
            if state is not None:
                saved = json.loads((state / "online_exp2_manifest.json").read_text())
                if saved["source_commit"] != source_commit:
                    raise ValueError(f"{root}: checkpoint requires verified source migration")
            elif not args.heldout and not fresh_pilot and not analysis.exists() and identity["reference_regime"] == "reference_small" and (
                identity["environment"] in {"dog-run", "humanoid-run"} and identity["seed"] in {1, 2, 3}
                or identity["environment"] == "myo-key-turn" and identity["seed"] in {1, 2}
            ):
                raise ValueError(f"{root}: expected migrated Modal checkpoint")
            if not analysis.exists():
                pending.append(identity)
            else:
                validate_completed(analysis.parent, identity, source_commit, args.heldout)
        return run(args, pending, source_commit)


if __name__ == "__main__":
    sys.exit(main())
