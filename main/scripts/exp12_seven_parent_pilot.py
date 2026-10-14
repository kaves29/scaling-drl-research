#!/usr/bin/env python3
"""Isolated two-GPU pilot worker over the approved 195-parent durable queue.

The training subprocesses run from a separate, clean checkout of the frozen
scientific commit. This script never creates a queue, changes a parent command,
or infers completion from an exit status alone.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import exp12_dynamic_scheduler as scheduler
from scripts.exp12_launch_plan import allocation_matches, require_pinned_stack, sha

FROZEN = "087feab341b9be1f377697668f2d1c750ec0467f"
PLACEMENT = (
    (0, "D4W1536", "dog-run", 1),
    (0, "D4W1536", "myo-key-turn", 1),
    (1, "D2W512", "humanoid-run", 1),
    (1, "D4W1024", "h1-run-v0", 1),
    (1, "D4W1024", "myo-key-turn", 1),
    (1, "D4W1024", "dog-trot", 1),
    (1, "D4W1024", "humanoid-walk", 1),
)
WALL_MINUTES = 600
GPU_HEADROOM_GB = 8
HOST_HEADROOM_GB = 16
STOP_MARGIN_SECONDS = 120


def git_head(checkout):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()


def require_clean(checkout, expected):
    if git_head(checkout) != expected:
        raise ValueError(
            f"{checkout}: expected source {expected}, got {git_head(checkout)}"
        )
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=checkout
    ).strip():
        raise ValueError(f"{checkout}: checkout is not clean")


def select(queue):
    if queue["commit"] != FROZEN:
        raise ValueError(
            "the 195-parent queue is not bound to the frozen scientific commit"
        )
    lookup = {
        (
            item["row"]["architecture"],
            item["row"]["environment"],
            item["row"]["seed"],
        ): item
        for item in queue["parents"]
    }
    if len(lookup) != 195:
        raise ValueError("parent queue is not unique")
    selected = []
    for gpu, arch, environment, seed in PLACEMENT:
        item = lookup.get((arch, environment, seed))
        if item is None or item["kind"] != "exp1":
            raise ValueError(f"missing requested parent: {arch}/{environment}/{seed}")
        if item["id"] != scheduler.identity("exp1", item["row"], FROZEN):
            raise ValueError("parent logical identity differs from frozen generator")
        selected.append((gpu, item))
    if len({item["id"] for _, item in selected}) != 7:
        raise ValueError("pilot parent identities are duplicated")
    return selected


def validate_resources(spec, selected):
    resources = spec["resources"]
    if (
        resources.get("gpus") != 2
        or resources.get("wall_minutes") != WALL_MINUTES
        or type(resources.get("cpus")) is not int
        or resources["cpus"] < 7
        or type(resources.get("memory_gb")) is not int
        or resources["memory_gb"] < 32
    ):
        raise ValueError("pilot requires exactly 2 A100s/10h and explicit CPU/RAM")
    estimates = spec["per_parent_resource_estimates"]
    if set(estimates) != {item["id"] for _, item in selected}:
        raise ValueError("every selected parent needs a reviewed CPU/RAM/VRAM estimate")
    for field in ("cpus", "host_ram_gb", "gpu_memory_gb"):
        if any(
            type(estimates[item["id"]].get(field)) is not int
            or estimates[item["id"]][field] <= 0
            for _, item in selected
        ):
            raise ValueError(f"invalid {field} estimate")
    if sum(estimates[item["id"]]["cpus"] for _, item in selected) > resources["cpus"]:
        raise ValueError("declared CPU demand exceeds allocation")
    if (
        sum(estimates[item["id"]]["host_ram_gb"] for _, item in selected)
        + HOST_HEADROOM_GB
        > resources["memory_gb"]
    ):
        raise ValueError("declared host RAM demand exceeds allocation")
    for gpu in (0, 1):
        if (
            sum(
                estimates[item["id"]]["gpu_memory_gb"]
                for assigned, item in selected
                if assigned == gpu
            )
            + GPU_HEADROOM_GB
            > 40
        ):
            raise ValueError(
                f"GPU {gpu} lacks the required {GPU_HEADROOM_GB} GiB memory margin"
            )


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    queue_root = Path(spec["queue"]).resolve()
    queue = scheduler.load_queue(queue_root)  # validates the approved 195-parent plan
    selected = select(queue)
    if (
        spec.get("schema_version") != 1
        or spec.get("frozen_commit") != FROZEN
        or spec.get("queue_sha256") != sha(queue_root / "queue.json")
        or spec.get("scheduler_commit") != git_head(scheduler.MAIN)
        or spec.get("training_checkout") == str(scheduler.MAIN)
    ):
        raise ValueError("pilot specification/source/queue binding mismatch")
    expected_placement = [
        {"gpu": gpu, "id": item["id"], "run_key": item["row"]["run_key"]}
        for gpu, item in selected
    ]
    if spec.get("placement") != expected_placement:
        raise ValueError(
            "pilot specification changed the seven frozen parent placements"
        )
    checkout = Path(spec["training_checkout"]).resolve()
    if not checkout.is_absolute() or not (checkout / "main" / "run.py").is_file():
        raise ValueError("separate frozen training checkout is unavailable")
    require_clean(checkout, FROZEN)
    require_clean(scheduler.MAIN.parent, spec["scheduler_commit"])
    validate_resources(spec, selected)
    return spec, queue, selected


def prepare(queue_root, training_checkout, out, cpus, memory_gb, estimates):
    queue_root = Path(queue_root).resolve()
    queue = scheduler.load_queue(queue_root)
    selected = select(queue)
    require_clean(training_checkout, FROZEN)
    require_clean(scheduler.MAIN.parent, git_head(scheduler.MAIN))
    spec = {
        "schema_version": 1,
        "queue": str(queue_root),
        "queue_sha256": sha(queue_root / "queue.json"),
        "scheduler_commit": git_head(scheduler.MAIN),
        "frozen_commit": FROZEN,
        "training_checkout": str(Path(training_checkout).resolve()),
        "resources": {
            "gpus": 2,
            "cpus": cpus,
            "memory_gb": memory_gb,
            "wall_minutes": WALL_MINUTES,
        },
        "per_parent_resource_estimates": estimates,
        "placement": [
            {"gpu": gpu, "id": item["id"], "run_key": item["row"]["run_key"]}
            for gpu, item in selected
        ],
    }
    validate_resources(spec, selected)
    scheduler._write_new(out, spec)
    return spec


def submission_command(spec_path):
    spec, _, _ = load_spec(spec_path)
    logs = Path(spec["queue"]) / "pilot_slurm"
    logs.mkdir(parents=True, exist_ok=True)
    resources = spec["resources"]
    import shlex

    command = [
        "env",
        f"EXP12_PILOT_SPEC={Path(spec_path).resolve()}",
        f"EXP12_PILOT_SPEC_SHA256={sha(spec_path)}",
        "sbatch",
        "--parsable",
        "--export=ALL",
        "--nodes=1",
        "--ntasks=1",
        "--gpus-per-node=2",
        f"--cpus-per-task={resources['cpus']}",
        f"--mem={resources['memory_gb']}G",
        f"--time={WALL_MINUTES}",
        f"--output={logs}/%x_%j.out",
        f"--error={logs}/%x_%j.err",
        "scripts/sbatch_exp12_seven_parent_pilot.sh",
    ]
    return shlex.join(command)


def _process_usage(pid):
    status = Path(f"/proc/{pid}/status")
    try:
        text = status.read_text()
        match = re.search(r"^VmRSS:\s+(\d+) kB", text, re.M)
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        # /proc stat fields 14 and 15 follow the (pid, comm) prefix.
        cpu_seconds = (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK")
        return {
            "rss_gib": int(match[1]) / 1024**2 if match else None,
            "cpu_seconds": cpu_seconds,
        }
    except FileNotFoundError:
        return None


def _telemetry(active, path, tokens, peaks):
    devices = {}
    for token in tokens:
        total, used = scheduler.gpu_snapshot(token)
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(token),
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        peaks[str(token)] = max(peaks.get(str(token), 0), used)
        devices[str(token)] = {
            "total_mib": total,
            "used_mib": used,
            "peak_observed_mib": peaks[str(token)],
            "utilization_pct": int(result.stdout.strip()),
        }
    now = time.time()
    processes = []
    for job in active:
        usage = _process_usage(job["process"].pid)
        if usage is not None:
            previous = job.get("previous_cpu_sample")
            usage["cpu_pct_one_core"] = (
                (100 * (usage["cpu_seconds"] - previous[1]) / (now - previous[0]))
                if previous and now > previous[0]
                else None
            )
            job["previous_cpu_sample"] = (now, usage["cpu_seconds"])
        processes.append(
            {
                "id": job["item"]["id"],
                "pid": job["process"].pid,
                "gpu": job["gpu"],
                "usage": usage,
                "checkpoint": scheduler.checkpoint_progress(job["item"]),
                "interactions_per_second": job.get("rate"),
            }
        )
    row = {
        "time": now,
        "gpus": devices,
        "available_host_gb": scheduler.available_host_gb(),
        "aggregate_interactions_per_second": {
            str(token): sum(
                job.get("rate") or 0 for job in active if job["gpu"] == token
            )
            for token in tokens
        },
        "processes": processes,
    }
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
    return devices


def worker(spec_path):
    if sha(spec_path) != os.environ.get("EXP12_PILOT_SPEC_SHA256"):
        raise ValueError("pilot specification changed after submission")
    spec, queue, selected = load_spec(spec_path)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("real Slurm allocation required")
    description = subprocess.check_output(
        ["scontrol", "show", "job", "--oneliner", job_id], text=True
    )
    allocation_matches(
        {
            "cpus": spec["resources"]["cpus"],
            "memory_gb": spec["resources"]["memory_gb"],
            "wall_minutes": WALL_MINUTES,
        },
        os.environ,
        description,
    )
    tokens = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (
        int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != 1
        or int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) != 2
        or len(tokens) != 2
        or len(set(tokens)) != 2
        or not all(
            re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", token) for token in tokens
        )
        or len(os.sched_getaffinity(0)) < spec["resources"]["cpus"]
    ):
        raise ValueError(
            "expected one node, two distinct visible A100s and approved CPU affinity"
        )
    require_pinned_stack()
    import jax
    import jaxlib

    if (
        jax.__version__ != "0.4.34"
        or jaxlib.__version__ != "0.4.34"
        or jax.default_backend() != "gpu"
        or len(jax.devices()) != 2
        or not jax.local_devices(backend="cpu")
    ):
        raise ValueError("pinned CPU/CUDA backends unavailable")
    for token in tokens:
        scheduler.gpu_snapshot(token)
    hb_python = os.environ.get("EXP12_HB_PYTHON", "")
    if not Path(hb_python).is_file() or not Path(hb_python).is_absolute():
        raise ValueError("absolute pinned HumanoidBench interpreter required")
    approval = {
        "expected_commit": FROZEN,
        "resources": {"humanoid_bench_python": hb_python},
        "admission": {
            "architectures": {
                arch: {"cpus": 1, "host_ram_gb": 1, "gpu_memory_gb": 1}
                for arch in scheduler.WEIGHTS
            }
        },
    }
    queue_root = Path(spec["queue"])
    attempt_root = queue_root / "pilot" / job_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    reports = attempt_root / "preflight"
    reports.mkdir()
    main_report = reports / "main_environment.json"
    for executable, report, extra in (
        (sys.executable, main_report, []),
        (hb_python, reports / "hb_environment.json", ["--main", str(main_report)]),
    ):
        with (reports / (report.stem + ".log")).open("xb") as log:
            subprocess.run(
                [
                    executable,
                    "scripts/check_exp12_hb_environment.py",
                    "--gpu-model",
                    scheduler.GPU_KIND,
                    *extra,
                    "--out",
                    str(report),
                ],
                cwd=scheduler.MAIN,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    ledger = scheduler.Ledger(queue_root)
    active = []
    pending = []
    for gpu, item in selected:
        if ledger.state(item) == "complete":
            print(
                f"[pilot] already completed; never restarting {item['id']}", flush=True
            )
            continue
        if ledger.state(item) != "pending":
            raise ValueError(f"selected parent is blocked: {item['id']}")
        if (Path(item["row"]["checkpoint_dir"]) / "DONE").exists():
            owner = scheduler._owner(job_id)
            if ledger.claim(
                item,
                owner,
                lambda row: scheduler.validate_done(row, FROZEN, queue_root),
            ):
                raise ValueError("DONE reconciliation unexpectedly made a new claim")
            if ledger.state(item) == "complete":
                print(
                    f"[pilot] reconciled validated completion {item['id']}", flush=True
                )
                continue
            raise ValueError(f"unreconciled terminal evidence for {item['id']}")
        if scheduler.validate_resume(item, FROZEN):
            print(f"[pilot] valid resumable checkpoint: {item['id']}", flush=True)
        pending.append((gpu, item))
    # Check declared resources and live headroom immediately before dispatch.
    validate_resources(spec, selected)
    for gpu, token in enumerate(tokens):
        _, used = scheduler.gpu_snapshot(token)
        need = sum(
            spec["per_parent_resource_estimates"][item["id"]]["gpu_memory_gb"]
            for assigned, item in selected
            if assigned == gpu
        )
        if used + 1024 * (need + GPU_HEADROOM_GB) > 40 * 1024:
            raise ValueError(f"GPU {gpu}: live VRAM use prevents requested placement")
    if scheduler.available_host_gb() < HOST_HEADROOM_GB + sum(
        spec["per_parent_resource_estimates"][item["id"]]["host_ram_gb"]
        for _, item in selected
    ):
        raise ValueError("live host RAM prevents requested placement")
    for gpu, item in pending:
        token = tokens[gpu]
        already = [j for j in active if j["gpu"] == token]
        _, used = scheduler.gpu_snapshot(token)
        reserved = sum(
            spec["per_parent_resource_estimates"][j["item"]["id"]]["gpu_memory_gb"]
            for j in already
        )
        next_gb = spec["per_parent_resource_estimates"][item["id"]]["gpu_memory_gb"]
        if max(used, 1024 * reserved) + 1024 * (next_gb + GPU_HEADROOM_GB) > 40 * 1024:
            raise ValueError(
                f"GPU {gpu}: insufficient live VRAM for next requested parent"
            )
        job = scheduler._start(
            item,
            token,
            approval,
            queue_root,
            ledger,
            job_id,
            training_checkout=str(Path(spec["training_checkout"]) / "main"),
        )
        if job is None:
            raise ValueError(
                f"could not claim/start parent {item['id']}; inspect durable ledger"
            )
        job["profile"] = spec["per_parent_resource_estimates"][item["id"]]
        pointer = Path(item["row"]["checkpoint_dir"]) / "state" / "LATEST"
        if pointer.is_file():
            job["last_observed_state"] = pointer.read_text().strip()
        active.append(job)
        time.sleep(5)
    end_match = re.search(r"\bEndTime=(\S+)", description)
    if not end_match:
        raise ValueError("Slurm end time unavailable")
    deadline = datetime.strptime(end_match[1], "%Y-%m-%dT%H:%M:%S").timestamp()
    telemetry = attempt_root / "telemetry.jsonl"
    save_windows = attempt_root / "checkpoint_publication_windows.jsonl"
    peaks = {}
    stopped = False
    warned_walltime = False
    last_sample = 0.0
    while active:
        if sha(spec_path) != os.environ["EXP12_PILOT_SPEC_SHA256"]:
            raise ValueError("pilot specification changed during allocation")
        # One-second filesystem observation measures a lower bound on each
        # checkpoint save's publication window without touching training code.
        for job in active:
            state_root = Path(job["item"]["row"]["checkpoint_dir"]) / "state"
            seen = job.setdefault("seen_state_dirs", {})
            for state in state_root.glob("step_*"):
                seen.setdefault(state.name, time.monotonic())
            pointer = state_root / "LATEST"
            if pointer.is_file():
                name = pointer.read_text().strip()
                if name != job.get("last_observed_state"):
                    job["last_observed_state"] = name
                    if name in seen:
                        with save_windows.open("a") as stream:
                            stream.write(
                                json.dumps(
                                    {
                                        "id": job["item"]["id"],
                                        "step_dir": name,
                                        "observed_window_lower_bound_s": max(
                                            0, time.monotonic() - seen[name]
                                        ),
                                        "sampling_interval_s": 1,
                                        "time": time.time(),
                                    }
                                )
                                + "\n"
                            )
                            stream.flush()
        scheduler.update_rates(active)
        if time.monotonic() - last_sample < 15:
            time.sleep(1)
            continue
        last_sample = time.monotonic()
        devices = _telemetry(active, telemetry, tokens, peaks)
        if not warned_walltime and time.time() >= deadline - STOP_MARGIN_SECONDS:
            warned_walltime = True
            print(
                "[pilot] walltime near: unfinished runs retain their last committed checkpoints; "
                "Slurm will end the allocation",
                flush=True,
            )
        if not stopped and (
            any(info["used_mib"] > 36 * 1024 for info in devices.values())
            or scheduler.available_host_gb() < HOST_HEADROOM_GB / 2
        ):
            stopped = True
            print(
                "[pilot] resource stop: terminating running attempts; committed checkpoints remain",
                flush=True,
            )
            for job in active:
                if job["process"].poll() is None:
                    os.killpg(job["process"].pid, signal.SIGTERM)
        for job in list(active):
            if job["process"].poll() is None:
                continue
            try:
                scheduler._finish(job, ledger, FROZEN, queue_root)
            except scheduler.LedgerContention as exc:
                print(
                    f"[pilot] ledger finish pending for {job['item']['id']}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            active.remove(job)
        time.sleep(1)
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("prepare")
    for flag in ("queue", "training-checkout", "out", "estimates"):
        make.add_argument("--" + flag, required=True)
    make.add_argument("--cpus", required=True, type=int)
    make.add_argument("--memory-gb", required=True, type=int)
    command = sub.add_parser("command")
    command.add_argument("--spec", required=True)
    run = sub.add_parser("worker")
    run.add_argument("--spec", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(
            args.queue,
            args.training_checkout,
            args.out,
            args.cpus,
            args.memory_gb,
            json.loads(Path(args.estimates).read_text()),
        )
    elif args.action == "command":
        print(submission_command(args.spec))
    else:
        return worker(args.spec)


if __name__ == "__main__":
    sys.exit(main())
