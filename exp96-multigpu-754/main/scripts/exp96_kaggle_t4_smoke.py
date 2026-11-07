#!/usr/bin/env python3
"""T4 engineering subset of the existing Exp96 A100 smoke; never an approval gate.

The existing four development commands and scientific probe settings are reused.
Each architecture runs alone through its first full probe before packing is
considered. All output is outside the source checkout. No official parent is run.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from scripts.exp96_campaign import MAIN, VERSION, _source
from scripts.exp96_gpu_smoke import CASES, command
from scripts.exp12_launch_plan import require_pinned_stack

HEADROOM_MB = 2048
POLL_SECONDS = 10


def gpu_measurement():
    fields = "name,driver_version,memory.total,memory.used,utilization.gpu"
    raw = subprocess.check_output(
        ["nvidia-smi", "--id=0", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip().splitlines()
    if len(raw) != 1:
        raise RuntimeError("exactly one visible NVIDIA GPU is required")
    name, driver, total, used, util = [part.strip() for part in raw[0].split(",")]
    return {"name": name, "driver": driver, "total_mb": int(total),
            "used_mb": int(used), "utilization_percent": int(util)}


def gpu_preflight():
    import jax
    import jax.numpy as jnp
    require_pinned_stack()
    info = gpu_measurement()
    if "T4" not in info["name"]:
        raise RuntimeError(f"Kaggle T4 required; detected {info['name']}")
    if jax.default_backend() != "gpu" or len(jax.devices("gpu")) != 1:
        raise RuntimeError("JAX CUDA backend did not initialize on the T4")
    if not jax.local_devices(backend="cpu"):
        raise RuntimeError("JAX CPU backend unavailable for explicit CPU initialization")
    x = jax.device_put(jnp.ones((64, 64), dtype=jnp.float32), jax.devices("gpu")[0])
    y = jax.jit(lambda a: a @ a)(x)
    y.block_until_ready()
    if y.device.platform != "gpu" or float(y[0, 0]) != 64.0:
        raise RuntimeError("GPU JAX matmul failed or silently fell back to CPU")
    from experiments.exp12.precision import set_matmul_precision
    from jax.extend.backend import get_backend
    set_matmul_precision()
    info.update(jax=jax.__version__, jaxlib=importlib.metadata.version("jaxlib"),
                jax_device=jax.devices("gpu")[0].device_kind,
                configured_matmul_precision=str(jax.config.jax_default_matmul_precision),
                cuda_plugin=importlib.metadata.version("jax-cuda12-plugin"),
                cuda_backend_version=get_backend().platform_version,
                python=platform.python_version())
    if info["configured_matmul_precision"] != "tensorfloat32":
        raise RuntimeError("production TF32 precision request was not preserved")
    # The T4 does not implement the A100 TF32 math path. This value records
    # the requested policy; it is not a numerical equivalence claim.
    import dm_control  # noqa: F401
    import myosuite  # noqa: F401
    import psutil
    info["host_total_mb"] = psutil.virtual_memory().total // 1048576
    return info


def _rss_mb(processes):
    try:
        import psutil
        return sum(psutil.Process(proc.pid).memory_info().rss for proc, _ in processes.values()) // 1048576
    except (ImportError, OSError):
        return None


def _stop(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    return proc.returncode


def _launch(case, root, attempts, stage):
    label = case[0]
    attempt = 1 + sum(record["case"] == label for record in attempts)
    logfile = root / f"{label}_attempt{attempt}.log"
    cmd = command(case, root)  # exact reviewed architecture, 20k, 10 targets, 2k fits
    env = dict(os.environ, WANDB_MODE="offline", WANDB_DIR=str(root / "wandb"),
               JAX_PLATFORMS="cuda,cpu", JAX_ENABLE_X64="false", MUJOCO_GL="disable",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", EXP12_JAX_CACHE_ROOT=str(root / "jax_cache"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("JAX_PLATFORM_NAME", None)
    env.pop("NVIDIA_TF32_OVERRIDE", None)
    with logfile.open("xb") as stream:
        proc = subprocess.Popen(cmd, cwd=MAIN, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
    record = {"case": label, "architecture": case[1], "stage": stage,
              "attempt": attempt, "command": cmd, "log": str(logfile),
              "pid": proc.pid, "started": time.time()}
    attempts.append(record)
    print(f"[{stage}] started {label} pid={proc.pid} log={logfile}", flush=True)
    return proc, record


def _sample(root, processes, stage, peaks):
    gpu = gpu_measurement()
    rss = _rss_mb(processes)
    point = {"time": time.time(), "stage": stage, "gpu": gpu,
             "host_process_rss_mb": rss, "active": list(processes)}
    with (root / "resources.jsonl").open("a") as stream:
        stream.write(json.dumps(point, sort_keys=True) + "\n")
    if len(processes) == 1:
        label = next(iter(processes))
        peaks[label] = max(peaks.get(label, 0), gpu["used_mb"])
    return point


def _first_probe_ready(root, label):
    pointer = root / label / "state" / "LATEST"
    probe = root / label / "probes" / "check_01.npz"
    return (pointer.is_file() and probe.is_file()
            and pointer.read_text().strip().startswith("step_000010000_"))


def _probe_artifacts(root, label):
    import numpy as np
    probes = root / label / "probes"
    checks = {}
    for index in range(3):
        path = probes / f"check_{index:02d}.npz"
        if not path.is_file():
            checks[str(index)] = {"present": False}
            continue
        with np.load(path) as payload:
            expected = ("fresh_q1_losses", "fresh_q2_losses") if index == 0 else (
                "fresh_q1_losses", "fresh_q2_losses", "current_q1_losses", "current_q2_losses")
            shapes = {name: list(payload[name].shape) for name in expected if name in payload}
            checks[str(index)] = {"present": True, "shapes": shapes,
                                  "full_fidelity": all(shapes.get(n) == [10, 2000] for n in expected)}
    run = root / label
    done = run / "DONE"
    try:
        endpoint = json.loads(done.read_text()).get("interaction_step") == 20_000
    except (OSError, ValueError):
        endpoint = False
    try:
        launches = json.loads((run / "run_metadata.json").read_text()).get("launches", [])
        resumed_launch = len(launches) >= 2 and launches[-1].get("resumed") is True
    except (OSError, ValueError):
        resumed_launch = False
    return {"checks": checks, "csv": (probes / "probe_checks.csv").is_file(),
            "latest": (run / "state" / "LATEST").is_file(),
            "endpoint_20000": endpoint, "resumed_launch": resumed_launch}


def _package(root, report):
    root.joinpath("summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    inventory = []
    for run in root.iterdir():
        if run.is_dir() and run.name in {c[0] for c in CASES}:
            for path in run.rglob("*"):
                if path.is_file() and "state" in path.relative_to(run).parts:
                    inventory.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})
    root.joinpath("checkpoint_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    bundle = root.parent / f"{root.name}_evidence.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if "jax_cache" in rel.parts or "media" in rel.parts:
                continue
            if "state" in rel.parts and rel.suffix != ".json" and rel.name != "LATEST":
                continue
            archive.add(path, arcname=str(Path(root.name) / rel), recursive=False)
    digest = hashlib.sha256()
    with bundle.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    bundle.with_name(bundle.name + ".sha256").write_text(f"{digest.hexdigest()}  {bundle.name}\n")
    print(f"evidence bundle: {bundle}", flush=True)
    return str(bundle)


def run(root, commit, max_seconds, phase_seconds):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"study_version": VERSION, "source_commit": commit, "platform": "Kaggle T4",
              "official_parent_training": False, "verdict": "INCOMPLETE", "attempts": [],
              "baseline": {}, "finished": {}, "limits": [], "resource_peaks_mb": {}}
    (root / "wandb").mkdir()
    deadline = time.monotonic() + max_seconds - 300  # five minutes for evidence publication
    attempts, peaks = report["attempts"], report["resource_peaks_mb"]
    launched = []

    def launch(case, stage):
        process, record = _launch(case, root, attempts, stage)
        launched.append((process, record))
        return process, record

    try:
        _source(commit)
        for name in ("base_exp12.yaml", "base_exp12_ten.yaml"):
            shutil.copy2(MAIN / "configs" / name, root / name)
        report["backend"] = gpu_preflight()
        report["package_versions"] = {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
        report["source_status"] = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=MAIN, text=True)
        if report["source_status"].strip():
            raise RuntimeError("source checkout is not clean")
        # Each architecture is measured alone through its first complete probe.
        # A completed case is retained; a paused case resumes from its checkpoint.
        candidates = []
        for case in CASES:
            label = case[0]
            if time.monotonic() >= deadline:
                report["limits"].append("overall time cap before " + label)
                break
            process, rec = launch(case, "single_baseline")
            stage_end = min(deadline, time.monotonic() + phase_seconds)
            samples = 0
            while time.monotonic() < stage_end and process.poll() is None:
                point = _sample(root, {label: (process, rec)}, "single_baseline", peaks)
                samples += 1
                if samples % 6 == 1:
                    print(f"[single_baseline] {label}: GPU {point['gpu']['used_mb']} MiB; "
                          f"first probe {'ready' if _first_probe_ready(root, label) else 'pending'}",
                          flush=True)
                if _first_probe_ready(root, label):
                    break
                time.sleep(POLL_SECONDS)
            complete = process.poll() == 0
            ready = _first_probe_ready(root, label)
            rec["exit_code"] = _stop(process)
            rec["ended"] = time.time()
            rec["checkpoint_10k_probe"] = ready
            report["baseline"][label] = {"ready": ready, "completed": complete,
                                          "peak_gpu_mb": peaks.get(label),
                                          "exit_code": rec["exit_code"]}
            if rec["exit_code"] not in (0, -signal.SIGTERM, -signal.SIGKILL):
                log_tail = Path(rec["log"]).read_text(errors="replace")[-4000:]
                if any(mark in log_tail for mark in (
                        "RESOURCE_EXHAUSTED", "out of memory", "CUDA_ERROR_OUT_OF_MEMORY")):
                    report["limits"].append(f"{label}: T4 out of memory")
                    continue
                raise RuntimeError(f"{label} failed: {log_tail[-800:]}")
            if complete:
                report["finished"][label] = 0
            elif ready:
                candidates.append(case)
            else:
                report["limits"].append(f"{label}: no complete 10k probe/checkpoint within baseline cap")
        # Only measured, checkpoint-ready runs may share the T4. If the three
        # measured peaks do not fit, carry on with the largest safe subset.
        memory_limit = report["backend"]["total_mb"] - HEADROOM_MB
        selected, waiting = [], []
        for case in (c for c in candidates if c[0] != "replacement"):
            if sum(peaks[c[0]] for c in selected) + peaks[case[0]] <= memory_limit:
                selected.append(case)
            else:
                waiting.append(case)
        if len(selected) < 3:
            report["limits"].append("three-architecture concurrency not supported by measured T4 memory")
        active = {c[0]: launch(c, "resume_packed") for c in selected}
        replacement = CASES[3]
        replacement_ready = replacement in candidates
        if not replacement_ready and report["finished"].get("replacement") != 0:
            report["limits"].append("replacement MyoSuite small critic lacks its own measured first-probe memory")
        if not selected and replacement_ready:
            waiting.append(replacement)
            replacement_ready = False
        replacement_started = False
        replacement_while_other_active = False
        while time.monotonic() < deadline and (active or waiting or replacement_ready):
            point = _sample(root, active, "packed_or_serial", peaks)
            if int(time.monotonic()) % 60 < POLL_SECONDS:
                print(f"[packed_or_serial] active={list(active)} GPU={point['gpu']['used_mb']} MiB",
                      flush=True)
            for label, (process, rec) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                rec.update(exit_code=code, ended=time.time())
                del active[label]
                report["finished"][label] = code
                print(f"[run] {label} exited {code}; scheduling next compatible case", flush=True)
                if code != 0:
                    log_tail = Path(rec["log"]).read_text(errors="replace")[-4000:]
                    if any(mark in log_tail for mark in (
                            "RESOURCE_EXHAUSTED", "out of memory", "CUDA_ERROR_OUT_OF_MEMORY")):
                        report["limits"].append(f"{label}: T4 out of memory during continuation")
                    else:
                        raise RuntimeError(f"{label} continuation failed: {log_tail[-800:]}")
            # The fourth small case is admitted immediately after a slot frees.
            if replacement_ready and any(c[0] in report["finished"] for c in selected):
                waiting.insert(0, replacement)
                replacement_ready = False
            for case in list(waiting):
                label = case[0]
                estimate = peaks.get(label, memory_limit + 1)
                if sum(peaks.get(name, memory_limit + 1) for name in active) + estimate > memory_limit:
                    continue
                if label == "replacement" and active:
                    replacement_while_other_active = True
                active[label] = launch(case, "replacement_or_serial")
                if label == "replacement" and selected:
                    replacement_started = True
                waiting.remove(case)
                break
            if not active and waiting and all(peaks.get(c[0], 0) > memory_limit for c in waiting):
                break
            time.sleep(POLL_SECONDS)
        for label, (process, rec) in active.items():
            rec.update(exit_code=_stop(process), ended=time.time(), stopped_at_cap=True)
        if active or waiting:
            report["limits"].append("overall time or T4 memory cap left cases incomplete")
        report["artifacts"] = {case[0]: _probe_artifacts(root, case[0]) for case in CASES}
        report["restarted_medium"] = (any(a["case"] == "medium" and a["stage"] == "resume_packed"
                                          for a in attempts)
                                      and report["artifacts"]["medium"]["resumed_launch"])
        completed = all(report["finished"].get(c[0]) == 0 for c in CASES)
        probes = all(all(report["artifacts"][c[0]]["checks"][str(i)].get("full_fidelity")
                         for i in range(3)) and report["artifacts"][c[0]]["csv"]
                     and report["artifacts"][c[0]]["latest"]
                     and report["artifacts"][c[0]]["endpoint_20000"] for c in CASES)
        if any(report["finished"].get(c[0]) == 0 and not (
                all(report["artifacts"][c[0]]["checks"][str(i)].get("full_fidelity") for i in range(3))
                and report["artifacts"][c[0]]["endpoint_20000"])
               for c in CASES):
            raise RuntimeError("a completed development run lacks full probe or endpoint evidence")
        if completed and probes:
            from scripts.exp96_campaign import validate_probe_artifacts
            for case in CASES:
                validate_probe_artifacts(root / case[0], {"buffer": {"min_length": 5000}}, 20_000)
        concurrent = len(selected) == 3
        report["coverage"] = {"all_four_completed": completed, "all_full_probes": probes,
                              "three_initial_architectures_concurrent": concurrent,
                              "replacement_started": replacement_started,
                              "replacement_while_other_active": replacement_while_other_active,
                              "durable_scheduler_claims": "NOT_EXERCISED",
                              "medium_restarted": report["restarted_medium"]}
        report["throughput_interactions_per_second"] = {
            label: 20_000 / sum(a["ended"] - a["started"] for a in attempts if a["case"] == label)
            for label, code in report["finished"].items() if code == 0}
        report["verdict"] = "PASS" if (completed and probes and concurrent
                                        and replacement_while_other_active
                                        and report["restarted_medium"]) else "INCOMPLETE"
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        for process, record in launched:
            if process.poll() is None:
                record.update(exit_code=_stop(process), ended=time.time(), stopped_after_error=True)
        report["elapsed_seconds"] = max_seconds - 300 - deadline + time.monotonic()
        resource_path = root / "resources.jsonl"
        if resource_path.is_file():
            samples = [json.loads(line) for line in resource_path.read_text().splitlines()]
            report["peak_gpu_used_mb"] = max((s["gpu"]["used_mb"] for s in samples), default=None)
            report["peak_host_process_rss_mb"] = max(
                (s["host_process_rss_mb"] for s in samples if s["host_process_rss_mb"] is not None),
                default=None)
        report["a100_qualification"] = "UNVERIFIED"
        report["bundle"] = str(root.parent / f"{root.name}_evidence.tar.gz")
        _package(root, report)
    return {"PASS": 0, "INCOMPLETE": 3, "FAIL": 2}[report["verdict"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--max-seconds", type=int, default=18_000)
    parser.add_argument("--phase-seconds", type=int, default=2_700)
    args = parser.parse_args()
    if not 600 <= args.phase_seconds <= args.max_seconds <= 18_000:
        parser.error("Kaggle engineering cap is five hours with a bounded baseline")
    return run(args.root, args.commit, args.max_seconds, args.phase_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
