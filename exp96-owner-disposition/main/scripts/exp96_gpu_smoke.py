#!/usr/bin/env python3
"""Bounded one-A100 engineering smoke for the Method C ten-target 96 study.

Development-only 20k interaction runs have two scheduled full ten-target probes.
Three sizes start together; a fourth small run replaces the first completion.
The medium process is interrupted once after a committed 10k checkpoint and
resumed from that checkpoint. No result is eligible for the official study.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from scripts.exp96_campaign import GPU, MAIN, VERSION, _source
from scripts.exp12_dynamic_scheduler import gpu_snapshot
from scripts.exp12_launch_plan import require_pinned_stack

CASES = (("small", "D2W512", 2, 512, "dog-run", "dmc_hard", 1),
         ("medium", "D4W1024", 4, 1024, "humanoid-run", "dmc_hard", 2),
         ("large", "D4W1536", 4, 1536, "myo-key-turn", "myosuite_simba", 3),
         ("replacement", "D2W512", 2, 512, "myo-pen-twirl", "myosuite_simba", 4))


def command(case, root):
    label, _, blocks, width, env, group, seed = case
    overrides = (f"critic_num_blocks={blocks}", f"critic_hidden_dim={width}",
                 f"env_name={env}", f"env={group}", f"seed={seed}",
                 f"results_root={root / 'results'}", "env.num_env_steps=40000",
                 "probe.checks=2", "run_role=dev", "fork.enabled=false")
    cmd = [sys.executable, "-u", "run.py", "--experiment", "exp1",
           "--config_name", "base_exp12_ten"]
    for override in overrides:
        cmd.extend(("--overrides", override))
    cmd.extend(("--checkpoint_dir", str(root / label), "--checkpoint_interval", "10000",
                "--checkpoint_start_frac", "0.0"))
    return cmd


def run(root, commit, cap_seconds=5100):
    import jax
    _source(commit)
    require_pinned_stack()
    if jax.default_backend() != "gpu" or not jax.local_devices(backend="cpu"):
        raise ValueError("both CUDA and CPU JAX backends required")
    if len(jax.devices("gpu")) != 1 or gpu_snapshot(os.environ["CUDA_VISIBLE_DEVICES"])[0] < 40_000:
        raise ValueError("one visible A100-40GB required")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + cap_seconds
    active, finished, attempts = {}, {}, []
    pending = [CASES[3]]
    restarted = False
    peak_gpu_mb, peak_host_mb = 0, 0

    def start(case, resumed=False):
        label = case[0]
        attempt = len([a for a in attempts if a["case"] == label]) + 1
        log = root / f"{label}_attempt{attempt}.log"
        stream = log.open("xb")
        env = dict(os.environ, WANDB_MODE="offline", WANDB_DIR=str(root), PYTHONUNBUFFERED="1")
        cmd = command(case, root)
        proc = subprocess.Popen(cmd, cwd=MAIN, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        stream.close()
        record = {"case": label, "attempt": attempt, "resumed": resumed,
                  "log": str(log), "command": cmd, "started": time.time(), "pid": proc.pid}
        attempts.append(record)
        active[label] = (case, proc, record)

    for case in CASES[:3]:
        start(case)
    while active and time.monotonic() < deadline:
        total, used = gpu_snapshot(os.environ["CUDA_VISIBLE_DEVICES"])
        peak_gpu_mb = max(peak_gpu_mb, used)
        try:
            import psutil
            host = sum(psutil.Process(p.pid).memory_info().rss for _, p, _ in active.values()) // 1048576
            peak_host_mb = max(peak_host_mb, host)
        except (ImportError, OSError):
            pass
        for label, (case, process, record) in list(active.items()):
            if label == "medium" and not restarted and process.poll() is None:
                pointer = root / label / "state" / "LATEST"
                if pointer.is_file() and pointer.read_text().startswith("step_000010000_"):
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=30)
                    record.update(exit_code=process.returncode, interrupted_at_checkpoint=pointer.read_text().strip(),
                                  ended=time.time())
                    del active[label]
                    restarted = True
                    start(case, resumed=True)
                    continue
            code = process.poll()
            if code is None:
                continue
            record.update(exit_code=code, ended=time.time())
            del active[label]
            finished[label] = code
            if pending:
                start(pending.pop(0))  # immediate replacement after any completion
        time.sleep(2)
    timed_out = bool(active)
    for _, process, record in active.values():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        record.update(exit_code=process.returncode, ended=time.time(), timed_out=True)
    checks = {}
    for case in CASES:
        label = case[0]
        probe_dir = root / label / "probes"
        checks[label] = {"done": (root / label / "DONE").is_file(),
                         "check_0": (probe_dir / "check_00.npz").is_file(),
                         "check_1": (probe_dir / "check_01.npz").is_file(),
                         "check_2": (probe_dir / "check_02.npz").is_file(),
                         "csv": (probe_dir / "probe_checks.csv").is_file(),
                         "latest": (root / label / "state" / "LATEST").is_file()}
    verdict = "PASS" if (not timed_out and restarted and len(finished) == 4
                          and all(code == 0 for code in finished.values())
                          and all(all(value.values()) for value in checks.values())) else "INCOMPLETE"
    throughput = {label: 20_000 / sum(record["ended"] - record["started"]
                                          for record in attempts if record["case"] == label)
                  for label, code in finished.items() if code == 0}
    report = {"study_version": VERSION, "expected_commit": commit, "verdict": verdict,
              "allocation_seconds": cap_seconds, "gpu": GPU, "peak_gpu_mb": peak_gpu_mb,
              "peak_process_rss_mb": peak_host_mb, "restarted_medium": restarted,
              "throughput_interactions_per_second": throughput,
              "attempts": attempts, "artifacts": checks, "completed": finished}
    (root / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if verdict == "PASS" else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--cap-seconds", type=int, default=5100)
    args = parser.parse_args()
    if args.cap_seconds != 5100:
        raise ValueError("reviewed smoke workload cap is exactly 85 minutes")
    return run(args.root, args.commit, args.cap_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
