#!/usr/bin/env python3
"""Solo-vs-contended throughput profiler (Step 0, throughput investigation,
2026-09-24). Extends scripts/preflight_checkpoint_check.py's launch
pattern: measures steady-state it/s (past buffer warmup at min_length and
past JIT compilation) over a fixed step window, optionally launching
--concurrency identical copies simultaneously to measure contention.

No real GPU was available to actually run this during the session that
wrote it - it has NOT been run for real. Verified only via: syntax/import
checks, and the progress-line regex tested directly against a real
historical log from angle1_logs/ (see .claude/research-methodology.md's
2026-09-21 incident update) to confirm it parses real tqdm output
correctly. Run it for real before trusting its numbers.

Usage:
    # Solo baseline - one job alone on one GPU:
    python scripts/profile_throughput.py \\
        --override critic_num_blocks=2 --override critic_hidden_dim=512 \\
        --override updates_per_interaction_step=5 \\
        --override env_name=dog-run --override env=dmc_hard \\
        --override seed=997 --override critic_degradation=true \\
        --override pathology_prop=true \\
        --override project_name=EchoCritic-throughput-profile \\
        --concurrency 1 --num_gpus 1

    # Contended - 8 identical copies sharing one GPU (matches the original
    # campaign's real density: 64 processes / 8 GPUs = 8 per GPU):
    python scripts/profile_throughput.py ... --concurrency 8 --num_gpus 1

    # Aggregate throughput across concurrency=8 on a 4-GPU allocation
    # (2 processes/GPU) - for Step 4/5's aggregate comparisons:
    python scripts/profile_throughput.py ... --concurrency 8 --num_gpus 4
"""
import argparse
import os
import re
import subprocess
import sys
import time

PROGRESS_RE = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^<]+)<([^,]+),\s*([\d.]+)it/s\]"
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="angle_1")
    parser.add_argument("--config_name", default="base_sac")
    parser.add_argument(
        "--override", action="append", default=[], dest="overrides",
        help="Extra --overrides to forward to run.py (repeatable).",
    )
    parser.add_argument(
        "--min_length", type=int, default=5000,
        help="Buffer min_length (configs/buffer/numpy_uniform.yaml) - no real "
        "gradient updates happen before this many interaction steps, so "
        "throughput before it reflects pure env-stepping, not training.",
    )
    parser.add_argument(
        "--warmup_margin_steps", type=int, default=200,
        help="Extra steps past min_length to also clear JIT compilation "
        "overhead, before the measurement window starts.",
    )
    parser.add_argument(
        "--window_steps", type=int, default=1000,
        help="Steps to measure steady-state it/s over, after warmup.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="GPUs to spread --concurrency copies across, via i %% num_gpus. "
        "Use 1 to put every copy on a single GPU (matches the original "
        "campaign's 8-processes-per-GPU density at --concurrency 8); use "
        "the allocation's real GPU count to spread them out instead.",
    )
    parser.add_argument("--timeout", type=int, default=3000, help="Seconds to wait before giving up on a job.")
    parser.add_argument("--out_dir", default=None, help="Directory for per-job logs. Defaults to a fresh tempdir.")
    parser.add_argument(
        "--checkpoint_dir_root", default=None,
        help="Absolute dir to checkpoint into (each job gets its own subdir). "
        "Defaults to a fresh tempdir. Never a relative path - see the "
        "2026-09-21 incident.",
    )
    return parser


def _tqdm_elapsed_to_seconds(s):
    """Parses tqdm's own elapsed-time field (e.g. "00:11", "3:21:45", or
    "1 day, 3:21:45") into seconds. Same logic used earlier in this
    investigation to parse angle1_logs/*.log for the incident report."""
    s = s.strip()
    dayval = 0
    if "day" in s:
        m = re.match(r"(\d+)\s*day[s]?,\s*(.*)", s)
        if m:
            dayval = int(m.group(1))
            s = m.group(2)
    parts = [float(p) for p in s.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return dayval * 86400 + sec


def measure_one_job(log_path, min_length, warmup_margin_steps, window_steps, timeout):
    """Polls log_path until steady-state warmup completes, then until the
    measurement window completes. Returns a dict of results, or a dict
    with "error" set if the job exited/timed out before finishing the
    window.

    Timing uses tqdm's OWN embedded elapsed-time field (parsed from each
    progress line, e.g. "00:11" in "5200/625000 [00:11<...]"), not this
    script's own wall-clock polling time - verified against a real
    historical log (angle1_logs/D2W512_cheetah-run_seed1.log) during
    development: using polling wall-clock time instead measured an
    impossible ~1,024,500 it/s against that (already-complete, so
    instantly-readable) file, since polling time only approximates real
    elapsed time for a log that's genuinely still growing in real time.
    Reading the log's own elapsed field is correct regardless of how fast
    this script happens to poll, and works identically whether the log is
    live or already complete.
    """
    target_start = min_length + warmup_margin_steps
    target_end = target_start + window_steps
    start_sample = None  # (step, tqdm_elapsed_seconds)
    end_sample = None
    deadline = time.time() + timeout

    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path, "r", errors="ignore") as f:
                text = f.read()
            for line in text.splitlines():
                m = PROGRESS_RE.search(line)
                if not m:
                    continue
                cur = int(m.group(2))
                elapsed_s = _tqdm_elapsed_to_seconds(m.group(4))
                if start_sample is None and cur >= target_start:
                    start_sample = (cur, elapsed_s)
                if start_sample is not None and end_sample is None and cur >= target_end:
                    end_sample = (cur, elapsed_s)
            if end_sample is not None:
                break
            if "Traceback (most recent call last)" in text:
                return {"error": "job crashed before reaching the measurement window", "log_path": log_path}
        time.sleep(3)

    if start_sample is None or end_sample is None:
        return {"error": f"did not reach step {target_end} within {timeout}s", "log_path": log_path}

    steps_elapsed = end_sample[0] - start_sample[0]
    wall_elapsed = end_sample[1] - start_sample[1]
    if wall_elapsed <= 0 or steps_elapsed <= 0:
        return {"error": "degenerate measurement window (zero elapsed time/steps)", "log_path": log_path}

    return {
        "start_step": start_sample[0],
        "end_step": end_sample[0],
        "wall_elapsed_s": wall_elapsed,
        "steady_state_it_s": steps_elapsed / wall_elapsed,
        "log_path": log_path,
    }


def main():
    args = build_parser().parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(repo_root, ".profile_throughput_runs", f"run_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_dir_root = args.checkpoint_dir_root or os.path.join(out_dir, "checkpoints")
    os.makedirs(checkpoint_dir_root, exist_ok=True)

    print(f"[profile] repo_root   = {repo_root}")
    print(f"[profile] out_dir     = {out_dir}")
    print(f"[profile] concurrency = {args.concurrency}")
    print(f"[profile] num_gpus    = {args.num_gpus}")

    procs = []
    log_paths = []
    for i in range(args.concurrency):
        gpu_id = i % args.num_gpus
        job_checkpoint_dir = os.path.join(checkpoint_dir_root, f"job{i}")
        os.makedirs(job_checkpoint_dir, exist_ok=True)
        log_path = os.path.join(out_dir, f"job{i}.log")
        log_paths.append(log_path)

        cmd = [
            sys.executable, "-u", "run.py",
            "--experiment", args.experiment,
            "--config_name", args.config_name,
        ]
        for ov in args.overrides:
            cmd += ["--overrides", ov]
        # unique seed per parallel copy so they aren't bit-identical runs -
        # matches the real campaign's per-job seed separation convention
        # (.claude/compute-and-data-safety.md's "Seed and Process Separation").
        cmd += ["--overrides", f"seed={900 + i}"]
        cmd += [
            "--checkpoint_dir", job_checkpoint_dir,
            "--checkpoint_interval", "100000",
            "--checkpoint_start_frac", "0.99",  # not testing checkpointing here - keep it out of the way
        ]

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"[profile] launching job{i} on GPU {gpu_id}: {' '.join(cmd)}")
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=logf, stderr=subprocess.STDOUT)
        procs.append(proc)

    results = []
    for i, log_path in enumerate(log_paths):
        print(f"[profile] measuring job{i} ({log_path})...")
        result = measure_one_job(log_path, args.min_length, args.warmup_margin_steps, args.window_steps, args.timeout)
        result["job_index"] = i
        results.append(result)
        status = "OK" if "error" not in result else f"ERROR: {result['error']}"
        print(f"[profile] job{i}: {status}")

    for proc in procs:
        if proc.poll() is None:
            proc.kill()

    print("\n=== RESULTS ===")
    ok_results = [r for r in results if "error" not in r]
    for r in results:
        if "error" in r:
            print(f"job{r['job_index']}: FAILED - {r['error']} (log: {r['log_path']})")
        else:
            print(
                f"job{r['job_index']}: steady_state_it_s={r['steady_state_it_s']:.3f} "
                f"(steps {r['start_step']}->{r['end_step']}, {r['wall_elapsed_s']:.1f}s)"
            )

    if ok_results:
        per_job = [r["steady_state_it_s"] for r in ok_results]
        aggregate = sum(per_job)
        print(f"\nper-job it/s: {[f'{v:.3f}' for v in per_job]}")
        print(f"aggregate it/s (sum across {len(ok_results)} successful jobs): {aggregate:.3f}")
        if len(ok_results) < args.concurrency:
            print(f"WARNING: only {len(ok_results)}/{args.concurrency} jobs completed the measurement window.")
    else:
        print("\nNo job completed the measurement window - nothing to report.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
