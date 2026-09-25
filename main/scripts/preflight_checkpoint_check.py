#!/usr/bin/env python3
"""Mandatory pre-launch smoke test for checkpoint correctness.

Launches exactly one real run.py job at a caller-specified architecture/
environment config, with a small step count and checkpoint_interval so a
real checkpoint is reachable in minutes, and verifies a real checkpoint
(meta.pkl, not just a directory) actually lands on disk.

Must be run and return PASS before any multi-job parallel launch is
considered safe. See .claude/research-methodology.md's "Infrastructure
Safety Rules" section (added 2026-09-21, after a 150-run/35-hour campaign
produced zero completed checkpoints due to a relative --checkpoint_dir
being rejected by orbax on every single run).

Usage (run from a GPU-visible session, i.e. inside a Slurm allocation with
the project's conda env activated):

    python scripts/preflight_checkpoint_check.py \\
        --override critic_num_blocks=2 --override critic_hidden_dim=512 \\
        --override updates_per_interaction_step=5 \\
        --override env_name=dog-run --override env=dmc_hard \\
        --override seed=999 --override project_name=EchoCritic-preflight-test
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="angle_1")
    parser.add_argument("--config_name", default="base_sac")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        dest="overrides",
        help="Extra --overrides to forward to run.py (repeatable), e.g. "
        "--override env_name=dog-run --override env=dmc_hard",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=1000,
        help="Small step count so the run finishes in minutes, not hours.",
    )
    parser.add_argument(
        "--checkpoint_start_frac",
        type=float,
        default=0.2,
        help="Matches the production default so the test exercises the real trigger fraction.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=200,
        help="Small interval so a checkpoint step is actually reachable within num_env_steps.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Seconds to wait before declaring FAIL (timeout).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Absolute dir to checkpoint into. Defaults to a fresh tempdir (auto-created, absolute).",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if args.checkpoint_dir is None:
        checkpoint_dir = tempfile.mkdtemp(prefix="preflight_ckpt_")
    else:
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = os.path.join(checkpoint_dir, "preflight_run.log")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cmd = [
        sys.executable, "-u", "run.py",
        "--experiment", args.experiment,
        "--config_name", args.config_name,
        "--overrides", f"num_env_steps={args.num_env_steps}",
    ]
    for ov in args.overrides:
        cmd += ["--overrides", ov]
    cmd += [
        "--checkpoint_dir", checkpoint_dir,
        "--checkpoint_interval", str(args.checkpoint_interval),
        "--checkpoint_start_frac", str(args.checkpoint_start_frac),
    ]

    print(f"[preflight] repo_root      = {repo_root}")
    print(f"[preflight] checkpoint_dir = {checkpoint_dir}")
    print(f"[preflight] log            = {log_path}")
    print(f"[preflight] command        = {' '.join(cmd)}")
    print(f"[preflight] timeout        = {args.timeout}s")

    start = time.time()
    timed_out = False
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT)
        try:
            returncode = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            returncode = None
            timed_out = True
    elapsed = time.time() - start

    meta_path = os.path.join(checkpoint_dir, "meta.pkl")
    ckpt_ok = os.path.isfile(meta_path) and os.path.getsize(meta_path) > 0

    print(f"[preflight] elapsed        = {elapsed:.1f}s")
    print(f"[preflight] returncode     = {returncode}")
    print(f"[preflight] timed_out      = {timed_out}")
    print(f"[preflight] meta.pkl found = {ckpt_ok} ({meta_path})")

    if ckpt_ok and not timed_out:
        print("\n=== PASS ===")
        print(f"Real checkpoint written to: {checkpoint_dir}")
        print(
            f"meta.pkl: {meta_path} "
            f"({os.path.getsize(meta_path)} bytes, mtime {time.ctime(os.path.getmtime(meta_path))})"
        )
        return 0

    print("\n=== FAIL ===")
    if timed_out:
        print(f"Run did not finish within {args.timeout}s and was killed.")
    elif returncode != 0:
        print(f"run.py exited with code {returncode}.")
    else:
        print("run.py exited 0 but no checkpoint (meta.pkl) was found - checkpoint logic may not have fired.")

    print(f"\n--- traceback / tail from {log_path} ---")
    try:
        with open(log_path, "r", errors="ignore") as f:
            text = f.read()
    except OSError as e:
        print(f"(could not read log: {e})")
        return 1

    lines = text.splitlines()
    tb_idx = None
    for i, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            tb_idx = i
    if tb_idx is not None:
        for line in lines[tb_idx:]:
            print(line)
    else:
        for line in lines[-40:]:
            print(line)

    return 1


if __name__ == "__main__":
    sys.exit(main())
