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

Experiments 1 and 2 (`--experiment exp1`): a dev-role smoke run of the given
architecture/environment with SMOKE_OVERRIDES_EXP12 (4,000 env steps, tiny probe
settings so 20 checks fit; plumbing only, not the real probe cost). PASS needs a
complete state under state/LATEST and the run's ledger. With --with-fork, f*_run is
forced at check 2 (test hook, dev only), the run must fork, and the injected arm
(exp2_arm) must then complete with Check 1 passing on this device:

    python scripts/preflight_checkpoint_check.py --experiment exp1 --with-fork \\
        --override critic_num_blocks=4 --override critic_hidden_dim=1536 \\
        --override env_name=dog-run --override env=dmc_hard --override seed=999
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

SMOKE_OVERRIDES_EXP12 = [
    "run_role=dev", "num_env_steps=4000", "buffer.min_length=50",
    "probe.steps=10", "probe.pool_size=256", "probe.batch_size=32", "probe.eval_chunk=256",
    "fork.eval_episodes=1",
]


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
    parser.add_argument("--with-fork", action="store_true",
                        help="exp1 only: force a fork and run the injected arm to completion.")
    parser.add_argument("--injection-m", default="last", help="exp1 --with-fork: m for the smoke arm.")
    return parser


def _run(cmd, repo_root, log_path, timeout):
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT)
        try:
            return proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return None, True


def exp12_preflight(args, checkpoint_dir, repo_root):
    """Exp 1/2 smoke run (and optionally fork + injected arm); returns the list of failures."""
    import json
    from pathlib import Path

    run_dir, arm_dir = Path(checkpoint_dir) / "run", Path(checkpoint_dir) / "arm_injected"
    results = Path(checkpoint_dir) / "results"
    overrides = SMOKE_OVERRIDES_EXP12 + list(args.overrides)
    overrides.append(f"results_root={results}")
    if args.with_fork:
        overrides.append("testing.force_trigger_check=2")
    base = [sys.executable, "-u", "run.py", "--config_name", "base_exp12"]
    flags = ["--checkpoint_interval", str(args.checkpoint_interval), "--checkpoint_start_frac", "0.0"]
    ov = [x for o in overrides for x in ("--overrides", o)]
    failures = []
    rc, timed_out = _run(base + ["--experiment", "exp1"] + ov + ["--checkpoint_dir", str(run_dir)] + flags,
                         repo_root, run_dir.parent / "preflight_exp1.log", args.timeout)
    latest = run_dir / "state" / "LATEST"
    print(f"[preflight] exp1 returncode = {rc}, timed_out = {timed_out}")
    if rc != 0 or timed_out:
        failures.append(f"exp1 exited with {rc} (timed out: {timed_out}); see {run_dir.parent / 'preflight_exp1.log'}")
    if not latest.exists() or not (run_dir / "state" / latest.read_text().strip() / "meta.pkl").exists():
        failures.append(f"no complete state under {run_dir / 'state'}")
    if not list((results / "exp12" / "exp1" / "runs").glob("*/run.csv")):
        failures.append(f"no Exp 1 ledger under {results}")
    if args.with_fork and not failures:
        if not (run_dir / "fork" / "FORK_READY").exists():
            failures.append("the run did not fork (is the architecture in fork.architectures?)")
        else:
            arm = ov + ["--overrides", f"fork.source={run_dir}", "--overrides", "fork.arm=injected",
                        "--overrides", f"injection.m={args.injection_m}"]
            rc, timed_out = _run(base + ["--experiment", "exp2_arm"] + arm + ["--checkpoint_dir", str(arm_dir)] + flags,
                                 repo_root, run_dir.parent / "preflight_exp2_arm.log", args.timeout)
            print(f"[preflight] exp2_arm returncode = {rc}, timed_out = {timed_out}")
            check1 = list((results / "exp12" / "exp2").glob("*/check1_injected.json"))
            if rc != 0 or timed_out or not (arm_dir / "DONE").exists():
                failures.append(f"exp2_arm did not complete; see {run_dir.parent / 'preflight_exp2_arm.log'}")
            if not check1:
                failures.append("no check1_injected.json")
            else:
                c1 = json.loads(check1[0].read_text())
                print(f"[preflight] Check 1: pass={c1['pass']} max_eps_units={c1['max_eps_units']:.3g} "
                      f"check1_precision={c1['matmul_precision']}")
                if not c1["pass"]:
                    failures.append("Check 1 failed (see check1_injected.json)")
    return failures


def main():
    args = build_parser().parse_args()

    if args.checkpoint_dir is None:
        checkpoint_dir = tempfile.mkdtemp(prefix="preflight_ckpt_")
    else:
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = os.path.join(checkpoint_dir, "preflight_run.log")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.experiment == "exp1":
        print(f"[preflight] Exp 1/2 smoke run in {checkpoint_dir}")
        failures = exp12_preflight(args, checkpoint_dir, repo_root)
        print("\n=== PASS ===" if not failures else "\n=== FAIL ===\n" + "\n".join(failures))
        return 0 if not failures else 1

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
