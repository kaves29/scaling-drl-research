import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analysis.baseline_calibration_pool import BASELINE_ARCHITECTURE, POOL_EXPERIMENT, POOL_SEEDS
from analysis.metrics_store import RunIdentity, metrics_path
from utils.atomic_io import atomic_write_text
from utils.paths import RESULTS_ROOT, require_absolute, results_path

# Safe launch pattern (added 2026-09-21, after the 2026-09-20/21 incident:
# a relative --checkpoint_dir crashed 100% of a 150-run campaign, and a
# duplicated manifest launched 50 of those configs twice concurrently).
# No prior example of this launch pattern existed anywhere in this repo -
# this is the first one. Before running `parallel` against two or more
# manifests intended to run concurrently, or after regenerating a
# manifest, always:
#
#   python scripts/check_manifest_overlap.py job_list.txt job_list_part2.txt
#   python scripts/preflight_checkpoint_check.py --override env_name=... --override env=...
#
# and only launch if both return PASS/OK. Use a uniquely-named --joblog
# per invocation (never a fixed "joblog.txt" reused across launches -
# GNU parallel truncates --joblog on a fresh invocation unless --resume
# is passed, silently destroying the previous invocation's exit-code
# evidence), e.g.:
#
#   parallel -j 64 --joblog "joblog_$(date +%Y%m%d_%H%M%S).txt" \
#       CUDA_VISIBLE_DEVICES='$(( ({%}-1) % 8 ))' bash -c {} :::: job_list.txt
#
# Throughput Step 4 (2026-09-24, added to the launch environment, not
# training code - not yet measured on real GPU hardware; see
# scripts/profile_throughput.py's aggregate-throughput comparison before
# relying on this): under many concurrent single-GPU JAX processes sharing
# a node's CPUs, each process's host-side numpy/Eigen/BLAS calls may
# default to spawning as many threads as visible CPUs, causing severe
# oversubscription (64 processes x many threads each, on a node with far
# fewer physical CPUs than that product). Cap each process to one thread
# per library before launching:
#
#   export OMP_NUM_THREADS=1
#   export MKL_NUM_THREADS=1
#   export OPENBLAS_NUM_THREADS=1
#   export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

ARCHS = [("D2W512", 2, 512), ("D5W768", 5, 768), ("D7W1024", 7, 1024)]
DMC_HARD = [("dog-run", "dmc_hard"), ("dog-trot", "dmc_hard"), ("humanoid-run", "dmc_hard")]
DMC_MEDIUM = [("cheetah-run", None), ("quadruped-run", None), ("manipulator-bring_ball", None)]
MYO_HARD = [("myo-reach", "myosuite_hard"), ("myo-key-turn", "myosuite_hard"), ("myo-leg-walk", "myosuite_hard")]
MYO_MEDIUM = [("myo-elbow-pose-random", "myosuite_medium")]

SEEDS = [1, 2, 3, 4, 5]
# Raw env steps, matching the original SimBa paper's benchmarks (see
# research-methodology.md). All MyoSuite tasks share one budget.
HARD_STEPS = 1_000_000
MED_STEPS = 500_000
MYO_STEPS = 1_000_000

SNAPSHOT_FLAG = "+save_probe_capture_snapshot=true"
DONE_MARKER = "DONE"  # experiments/angle_1.py's DONE_MARKER


def _last_env_step(csv_path):
    try:
        df = pd.read_csv(csv_path, usecols=["env_step"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None
    return int(df["env_step"].max()) if len(df) else None


def _has_any_files(path):
    """True if `path` contains at least one regular file, at any depth.
    A checkpoint dir holding only empty subdirectories (however nested -
    e.g. a leftover empty logs/ from a crashed campaign that never wrote a
    CSV) has no real training artifacts and must be indistinguishable from
    a never-started one - `any(path.iterdir())` alone doesn't catch this,
    since an empty subdirectory is still one entry."""
    return any(p.is_file() for p in path.rglob("*"))


def classify(ckpt_dir, experiment, arch_name, env_name, seed, steps, snapshot, results_root=RESULTS_ROOT):
    """Returns (status, reason); status is one of fresh, resume, done,
    done_legacy, review. Only fresh and resume jobs are emitted."""
    ckpt = Path(ckpt_dir)
    if not ckpt.exists() or not _has_any_files(ckpt):
        return "fresh", ""
    if (ckpt / DONE_MARKER).exists():
        return "done", ""
    if not (ckpt / "meta.pkl").exists():
        return "review", "started but never checkpointed (no meta.pkl)"
    try:
        with open(ckpt / "meta.pkl", "rb") as f:
            int(pickle.load(f)["interaction_step"])
    except Exception as e:
        return "review", f"unreadable meta.pkl: {e!r}"

    logs = sorted((ckpt / "logs").glob("*.csv"))
    if len(logs) != 1:
        return "review", f"expected exactly one logs/*.csv, found {len(logs)}"
    last = _last_env_step(logs[0])
    if last is None or last < steps:
        return "resume", f"resumes from the checkpoint in meta.pkl (log at env_step {last})"

    # Training finished but no DONE marker: a run from before the marker
    # existed, or one that crashed during its end-of-run writes. Never
    # resume these - resuming retrains the tail and overwrites final data.
    identity = RunIdentity(experiment=experiment, architecture=arch_name, environment=env_name, seed=seed)
    metrics_csv = metrics_path(identity, root=results_path("metrics", results_root=results_root))
    missing = []
    if _last_env_step(metrics_csv) != steps:
        missing.append(f"metrics CSV at final env_step ({metrics_csv})")
    if snapshot:
        snap = (
            Path(results_path("baseline_calibration_pool", results_root=results_root)) / env_name / arch_name / f"seed{seed}"
            / "baseline_pool" / "checkpoints" / "pool" / "probe_capture_state.pkl"
        )
        if not snap.exists():
            missing.append(f"pool snapshot ({snap})")
    if missing:
        return "review", "training finished but end-of-run artifacts missing: " + "; ".join(missing)
    return "done_legacy", "pre-DONE-marker run; all end-of-run artifacts verified"


def add_jobs(
    jobs, status, env_list, steps, archs=ARCHS, seeds=SEEDS, experiment="angle_1", prefix="", extra=(),
    results_root=RESULTS_ROOT,
):
    for env_name, env_type in env_list:
        for arch_name, blocks, hidden in archs:
            for seed in seeds:
                ckpt_dir = os.path.abspath(f"./angle1_prod/{prefix}{arch_name}/{env_name}/seed_{seed}")
                log_path = f"./angle1_logs/{prefix.replace('/', '_')}{arch_name}_{env_name}_seed{seed}.log"
                cmd = (
                    f"python -u run.py --experiment {experiment} --config_name base_sac "
                    f"--overrides critic_num_blocks={blocks} "
                    f"--overrides critic_hidden_dim={hidden} "
                    f"--overrides updates_per_interaction_step=5 "
                    f"--overrides num_env_steps={steps} "
                    f"--overrides env_name={env_name} "
                )
                if env_type:
                    cmd += f"--overrides env={env_type} "
                if arch_name == "D2W512" and seed in (1, 2, 3, 4, 5):
                    cmd += f"--overrides {SNAPSHOT_FLAG} "
                cmd += (
                    f"--overrides seed={seed} "
                    f"--overrides critic_degradation=true "
                    f"--overrides pathology_prop=true "
                )
                for override in extra:
                    cmd += f"--overrides {override} "
                if results_root != RESULTS_ROOT:
                    cmd += f"--overrides results_root={results_root} "
                cmd += (
                    f"--overrides project_name=EchoCritic-{experiment} "
                    f"--checkpoint_dir {ckpt_dir} "
                    f"--checkpoint_interval 12501 "
                    f"--checkpoint_start_frac 0.15 "
                    f"> {log_path} 2>&1"
                )
                state, reason = classify(
                    ckpt_dir, experiment, arch_name, env_name, seed, steps, snapshot=SNAPSHOT_FLAG in cmd,
                    results_root=results_root,
                )
                status.setdefault(state, []).append((ckpt_dir, reason))
                if state in ("fresh", "resume"):
                    jobs.append(cmd)


def add_grid(jobs, status, **kwargs):
    add_jobs(jobs, status, DMC_HARD, HARD_STEPS, **kwargs)
    add_jobs(jobs, status, MYO_HARD, MYO_STEPS, **kwargs)
    add_jobs(jobs, status, DMC_MEDIUM, MED_STEPS, **kwargs)
    add_jobs(jobs, status, MYO_MEDIUM, MYO_STEPS, **kwargs)


# Shared baseline-calibration pool's dedicated agents (seeds 6-10, see
# analysis/baseline_calibration_pool.py): same training as the D2W512 grid
# runs, under their own experiment name and checkpoint subtree.
POOL_ARCHS = [a for a in ARCHS if a[0] == BASELINE_ARCHITECTURE]
POOL_EXTRA = (SNAPSHOT_FLAG, "onset_detection.baseline_experiment=angle_1")


def backfill_done(status):
    """Writes DONE for runs classified done_legacy - verified here, on the
    machine holding their files - so later manifests skip them outright."""
    for ckpt_dir, reason in status.pop("done_legacy", []):
        atomic_write_text(Path(ckpt_dir) / DONE_MARKER, json.dumps({
            "backfilled_at": datetime.now(timezone.utc).isoformat(),
            "backfilled_by": "generate_manifest.py --backfill-done",
            "verified": reason,
        }))
        status.setdefault("done", []).append((ckpt_dir, "DONE backfilled"))
        print(f"wrote {Path(ckpt_dir) / DONE_MARKER}")


# ---------------------------------------------------------------------------
# Experiments 1 and 2 (.claude/methodology-exp1-exp2.md). `--grid exp12` writes
#   exp12_exp1_jobs.txt          the 195-run Exp 1 grid (3 critics x 13 envs x 5 seeds)
#   exp2_arms_<device>.txt       one injected-arm job per completed fork, grouped by the
#                                GPU model that produced the fork (amendment (p)); only
#                                with --injection-m, i.e. once the positive control froze m
# The control arm is the Exp 1 job itself. Every path is absolute; logs go next to the
# checkpoints. Budgets come from the env groups (500k swimmer/hopper, 2M HumanoidBench,
# 1M otherwise; checked by tests/test_exp12_manifest.py).
EXP12_ARCHS = [("D2W512", 2, 512), ("D4W1024", 4, 1024), ("D6W1536", 6, 1536)]
EXP12_ENVS = [
    ("dog-run", "dmc_hard"), ("dog-trot", "dmc_hard"), ("humanoid-run", "dmc_hard"),
    ("humanoid-walk", "dmc_hard"), ("humanoid-stand", "dmc_hard"),
    ("swimmer-swimmer15", "dmc_medium"), ("hopper-hop", "dmc_medium"),
    ("myo-key-turn", "myosuite_simba"), ("myo-pen-twirl", "myosuite_simba"),
    ("myo-pose-hard", "myosuite_simba"), ("myo-reach", "myosuite_simba"),
    ("h1-reach-v0", "humanoid_bench"), ("h1-run-v0", "humanoid_bench"),
]
EXP12_BUDGETS = {"dmc_hard": 1_000_000, "dmc_medium": 500_000, "myosuite_simba": 1_000_000,
                 "humanoid_bench": 2_000_000}
EXP12_FORKING = ("D4W1024", "D6W1536")
EXP12_SEEDS = [1, 2, 3, 4, 5]
EXP12_CHECKS = 20
EXP12_ACTION_REPEAT = 2


def exp12_paths(ckpt_root, arch_name, env_name, seed):
    run_dir = os.path.join(ckpt_root, "exp1", arch_name, env_name, f"seed_{seed}")
    arm_dir = os.path.join(ckpt_root, "exp2_arm", arch_name, env_name, f"seed_{seed}", "injected")
    return run_dir, arm_dir


def exp12_overrides(arch, env_name, env_group, seed, results_root):
    _, blocks, hidden = arch
    return [f"critic_num_blocks={blocks}", f"critic_hidden_dim={hidden}", f"env_name={env_name}",
            f"env={env_group}", f"seed={seed}", f"results_root={results_root}"]


def exp12_command(experiment, overrides, ckpt_dir, interval, log_path):
    cmd = f"python -u run.py --experiment {experiment} --config_name base_exp12 "
    cmd += "".join(f"--overrides {o} " for o in overrides)
    return (cmd + f"--checkpoint_dir {ckpt_dir} --checkpoint_interval {interval} "
            f"--checkpoint_start_frac 0.0 > {log_path} 2>&1")


def classify_exp12(ckpt_dir):
    """done (DONE marker), fresh (nothing on disk) or resume (exp1/exp2_arm resume from
    state/LATEST, from the saved fork state, or start over before their first save)."""
    ckpt = Path(ckpt_dir)
    if (ckpt / DONE_MARKER).exists():
        return "done"
    if not ckpt.exists() or not _has_any_files(ckpt):
        return "fresh"
    return "resume"


def _device_slug(device_kind):
    return "".join(c if c.isalnum() else "_" for c in device_kind).strip("_") or "unknown"


def add_exp12_grid(manifests, status, ckpt_root, results_root, injection_m=None, seeds=EXP12_SEEDS):
    """Exp 1 jobs into manifests["exp12_exp1_jobs.txt"]; injected-arm jobs (forks only) into
    manifests["exp2_arms_<device>.txt"]. status: state -> list of checkpoint dirs."""
    if injection_m is not None and injection_m not in ("last", "half", "all"):
        raise ValueError("--injection-m must be last, half or all")
    logs = os.path.join(ckpt_root, "logs")
    exp1_jobs = manifests.setdefault("exp12_exp1_jobs.txt", [])
    for env_name, env_group in EXP12_ENVS:
        interval = EXP12_BUDGETS[env_group] // EXP12_ACTION_REPEAT // EXP12_CHECKS  # one save per check
        for arch in EXP12_ARCHS:
            for seed in seeds:
                run_dir, arm_dir = exp12_paths(ckpt_root, arch[0], env_name, seed)
                overrides = exp12_overrides(arch, env_name, env_group, seed, results_root)
                state = classify_exp12(run_dir)
                status.setdefault(f"exp1 {state}", []).append(run_dir)
                if state != "done":
                    log = os.path.join(logs, f"exp1_{arch[0]}_{env_name}_seed{seed}.log")
                    exp1_jobs.append(exp12_command("exp1", overrides, run_dir, interval, log))
                fork_json = Path(run_dir) / "fork" / "fork.json"
                if arch[0] not in EXP12_FORKING or not (Path(run_dir) / "fork" / "FORK_READY").exists():
                    continue
                arm_state = classify_exp12(arm_dir)
                status.setdefault(f"exp2 arm {arm_state}", []).append(arm_dir)
                if arm_state == "done":
                    continue
                if injection_m is None:
                    status.setdefault("exp2 arm waiting for --injection-m", []).append(arm_dir)
                    continue
                device = json.loads(fork_json.read_text()).get("device", {}).get("device_kind", "unknown")
                arm_overrides = overrides + [f"fork.source={run_dir}", "fork.arm=injected", f"injection.m={injection_m}"]
                log = os.path.join(logs, f"exp2_injected_{arch[0]}_{env_name}_seed{seed}.log")
                manifests.setdefault(f"exp2_arms_{_device_slug(device)}.txt", []).append(
                    exp12_command("exp2_arm", arm_overrides, arm_dir, interval, log))
    os.makedirs(logs, exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backfill-done", action="store_true",
        help="Write DONE into every run verified complete that predates the marker.",
    )
    parser.add_argument(
        "--results-root", default=RESULTS_ROOT,
        help="Absolute results root the jobs write to and completed runs are verified against.",
    )
    parser.add_argument("--grid", choices=("angle_1", "exp12"), default="angle_1",
                        help="exp12: the Experiment 1/2 grid (needs --ckpt-root).")
    parser.add_argument("--ckpt-root", default=None, help="exp12: absolute root for checkpoints and logs.")
    parser.add_argument("--injection-m", default=None,
                        help="exp12: the frozen m (last|half|all); without it no arm jobs are written.")
    args = parser.parse_args(argv)
    results_root = require_absolute(args.results_root, "--results-root")

    if args.grid == "exp12":
        ckpt_root = require_absolute(args.ckpt_root or "", "--ckpt-root")
        status, manifests = {}, {}
        add_exp12_grid(manifests, status, ckpt_root, results_root, args.injection_m)
        for path, jobs in manifests.items():
            with open(path, "w") as f:
                f.write("\n".join(jobs) + ("\n" if jobs else ""))
            print(f"{path}: queued {len(jobs)}")
        for state, dirs in sorted(status.items()):
            print(f"{state}: {len(dirs)}")
        return

    status = {}
    manifests = {"job_list.txt": [], "job_list_pool.txt": []}
    add_grid(manifests["job_list.txt"], status, results_root=results_root)
    add_grid(
        manifests["job_list_pool.txt"], status, archs=POOL_ARCHS, seeds=POOL_SEEDS,
        experiment=POOL_EXPERIMENT, prefix=f"{POOL_EXPERIMENT}/", extra=POOL_EXTRA,
        results_root=results_root,
    )
    if args.backfill_done:
        backfill_done(status)

    os.makedirs("./angle1_logs", exist_ok=True)
    for path, jobs in manifests.items():
        with open(path, "w") as f:
            f.write("\n".join(jobs) + "\n")
        print(f"{path}: queued {len(jobs)}")

    for state in ("fresh", "resume", "done", "done_legacy", "review"):
        entries = status.get(state, [])
        print(f"{state}: {len(entries)}")
        if state in ("resume", "done_legacy", "review"):
            for ckpt_dir, reason in entries:
                print(f"  {ckpt_dir}: {reason}")


if __name__ == "__main__":
    main()
