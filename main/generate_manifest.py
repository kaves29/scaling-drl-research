import os
import pickle
from pathlib import Path

import pandas as pd

from analysis.baseline_calibration_pool import BASELINE_ARCHITECTURE, POOL_EXPERIMENT, POOL_SEEDS, POOL_STORAGE_ROOT
from analysis.metrics_store import RunIdentity, metrics_path

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


def classify(ckpt_dir, experiment, arch_name, env_name, seed, steps, snapshot):
    """Returns (status, reason); status is one of fresh, resume, done,
    done_legacy, review. Only fresh and resume jobs are emitted."""
    ckpt = Path(ckpt_dir)
    if not ckpt.exists() or not any(ckpt.iterdir()):
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
    missing = []
    if _last_env_step(metrics_path(identity)) != steps:
        missing.append(f"metrics CSV at final env_step ({metrics_path(identity)})")
    if snapshot:
        snap = (
            Path(POOL_STORAGE_ROOT) / env_name / arch_name / f"seed{seed}"
            / "baseline_pool" / "checkpoints" / "pool" / "probe_capture_state.pkl"
        )
        if not snap.exists():
            missing.append(f"pool snapshot ({snap})")
    if missing:
        return "review", "training finished but end-of-run artifacts missing: " + "; ".join(missing)
    return "done_legacy", "pre-DONE-marker run; all end-of-run artifacts verified"


def add_jobs(jobs, status, env_list, steps, archs=ARCHS, seeds=SEEDS, experiment="angle_1", prefix="", extra=()):
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
                cmd += (
                    f"--overrides project_name=EchoCritic-{experiment} "
                    f"--checkpoint_dir {ckpt_dir} "
                    f"--checkpoint_interval 12501 "
                    f"--checkpoint_start_frac 0.15 "
                    f"> {log_path} 2>&1"
                )
                state, reason = classify(
                    ckpt_dir, experiment, arch_name, env_name, seed, steps, snapshot=SNAPSHOT_FLAG in cmd,
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


def main():
    status = {}
    manifests = {"job_list.txt": [], "job_list_pool.txt": []}
    add_grid(manifests["job_list.txt"], status)
    add_grid(
        manifests["job_list_pool.txt"], status, archs=POOL_ARCHS, seeds=POOL_SEEDS,
        experiment=POOL_EXPERIMENT, prefix=f"{POOL_EXPERIMENT}/", extra=POOL_EXTRA,
    )

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
