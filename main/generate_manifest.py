import os

from analysis.baseline_calibration_pool import BASELINE_ARCHITECTURE, POOL_EXPERIMENT, POOL_SEEDS

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

skipped_done, skipped_review = [], []


def add_jobs(jobs, env_list, steps, archs=ARCHS, seeds=SEEDS, experiment="angle_1", prefix="", extra=()):
    for env_name, env_type in env_list:
        for arch_name, blocks, hidden in archs:
            for seed in seeds:
                ckpt_dir = os.path.abspath(f"./angle1_prod/{prefix}{arch_name}/{env_name}/seed_{seed}")
                if os.path.exists(f"{ckpt_dir}/DONE"):
                    skipped_done.append(ckpt_dir); continue
                if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
                    skipped_review.append(ckpt_dir); continue

                os.makedirs("./angle1_logs", exist_ok=True)
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
                    cmd += "--overrides +save_probe_capture_snapshot=true "
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
                jobs.append(cmd)


def add_grid(jobs, **kwargs):
    add_jobs(jobs, DMC_HARD, HARD_STEPS, **kwargs)
    add_jobs(jobs, MYO_HARD, MYO_STEPS, **kwargs)
    add_jobs(jobs, DMC_MEDIUM, MED_STEPS, **kwargs)
    add_jobs(jobs, MYO_MEDIUM, MYO_STEPS, **kwargs)


# Shared baseline-calibration pool's dedicated agents (seeds 6-10, see
# analysis/baseline_calibration_pool.py): same training as the D2W512 grid
# runs, under their own experiment name and checkpoint subtree.
POOL_ARCHS = [a for a in ARCHS if a[0] == BASELINE_ARCHITECTURE]
POOL_EXTRA = ("+save_probe_capture_snapshot=true", "onset_detection.baseline_experiment=angle_1")

manifests = {"job_list.txt": [], "job_list_pool.txt": []}
add_grid(manifests["job_list.txt"])
add_grid(
    manifests["job_list_pool.txt"], archs=POOL_ARCHS, seeds=POOL_SEEDS,
    experiment=POOL_EXPERIMENT, prefix=f"{POOL_EXPERIMENT}/", extra=POOL_EXTRA,
)

for path, jobs in manifests.items():
    with open(path, "w") as f:
        f.write("\n".join(jobs) + "\n")
    print(f"{path}: queued {len(jobs)}")

print(f"Already done, skipped: {len(skipped_done)}")
print(f"Partial/needs review, skipped: {len(skipped_review)}")
for p in skipped_review:
    print(f"  REVIEW: {p}")
