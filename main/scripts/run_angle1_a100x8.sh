#!/bin/bash
#SBATCH --job-name=angle1_phase1
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=35:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=ALL

# set -e placed here (after the #SBATCH block, not right after the shebang):
# sbatch stops reading #SBATCH directives at the first line that isn't
# blank or a comment, so any real command placed before them would cause
# every directive above to be silently ignored (no GPU, no partition, no
# time limit requested). This is still "immediately after the shebang" in
# effect - the first executable line - it just has to come after the
# directive block, not before it.
#
# Fixes 2026-09-25/26 incident (job 22338067, exit 101, on the earlier
# gpuA100x4 version of this script): conda activate was called without
# sourcing conda's shell hook first, which fails non-interactively
# ("CondaError: Run 'conda init' before 'conda activate'") - and because
# nothing halted the script, it went on to launch every parallel job
# anyway, against an unactivated environment, so all of them failed too.
# set -e ensures a failure like that stops the script immediately instead
# of silently continuing.
set -e

# WANDB_API_KEY must be exported in the submitting shell before running
# sbatch - e.g.:
#   export WANDB_API_KEY=... && sbatch scripts/run_angle1_a100x8.sh
# --export=ALL above (SLURM's own default; made explicit here, confirmed
# working this way earlier in the same investigation) propagates it into
# the job. Never hardcode a real key in this file.

module reset
module load parallel/20240822
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.10

cd /work/hdd/biqc/skaveti1/scaling-drl-research/main

# Concurrency: 32 total slots (-j 32), 4 jobs sharing each of the 8 GPUs.
#
# Reasoning (2026-09-26, verified against real Delta node specs, not
# assumed): Lightning AI validated 3 jobs/GPU as safe on L4s, which have
# 12 CPUs/GPU there - so 12/3 = 4 CPUs/job is the proven-safe per-job CPU
# budget. VRAM is not the tightening constraint moving to A100 (40/80GB,
# far more headroom than an L4's 24GB), so that 4-CPUs/job budget carries
# over directly. gpuA100x8 nodes have 128 CPUs / 8 GPUs = 16 CPUs/GPU
# (confirmed via `scontrol show node`, not assumed from the partition
# name) - at 4 CPUs/job, that's 16/4 = 4 jobs/GPU, x 8 GPUs = 32 total
# slots. (This happens to match the -j 32 the earlier gpuA100x4 version
# of this script already used - independently re-derived here, not just
# carried forward - gpuA100x4 nodes use the same 16-CPUs/GPU ratio: 64
# CPUs / 4 GPUs.)
#
# Phase ordering (strict size-first, to protect onset-analysis validity -
# see .claude/research-methodology.md): finish ALL of phase1_jobs.txt
# (every remaining D2W512 job + all 50 baseline-calibration-pool jobs)
# before starting phase2_jobs.txt (D5W768), and finish all of phase2
# before phase3_jobs.txt (D7W1024). To launch a later phase, swap the
# filename on the line below - e.g.:
#   parallel ... :::: phase2_jobs.txt
#   parallel ... :::: phase3_jobs.txt
# Only launch the next phase once scripts/check_manifest_overlap.py and a
# manual review confirm the prior phase's jobs are actually done - this
# script does not check that for you.

parallel -j 32 --delay 5 --joblog joblog_$(date +%Y%m%d_%H%M%S).txt \
  'CUDA_VISIBLE_DEVICES=$(( ({%} - 1) % 8 )) bash -c {}' :::: phase1_jobs.txt
