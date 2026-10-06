#!/bin/bash
#SBATCH --job-name=angle1_a40x4
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=47:00:00
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
#   export WANDB_API_KEY=... && sbatch scripts/run_angle1_a40x4.sh
# --export=ALL above (SLURM's own default; made explicit here, confirmed
# working this way earlier in the same investigation) propagates it into
# the job. Never hardcode a real key in this file.

module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# No memory fraction (2026-10-05): .10 capped each process at ~4 GB of a 40 GB A100, and D6W1536 peaks at
# 4.01 GiB (Block A). Jobs sharing a GPU rely on PREALLOCATE=false alone.
unset XLA_PYTHON_CLIENT_MEM_FRACTION

cd /work/hdd/biqc/skaveti1/scaling-drl-research/main

# Concurrency: 16 total slots, 4 jobs sharing each of the 4 GPUs (same
# 4-jobs/GPU ratio as run_angle1_a100x8.sh - gpuA40x4 nodes have 64 CPUs /
# 4 GPUs = 16 CPUs/GPU too, confirmed via `scontrol show node`, so the
# same 4-CPUs/job budget carries over directly here as well).
#
# scripts/claim_launcher.py (2026-09-27) replaces the earlier plain
# `parallel` invocation: this node may be running concurrently with a
# gpuA100x8 allocation (scripts/run_angle1_a100x8.sh) sharing the same
# /work/hdd job queues, so job claiming (atomic os.mkdir per
# --checkpoint_dir, so the same job is never run twice) and phase-ordering
# (no phase2/phase3 job starts until every phase1 job is DONE, checked
# fresh - not just at manifest-generation time, since the OTHER node may
# still be working through phase1) now live in the launcher itself, not
# in this script. It works through phase1_jobs.txt -> phase2_jobs.txt ->
# phase3_jobs.txt automatically in one invocation - no more manually
# swapping filenames between phases.
python scripts/claim_launcher.py --concurrency 16 --num-gpus 4
