#!/bin/bash
#SBATCH --job-name=angle1_retry
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
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
# Fixes 2026-09-25/26 incident (job 22338067, exit 101): conda activate
# was called without sourcing conda's shell hook first, which fails
# non-interactively ("CondaError: Run 'conda init' before 'conda
# activate'") - and because nothing halted the script, it went on to
# launch all ~150 parallel jobs anyway, against an unactivated
# environment, so every one of them failed too. set -e ensures a failure
# like that stops the script immediately instead of silently continuing.
set -e

# WANDB_API_KEY must be exported in the submitting shell before running
# sbatch - e.g.:
#   export WANDB_API_KEY=... && sbatch scripts/run_angle1_a100x4_full.sh
# --export=ALL above (SLURM's own default; made explicit here, confirmed
# working this way earlier in the same investigation) propagates it into
# the job. Never hardcode a real key in this file - the previous version
# of this script had a literal placeholder value here instead
# (WANDB_API_KEY=your_actual_key_here), which would have been the next
# failure once the conda issue above was fixed.

module reset
module load parallel/20240822
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.10

cd /work/hdd/biqc/skaveti1/scaling-drl-research/main

parallel -j 32 --delay 5 --joblog joblog_$(date +%Y%m%d_%H%M%S).txt \
  'CUDA_VISIBLE_DEVICES=$(( ({%} - 1) % 4 )) bash -c {}' :::: job_list.txt
