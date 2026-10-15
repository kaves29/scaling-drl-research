#!/usr/bin/env bash
# Versioned Method C ten-function parent worker. Submission is external and approved.
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${EXP96_QUEUE:?}" "${EXP96_ALLOCATION_APPROVAL:?}"
: "${EXP96_APPROVAL_SHA256:?}" "${SLURM_JOB_ID:?}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/exp96_campaign.py scripts/exp12_dynamic_scheduler.py \
  scripts/sbatch_exp96_allocation.sh run.py configs/base_exp12_ten.yaml
test -z "$(git status --porcelain --untracked-files=all)"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=online
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" --ntasks=1 --gpus-per-task=1 \
  python -u scripts/exp96_campaign.py worker \
  --queue "$EXP96_QUEUE" --approval "$EXP96_ALLOCATION_APPROVAL"
