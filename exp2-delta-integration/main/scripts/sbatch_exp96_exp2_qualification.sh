#!/usr/bin/env bash
# Prepared only: no Exp2 GPU spending is authorized by this file.
set -euo pipefail
: "${EXP2_QUALIFICATION_PLAN:?}" "${EXP2_PHASE:?}" "${SLURM_JOB_ID:?}"
cd "${EXP2_SCHEDULER_MAIN:?}"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" --gpus-per-task=1 \
  python -u -m scripts.exp96_exp2_delta qualification \
    --plan "$EXP2_QUALIFICATION_PLAN" --phase "$EXP2_PHASE"
