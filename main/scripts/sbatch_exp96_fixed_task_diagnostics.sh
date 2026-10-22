#!/usr/bin/env bash
# Single-GPU diagnostics only. Scheduling resources are supplied by the submitter.
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${DIAGNOSTIC_ROOT:?}" "${FROZEN_CAMPAIGN:?}" "${SLURM_JOB_ID:?}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/exp96_fixed_task_diagnostics.py experiments/exp12/fixed_task.py \
  experiments/exp12/diagnostic_artifacts.py scripts/sbatch_exp96_fixed_task_diagnostics.sh
test -z "$(git status --porcelain --untracked-files=all)"
test "${SLURM_GPUS_ON_NODE:?}" = 1
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled
export EXP12_JAX_CACHE_DIR=off
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" --gpus-per-task=1 \
  timeout --signal=TERM --kill-after=60 6900 \
  python -u scripts/exp96_fixed_task_diagnostics.py --root "$DIAGNOSTIC_ROOT" \
  --campaign "$FROZEN_CAMPAIGN" --commit "$EXPECTED_COMMIT" --cap-seconds 6600
