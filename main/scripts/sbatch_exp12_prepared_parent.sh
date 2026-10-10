#!/usr/bin/env bash
# No resource defaults: pass the exact approved plan allocation to sbatch.
# This script never submits, retries, steals claims or launches Exp2.
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${LAUNCH_PLAN:?}" "${SLURM_JOB_ID:?}"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/exp12_launch_plan.py run.py experiments/exp1.py configs/base_exp12.yaml
test -z "$(git status --porcelain --untracked-files=all)"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
python -c 'import sys; assert sys.version_info[:3] == (3,12,13), sys.version'
index="${SLURM_ARRAY_TASK_ID:-0}"
extra=()
if [[ "${EXPLICIT_RESUME:-false}" == true ]]; then extra+=(--resume); fi
python -u scripts/exp12_launch_plan.py execute --plan "$LAUNCH_PLAN" --index "$index" "${extra[@]}"
