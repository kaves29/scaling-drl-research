#!/usr/bin/env bash
# Separate owner-approved Exp2 arm allocation. No parent training starts here.
set -euo pipefail
cd "${EXP2_SCHEDULER_MAIN:?}"
test -n "${EXP2_QUEUE:-}" && test -n "${EXP2_APPROVAL:-}"
test -n "${EXP2_APPROVAL_SHA256:-}" && test -n "${EXP2_SCHEDULER_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$EXP2_SCHEDULER_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
test "$(sha256sum "$EXP2_APPROVAL" | cut -d' ' -f1)" = "$EXP2_APPROVAL_SHA256"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
python -m scripts.exp96_exp2_arms discover --queue "$EXP2_QUEUE" --approval "$EXP2_APPROVAL"
srun --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" \
  python -u -m scripts.exp96_exp2_arms worker --queue "$EXP2_QUEUE" --approval "$EXP2_APPROVAL"
