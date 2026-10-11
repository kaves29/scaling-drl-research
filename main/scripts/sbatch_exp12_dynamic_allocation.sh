#!/usr/bin/env bash
# Exact GPUs/nodes/CPU/RAM/time are supplied by a reviewed allocation approval.
# This is an allocation worker only; it never submits another Slurm job.
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${EXP12_QUEUE:?}" "${EXP12_ALLOCATION_APPROVAL:?}"
: "${EXP12_APPROVAL_SHA256:?}" "${EXP12_ALLOCATION:?}" "${SLURM_JOB_ID:?}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/exp12_dynamic_scheduler.py scripts/exp12_launch_plan.py \
  scripts/sbatch_exp12_dynamic_allocation.sh scripts/check_exp12_hb_environment.py \
  run.py configs/base_exp12.yaml
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
case "$EXP12_ALLOCATION" in
  A) nodes=1;;
  B) nodes=2;;
  *) echo "invalid allocation name: $EXP12_ALLOCATION" >&2; exit 2;;
esac
test "$(sha256sum "$EXP12_ALLOCATION_APPROVAL" | cut -d' ' -f1)" = "$EXP12_APPROVAL_SHA256"
# One persistent worker on each allocated four-A100 node. The shared durable
# queue/claims coordinate workers within B and between overlapping A and B.
srun --ntasks="$nodes" --ntasks-per-node=1 --gpus-per-task=4 \
  python -u scripts/exp12_dynamic_scheduler.py worker \
  --queue "$EXP12_QUEUE" --approval "$EXP12_ALLOCATION_APPROVAL" \
  --allocation "$EXP12_ALLOCATION"
