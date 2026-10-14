#!/usr/bin/env bash
# Scheduler-only pilot. The seven training subprocesses execute from a separate
# clean checkout of the frozen scientific source, verified by the Python worker.
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
set -euo pipefail
: "${EXP12_PILOT_SPEC:?}" "${EXP12_PILOT_SPEC_SHA256:?}" "${SLURM_JOB_ID:?}"
cd "${SLURM_SUBMIT_DIR:?}"
test -z "$(git status --porcelain --untracked-files=all)"
test "$(sha256sum "$EXP12_PILOT_SPEC" | cut -d' ' -f1)" = "$EXP12_PILOT_SPEC_SHA256"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=online
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
python -u scripts/exp12_seven_parent_pilot.py worker --spec "$EXP12_PILOT_SPEC"
