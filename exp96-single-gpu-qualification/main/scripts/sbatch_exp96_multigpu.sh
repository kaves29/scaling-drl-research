#!/usr/bin/env bash
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x8
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${EXP96_ROOT:?}" "${EXP96_ACTION:?}" "${EXP96_BUDGET:?}" "${EXP96_RESERVATION:?}" "${SLURM_JOB_ID:?}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/exp96_multigpu.py scripts/exp96_multigpu_qualification.py \
  scripts/exp96_campaign.py scripts/exp12_dynamic_scheduler.py scripts/sbatch_exp96_multigpu.sh \
  run.py configs/base_exp12_ten.yaml
test -z "$(git status --porcelain --untracked-files=all)"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
# Admission/qualification controllers remain CPU-only. Each child gets one
# physical GPU UUID, disjoint allocated CPUs and a private compilation cache.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" --gpus-per-task="${SLURM_GPUS_ON_NODE:?}" \
  python -u scripts/exp96_multigpu.py "$EXP96_ACTION" --root "$EXP96_ROOT"
