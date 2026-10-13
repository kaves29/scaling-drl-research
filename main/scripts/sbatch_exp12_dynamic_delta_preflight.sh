#!/usr/bin/env bash
# Diagnostic only: two four-A100 nodes, atomic shared-mount claims and binding.
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${EXP12_PREFLIGHT_ROOT:?}" "${SLURM_JOB_ID:?}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/exp12_dynamic_delta_preflight.py \
  scripts/sbatch_exp12_dynamic_delta_preflight.sh scripts/exp12_dynamic_scheduler.py
test -z "$(git status --porcelain --untracked-files=all)"
[[ "$EXP12_PREFLIGHT_ROOT" = /* ]] && [ ! -e "$EXP12_PREFLIGHT_ROOT" ]
repo=$(git rev-parse --show-toplevel)
case "$(realpath -m "$EXP12_PREFLIGHT_ROOT")/" in "$repo/"*)
  echo 'preflight output must be outside the checkout' >&2; exit 2;; esac
mkdir "$EXP12_PREFLIGHT_ROOT"
trap 'code=$?; printf "%s\n" "$code" > "$EXP12_PREFLIGHT_ROOT/exit_status.txt"' EXIT
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
test "${SLURM_JOB_NUM_NODES:?}" = 2
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=4 \
  python scripts/exp12_dynamic_delta_preflight.py worker --root "$EXP12_PREFLIGHT_ROOT"
python scripts/exp12_dynamic_delta_preflight.py verify --root "$EXP12_PREFLIGHT_ROOT"
