#!/usr/bin/env bash
# Exploratory Method C detector validation only; no completed Exp1 parent or Exp2 arm.
#SBATCH --job-name=exp12_method_c_hopper_dev
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${METHOD_C_OUT:?}" "${SLURM_SUBMIT_DIR:?}"
cd "$SLURM_SUBMIT_DIR"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/sci_investigation/method_c_trained_pilot.py \
  scripts/sbatch_exp12_method_c_trained_pilot.sh
test -z "$(git status --porcelain --untracked-files=all)"
[[ "$METHOD_C_OUT" = /* ]] && [ ! -e "$METHOD_C_OUT" ]
repo=$(git rev-parse --show-toplevel)
case "$(realpath -m "$METHOD_C_OUT")/" in "$repo/"*) echo 'output must be outside checkout' >&2; exit 2;; esac
mkdir "$METHOD_C_OUT"
finish() {
  code=$?
  printf '%s\n' "$code" > "$METHOD_C_OUT/exit_status.txt"
  find "$METHOD_C_OUT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > "$METHOD_C_OUT/SHA256SUMS"
}
trap finish EXIT
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
timeout --signal=TERM --kill-after=30s 5400 \
  python -u scripts/sci_investigation/method_c_trained_pilot.py \
  --expected-commit "$EXPECTED_COMMIT" --out-dir "$METHOD_C_OUT/pilot" \
  > "$METHOD_C_OUT/driver.log" 2>&1
