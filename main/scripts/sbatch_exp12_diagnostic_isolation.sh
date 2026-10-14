#!/usr/bin/env bash
# Diagnostic harness only. The frozen Exp1 implementation is imported from FROZEN_DIR/main.
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
: "${FROZEN_DIR:?}" "${HARNESS_DIR:?}" "${OUT_ROOT:?}" "${FROZEN_COMMIT:?}" "${HARNESS_COMMIT:?}"
[[ "$FROZEN_DIR" = /* && "$HARNESS_DIR" = /* && "$OUT_ROOT" = /* ]]
[[ "$(git -C "$FROZEN_DIR" rev-parse HEAD)" = "$FROZEN_COMMIT" ]]
[[ "$(git -C "$HARNESS_DIR" rev-parse HEAD)" = "$HARNESS_COMMIT" ]]
[[ -z "$(git -C "$FROZEN_DIR" status --porcelain --untracked-files=all)" ]]
[[ -z "$(git -C "$HARNESS_DIR" status --porcelain --untracked-files=all)" ]]
[[ ! -e "$OUT_ROOT/evidence" ]]
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export EXP12_JAX_CACHE_DIR="$OUT_ROOT/jax_cache"
unset JAX_PLATFORM_NAME NVIDIA_TF32_OVERRIDE
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
cd "$FROZEN_DIR/main"
export PYTHONPATH="$FROZEN_DIR/main"
printf 'frozen=%s harness=%s job=%s\n' "$FROZEN_COMMIT" "$HARNESS_COMMIT" "$SLURM_JOB_ID"
sha256sum "$HARNESS_DIR/main/scripts/exp12_diagnostic_isolation.py" \
  "$HARNESS_DIR/main/scripts/sbatch_exp12_diagnostic_isolation.sh" > "$OUT_ROOT/harness.sha256"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$OUT_ROOT/gpu.csv"
finish() { code=$?; trap - EXIT; printf '%s\n' "$code" > "$OUT_ROOT/exit_status.txt"; exit "$code"; }
trap finish EXIT
python "$HARNESS_DIR/main/scripts/exp12_diagnostic_isolation.py" \
  --out "$OUT_ROOT/evidence" --frozen-commit "$FROZEN_COMMIT" \
  --harness-commit "$HARNESS_COMMIT" > "$OUT_ROOT/harness.log" 2>&1
