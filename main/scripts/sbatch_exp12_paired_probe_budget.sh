#!/usr/bin/env bash
# Exploratory D1/D2 only. This never changes or launches production parents.
#SBATCH --job-name=exp12_paired_probe_budget
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${PROBE_DIAGNOSTIC_ROOT:?}" "${HOPPER_CAPTURE_ARCHIVE:?}" "${SLURM_SUBMIT_DIR:?}"
cd "$SLURM_SUBMIT_DIR"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/sci_investigation/paired_probe_budget.py \
  scripts/sbatch_exp12_paired_probe_budget.sh
test -z "$(git status --porcelain --untracked-files=all)"
[[ "$PROBE_DIAGNOSTIC_ROOT" = /* ]] && [ ! -e "$PROBE_DIAGNOSTIC_ROOT" ]
repo=$(git rev-parse --show-toplevel)
case "$(realpath -m "$PROBE_DIAGNOSTIC_ROOT")/" in "$repo/"*)
  echo 'diagnostic output must be outside the checkout' >&2; exit 2;; esac
test -f "$HOPPER_CAPTURE_ARCHIVE"
echo 'e96cb06b953ed1acef29850861b5371c91766c58e1b4e0eb9a0b86acec0d4868  '"$HOPPER_CAPTURE_ARCHIVE" | sha256sum -c -
mkdir "$PROBE_DIAGNOSTIC_ROOT"
trap 'code=$?; printf "%s\n" "$code" > "$PROBE_DIAGNOSTIC_ROOT/exit_status.txt"; find "$PROBE_DIAGNOSTIC_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > "$PROBE_DIAGNOSTIC_ROOT/SHA256SUMS"' EXIT
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
python -u scripts/sci_investigation/paired_probe_budget.py \
  --mode hopper --seed 990 --capture-archive "$HOPPER_CAPTURE_ARCHIVE" \
  --expected-commit "$EXPECTED_COMMIT" --out-dir "$PROBE_DIAGNOSTIC_ROOT/hopper" \
  > "$PROBE_DIAGNOSTIC_ROOT/hopper.log" 2>&1
python -u scripts/sci_investigation/paired_probe_budget.py \
  --mode null --seed 991 --pairs 100 \
  --expected-commit "$EXPECTED_COMMIT" --out-dir "$PROBE_DIAGNOSTIC_ROOT/null_seed991" \
  > "$PROBE_DIAGNOSTIC_ROOT/null.log" 2>&1
