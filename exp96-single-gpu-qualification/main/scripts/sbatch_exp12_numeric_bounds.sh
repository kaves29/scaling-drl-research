#!/usr/bin/env bash
# Measure the four diagnostics-isolation deviations on the final source (one A100).
# Diagnostic only: tests skip while GPU_TOLERANCES are None and record NUMERICS lines, each
# naming this source commit, the backend, device and mode; the proposal refuses anything else.
# Resources are supplied by a separately reviewed sbatch command; this file never submits.
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}" "${NUMERICS_ROOT:?}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/sbatch_exp12_numeric_bounds.sh scripts/exp12_numeric_bounds.py \
  tests/exp12_helpers.py tests/test_exp12_diagnostics.py tests/test_exp12_twin_critic.py
test -z "$(git status --porcelain --untracked-files=all)"
[[ "$NUMERICS_ROOT" = /* ]] && [ ! -e "$NUMERICS_ROOT" ] || { echo 'NUMERICS_ROOT must be new and absolute' >&2; exit 2; }
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE XLA_FLAGS
export EXP12_NUMERICS_COMMIT="$EXPECTED_COMMIT"  # every record names its source
mkdir "$NUMERICS_ROOT"
finish() { code=$?; trap - EXIT; printf '%s\n' "$code" > "$NUMERICS_ROOT/exit_status.txt"; exit "$code"; }
trap finish EXIT
printf '{"commit":"%s","job_id":"%s"}\n' "$EXPECTED_COMMIT" "$SLURM_JOB_ID" > "$NUMERICS_ROOT/submission.json"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$NUMERICS_ROOT/gpu.csv"
tests=(tests.test_exp12_diagnostics.KnownAnswerTest.test_diagnostics_never_change_the_update
       tests.test_exp12_diagnostics.TrainingRunTest.test_training_is_identical_with_diagnostics_on_and_off
       tests.test_exp12_twin_critic.TwinDiagnosticsTest.test_diagnostics_never_change_the_update)
EXP12_NUMERICS_OUT="$NUMERICS_ROOT/numerics_tf32.jsonl" \
  python -m unittest -v "${tests[@]}" > "$NUMERICS_ROOT/tests_tf32.log" 2>&1
EXP12_NUMERICS_OUT="$NUMERICS_ROOT/numerics_deterministic.jsonl" XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  python -m unittest -v "${tests[@]}" > "$NUMERICS_ROOT/tests_deterministic.log" 2>&1
python scripts/exp12_numeric_bounds.py --root "$NUMERICS_ROOT" \
  --expected-commit "$EXPECTED_COMMIT" --out "$NUMERICS_ROOT/proposed_bounds.json"
