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
#SBATCH --time=04:30:00
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${PROBE_DIAGNOSTIC_ROOT:?}" "${HOPPER_CAPTURE_ARCHIVE:?}" "${NULL_991_JSONL:?}" "${SLURM_SUBMIT_DIR:?}"
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
echo 'd6dc126939d82f145f1ea1b4be60d87a36751970c365bdaf5ae5d8f5333ca84e  '"$NULL_991_JSONL" | sha256sum -c -
mkdir "$PROBE_DIAGNOSTIC_ROOT"
telemetry_pid=''
finish() {
  code=$?
  if [ -n "$telemetry_pid" ]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  printf '%s\n' "$code" > "$PROBE_DIAGNOSTIC_ROOT/exit_status.txt"
  find "$PROBE_DIAGNOSTIC_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > "$PROBE_DIAGNOSTIC_ROOT/SHA256SUMS"
}
trap finish EXIT
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
(
  while :; do
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits || true
    nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits || true
    free -b | head -2 || true
    sleep 5
  done
) > "$PROBE_DIAGNOSTIC_ROOT/resource_telemetry.log" 2>&1 &
telemetry_pid=$!
failed=0
run_stage() {
  label=$1 cap=$2
  shift 2
  if timeout --signal=TERM --kill-after=30s "$cap" \
    python -u scripts/sci_investigation/paired_probe_budget.py \
    --expected-commit "$EXPECTED_COMMIT" "$@" \
    > "$PROBE_DIAGNOSTIC_ROOT/$label.log" 2>&1; then
    printf '0\n' > "$PROBE_DIAGNOSTIC_ROOT/$label.exit_status.txt"
  else
    code=$?
    printf '%s\n' "$code" > "$PROBE_DIAGNOSTIC_ROOT/$label.exit_status.txt"
    failed=1
  fi
}
# Fixed per-stage caps; completed earlier stages never wait for later stages.
run_stage hopper_990 900 --mode hopper --seed 990 --capture-archive "$HOPPER_CAPTURE_ARCHIVE" \
  --max-elapsed-seconds 840 --out-dir "$PROBE_DIAGNOSTIC_ROOT/hopper_990"
run_stage hopper_992 900 --mode hopper --seed 992 \
  --max-elapsed-seconds 840 --out-dir "$PROBE_DIAGNOSTIC_ROOT/hopper_992"
run_stage hopper_993 900 --mode hopper --seed 993 \
  --max-elapsed-seconds 840 --out-dir "$PROBE_DIAGNOSTIC_ROOT/hopper_993"
run_stage null_991 12000 --mode null --seed 991 --pairs 100 \
  --pooled-threshold 0.0009080017606417335 --reference-null-jsonl "$NULL_991_JSONL" \
  --max-elapsed-seconds 11940 \
  --out-dir "$PROBE_DIAGNOSTIC_ROOT/null_991"
exit "$failed"
