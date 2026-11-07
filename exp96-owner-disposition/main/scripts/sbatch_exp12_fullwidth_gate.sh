#!/usr/bin/env bash
# Prepared only. Each stage requires a separate, explicit submission authorization.
#SBATCH --job-name=exp12_fullwidth
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:07:00
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${GATE_SPEC:?}" "${GATE_ROOT:?}" "${GATE_STAGE:?}" "${SLURM_JOB_ID:?}"
case "$GATE_STAGE" in reference_cold|reference_warm|crash|resume) ;; *) echo 'invalid stage' >&2; exit 2;; esac
mkdir -p "$GATE_ROOT/attempts"
attempt="$GATE_ROOT/attempts/${GATE_STAGE}_${SLURM_JOB_ID}"
mkdir "$attempt"
finish() {
  local code=$?
  trap - EXIT
  set +e
  printf '%s\n' "$code" > "$attempt/exit_status.txt"
  local verdict=INCOMPLETE
  case "$code" in
    0) verdict=PROCESS_COMPLETE;;
    2) verdict=INFRASTRUCTURE_FAILURE;;
    3) [[ "$GATE_STAGE" == crash ]] && verdict=EXPECTED_CRASH;;
    124) verdict=TIMEOUT;;
    137) verdict=KILLED_OR_TIMEOUT;;
  esac
  # Exit3 is only the declared abrupt crash, checked later against its receipt/state.
  printf '{"exit":%s,"stage":"%s","verdict":"%s","timeout":%s}\n' "$code" "$GATE_STAGE" "$verdict" \
    "$([[ $code == 124 || $code == 137 ]] && echo true || echo false)" > "$attempt/process.json"
  (cd "$GATE_ROOT"; find . -type f ! -path './persistent_cache/*' ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$attempt/SHA256SUMS")
  exit "$code"
}
trap finish EXIT
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/exp12_fullwidth_gate.py configs/base_exp12.yaml
test -z "$(git status --porcelain --untracked-files=all)"
git rev-parse HEAD > "$attempt/commit.txt"
cp "$GATE_SPEC" "$attempt/submitted-spec.json"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
python -m pip freeze > "$attempt/dependencies.txt"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$attempt/nvidia-smi.csv"
python -c 'import sys; assert sys.version_info[:3] == (3,12,13), sys.version'
timeout -k 15 300 python -u scripts/exp12_fullwidth_gate.py \
  --spec "$GATE_SPEC" --out "$GATE_ROOT" --stage "$GATE_STAGE" > "$attempt/workload.log" 2>&1
