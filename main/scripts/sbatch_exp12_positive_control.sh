#!/usr/bin/env bash
# Run the canonical positive control for one natural, same-source dev fork.
# Resources are supplied by a separately reviewed sbatch command.
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}" "${PC_RUN_DIR:?}" "${PC_OUTPUT_ROOT:?}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/sbatch_exp12_positive_control.sh \
  scripts/positive_control.py experiments/exp12/probe.py configs/base_exp12.yaml \
  scripts/exp12_source_compatibility.py
test -z "$(git status --porcelain --untracked-files=all)"
[[ "$PC_RUN_DIR" = /* && "$PC_OUTPUT_ROOT" = /* && ! -e "$PC_OUTPUT_ROOT" ]] || {
  echo 'positive control requires absolute source and new absolute evidence root' >&2
  exit 2
}
repo=$(git rev-parse --show-toplevel)
case "$(realpath -m "$PC_OUTPUT_ROOT")/" in
  "$repo/"*) echo 'positive-control output must be outside checkout' >&2; exit 2;;
esac
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
# Optional: a fork from an earlier commit (e.g. pilot 22800749 at 7d0883c) is accepted
# only with a hash-pinned science-source compatibility certificate, re-verified here.
python - "$PC_RUN_DIR" "$EXPECTED_COMMIT" "${PC_SOURCE_COMPATIBILITY:-}" \
  "${PC_SOURCE_COMPATIBILITY_SHA256:-}" <<'PY'
import sys
from pathlib import Path
import jax
from scripts.exp12_launch_plan import require_pinned_stack
from utils.run_metadata import load_run_metadata

require_pinned_stack()
if sys.version_info[:3] != (3, 12, 13):
    raise ValueError('pinned Python 3.12.13 required')
devices = jax.devices()
if (len(devices) != 1 or devices[0].device_kind != 'NVIDIA A100-SXM4-40GB'
        or not jax.local_devices(backend='cpu')):
    raise ValueError('positive control requires one A100 and the CPU backend')
run = Path(sys.argv[1])
metadata = load_run_metadata(run / 'run_metadata.json')
code = (metadata or {}).get('code') or {}
if code.get('dirty') is not False:
    raise ValueError('development fork source is not a clean commit')
if code.get('commit') != sys.argv[2]:
    if not sys.argv[3]:
        raise ValueError('development fork source differs from approved checkout')
    from scripts.exp12_source_compatibility import verify
    verify(sys.argv[3], sys.argv[4], code['commit'], sys.argv[2])
    print('fork source', code['commit'], 'certified compatible with', sys.argv[2])
if not (run / 'fork' / 'FORK_READY').is_file():
    raise ValueError('development run has no complete natural fork marker')
PY
mkdir "$PC_OUTPUT_ROOT"
finish() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$PC_OUTPUT_ROOT/exit_status.txt"
  find "$PC_OUTPUT_ROOT" -type f ! -name SHA256SUMS ! -name exit_status.txt -print0 |
    sort -z | xargs -0 -r sha256sum > "$PC_OUTPUT_ROOT/SHA256SUMS"
  exit "$code"
}
trap finish EXIT
printf '{"commit":"%s","job_id":"%s","run_dir":"%s","source_compatibility":"%s","source_compatibility_sha256":"%s"}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$PC_RUN_DIR" "${PC_SOURCE_COMPATIBILITY:-}" \
  "${PC_SOURCE_COMPATIBILITY_SHA256:-}" > "$PC_OUTPUT_ROOT/submission.json"
python -u scripts/positive_control.py --run_dir "$PC_RUN_DIR" \
  --out_dir "$PC_OUTPUT_ROOT/result" > "$PC_OUTPUT_ROOT/positive_control.log" 2>&1
