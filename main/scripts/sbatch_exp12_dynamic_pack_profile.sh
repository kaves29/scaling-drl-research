#!/usr/bin/env bash
# Diagnostic only: serial versus mixed-architecture processes on one A100.
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${EXP12_PACK_ROOT:?}" "${SLURM_JOB_ID:?}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/sbatch_exp12_dynamic_pack_profile.sh \
  scripts/profile_exp12.py experiments/exp12/probe.py configs/base_exp12.yaml
test -z "$(git status --porcelain --untracked-files=all)"
[[ "$EXP12_PACK_ROOT" = /* ]] && [ ! -e "$EXP12_PACK_ROOT" ]
repo=$(git rev-parse --show-toplevel)
case "$(realpath -m "$EXP12_PACK_ROOT")/" in "$repo/"*)
  echo 'profile output must be outside the checkout' >&2; exit 2;; esac
mkdir "$EXP12_PACK_ROOT"
finish() {
  code=$?
  trap - EXIT
  if [ -n "${sampler:-}" ]; then kill "$sampler" 2>/dev/null || true; fi
  printf '%s\n' "$code" > "$EXP12_PACK_ROOT/exit_status.txt"
  exit "$code"
}
trap finish EXIT
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
python - <<'PY'
import jax
from scripts.exp12_launch_plan import require_pinned_stack
require_pinned_stack()
d = jax.devices()
assert len(d) == 1 and d[0].device_kind == "NVIDIA A100-SXM4-40GB", d
assert jax.local_devices(backend="cpu")
PY
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > "$EXP12_PACK_ROOT/gpu.csv"
(
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits
    sleep 5
  done
) > "$EXP12_PACK_ROOT/gpu_samples.log" 2>&1 &
sampler=$!
archs=(D2W512 D4W1024 D4W1536)
run_one() {
  local mode=$1 arch=$2
  mkdir "$EXP12_PACK_ROOT/cache_${mode}_${arch}" "$EXP12_PACK_ROOT/tmp_${mode}_${arch}"
  EXP12_JAX_CACHE_DIR="$EXP12_PACK_ROOT/cache_${mode}_${arch}" \
  TMPDIR="$EXP12_PACK_ROOT/tmp_${mode}_${arch}" \
    python -u scripts/profile_exp12.py --env dog-run --env_group dmc_hard \
    --archs "$arch" --seed 990 --warmup_steps 1001 --train_steps 500 \
    --probe_repeats 1 --out "$EXP12_PACK_ROOT/${mode}_${arch}.json" \
    > "$EXP12_PACK_ROOT/${mode}_${arch}.log" 2>&1
}
serial_start=$(date +%s.%N)
for arch in "${archs[@]}"; do run_one serial "$arch"; done
serial_end=$(date +%s.%N)
declare -a pids=()
mixed_start=$(date +%s.%N)
for arch in "${archs[@]}"; do
  run_one mixed "$arch" &
  pids+=("$!")
done
failure=0
for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
mixed_end=$(date +%s.%N)
test "$failure" = 0
python - "$EXP12_PACK_ROOT" "$serial_start" "$serial_end" "$mixed_start" "$mixed_end" <<'PY'
import json, math, sys
from pathlib import Path
root = Path(sys.argv[1])
archs = ("D2W512", "D4W1024", "D4W1536")
rates = {}
for mode in ("serial", "mixed"):
    rates[mode] = {}
    for arch in archs:
        rows = json.loads((root / f"{mode}_{arch}.json").read_text())
        if len(rows) != 1 or rows[0]["arch"] != arch:
            raise ValueError(f"incomplete {mode}/{arch} profile")
        rate = rows[0]["train_it_per_s_probes_off"]
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"invalid {mode}/{arch} training rate")
        rates[mode][arch] = rate
serial_wall = float(sys.argv[3]) - float(sys.argv[2])
mixed_wall = float(sys.argv[5]) - float(sys.argv[4])
if serial_wall <= 0 or mixed_wall <= 0:
    raise ValueError("invalid profile walltime")
(root / "comparison.json").write_text(json.dumps({
    "source": "one cold-cache diagnostic per architecture in each mode; includes startup and probes",
    "rates_interactions_per_second": rates,
    "serial_wall_seconds": serial_wall,
    "mixed_wall_seconds": mixed_wall,
    "observed_end_to_end_speedup": serial_wall / mixed_wall,
    "interpretation": "one workload observation only; no packing ceiling is qualified automatically",
}, indent=2))
PY
