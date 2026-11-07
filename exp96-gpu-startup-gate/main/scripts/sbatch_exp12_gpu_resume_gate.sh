#!/usr/bin/env bash
# Prepared qualification only; this script never calls sbatch itself.
#SBATCH --job-name=exp12_gpu_resume95
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:07:00
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${VALIDATION_ROOT:?}" "${SLURM_JOB_ID:?}"
cd "$VALIDATION_ROOT/checkout/main"
export OUT="$VALIDATION_ROOT/gpu_resume_95_$SLURM_JOB_ID"
mkdir "$OUT"
finish() {
  local status=$?
  trap - EXIT
  set +e
  printf '%s\n' "$status" > "$OUT/exit_status.txt"
  (cd "$OUT"; find . -type f ! -path './jax_cache/*' ! -name SHA256SUMS \
    -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  exit "$status"
}
trap finish EXIT
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/sci_investigation/gpu_resume_probe.py \
  tests/exp12_subprocess_runner.py configs/base_exp12.yaml
test -z "$(git status --porcelain --untracked-files=all)"
git rev-parse HEAD > "$OUT/commit.txt"
git status --porcelain --untracked-files=all > "$OUT/git-status.txt"
cp configs/base_exp12.yaml "$OUT/base_exp12.yaml"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
mkdir "$OUT/tmp" "$OUT/jax_cache"
export TMPDIR="$OUT/tmp" EXP12_JAX_CACHE_DIR="$OUT/jax_cache"
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
python -m pip freeze > "$OUT/dependencies.txt"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$OUT/nvidia-smi.csv"
cat > "$OUT/workload.py" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    import jax
    import jaxlib

    out = Path(os.environ["OUT"])
    devices = jax.devices()
    assert sys.version_info[:2] == (3, 12), sys.version
    assert jax.__version__ == jaxlib.__version__ == "0.4.34"
    assert len(devices) == 1 and devices[0].platform == "gpu", devices
    assert devices[0].device_kind == "NVIDIA A100-SXM4-40GB", devices
    assert jax.local_devices(backend="cpu")
    from experiments.exp12.precision import set_matmul_precision, configure_compilation_cache

    set_matmul_precision()
    configure_compilation_cache()
    assert jax.config.jax_compilation_cache_max_size == -1
    (out / "backend.json").write_text(json.dumps({
        "python": sys.version, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "devices": [str(d) for d in devices], "device_kind": devices[0].device_kind,
        "cpu_devices": [str(d) for d in jax.local_devices(backend="cpu")],
        "x64": jax.config.jax_enable_x64,
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "cache_max_size": jax.config.jax_compilation_cache_max_size}, indent=2))
    sys.exit(subprocess.call([sys.executable, "-u",
        "scripts/sci_investigation/gpu_resume_probe.py", "--out", str(out / "resume_probe"),
        "--require-backend", "gpu", "--crash-step", "95"]))
PY
# One total cap, including startup and all four children; no per-child budget multiplication.
timeout -k 15 300 python -u "$OUT/workload.py" > "$OUT/tests.log" 2>&1
