#!/bin/bash
#SBATCH --job-name=exp12_blockA
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e
cd /work/hdd/biqc/skaveti1/exp12/main
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
unset JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_MEM_FRACTION
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled
echo "=== job $SLURM_JOB_ID on $(hostname), $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "branch: $(git rev-parse --abbrev-ref HEAD)  commit: $(git rev-parse HEAD)"
git status --short | head -20
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]; then
  echo "ERROR: HEAD is not $EXPECTED_COMMIT"; exit 2
fi
python - <<'EOF'
import sys, jax
print("python", sys.version.split()[0], "| jax", jax.__version__, "| devices", jax.devices())
assert jax.devices()[0].platform == "gpu", "JAX does not see the GPU"
EOF
pip list 2>/dev/null | grep -Ei "^(jax|jaxlib|flax|optax|orbax|numpy|pandas|rliable) " || true
env | grep -Ei "precision|tf32|xla" || true
set +e
bash scripts/exp12_blockA.sh
status=$?
echo "=== Block A runner exit status $status at $(date) ==="
exit $status
