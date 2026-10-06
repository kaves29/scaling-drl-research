#!/bin/bash
#SBATCH --job-name=exp12_blockB
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Request, from Block A's measured D6W1536 point (34.6 it/s, 50 s per probe check; docs/exp12_cuda_commands.md):
#  - GPU 0, the critical path: dev run to 120% of B after a late fork ~5.9 h -> positive control ~0.2 h ->
#    preflight ~0.25 h = ~6.4 h. The injected arm (ARM_M=half, the default) starts at the fork and ends by ~5.7 h;
#  - GPU 2 ~4.6 h (identity forks run twice, cold and warm cache), GPU 3 ~2.4 h (null, then per-suite speed);
#  - 10 h leaves ~55% over the critical path for what Block A did not measure (Delta's host speed, GPU
#    evaluations, compile times), and the driver's own 9.5 h deadline keeps time for the report;
#  - memory: ~33 GB peak estimated, so 128G; CPUs: 64 = 16 per GPU lane (4 per packing job).
# Submit from main/ after `mkdir -p logs`: Slurm opens logs/%x_%j.out before the job starts.
# Optional: HB_ENV=<conda env with HumanoidBench> for the HumanoidBench steps (unset: recorded unavailable).
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
echo "=== job $SLURM_JOB_ID on $(hostname), $(date); ARM_M=${ARM_M:-half} HB_ENV=${HB_ENV:-unset} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "branch: $(git rev-parse --abbrev-ref HEAD)  commit: $(git rev-parse HEAD)"
git status --short | head -20
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]; then
  echo "ERROR: HEAD is not $EXPECTED_COMMIT"; exit 2
fi
python - <<'EOF'
import sys, jax
d = jax.devices()
print("python", sys.version.split()[0], "| jax", jax.__version__, "| devices", d)
assert len(d) == 4 and all(x.platform == "gpu" for x in d), "JAX does not see 4 GPUs"
assert len({x.device_kind for x in d}) == 1, "the 4 GPUs are not one model"
EOF
pip list 2>/dev/null | grep -Ei "^(jax|jaxlib|flax|optax|orbax|numpy|pandas|rliable|myosuite|humanoid.bench) " || true
env | grep -Ei "precision|tf32|xla" || true
set +e
bash scripts/exp12_blockB.sh
status=$?
bash scripts/collect_report.sh "logs/blockB_$SLURM_JOB_ID"
echo "=== Block B driver exit status $status at $(date) ==="
exit $status
