#!/usr/bin/env bash
# One approved full-setting CUDA identity case. Submit cold and warm separately.
# Resource limits must be supplied explicitly to sbatch; this file never submits.
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}" "${IDENTITY_CASE:?}" "${IDENTITY_CACHE_MODE:?}" "${IDENTITY_ROOT:?}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/sbatch_exp12_identity_case.sh scripts/exp12_launch_plan.py run.py configs/base_exp12.yaml
test -z "$(git status --porcelain --untracked-files=all)"
case "$IDENTITY_CASE" in
  D4W1024_dog-run) width=1024; environment=dog-run; group=dmc_hard;;
  D4W1536_dog-run) width=1536; environment=dog-run; group=dmc_hard;;
  D4W1024_myo-key-turn) width=1024; environment=myo-key-turn; group=myosuite_simba;;
  D4W1536_myo-key-turn) width=1536; environment=myo-key-turn; group=myosuite_simba;;
  D4W1024_h1-run-v0) width=1024; environment=h1-run-v0; group=humanoid_bench;;
  D4W1536_h1-run-v0) width=1536; environment=h1-run-v0; group=humanoid_bench;;
  *) echo "invalid identity case: $IDENTITY_CASE" >&2; exit 2;;
esac
case "$IDENTITY_CACHE_MODE" in cold|warm) ;; *) echo 'cache mode must be cold or warm' >&2; exit 2;; esac
[[ "$IDENTITY_ROOT" = /* ]] && [ ! -e "$IDENTITY_ROOT" ] || { echo 'identity root must be new and absolute' >&2; exit 2; }
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
py=$(command -v python)
if [ "$group" = humanoid_bench ]; then
  : "${HB_ENV:?HB_ENV must point to the reviewed HumanoidBench environment}"
  py="$HB_ENV/bin/python"
  [ -x "$py" ] || { echo "missing HumanoidBench Python: $py" >&2; exit 2; }
fi
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false
if [ "$group" = humanoid_bench ]; then
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
else
  export MUJOCO_GL=disable
  unset PYOPENGL_PLATFORM
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
"$py" - <<'PY'
import sys
import jax
from scripts.exp12_launch_plan import require_pinned_stack
require_pinned_stack()
assert sys.version_info[:3] == (3, 12, 13), sys.version
devices = jax.devices()
assert len(devices) == 1 and devices[0].device_kind == 'NVIDIA A100-SXM4-40GB', devices
assert jax.local_devices(backend='cpu')
assert not jax.config.jax_enable_x64
PY
mkdir "$IDENTITY_ROOT"
# Resource evidence only (never read by training): GPU memory/utilisation every 5 s,
# GNU time per phase (max RSS, wall), phase timestamps for end-to-end throughput.
(
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    sleep 5
  done
) > "$IDENTITY_ROOT/gpu_samples.csv" 2>&1 &
sampler=$!
timer=()
[ -x /usr/bin/time ] && timer=(/usr/bin/time -v -o)
phase() {  # phase NAME command...
  local name=$1; shift
  date +%s.%N > "$IDENTITY_ROOT/${name}.start"
  if [ ${#timer[@]} -gt 0 ]; then "${timer[@]}" "$IDENTITY_ROOT/${name}.time" "$@"; else "$@"; fi
  date +%s.%N > "$IDENTITY_ROOT/${name}.end"
}
finish() {
  code=$?; trap - EXIT
  kill "$sampler" 2>/dev/null || true
  printf '%s\n' "$code" > "$IDENTITY_ROOT/exit_status.txt"; exit "$code"
}
trap finish EXIT
printf '{"commit":"%s","case":"%s","cache_mode":"%s","job_id":"%s"}\n' \
  "$EXPECTED_COMMIT" "$IDENTITY_CASE" "$IDENTITY_CACHE_MODE" "$SLURM_JOB_ID" > "$IDENTITY_ROOT/submission.json"
"$py" -m pip freeze > "$IDENTITY_ROOT/dependencies.txt"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$IDENTITY_ROOT/gpu.csv"
common=(--config_name base_exp12
  --overrides "env_name=$environment" --overrides "env=$group"
  --overrides seed=101 --overrides critic_num_blocks=4 --overrides "critic_hidden_dim=$width"
  --overrides run_role=dev --overrides testing.force_trigger_check=2
  --overrides fork.identity_snapshot_steps=1000
  --overrides testing.stop_after_identity_snapshot=true
  --overrides "results_root=$IDENTITY_ROOT/results")
parent_cache="$IDENTITY_ROOT/cache_parent"
arm_cache="$IDENTITY_ROOT/cache_arm"
if [ "$IDENTITY_CACHE_MODE" = warm ]; then
  parent_cache="$IDENTITY_ROOT/cache_shared"
  arm_cache="$parent_cache"
fi
phase parent env EXP12_JAX_CACHE_DIR="$parent_cache" "$py" -u run.py --experiment exp1 "${common[@]}" \
  --checkpoint_dir "$IDENTITY_ROOT/parent" > "$IDENTITY_ROOT/parent.log" 2>&1
phase identity env EXP12_JAX_CACHE_DIR="$arm_cache" "$py" -u run.py --experiment exp2_arm "${common[@]}" \
  --overrides "fork.source=$IDENTITY_ROOT/parent" --overrides fork.arm=identity \
  --checkpoint_dir "$IDENTITY_ROOT/identity" > "$IDENTITY_ROOT/identity.log" 2>&1
"$py" scripts/exp12_launch_plan.py certify-identity \
  --run "$IDENTITY_ROOT/parent" --arm "$IDENTITY_ROOT/identity" \
  --commit "$EXPECTED_COMMIT" --out "$IDENTITY_ROOT/certificate.json" \
  > "$IDENTITY_ROOT/certify.log" 2>&1
sha256sum "$IDENTITY_ROOT/certificate.json" "$IDENTITY_ROOT/submission.json" \
  "$IDENTITY_ROOT/parent.log" "$IDENTITY_ROOT/identity.log" > "$IDENTITY_ROOT/SHA256SUMS"
