#!/bin/bash
#SBATCH --job-name=exp12_hb
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# HumanoidBench with twin critics (amendment (z)) and the identity-fork gate on the final commit, one A100:
#   1. HumanoidBench and twin-critic tests                                        (HB_ENV)
#   2. identity forks, reduced budget, D4W1024 and D4W1536, on dog-run and myo-key-turn (main env) and h1-run-v0
#      (HB_ENV, twin critics)
#   3. h1-run-v0 training speed, probe-check time and peak GPU memory per size, twin critics (HB_ENV)
# Estimate ~1.5 h (docs/exp12_cuda_commands.md); 4 h requested. Every step has its own log and timeout under
# logs/hb_<jobid>/; a failing step does not stop the others.
# Submit from main/ after `mkdir -p logs`:  HB_ENV=<env> EXPECTED_COMMIT=<hash> EXPECTED_GPU_MODEL=<device_kind> sbatch scripts/sbatch_exp12_hb.sh
# CPU plumbing test: HB_ENV=<env> HB_TEST_HOOKS=<file that shrinks the settings> bash scripts/sbatch_exp12_hb.sh
set -uo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)" || exit 2
cd "$SCRIPT_DIR/.." || exit 2
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the reviewed clean final-tree revision}"
: "${EXPECTED_GPU_MODEL:?set EXPECTED_GPU_MODEL to the grid GPU device_kind}"
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${HB_TEST_HOOKS:-}" ]; then
  echo "ERROR: local plumbing hooks are prohibited in a Slurm qualification job"; exit 2
fi
: "${HB_ENV:?set HB_ENV to the conda env with HumanoidBench (docs/exp12_cuda_commands.md, Setup)}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  module reset || exit 2
  source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh || exit 2
  conda activate scaling-drl-py31213 || exit 2
fi
MAIN_PY="$(command -v python)"
HB_PY="$HB_ENV/bin/python"
unset JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE XLA_PYTHON_CLIENT_MEM_FRACTION
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled PYTHONUNBUFFERED=1 MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [ "$MUJOCO_GL" != egl ] || [ "$PYOPENGL_PLATFORM" != egl ]; then
  echo "ERROR: HB CUDA qualification requires MUJOCO_GL=egl PYOPENGL_PLATFORM=egl"; exit 2
fi
OUT="$(pwd)/logs/hb_${SLURM_JOB_ID:-local}"
IDENTITY_ARCHS="4:1024 4:1536" IDENTITY_BUDGET=240000 SPEED_ARCHS="D2W512 D4W1024 D4W1536"
TESTS="tests.test_exp12_pipeline.PipelinePerSuiteTest.test_humanoid_bench tests.test_exp12_fork.HumanoidBenchReachEvalSeedingTest tests.test_exp12_twin_critic"
EXTRA_OVERRIDES="" SPEED_EXTRA="--train_steps 600 --warmup_steps 100 --probe_repeats 1"
TIMEOUT_TESTS=2700 TIMEOUT_IDENTITY=3600 TIMEOUT_SPEED=3600
[ -n "${HB_TEST_HOOKS:-}" ] && source "$HB_TEST_HOOKS"
[ ! -e "$OUT/status.tsv" ] || { echo "ERROR: status.tsv already exists: $OUT"; exit 2; }
mkdir -p "$OUT"

echo "=== exp12_hb job ${SLURM_JOB_ID:-local} on $(hostname), $(date); HB_ENV=$HB_ENV ==="
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "commit: $(git rev-parse HEAD)"; git status --short | head -20
if [ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ] || [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "ERROR: expected a clean tree at $EXPECTED_COMMIT"; exit 2
fi
"$MAIN_PY" scripts/check_exp12_hb_environment.py --gpu-model "$EXPECTED_GPU_MODEL" --out "$OUT/main_environment.json" || exit 2
"$HB_PY" scripts/check_exp12_hb_environment.py --gpu-model "$EXPECTED_GPU_MODEL" --main "$OUT/main_environment.json" --out "$OUT/hb_environment.json" || exit 2
# Separate writable caches for ordinary test/profile workers. Identity scenarios retain
# their explicit caller cache setting until the cold/warm plan is approved.
export EXP12_HB_CACHE_ROOT="$OUT/cache"

ov() { local x; for x in "$@"; do printf -- "--overrides %s " "$x"; done; }
step() {  # name timeout_key command...
  local name=$1 key=$2 s=$SECONDS; shift 2
  local limit
  case "$key" in tests) limit=$TIMEOUT_TESTS;; identity) limit=$TIMEOUT_IDENTITY;; speed) limit=$TIMEOUT_SPEED;; *) return 2;; esac
  if [ "$key" = identity ]; then
    timeout -k 30 "$limit" "$@" > "$OUT/$name.log" 2>&1
  else
    EXP12_JAX_CACHE_DIR="$EXP12_HB_CACHE_ROOT/$name" timeout -k 30 "$limit" "$@" > "$OUT/$name.log" 2>&1
  fi
  local code=$?
  printf "%s\t%s\t%ss\n" "$name" "$code" "$((SECONDS - s))" | tee -a "$OUT/status.tsv"
}
identity() {  # python blocks width env group
  local py=$1 base="$OUT/identity/D$2W$3_$4"
  local common="--config_name base_exp12 $(ov env_name=$4 env=$5 seed=101 critic_num_blocks=$2 critic_hidden_dim=$3 \
    run_role=dev testing.force_trigger_check=2 fork.identity_snapshot_steps=1000 testing.stop_after_identity_snapshot=true \
    results_root=$base/results num_env_steps=$IDENTITY_BUDGET $EXTRA_OVERRIDES)"
  mkdir -p "$base"
  "$py" run.py --experiment exp1 $common --checkpoint_dir "$base/parent" || return 1
  "$py" run.py --experiment exp2_arm $common --overrides fork.source="$base/parent" --overrides fork.arm=identity \
    --checkpoint_dir "$base/identity" || return 1
  "$py" scripts/compare_identity_fork.py --run_dir "$base/parent" --arm_dir "$base/identity" --out "$base.json"
}
export -f ov identity
export OUT IDENTITY_BUDGET EXTRA_OVERRIDES

step tests tests "$HB_PY" -m unittest -v $TESTS
for a in $IDENTITY_ARCHS; do
  step "identity_D${a%%:*}W${a##*:}_dog-run" identity bash -c 'identity "$@"' _ "$MAIN_PY" "${a%%:*}" "${a##*:}" dog-run dmc_hard
  step "identity_D${a%%:*}W${a##*:}_myo-key-turn" identity bash -c 'identity "$@"' _ "$MAIN_PY" "${a%%:*}" "${a##*:}" myo-key-turn myosuite_simba
  step "identity_D${a%%:*}W${a##*:}_h1-run-v0" identity bash -c 'identity "$@"' _ "$HB_PY" "${a%%:*}" "${a##*:}" h1-run-v0 humanoid_bench
done
step speed_h1-run-v0 speed "$HB_PY" scripts/profile_exp12.py --env h1-run-v0 --env_group humanoid_bench \
  --archs $SPEED_ARCHS $SPEED_EXTRA --out "$OUT/speed_h1-run-v0.json"

{ echo "== status (step, exit code, wall) =="; cat "$OUT/status.tsv"
  echo "== tests =="; grep -E "^(Ran|OK|FAILED)" "$OUT/tests.log"
  echo "== identity forks (twin critics on h1-run-v0) =="
  for f in "$OUT"/identity/*.json; do
    [ -f "$f" ] || continue
    "$MAIN_PY" -c "import json, sys; d = json.load(open(sys.argv[1])); print(sys.argv[1].rsplit('/', 1)[1], 'PASS' if d['pass'] else 'FAIL', d.get('differences', d.get('error', ''))[:5])" "$f"
  done
  echo "== h1-run-v0 with twin critics: it/s (probes off), probe check s, overhead %, peak GPU GiB =="
  "$MAIN_PY" -c "
import json, sys
for r in json.load(open(sys.argv[1])):
    peak = r.get('peak_device_bytes')
    print(r['arch'], round(r['train_it_per_s_probes_off'], 1), round(r['probe_check_s'], 1),
          round(r['probe_overhead_pct_of_wallclock'], 2), None if peak is None else round(peak / 2**30, 2))" "$OUT/speed_h1-run-v0.json"
} 2>&1 | tee "$OUT/summary.txt"

"$MAIN_PY" scripts/check_exp12_hb_status.py --out "$OUT" --identity-archs $IDENTITY_ARCHS --speed-archs $SPEED_ARCHS
exit $?
