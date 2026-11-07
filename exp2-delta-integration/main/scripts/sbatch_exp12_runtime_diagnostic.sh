#!/usr/bin/env bash
#SBATCH --job-name=exp12_runtime_diagnostic
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:07:00
# Manual diagnostic only. Activate the reviewed CUDA environment before sbatch.
# Set external --output/--error paths; see docs/exp12_overnight_readiness.md.
set -euo pipefail
fail() {
  local status=$1
  shift
  printf 'exp12_runtime_diagnostic: preflight failed: %s\n' "$*" >&2
  exit "$status"
}
[ -n "${SLURM_JOB_ID:-}" ] || fail 1 "SLURM_JOB_ID is required; run only in an explicitly requested Slurm allocation"
[ -n "${SLURM_SUBMIT_DIR:-}" ] || fail 1 "SLURM_SUBMIT_DIR is required; submit from the reviewed main directory"
[ -n "${OUT:-}" ] || fail 1 "OUT is required; use a new absolute persistent output directory"
[ -n "${EXPECTED_COMMIT:-}" ] || fail 1 "EXPECTED_COMMIT is required; use the reviewed full source hash"
[ -n "${EXPECTED_GPU_MODEL:-}" ] || fail 1 "EXPECTED_GPU_MODEL is required; use the reviewed literal A100 device_kind"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail 2 "EXPECTED_COMMIT must be a full 40-character lowercase hexadecimal commit hash"
DIAGNOSTIC_MODE=${DIAGNOSTIC_MODE:-profile}
DIAGNOSTIC_ARCH=${DIAGNOSTIC_ARCH:-D4W1536}
case "$DIAGNOSTIC_MODE" in
  profile|identity_cold|identity_warm) ;;
  *) fail 2 "unsupported DIAGNOSTIC_MODE: $DIAGNOSTIC_MODE" ;;
esac
case "$DIAGNOSTIC_ARCH" in
  D2W512) DIAGNOSTIC_BLOCKS=2; DIAGNOSTIC_WIDTH=512 ;;
  D4W1024) DIAGNOSTIC_BLOCKS=4; DIAGNOSTIC_WIDTH=1024 ;;
  D4W1536) DIAGNOSTIC_BLOCKS=4; DIAGNOSTIC_WIDTH=1536 ;;
  *) fail 2 "unsupported DIAGNOSTIC_ARCH: $DIAGNOSTIC_ARCH" ;;
esac
[ "$DIAGNOSTIC_MODE" = profile ] || [ "$DIAGNOSTIC_BLOCKS" = 4 ] || fail 2 "identity mode requires a D4 architecture"
cd "$SLURM_SUBMIT_DIR" || fail 1 "cannot enter SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
required_files=(scripts/profile_exp12.py scripts/trace_exp12_runtime.py)
if [ "$DIAGNOSTIC_MODE" != profile ]; then
  required_files+=(run.py scripts/compare_identity_fork.py)
fi
for path in "${required_files[@]}"; do
  [ -f "$path" ] || fail 1 "missing required file: $path (submit from the reviewed main directory)"
done
actual_commit=$(git rev-parse --verify HEAD) || fail 1 "cannot resolve Git HEAD in SLURM_SUBMIT_DIR"
[ "$actual_commit" = "$EXPECTED_COMMIT" ] || fail 1 "HEAD mismatch: expected $EXPECTED_COMMIT, found $actual_commit"
tree_status=$(git status --porcelain --untracked-files=all) || fail 1 "cannot inspect Git working tree"
[ -z "$tree_status" ] || fail 1 "working tree is not clean (tracked, staged, or untracked changes)"
case "$EXPECTED_GPU_MODEL" in *A100*) ;; *) fail 2 "EXPECTED_GPU_MODEL must name an A100: $EXPECTED_GPU_MODEL" ;; esac
case "$OUT" in /*) ;; *) fail 2 "OUT must be absolute: $OUT" ;; esac
OUT=$(realpath -m "$OUT") || fail 1 "cannot resolve OUT path: $OUT"
repo=$(git rev-parse --show-toplevel) || fail 1 "cannot resolve Git checkout root"
case "$OUT/" in "$repo/"*) fail 2 "OUT must be outside the checkout: $OUT" ;; esac
[ -d "$(dirname "$OUT")" ] || fail 1 "OUT parent does not exist: $(dirname "$OUT")"
mkdir "$OUT" || fail 1 "cannot create new OUT directory (must not already exist): $OUT"
mkdir "$OUT/temp" "$OUT/cache_single_writer" || fail 1 "cannot create temporary/cache directories beneath OUT: $OUT"
export OUT EXPECTED_COMMIT EXPECTED_GPU_MODEL
export DIAGNOSTIC_MODE DIAGNOSTIC_ARCH DIAGNOSTIC_BLOCKS DIAGNOSTIC_WIDTH
export TMPDIR="$OUT/temp" WANDB_MODE=disabled PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda,cpu MUJOCO_GL=disable JAX_ENABLE_X64=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_MEM_FRACTION JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE JAX_PLATFORM_NAME
export EXP12_JAX_CACHE_DIR="$OUT/cache_single_writer"
git rev-parse HEAD > "$OUT/commit.txt"
set +e
timeout -k 15 300 bash <<'DIAGNOSTIC' > "$OUT/profile.log" 2>&1
set -euo pipefail
python - <<'PY'
import json
import os
import sys
from pathlib import Path
import jax
import jaxlib

assert sys.version_info[:2] == (3, 12)
assert jax.__version__ == jaxlib.__version__ == "0.4.34"
devices = jax.devices()
assert len(devices) == 1 and devices[0].platform == "gpu"
assert devices[0].device_kind == os.environ["EXPECTED_GPU_MODEL"]
cpu_devices = jax.local_devices(backend="cpu")
assert cpu_devices and all(d.platform == "cpu" for d in cpu_devices), "SAC initialization requires a local CPU backend"
metadata = dict(
    python=sys.version,
    jax=jax.__version__,
    jaxlib=jaxlib.__version__,
    device_kind=devices[0].device_kind,
    device=str(devices[0]),
    cpu_devices=[str(d) for d in cpu_devices],
    jax_platforms=jax.config.jax_platforms,
    commit=os.environ["EXPECTED_COMMIT"],
)
Path(os.environ["OUT"], "backend.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata), flush=True)
PY
if [ "$DIAGNOSTIC_MODE" != profile ]; then
  common=(--config_name base_exp12
    --overrides env_name=dog-run --overrides env=dmc_hard --overrides seed=101
    --overrides "critic_num_blocks=$DIAGNOSTIC_BLOCKS"
    --overrides "critic_hidden_dim=$DIAGNOSTIC_WIDTH"
    --overrides run_role=dev --overrides testing.force_trigger_check=2
    --overrides fork.identity_snapshot_steps=1000
    --overrides testing.stop_after_identity_snapshot=true
    --overrides "results_root=$OUT/results" --overrides num_env_steps=240000)
  parent_cache="$OUT/cache_parent"; arm_cache="$OUT/cache_arm"
  if [ "$DIAGNOSTIC_MODE" = identity_warm ]; then
    parent_cache="$OUT/cache_single_writer"; arm_cache="$parent_cache"
  fi
  EXP12_JAX_CACHE_DIR="$parent_cache" python scripts/trace_exp12_runtime.py \
    --out "$OUT/parent_trace.jsonl" --synchronize --progress-every 100 --boundary-detail --stack-after 60 -- \
    run.py --experiment exp1 "${common[@]}" --checkpoint_dir "$OUT/parent"
  EXP12_JAX_CACHE_DIR="$arm_cache" python scripts/trace_exp12_runtime.py \
    --out "$OUT/identity_trace.jsonl" --synchronize --progress-every 100 --boundary-detail --stack-after 60 -- \
    run.py --experiment exp2_arm "${common[@]}" --overrides "fork.source=$OUT/parent" \
    --overrides fork.arm=identity --checkpoint_dir "$OUT/identity"
  python scripts/compare_identity_fork.py --run_dir "$OUT/parent" --arm_dir "$OUT/identity" \
    --out "$OUT/identity.json"
  python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT"])
result = json.loads((root / "identity.json").read_text())
assert result["pass"] and result["differences"] == []
assert result["fork_step"] == 12000
assert result["interaction_step"] == result["identity_interaction_step"] == 13000
for name in ("parent_trace.jsonl", "identity_trace.jsonl"):
    events = [json.loads(line) for line in (root / name).read_text().splitlines()]
    assert events[-1]["event"] == "summary"
    assert all(event.get("error") is None for event in events)
(root / "validation.json").write_text(
    json.dumps(
        dict(
            status="existing exact identity gate passed; not full campaign qualification",
            mode=os.environ["DIAGNOSTIC_MODE"],
            arch=os.environ["DIAGNOSTIC_ARCH"],
        ),
        indent=2,
    )
)
PY
  exit 0
fi
python scripts/trace_exp12_runtime.py \
  --out "$OUT/trace.jsonl" --synchronize --progress-every 100 --boundary-detail --stack-after 60 -- \
  scripts/profile_exp12.py --env dog-run --env_group dmc_hard \
  --archs "$DIAGNOSTIC_ARCH" --seed 990 --warmup_steps 1001 --train_steps 60 \
  --probe_repeats 1 --fork_timing --eval_cost --out "$OUT/profile.json"
python - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["OUT"])


def invalid_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def read_json(text):
    return json.loads(text, parse_constant=invalid_constant)


rows = read_json((root / "profile.json").read_text())
assert len(rows) == 1
row = rows[0]
assert (row["arch"], row["env"], row["num_interaction_steps"]) == (
    os.environ["DIAGNOSTIC_ARCH"],
    "dog-run",
    500000,
)
for name in (
    "train_it_per_s_probes_off",
    "probe_check_s",
    "fork_save_s",
    "fork_restore_s",
    "fork_state_bytes",
    "post_fork_eval_s",
):
    assert math.isfinite(row[name]) and row[name] > 0, name
assert row["fork_buffer_transitions"] == 475000
assert row["post_fork_eval_episodes"] == 10
assert len(row["probe_check_s_all"]) == 1
events = [read_json(line) for line in (root / "trace.jsonl").read_text().splitlines()]
assert events[-1]["event"] == "summary"
assert all(event.get("error") is None for event in events)
progress = [event for event in events if event["event"] == "progress"]
assert (progress[-1]["interaction_step"], progress[-1]["update_step"]) == (6061, 2124)
totals = events[-1]["totals"]
for stage, count in [("update_many", 1062), ("probe_round", 10), ("_fit", 20)]:
    assert totals[stage]["calls"] == count, stage
result = dict(
    status="complete diagnostic; not scientific qualification",
    memory_measurement_available=row.get("peak_device_bytes") is not None,
    interaction_step=6061,
    sac_updates=2124,
)
(root / "validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
PY
DIAGNOSTIC
rc=$?
printf '%s\n' "$rc" > "$OUT/exit_status.txt"
exit "$rc"
