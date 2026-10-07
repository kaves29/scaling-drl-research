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
: "${SLURM_JOB_ID:?run only inside an explicitly requested Slurm allocation}"
: "${SLURM_SUBMIT_DIR:?submit from the reviewed main directory}"
: "${OUT:?new absolute persistent output directory required}"
: "${EXPECTED_COMMIT:?reviewed full source hash required}"
: "${EXPECTED_GPU_MODEL:?reviewed literal A100 device_kind required}"
cd "$SLURM_SUBMIT_DIR"
test -f scripts/profile_exp12.py
test -f scripts/trace_exp12_runtime.py
test "$(git branch --show-current)" = claude/eloquent-fermat-inxqlt
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
case "$EXPECTED_GPU_MODEL" in *A100*) ;; *) exit 2 ;; esac
case "$OUT" in /*) ;; *) exit 2 ;; esac
OUT=$(realpath -m "$OUT")
repo=$(git rev-parse --show-toplevel)
case "$OUT/" in "$repo/"*) exit 2 ;; esac
test -d "$(dirname "$OUT")"
mkdir "$OUT"
mkdir "$OUT/temp" "$OUT/cache_single_writer"
export OUT EXPECTED_COMMIT EXPECTED_GPU_MODEL
export TMPDIR="$OUT/temp" WANDB_MODE=disabled PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda MUJOCO_GL=disable JAX_ENABLE_X64=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_MEM_FRACTION JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
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
metadata = dict(
    python=sys.version,
    jax=jax.__version__,
    jaxlib=jaxlib.__version__,
    device_kind=devices[0].device_kind,
    device=str(devices[0]),
    commit=os.environ["EXPECTED_COMMIT"],
)
Path(os.environ["OUT"], "backend.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata), flush=True)
PY
python scripts/trace_exp12_runtime.py \
  --out "$OUT/trace.jsonl" --synchronize --progress-every 100 -- \
  scripts/profile_exp12.py --env dog-run --env_group dmc_hard \
  --archs D4W1536 --seed 990 --warmup_steps 1001 --train_steps 60 \
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
    "D4W1536",
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
