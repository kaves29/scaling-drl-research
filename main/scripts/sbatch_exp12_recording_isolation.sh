#!/usr/bin/env bash
# One A100 qualification of recording isolation, true-OFF evidence, and ON performance.
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
: "${FROZEN_DIR:?}" "${CANDIDATE_DIR:?}" "${ORIGINAL_EVIDENCE:?}" "${OUT_ROOT:?}"
: "${FROZEN_COMMIT:?}" "${CANDIDATE_COMMIT:?}"
for path in "$FROZEN_DIR" "$CANDIDATE_DIR" "$ORIGINAL_EVIDENCE" "$OUT_ROOT"; do
  [[ "$path" = /* ]] || { echo "absolute path required: $path" >&2; exit 2; }
done
[[ "$FROZEN_COMMIT" = 087feab341b9be1f377697668f2d1c750ec0467f ]]
[[ "$(git -C "$FROZEN_DIR" rev-parse HEAD)" = "$FROZEN_COMMIT" ]]
[[ "$(git -C "$CANDIDATE_DIR" rev-parse HEAD)" = "$CANDIDATE_COMMIT" ]]
[[ -z "$(git -C "$FROZEN_DIR" status --porcelain --untracked-files=all)" ]]
[[ -z "$(git -C "$CANDIDATE_DIR" status --porcelain --untracked-files=all)" ]]
[[ -f "$ORIGINAL_EVIDENCE/evidence/receipt.json" ]]
[[ -f "$ORIGINAL_EVIDENCE/evidence.sha256" ]]
[[ ! -e "$OUT_ROOT/evidence" ]]
mkdir -p "$OUT_ROOT/evidence"
sha256sum "$CANDIDATE_DIR/main/scripts/sbatch_exp12_recording_isolation.sh" \
  "$CANDIDATE_DIR/main/scripts/exp12_recording_isolation_qualification.py" \
  > "$OUT_ROOT/evidence/harness.sha256"

module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_deterministic_ops=true JAX_LOG_COMPILES=1
unset JAX_PLATFORM_NAME NVIDIA_TF32_OVERRIDE
nvidia-smi -L > "$OUT_ROOT/evidence/gpu.txt"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits -l 1 > "$OUT_ROOT/evidence/gpu_monitor.csv" &
monitor_pid=$!

finish() {
  code=$?
  trap - EXIT
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  if [[ -d "$OUT_ROOT/evidence" ]]; then
    (cd "$OUT_ROOT" && find evidence -type f -print0 | sort -z | xargs -0 -r sha256sum) \
      > "$OUT_ROOT/evidence.sha256" || code=2
  fi
  printf '%s\n' "$code" > "$OUT_ROOT/exit_status.txt"
  exit "$code"
}
trap finish EXIT

(
  cd "$ORIGINAL_EVIDENCE"
  sha256sum -c evidence.sha256 > "$OUT_ROOT/evidence/original_checksums.log"
)
python - "$ORIGINAL_EVIDENCE" "$OUT_ROOT/evidence" <<'PY'
import hashlib, json, pathlib, sys
source, out = map(pathlib.Path, sys.argv[1:])
receipt_path = source / "evidence/receipt.json"
receipt = json.loads(receipt_path.read_text())
assert receipt["frozen_commit"] == "087feab341b9be1f377697668f2d1c750ec0467f"
assert receipt["harness_commit"] == "60f2787df10619f8f4ea678f6ebe73fc55f3d35a"
assert receipt["verdict"] == "BLOCKED_DIAGNOSTICS_ASSOCIATED_DIVERGENCE"
(out / "original_true_off.json").write_text(json.dumps({
    "source": str(source), "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    "frozen_commit": receipt["frozen_commit"], "harness_commit": receipt["harness_commit"],
    "verdict": receipt["verdict"],
}, indent=2) + "\n")
PY

run_arm() {
  local source_dir=$1 source_commit=$2 cache_name=$3 label=$4 mode=$5 precision=$6
  export EXP12_JAX_CACHE_DIR="$OUT_ROOT/jax_cache/$cache_name"
  export PYTHONPATH="$source_dir/main"
  cd "$source_dir/main"
  local extra=()
  case "$mode" in
    triple) extra+=(--precision "$precision") ;;
    single_on) extra+=(--single-on) ;;
    single_off) extra+=(--single-off) ;;
    *) echo "unknown mode: $mode" >&2; exit 2 ;;
  esac
  /usr/bin/time -v -o "$OUT_ROOT/evidence/$label.resource" \
    python "$CANDIDATE_DIR/main/scripts/exp12_recording_isolation_qualification.py" \
      --out "$OUT_ROOT/evidence/$label" --source-commit "$source_commit" \
      --harness-commit "$CANDIDATE_COMMIT" "${extra[@]}" \
      > "$OUT_ROOT/evidence/$label.log" 2>&1
}

run_arm "$FROZEN_DIR" "$FROZEN_COMMIT" frozen frozen_cold single_on production_tf32
run_arm "$FROZEN_DIR" "$FROZEN_COMMIT" frozen frozen_warm single_on production_tf32
run_arm "$FROZEN_DIR" "$FROZEN_COMMIT" frozen_off frozen_off_cold single_off production_tf32
run_arm "$FROZEN_DIR" "$FROZEN_COMMIT" frozen_off frozen_off_warm single_off production_tf32
run_arm "$CANDIDATE_DIR" "$CANDIDATE_COMMIT" candidate candidate_cold single_on production_tf32
run_arm "$CANDIDATE_DIR" "$CANDIDATE_COMMIT" candidate candidate_warm single_on production_tf32
run_arm "$CANDIDATE_DIR" "$CANDIDATE_COMMIT" recording_highest candidate_recording_highest triple highest
run_arm "$CANDIDATE_DIR" "$CANDIDATE_COMMIT" recording_tf32 candidate_recording_tf32 triple production_tf32

export PYTHONPATH="$CANDIDATE_DIR/main"
cd "$CANDIDATE_DIR/main"
python - "$OUT_ROOT/evidence" <<'PY'
import json, pathlib, sys
from experiments.exp12.state import latest_state_dir, state_differences
from scripts.exp12_recording_isolation_qualification import agent_deviations
root = pathlib.Path(sys.argv[1])
recording = {mode: json.loads((root / f"candidate_recording_{mode}/receipt.json").read_text())
             for mode in ("highest", "tf32")}
assert all(r["verdict"] == "PASS_EXACT_ISOLATION_AT_TESTED_CONFIGURATION"
           for r in recording.values())
frozen = latest_state_dir(root / "frozen_warm/on/state")
candidate = latest_state_dir(root / "candidate_warm/on/state")
true_off = latest_state_dir(root / "frozen_off_warm/off/state")
differences = state_differences(candidate, frozen, ignore_meta=("wandb_run_id",))
true_off_comparisons = {
    "frozen_on": {"differences": state_differences(frozen, true_off,
                                                      ignore_meta=("wandb_run_id", "actor_diagnostics", "metrics_rows")),
                  "agent_deviations": agent_deviations(frozen, true_off)},
    "candidate_on": {"differences": state_differences(candidate, true_off,
                                                         ignore_meta=("wandb_run_id", "actor_diagnostics", "metrics_rows")),
                     "agent_deviations": agent_deviations(candidate, true_off)},
}
summary = {"recording_verdicts": {mode: r["verdict"] for mode, r in recording.items()},
           "production_on_vs_frozen_on_differences": differences,
           "true_no_diagnostics_comparisons": true_off_comparisons,
           "computational_noninvasiveness_verdict": (
               "EXACT_AT_TESTED_CONFIGURATION" if not true_off_comparisons["candidate_on"]["differences"]
               else "UNRESOLVED_NUMERICAL_DIVERGENCE"),
           "performance_receipts": {label: json.loads((root / f"{label}/receipt.json").read_text())
                                    for label in ("frozen_warm", "frozen_off_warm", "candidate_warm")}}
(root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
assert not differences, differences
PY

# This tests GPU probe isolation on the unchanged tiny Exp1 entry point, not range or sensitivity.
export EXP12_JAX_CACHE_DIR="$OUT_ROOT/jax_cache/probe"
/usr/bin/time -v -o "$OUT_ROOT/evidence/probe_isolation.resource" \
  python -B -m unittest \
    tests.test_exp12_probe.ProbeDoesNotChangeTrainingTest.test_probes_on_equals_probes_off \
    > "$OUT_ROOT/evidence/probe_isolation.log" 2>&1
