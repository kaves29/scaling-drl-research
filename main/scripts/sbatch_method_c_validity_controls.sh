#!/usr/bin/env bash
# Exploratory Method C construct controls. No production config or checkpoint is changed.
# Submit only after the owner approves the allocation and confirms the source artifacts.
#SBATCH --job-name=method-c-validity
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -Eeuo pipefail
: "${HARNESS_SHA:?set the exact reviewed diagnostic commit}"
: "${PILOT_ROOT:?set the existing pilot root}"
: "${EVIDENCE_ROOT:?set a new absolute evidence root}"
: "${PYTHON_BIN:?set the pinned Delta Python binary}"
[[ "$EVIDENCE_ROOT" == /* && ! -e "$EVIDENCE_ROOT" ]] || {
  echo "EVIDENCE_ROOT must be a new absolute path" >&2; exit 2;
}

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$HARNESS_SHA" ]] || {
  echo "diagnostic source SHA mismatch" >&2; exit 2;
}
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]] || {
  echo "diagnostic checkout is dirty" >&2; exit 2;
}
[[ -x "$PYTHON_BIN" ]] || { echo "pinned Python unavailable" >&2; exit 2; }
[[ -d "$PILOT_ROOT/checkpoints" ]] || { echo "pilot checkpoints unavailable" >&2; exit 2; }

mkdir -p "$EVIDENCE_ROOT"
exec > >(tee -a "$EVIDENCE_ROOT/driver.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$EVIDENCE_ROOT/driver_exit_status.txt"; exit "$status"' EXIT

cd "$source_root/main"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" - <<'PY'
import jax
print({"jax": jax.__version__, "devices": [str(x) for x in jax.local_devices()]}, flush=True)
if not any(x.platform == "gpu" for x in jax.local_devices()):
    raise SystemExit("CUDA GPU backend unavailable")
if not jax.local_devices(backend="cpu"):
    raise SystemExit("CPU backend required by SAC checkpoint initialization")
PY

d2="$PILOT_ROOT/checkpoints/run4_D2W512_myo-key-turn_seed1"
d4a="$PILOT_ROOT/checkpoints/run5_D4W1024_myo-key-turn_seed1"
d4b="$PILOT_ROOT/checkpoints/run7_D4W1536_myo-key-turn_seed1"
for run in "$d2" "$d4a" "$d4b"; do
  "$PYTHON_BIN" scripts/method_c_validity_controls.py --mode inventory --run-dir "$run"
done

run_control() {
  local label="$1" run="$2" mode="$3"
  shift 3
  echo "START $label $(date -u +%FT%TZ)" >&2
  "$PYTHON_BIN" scripts/method_c_validity_controls.py \
    --run-dir "$run" --output "$EVIDENCE_ROOT/$label" \
    --mode "$mode" --expected-harness-sha "$HARNESS_SHA" "$@"
  echo "END $label $(date -u +%FT%TZ)" >&2
}

# The core controls stop before the 6-hour allocation ends; every completed
# condition has its own receipt and curves even if a later condition times out.
core() {
  for run in "$d4a" "$d4b"; do
    label="$(basename "$run")"
    run_control "${label}_crossover" "$run" crossover
    "$PYTHON_BIN" - "$EVIDENCE_ROOT/${label}_crossover" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for point in ("early", "late"):
    report = json.loads((root / f"crossover_{point}" / "summary.json").read_text())
    if not report["source_replay_bitwise_verified"]:
        raise SystemExit(f"{point}: original probe reproduction is not bitwise exact; stop")
PY
    run_control "${label}_late_null" "$run" null --points late --null-pairs 4
    run_control "${label}_conditioning" "$run" conditioning
  done
  run_control "$(basename "$d2")_late_null" "$d2" null --points late --null-pairs 4
  if [[ -f "$d2/fork/state/LATEST" ]]; then
    run_control "$(basename "$d2")_early_null" "$d2" null --points early --null-pairs 4
  else
    echo "D2 early checkpoint absent: early-pool null control unavailable; no surrogate used" >&2
  fi
  for run in "$d4a" "$d4b"; do
    label="$(basename "$run")"
    run_control "${label}_early_null" "$run" null --points early --null-pairs 4
  done
}

export -f run_control core
export PYTHON_BIN PILOT_ROOT EVIDENCE_ROOT HARNESS_SHA
export d2 d4a d4b
timeout --signal=TERM --kill-after=30s 5h40m bash -c core
