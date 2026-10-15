#!/usr/bin/env bash
# Exploratory five-versus-ten Method C controls. Requires owner GPU approval.
#SBATCH --job-name=method-c-ten-validation
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00

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
[[ -x "$PYTHON_BIN" && -d "$PILOT_ROOT/checkpoints" ]] || {
  echo "pinned Python or pilot checkpoints unavailable" >&2; exit 2;
}
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

require_original_replay() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
root, mode = Path(sys.argv[1]), sys.argv[2]
for point in ("early", "late"):
    if mode == "crossover":
        report = json.loads((root / f"crossover_{point}" / "summary.json").read_text())
        ok = report["source_replay_bitwise_verified"]
    else:
        report = json.loads((root / f"five_ten_{point}_set0" / "summary.json").read_text())
        ok = all(all(fields.values()) for fields in report["original_first_five_bitwise_replay"].values())
    if not ok:
        raise SystemExit(f"{mode} {point}: five-function source reproduction failed")
PY
}

require_shared_targets() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for point in ("early", "late"):
    for anchor in ("current", "fresh", "mean"):
        report = json.loads((root / f"shared_{point}_{anchor}" / "summary.json").read_text())
        if not report["identical_numerical_targets"]:
            raise SystemExit(f"{point} {anchor}: shared target offsets differ")
        reproduced = report["anchored_curve_bitwise_reproduced"]
        if reproduced is not None and not all(reproduced.values()):
            raise SystemExit(f"{point} {anchor}: anchored original fit differs")
PY
}

core() {
  for run in "$d4a" "$d4b"; do
    local label
    label="$(basename "$run")"
    run_control "${label}_crossover" "$run" crossover
    require_original_replay "$EVIDENCE_ROOT/${label}_crossover" crossover
    run_control "${label}_shared" "$run" shared
    require_shared_targets "$EVIDENCE_ROOT/${label}_shared"
    run_control "${label}_conditioning" "$run" conditioning
    run_control "${label}_five_ten" "$run" five_ten --independent-sets 2
    require_original_replay "$EVIDENCE_ROOT/${label}_five_ten" five_ten
  done
  for run in "$d4a" "$d4b"; do
    local label
    label="$(basename "$run")"
    run_control "${label}_null_five_ten" "$run" null_five_ten --null-pairs 4
  done
  run_control "$(basename "$d2")_late_null_five_ten" "$d2" null_five_ten \
    --points late --null-pairs 4
  if [[ -f "$d2/fork/state/LATEST" ]]; then
    run_control "$(basename "$d2")_early_null_five_ten" "$d2" null_five_ten \
      --points early --null-pairs 4
  else
    echo "D2 early checkpoint absent; no surrogate state used" >&2
  fi
  for run in "$d4a" "$d4b"; do
    run_control "$(basename "$run")_extension_4000" "$run" extension --extension-steps 4000
  done
}

export -f run_control require_original_replay require_shared_targets core
export PYTHON_BIN PILOT_ROOT EVIDENCE_ROOT HARNESS_SHA d2 d4a d4b
timeout --signal=TERM --kill-after=30s 7h40m bash -c core
