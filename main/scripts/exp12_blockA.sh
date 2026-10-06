#!/bin/bash
# Exp 1/2 Block A (docs/exp12_cuda_commands.md) on one GPU, unattended: A0, A2, A3, A1, then A4 if RUN_A4=1.
#
#   bash scripts/exp12_blockA.sh            # run; logs and summary in logs/blockA_${SLURM_JOB_ID:-local}/
#   bash scripts/exp12_blockA.sh --dry-run  # print each step's command and timeout, check files; no jax
#
# No network, no installs, no interactive input; wandb is disabled. A0 failing (no TF32 behaviour at the
# run setting on the GPU) stops the runner; any other failed or timed-out step is logged and the runner
# continues. A2 fails when amendment (w)'s range criterion fails (P/b < 0.9 at the configured pool).
# Exit status: 0 only if every step that ran passed.

set -u
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.." || exit 2

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

OUT="$(pwd)/logs/blockA_${SLURM_JOB_ID:-local}"
export OUT
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1   # keeps prints and unittest status lines in order in the logs
export MUJOCO_GL="${MUJOCO_GL:-disable}"   # Block A renders nothing
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
unset JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE  # the code sets TF32 itself (amendment (v))

STEPS=(A0 A2 A3 A1)
[ "${RUN_A4:-0}" = 1 ] && STEPS+=(A4)
declare -A TIMEOUT=([A0]=300 [A2]=1200 [A3]=900 [A1]=1500 [A4]=600)
A0_MIN_TF32_REL_ERROR=1e-5  # float32 gives ~1e-7 to 1e-6 here, TF32 ~1e-4 to 1e-3

step_A0() {
  set -o pipefail
  timeout -k 30 "$1" python - <<'PY' | tee "$OUT/A0_matmul_precision.json"
import json, jax
from experiments.exp12.precision import set_matmul_precision
from experiments.exp12.fork import matmul_precision_report as report
set_matmul_precision()
out = {"run_setting": report()}
with jax.default_matmul_precision("highest"):
    out["check1_highest"] = report()
print(json.dumps(out, indent=2))
PY
}

step_A2() {
  timeout -k 30 "$1" python scripts/probe_fresh_checks.py --mode range --env dog-run --env_group dmc_hard \
    --archs D2W512 D4W1024 D4W1536 --pools 1600 6400 25600 --out_dir "$OUT/A2_range_dog_run"
}

step_A3() {
  timeout -k 30 "$1" python scripts/profile_exp12.py --env dog-run --env_group dmc_hard --archs D4W1536 \
    --train_steps 300 --warmup_steps 100 --probe_repeats 1 --out "$OUT/A3_profile_dog_run_D4W1536.json"
}

step_A1() {
  timeout -k 30 "$1" python -m unittest -v tests.test_exp12_probe tests.test_exp12_injection \
    tests.test_exp12_diagnostics tests.test_exp12_fork.ForkUnitTest tests.test_exp12_fork.IdentityValidationTest
}

step_A4() {
  timeout -k 30 "$1" python - <<'PY'
import os, random, tempfile, time
import jax, numpy as np
from experiments.exp12.precision import set_matmul_precision
set_matmul_precision()
from experiments.exp1 import compose_config
from experiments.exp12.trainer import Exp12Trainer
cfg = compose_config(os.path.abspath("configs"), "base_exp12", [
    "env_name=dog-run", "env=dmc_hard", "critic_num_blocks=4", "critic_hidden_dim=1536", "seed=990", "run_role=dev"])
np.random.seed(cfg.seed); random.seed(cfg.seed)
t = Exp12Trainer(cfg, tempfile.mkdtemp(prefix="trace_"))
t.start()
warm = int(cfg.buffer.min_length) + 200
t.train(warm); jax.block_until_ready(t._sac_agent.critic.params)
with jax.profiler.trace(os.environ["OUT"] + "/A4_trace_D4W1536_dog_run"):
    t0 = time.perf_counter()
    t.train(warm + 300); jax.block_until_ready(t._sac_agent.critic.params)
    dt = time.perf_counter() - t0
print({"it_per_s_traced": 300 / dt, "matmul_precision": jax.config.jax_default_matmul_precision})
t.close()
PY
}

# Pass/fail rules beyond the exit code (no jax).
check_A0() {
  python - "$OUT/A0_matmul_precision.json" "$A0_MIN_TF32_REL_ERROR" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["run_setting"]
ok = (r["platform"] == "gpu" and r["matmul_precision"] == "tensorfloat32"
      and r["float32_matmul_max_rel_error"] >= float(sys.argv[2]))
print(f"A0 check: platform={r['platform']} matmul_precision={r['matmul_precision']} "
      f"rel_error={r['float32_matmul_max_rel_error']:.3g} (TF32 needs >= {sys.argv[2]}) -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
}

check_A2() {
  # Amendment (w): P/b >= 0.9 at the configured pool at every size; the old 10-90% rule is printed only.
  python scripts/exp12_reports.py range-check "$OUT/A2_range_dog_run" D2W512 D4W1024 D4W1536
}

REQUIRED_FILES=(
  experiments/exp12/precision.py experiments/exp12/fork.py experiments/exp12/trainer.py experiments/exp1.py
  scripts/probe_fresh_checks.py scripts/profile_exp12.py scripts/compare_identity_fork.py scripts/exp12_reports.py
  configs/base_exp12.yaml configs/env/dmc_hard.yaml configs/env/dmc_medium.yaml
  tests/__init__.py tests/exp12_helpers.py tests/test_exp12_probe.py tests/test_exp12_injection.py
  tests/test_exp12_diagnostics.py tests/test_exp12_fork.py
)

dry_run() {
  local fail=0
  echo "Block A dry run from $(pwd); logs would go to $OUT"
  echo "Environment: WANDB_MODE=$WANDB_MODE MUJOCO_GL=$MUJOCO_GL XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE" \
       "JAX_DEFAULT_MATMUL_PRECISION=${JAX_DEFAULT_MATMUL_PRECISION:-<unset>} NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-<unset>}"
  for s in "${STEPS[@]}"; do
    echo; echo "=== $s (timeout ${TIMEOUT[$s]} s) ==="
    declare -f "step_$s" | sed '1,2d;$d'
  done
  if [ "${RUN_A4:-0}" != 1 ]; then
    echo; echo "(A4, the GPU trace, is off; RUN_A4=1 runs it last with a ${TIMEOUT[A4]} s timeout)"
  fi
  echo
  for f in "${REQUIRED_FILES[@]}"; do
    [ -f "$f" ] || { echo "MISSING file: $f"; fail=1; }
  done
  for c in "class ForkUnitTest" "class IdentityValidationTest"; do
    grep -qs "^$c" tests/test_exp12_fork.py || { echo "MISSING in tests/test_exp12_fork.py: $c"; fail=1; }
  done
  for pair in "experiments/exp12/precision.py:def set_matmul_precision" \
              "experiments/exp12/fork.py:def matmul_precision_report" \
              "experiments/exp1.py:def compose_config" "experiments/exp12/trainer.py:class Exp12Trainer"; do
    grep -qs "^${pair#*:}" "${pair%%:*}" || { echo "MISSING in ${pair%%:*}: ${pair#*:}"; fail=1; }
  done
  for cmd in python timeout; do
    command -v "$cmd" >/dev/null || { echo "MISSING command: $cmd"; fail=1; }
  done
  bash -n "$SELF" || { echo "bash -n failed on $SELF"; fail=1; }
  [ "$fail" = 0 ] && echo "DRY RUN OK: all referenced files exist and $SELF parses." || echo "DRY RUN FAILED"
  return "$fail"
}

[ "$DRY_RUN" = 1 ] && { dry_run; exit $?; }

mkdir -p "$OUT"
STATUS_FILE="$OUT/status.tsv"
printf "step\tresult\texit_code\twall_s\n" > "$STATUS_FILE"
any_failed=0
stop=0
for s in "${STEPS[@]}"; do
  if [ "$stop" = 1 ]; then
    printf "%s\tSKIPPED\t-\t-\n" "$s" >> "$STATUS_FILE"
    continue
  fi
  echo "=== $s start $(date) (timeout ${TIMEOUT[$s]} s) ==="
  start=$(date +%s)
  "step_$s" "${TIMEOUT[$s]}" > "$OUT/$s.log" 2>&1
  code=$?
  if [ "$code" = 0 ] && declare -F "check_$s" >/dev/null; then
    "check_$s" >> "$OUT/$s.log" 2>&1
    code=$?
    [ "$code" = 0 ] || code=3
  fi
  wall=$(( $(date +%s) - start ))
  case "$code" in
    0) result=PASS ;;
    124|137) result=TIMEOUT ;;
    3) result=FAIL_RULE ;;
    *) result=FAIL ;;
  esac
  printf "%s\t%s\t%s\t%s\n" "$s" "$result" "$code" "$wall" >> "$STATUS_FILE"
  echo "=== $s $result (exit $code, ${wall} s); log $OUT/$s.log ==="
  if [ "$code" != 0 ]; then
    any_failed=1
    tail -n 20 "$OUT/$s.log"
    if [ "$s" = A0 ]; then
      echo "A0 failed: TF32 is not confirmed at the run setting; skipping the remaining steps."
      stop=1
    fi
  fi
done

python scripts/exp12_reports.py blockA "$OUT" 2>&1 || cat "$OUT/summary.txt" 2>/dev/null
echo "Block A done: $([ "$any_failed" = 0 ] && echo "all steps passed" || echo "at least one step FAILED"); summary in $OUT/summary.txt"
exit "$any_failed"
