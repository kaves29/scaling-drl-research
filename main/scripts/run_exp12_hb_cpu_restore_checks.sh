#!/usr/bin/env bash
# HumanoidBench CPU restoration checks on Delta, with the reviewed HumanoidBench interpreter.
# No GPU, no Slurm submission. Run on a CPU host (a CPU-partition job, or briefly on a login
# node) from a clean detached checkout of the commit under review:
#   EXPECTED_COMMIT=<sha> HB_ENV=<prefix> HB_CHECK_ROOT=<new absolute dir> \
#     bash scripts/run_exp12_hb_cpu_restore_checks.sh
# MUJOCO_GL defaults to osmesa (CPU hosts); use egl only on GPU nodes. Any skipped test, any
# HumanoidBench task that is not PASS, or a missing humanoid_bench fails the check.
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${HB_ENV:?}" "${HB_CHECK_ROOT:?}"
cd "$(dirname "$0")/.."
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" || { echo "HEAD is not $EXPECTED_COMMIT" >&2; exit 2; }
test -z "$(git status --porcelain --untracked-files=all)" || { echo 'checkout is not clean' >&2; exit 2; }
[[ "$HB_CHECK_ROOT" = /* ]] && [ ! -e "$HB_CHECK_ROOT" ] || { echo 'HB_CHECK_ROOT must be new and absolute' >&2; exit 2; }
py="$HB_ENV/bin/python"
[ -x "$py" ] || { echo "missing HumanoidBench Python: $py" >&2; exit 2; }
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
export JAX_PLATFORMS=cpu EXP12_JAX_CACHE_DIR=off WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir "$HB_CHECK_ROOT"
finish() { code=$?; trap - EXIT; printf '%s\n' "$code" > "$HB_CHECK_ROOT/exit_status.txt"; exit "$code"; }
trap finish EXIT
printf '{"commit":"%s","host":"%s","mujoco_gl":"%s"}\n' "$EXPECTED_COMMIT" "$(hostname)" "$MUJOCO_GL" \
  > "$HB_CHECK_ROOT/submission.json"
"$py" -m pip freeze > "$HB_CHECK_ROOT/dependencies.txt"
"$py" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("humanoid_bench") is None)' \
  || { echo 'humanoid_bench is not importable from HB_ENV' >&2; exit 2; }

# 1. Simulator: exact next-transition restore for every task, HumanoidBench included.
"$py" scripts/check_exp12_cpu_environments.py > "$HB_CHECK_ROOT/environments.log" 2>&1
"$py" - "$HB_CHECK_ROOT/environments.log" <<'PY'
import json, sys
text = open(sys.argv[1]).read()
summary = json.loads(text[text.rindex('{\n  "label"'):])
hb = [r for r in summary["tasks"] if r["task"] in ("h1-reach-v0", "h1-run-v0")]
bad = [r for r in summary["tasks"] if r["status"] != "PASS"]
assert len(hb) == 10, f"expected 10 HumanoidBench task/seed cells, found {len(hb)}"
assert not bad, f"non-PASS environment cells: {bad}"
print(f"environment cells: {summary['passed']}/{len(summary['tasks'])} PASS")
PY

# 2. Trainer and fork restoration with the real H1 environments and the twin critic.
tests=(
  tests.test_exp12_hb_cpu_restore
  tests.test_exp12_foundations.EnvRestoreAcrossProcessesTest
  tests.test_exp12_foundations.EnvRestoreBreakTest
  tests.test_exp12_foundations.ResumeExactnessTest
  tests.test_exp12_foundations.KillAndResumeEntryPointTest
  tests.test_exp12_pipeline.PipelinePerSuiteTest.test_humanoid_bench
  tests.test_exp12_fork.HumanoidBenchReachEvalSeedingTest
  tests.test_exp12_twin_critic
)
set +e
"$py" -m unittest -v "${tests[@]}" > "$HB_CHECK_ROOT/tests.log" 2>&1
status=$?
set -e
tail -n 5 "$HB_CHECK_ROOT/tests.log"
[ "$status" = 0 ] || { echo 'HumanoidBench CPU tests failed' >&2; exit 1; }
if grep -E -q 'skipped=|\.\.\. skipped' "$HB_CHECK_ROOT/tests.log"; then
  echo 'a HumanoidBench CPU test was skipped; that is not a pass' >&2
  exit 1
fi
sha256sum "$HB_CHECK_ROOT"/*.log "$HB_CHECK_ROOT/submission.json" > "$HB_CHECK_ROOT/SHA256SUMS"
echo 'PASS: HumanoidBench CPU restoration checks (no CUDA qualification implied)'
