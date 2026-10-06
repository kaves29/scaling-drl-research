#!/bin/bash
# Exp 1/2 Block B (docs/exp12_cuda_commands.md) on one 4-GPU node, unattended, one lane per GPU:
#   GPU 0  dev run (D4W1536 dog-run, seed 102) -> positive control -> preflight (D2W512, D4W1024, D4W1536)
#   GPU 1  watcher: the dev run's injected arm (m = ARM_M, default half), once its fork state exists
#   GPU 2  packing test -> A1 follow-up measurements -> GPU test suite (default, deterministic ops)
#          + break checks + HumanoidBench tests -> hopper-hop range check -> identity forks (reduced
#          budget), once with a cold compilation cache per process and once with a shared warm cache
#   GPU 3  fresh-pair null -> per-suite training speed (myo-key-turn, h1-run-v0; dog-run comes from packing)
#
#   bash scripts/exp12_blockB.sh             # run; everything under logs/blockB_${SLURM_JOB_ID:-local}/
#   bash scripts/exp12_blockB.sh --dry-run   # print the layout, commands and timeouts; check files; no jax
#
# No network, no installs, no interactive input; wandb is disabled. Every step has its own timeout, log
# and status line; a failing step never stops another lane. Exit-3 stops (no trigger, L_trigger <= 0,
# probe noise) skip only the steps that depend on them. A final step writes report.txt.
# ARM_M=last|half|all|pc picks the injected arm's m (pc: the m this job's positive control chooses).
# HB_ENV=<path of a conda env with HumanoidBench>: used only by the HumanoidBench steps; unset, they are
# recorded SKIPPED_UNAVAILABLE.

set -u
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.." || exit 2

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

export OUT="$(pwd)/logs/blockB_${SLURM_JOB_ID:-local}"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE  # the code sets TF32 itself (amendment (v))

# ---- settings (the CPU plumbing test shrinks them through BLOCKB_TEST_HOOKS; never needed on the GPU) ----
export ARM_M="${ARM_M:-half}"
export HB_ENV="${HB_ENV:-}"
export DEV_OVERRIDES="env_name=dog-run env=dmc_hard seed=102 critic_num_blocks=4 critic_hidden_dim=1536 run_role=dev"
export DEV_CKPT_INTERVAL=25000          # one save per probe check (dog-run: 500,000 interaction steps / 20)
export EXTRA_OVERRIDES=""               # appended to every run.py job
export PROBE_EXTRA=""                   # appended (as --override) to probe_fresh_checks.py
export PC_EXTRA=""                      # appended to positive_control.py
export PACK_CONFIGS="D4W1536:1,2,3 D4W1024:1,2,4 D2W512:1,3,4" PACK_STEPS=600 PACK_WARMUP=100 PACK_CORES=4
export PACK_EXTRA=""                    # appended to the packing jobs' config overrides
export TEST_PATTERN="test_exp12_*.py" RUN_BREAK_CHECKS=1
export RANGE_ARCHS="D2W512 D4W1024 D4W1536" RANGE_POOLS="1600 6400 25600"
export NULL_ARCHS="D2W512 D4W1024 D4W1536" NULL_PAIRS=100
export IDENTITY_ARCHS="4:1024 4:1536" IDENTITY_SUITES="dog-run:dmc_hard myo-key-turn:myosuite_simba h1-run-v0:humanoid_bench"
export IDENTITY_BUDGET=240000           # reduced, equivalent budget (Block C)
export IDENTITY_CACHE_MODES="cold warm" # item 8b: separate cold caches per process, then one shared warm cache
export SPEED_SUITES="myo-key-turn:myosuite_simba h1-run-v0:humanoid_bench" SPEED_ARCHS="D2W512 D4W1024 D4W1536"
export HB_TESTS="tests.test_exp12_pipeline.PipelinePerSuiteTest.test_humanoid_bench tests.test_exp12_fork.HumanoidBenchReachEvalSeedingTest"
export PREFLIGHT_ARCHS="2:512 4:1024 4:1536" PREFLIGHT_EXTRA=""
export LANES="0 1 2 3"
export WATCH_POLL=60
export BLOCKB_WALL="${BLOCKB_WALL:-27000}"   # 7.5 h: internal deadline inside the 8 h Slurm limit
declare -A TIMEOUT=(
  [dev_run]=21600 [positive_control]=1800 [preflight]=1200 [arm]=10800
  [pack]=900 [a1_followup]=1200 [tests]=7200 [break_checks]=1800 [range_hopper]=1800
  [identity]=3600 [null]=14400 [speed]=900 [tests_hb]=1800
)
[ -n "${BLOCKB_TEST_HOOKS:-}" ] && source "$BLOCKB_TEST_HOOKS"

ov() { local x; for x in "$@"; do printf -- "--overrides %s " "$x"; done; }

# ---- steps: each runs under timeout in its own bash, from main/, with the lane's GPU and CPUs ----
step_dev_run() {
  python run.py --experiment exp1 --config_name base_exp12 $(ov $DEV_OVERRIDES $EXTRA_OVERRIDES) \
    --overrides results_root="$OUT/gpu0/results" --checkpoint_dir "$OUT/gpu0/dev_run" \
    --checkpoint_interval "$DEV_CKPT_INTERVAL" --checkpoint_start_frac 0.0
}

step_positive_control() {
  python scripts/positive_control.py --run_dir "$OUT/gpu0/dev_run" --out_dir "$OUT/gpu0/positive_control" $PC_EXTRA
}

step_preflight() {  # $1 blocks, $2 width
  local fork=""; [ "$1" != 2 ] && fork="--with-fork"
  python scripts/preflight_checkpoint_check.py --experiment exp1 $fork \
    --override critic_num_blocks="$1" --override critic_hidden_dim="$2" \
    --override env_name=dog-run --override env=dmc_hard --override seed=999 \
    $(for x in $PREFLIGHT_EXTRA; do printf -- "--override %s " "$x"; done) --checkpoint_dir "$OUT/gpu0/preflight_D$1W$2"
}

step_arm() {  # $1 m
  python run.py --experiment exp2_arm --config_name base_exp12 $(ov $DEV_OVERRIDES $EXTRA_OVERRIDES) \
    --overrides results_root="$OUT/gpu0/results" --overrides fork.source="$OUT/gpu0/dev_run" \
    --overrides fork.arm=injected --overrides injection.m="$1" --checkpoint_dir "$OUT/gpu1/arm_injected" \
    --checkpoint_interval "$DEV_CKPT_INTERVAL" --checkpoint_start_frac 0.0
}

step_pack() {  # $1 output dir, $2 arch, $3 concurrent jobs, $4 the lane's CPU list (comma-separated), $5 env, $6 group
  local dir=$1 arch=$2 n=$3 env=$5 group=$6 i pids=() cores
  mkdir -p "$dir"
  IFS=, read -r -a cpus <<< "$4"
  if [ -n "${LANE_GPU:-}" ] && command -v nvidia-smi >/dev/null; then
    ( while [ ! -e "$dir/done" ]; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$LANE_GPU" >> "$dir/gpu_mem_mib.txt" 2>/dev/null
        sleep 1
      done ) &
  fi
  for ((i = 0; i < n; i++)); do
    cores=$(for ((j = 0; j < PACK_CORES; j++)); do echo "${cpus[$(( (i * PACK_CORES + j) % ${#cpus[@]} ))]}"; done | paste -sd,)
    taskset -c "$cores" python - "$arch" "$n" "$i" "$PACK_STEPS" "$PACK_WARMUP" "$dir" "$env" "$group" $PACK_EXTRA \
      > "$dir/job_$i.log" 2>&1 <<'PY' &
import json, os, random, sys, tempfile, time
from pathlib import Path
import jax, numpy as np
from experiments.exp12.precision import configure_compilation_cache, set_matmul_precision
set_matmul_precision()
configure_compilation_cache()
from experiments.exp1 import compose_config
from experiments.exp12.trainer import Exp12Trainer

arch, n, i, steps, warmup, d = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), Path(sys.argv[6])
env, group = sys.argv[7], sys.argv[8]
blocks, width = {"D2W512": (2, 512), "D4W1024": (4, 1024), "D4W1536": (4, 1536)}[arch]
cfg = compose_config(os.path.abspath("configs"), "base_exp12", [
    f"env_name={env}", f"env={group}", f"critic_num_blocks={blocks}", f"critic_hidden_dim={width}",
    f"seed={990 + i}", "run_role=dev", "num_eval_episodes=1", *sys.argv[9:]])
np.random.seed(cfg.seed); random.seed(cfg.seed)
t = Exp12Trainer(cfg, tempfile.mkdtemp(prefix="pack_"))
t.start()
warm = int(cfg.buffer.min_length) + warmup
t.train(warm); jax.block_until_ready(t._sac_agent.critic.params)
(d / f"ready_{i}").touch()
deadline = time.time() + 600
while len(list(d.glob("ready_*"))) < n:  # start barrier: every job times the same window
    if time.time() > deadline:
        sys.exit("start barrier timed out (a sibling job did not reach it)")
    time.sleep(0.1)
start, p0 = time.time(), time.perf_counter()
t.train(warm + steps); jax.block_until_ready(t._sac_agent.critic.params)
elapsed, end = time.perf_counter() - p0, time.time()
stats = jax.devices()[0].memory_stats() or {}
json.dump({"arch": arch, "env": env, "jobs": n, "job": i, "timed_steps": steps, "it_per_s": steps / elapsed, "start": start,
           "end": end, "peak_bytes": stats.get("peak_bytes_in_use"), "bytes_limit": stats.get("bytes_limit"),
           "cores": sorted(os.sched_getaffinity(0)), "device": jax.devices()[0].device_kind,
           "matmul_precision": jax.config.jax_default_matmul_precision}, open(d / f"job_{i}.json", "w"), indent=2)
t.close()
PY
    pids+=($!)
  done
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  touch "$dir/done"
  wait
  return $rc
}

step_a1_followup() {  # measurements behind the A1 failure classification (decisions log, 2026-10-05)
  python - "$OUT/gpu2/a1_followup_${1}.json" <<'PY'
import json, os, sys
sys.path.insert(0, "tests")
import jax, jax.numpy as jnp, numpy as np
from experiments.exp12.precision import set_matmul_precision
set_matmul_precision()  # as in A1, where an earlier exp1.run had set it process-wide
from experiments.exp12 import injection
from test_exp12_injection import _batch, _critic, _inject
from test_exp12_diagnostics import OBS, _agent, _batches, _gaussian

EPS = float(np.finfo(np.float32).eps)
out = {"XLA_FLAGS": os.environ.get("XLA_FLAGS"), "device": jax.devices()[0].device_kind,
       "run_setting": jax.config.jax_default_matmul_precision, "injection": [], "diagnostics_on_off": {}, "policy_kl": {}}

def q_and_grad(tr, obs, act):
    q = tr.network_def.apply({"params": tr.params}, obs, act)
    g = jax.grad(lambda a: tr.network_def.apply({"params": tr.params}, obs, a).sum())(act)
    return np.asarray(q), np.asarray(g)

obs, act, _ = _batch(1, 256)
for blocks in (2, 4):
    critic, target = _critic(blocks)
    for m in injection.M_LABELS:
        inj, inj_t = _inject(critic, target, m)
        for which, before, after in (("critic", critic, inj), ("target", target, inj_t)):
            row = {"blocks": blocks, "m": m, "pair": which}
            for prec in ("tensorfloat32", "highest"):
                with jax.default_matmul_precision(prec):
                    q0, g0 = q_and_grad(before, obs, act)
                    q1, g1 = q_and_grad(after, obs, act)
                row[prec] = {"q_bit_identical": bool(np.array_equal(q0, q1)), "max_abs_dq": float(np.abs(q0 - q1).max()),
                             "dq_da_dev_eps_units": float(np.abs(g0 - g1).max() / (EPS * np.abs(g0).max()))}
            out["injection"].append(row)

ref = np.random.default_rng(1).normal(size=(64, OBS)).astype(np.float32)
leaves = lambda g: jax.tree_util.tree_leaves((g._actor, g._critic, g._target_critic, g._temperature, g._rng))
for prec in ("tensorfloat32", "highest"):
    with jax.default_matmul_precision(prec):
        agents = []
        for diag in (False, True, False):  # off, on, off again (the same compiled program as the first)
            g = _agent(); g = getattr(g, "agent", g)
            kw = {"saturation_threshold": 0.99, "kl_ref_observations": ref} if diag else {}
            g.update_many(0, _batches(4), 2, **kw)
            agents.append(g)
        def compare(a, b):
            d = [(float(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)).max()),
                  float(np.abs(np.asarray(x, np.float64)).max())) for x, y in zip(leaves(a), leaves(b))
                 if np.asarray(x).dtype.kind == "f"]
            return {"leaves_differing": sum(1 for v, _ in d if v > 0), "leaves": len(d),
                    "max_abs_dev": max(v for v, _ in d), "max_dev_over_leaf_max": max(v / s for v, s in d if s > 0)}
        out["diagnostics_on_off"][prec] = {"off_vs_on": compare(agents[0], agents[1]),
                                           "off_vs_off_same_program": compare(agents[0], agents[2])}
        g = _agent(); g = getattr(g, "agent", g)
        before = g.actor.params
        info = g.update_many(0, _batches(), 10**9, saturation_threshold=0.99, kl_ref_observations=ref)
        (m0, s0), (m1, s1) = _gaussian(g, before, ref), _gaussian(g, g.actor.params, ref)
        kl = float(np.sum(np.log(s0 / s1) + (s1 ** 2 + (m1 - m0) ** 2) / (2 * s0 ** 2) - 0.5, axis=-1).mean())
        diag_kl = float(info["train/policy_kl"][0])
        out["policy_kl"][prec] = {"diagnostic": diag_kl, "closed_form": kl, "rel_diff": abs(diag_kl - kl) / abs(kl),
                                  "min_sigma": float(s0.min())}
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(json.dumps(out["policy_kl"]), json.dumps(out["diagnostics_on_off"]))
PY
}

step_tests() {  # $1 default|deterministic
  if [ "$1" = deterministic ]; then export XLA_FLAGS=--xla_gpu_deterministic_ops=true; fi
  export EXP12_NUMERICS_OUT="$OUT/gpu2/numerics_$1.jsonl"   # measured GPU deviations (tests/exp12_helpers.py)
  python -m unittest discover -s tests -p "$TEST_PATTERN" -t . -v
}

step_tests_hb() { python -m unittest -v $HB_TESTS; }  # under HB_ENV (PATH), MUJOCO_GL=egl

step_break_checks() { python tests/exp12_break_checks.py; }

step_range_hopper() {
  local o=""; local x; for x in $PROBE_EXTRA; do o+="--override $x "; done
  python scripts/probe_fresh_checks.py --mode range --env hopper-hop --env_group dmc_medium \
    --archs $RANGE_ARCHS --pools $RANGE_POOLS --out_dir "$OUT/gpu2/range_hopper_hop" $o || return 1
  python scripts/exp12_reports.py range-check "$OUT/gpu2/range_hopper_hop" $RANGE_ARCHS || return 4  # amendment (w)
}

step_identity() {  # $1 blocks, $2 width, $3 env, $4 group, $5 cache mode (Block C, reduced budget; item 8b)
  local name="D$1W$2_$3" base="$OUT/gpu2/identity/$5/D$1W$2_$3" t
  local common="--config_name base_exp12 $(ov env_name=$3 env=$4 seed=101 critic_num_blocks=$1 critic_hidden_dim=$2 \
    run_role=dev testing.force_trigger_check=2 fork.identity_snapshot_steps=1000 testing.stop_after_identity_snapshot=true \
    results_root=$base/results num_env_steps=$IDENTITY_BUDGET $EXTRA_OVERRIDES)"
  local parent_cache="$base/cache_parent" arm_cache="$base/cache_arm"   # cold: each process compiles for itself
  [ "$5" = warm ] && parent_cache="$base/cache_shared" && arm_cache="$base/cache_shared"  # warm: the arm reuses
  mkdir -p "$base"
  t=$SECONDS
  EXP12_JAX_CACHE_DIR="$parent_cache" python run.py --experiment exp1 $common --checkpoint_dir "$base/parent" || return 1
  echo "[identity] parent process wall $((SECONDS - t)) s (cache $parent_cache)"
  t=$SECONDS
  EXP12_JAX_CACHE_DIR="$arm_cache" python run.py --experiment exp2_arm $common --overrides fork.source="$base/parent" \
    --overrides fork.arm=identity --checkpoint_dir "$base/identity" || return 1
  echo "[identity] identity-arm process wall $((SECONDS - t)) s (cache $arm_cache)"
  python scripts/compare_identity_fork.py --run_dir "$base/parent" --arm_dir "$base/identity" \
    --out "$OUT/gpu2/identity/$5/$name.json"
}

step_null() {
  local o=""; local x; for x in $PROBE_EXTRA; do o+="--override $x "; done
  python scripts/probe_fresh_checks.py --mode null --env dog-run --env_group dmc_hard \
    --archs $NULL_ARCHS --null_pairs "$NULL_PAIRS" --out_dir "$OUT/gpu3/null_dog_run" $o
}

suite_available() {  # $1 group: exits 0 if this install can run the suite (no jax imported)
  case "$1" in
    dmc_*) python -c "import dm_control.suite" ;;
    myosuite_*) python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('myosuite') is None)" ;;
    humanoid_bench)  # only through HB_ENV (item 6); the main environment is never used for HumanoidBench
      [ -n "$HB_ENV" ] && [ -x "$HB_ENV/bin/python" ] || { echo "HB_ENV is not set or has no bin/python"; return 1; }
      "$HB_ENV/bin/python" -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('humanoid_bench') is None)" &&
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$HB_ENV/bin/python" -c "from dm_control import _render; assert _render.BACKEND == 'egl'" ;;
    *) return 1 ;;
  esac
}

export -f ov step_dev_run step_positive_control step_preflight step_arm step_pack step_a1_followup step_tests \
  step_tests_hb step_break_checks step_range_hopper step_identity step_null suite_available

suite_run() {  # group run_step-args...: run_step in the environment the suite needs (HumanoidBench: HB_ENV, EGL)
  local group=$1; shift
  if [ "$group" = humanoid_bench ]; then
    PATH="$HB_ENV/bin:$PATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl run_step "$@"
  else
    run_step "$@"
  fi
}

# ---- runner helpers ----
START=$(date +%s); export START
remaining() { echo $(( START + BLOCKB_WALL - 300 - $(date +%s) )); }   # 5 min kept for the report

record() {  # lane step result exit wall start
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "$OUT/status/$1.tsv"
  echo "[$(date +%H:%M:%S)] $1 $2: $3 (exit $4, ${5}s)"
}

run_step() {  # lane step timeout_key fn args... ; returns the step's exit code
  local lane=$1 step=$2 key=$3 fn=$4; shift 4
  local t=${TIMEOUT[$key]} left; left=$(remaining)
  [ "$left" -lt "$t" ] && t=$left
  if [ "$t" -le 60 ]; then record "$lane" "$step" SKIPPED_NO_TIME - - "$(date +%s)"; return 125; fi
  local s; s=$(date +%s)
  taskset -c "$LANE_CPUS" timeout -k 30 "$t" bash -c '"$@"' _ "$fn" "$@" > "$OUT/$lane/$step.log" 2>&1
  local code=$? result
  case "$code" in
    0) result=PASS ;; 124|137) result=TIMEOUT ;; 3) result=STOP_EXIT3 ;; 4) result=FAIL_RULE ;; *) result=FAIL ;;
  esac
  record "$lane" "$step" "$result" "$code" "$(( $(date +%s) - s ))" "$s"
  return "$code"
}

lane_gpu0() {
  run_step gpu0 dev_run dev_run step_dev_run
  touch "$OUT/gpu0/dev_run.finished"
  if [ -e "$OUT/gpu0/dev_run/fork/FORK_READY" ]; then
    run_step gpu0 positive_control positive_control step_positive_control
  else
    record gpu0 positive_control "SKIPPED_NO_TRIGGER" - - "$(date +%s)"  # exit-3 rule: no fork, nothing to probe
  fi
  touch "$OUT/gpu0/positive_control.finished"
  local a
  for a in $PREFLIGHT_ARCHS; do run_step gpu0 "preflight_D${a%%:*}W${a##*:}" preflight step_preflight "${a%%:*}" "${a##*:}"; done
}

lane_gpu1() {  # the injected-arm watcher
  local s; s=$(date +%s)
  until [ -e "$OUT/gpu0/dev_run/fork/FORK_READY" ]; do
    if [ -e "$OUT/gpu0/dev_run.finished" ] && [ ! -e "$OUT/gpu0/dev_run/fork/FORK_READY" ]; then
      record gpu1 arm_injected SKIPPED_NO_TRIGGER - "$(( $(date +%s) - s ))" "$s"
      echo "the dev run ended without a fork (never triggered, or failed); no injected arm" > "$OUT/gpu1/arm_injected.log"
      return 0
    fi
    [ "$(remaining)" -le 60 ] && { record gpu1 arm_injected SKIPPED_NO_TIME - - "$s"; return 0; }
    sleep "$WATCH_POLL"
  done
  local m=$ARM_M
  if [ "$m" = pc ]; then
    until [ -e "$OUT/gpu0/positive_control.finished" ]; do
      [ "$(remaining)" -le 60 ] && { record gpu1 arm_injected SKIPPED_NO_TIME - - "$s"; return 0; }
      sleep "$WATCH_POLL"
    done
    m=$(python -c "import json, sys; print(json.load(open(sys.argv[1])).get('chosen_m') or '')" \
          "$OUT/gpu0/positive_control/positive_control.json" 2>/dev/null)
    if [ -z "$m" ]; then
      record gpu1 arm_injected SKIPPED_EXIT3 - "$(( $(date +%s) - s ))" "$s"
      echo "the positive control chose no m (exit-3 stop or failure; see gpu0/positive_control.log); no injected arm" \
        > "$OUT/gpu1/arm_injected.log"
      return 0
    fi
  fi
  echo "$m" > "$OUT/gpu1/arm_m.txt"
  run_step gpu1 arm_injected arm step_arm "$m"
}

lane_gpu2() {
  local cfg arch n
  for cfg in $PACK_CONFIGS; do
    arch=${cfg%%:*}
    for n in $(echo "${cfg#*:}" | tr , ' '); do
      run_step gpu2 "pack_${arch}_x$n" pack step_pack "$OUT/gpu2/packing/${arch}_x$n" "$arch" "$n" "$LANE_CPUS" dog-run dmc_hard
    done
  done
  run_step gpu2 a1_followup_default a1_followup step_a1_followup default
  XLA_FLAGS=--xla_gpu_deterministic_ops=true run_step gpu2 a1_followup_deterministic a1_followup step_a1_followup deterministic
  local hb_gl=()
  if suite_available humanoid_bench >/dev/null 2>&1; then hb_gl=(MUJOCO_GL=egl PYOPENGL_PLATFORM=egl); fi
  if [ ${#hb_gl[@]} -gt 0 ]; then export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl; fi  # HumanoidBench tests need EGL
  run_step gpu2 tests_default tests step_tests default
  [ "$RUN_BREAK_CHECKS" = 1 ] && run_step gpu2 break_checks break_checks step_break_checks
  run_step gpu2 tests_deterministic tests step_tests deterministic
  export MUJOCO_GL=disable; unset PYOPENGL_PLATFORM
  if suite_available humanoid_bench > "$OUT/gpu2/suite_humanoid_bench.log" 2>&1; then
    suite_run humanoid_bench gpu2 tests_humanoid_bench tests_hb step_tests_hb
  else
    record gpu2 tests_humanoid_bench SKIPPED_UNAVAILABLE - - "$(date +%s)"
  fi
  run_step gpu2 range_hopper range_hopper step_range_hopper
  local a s env group mode
  for mode in $IDENTITY_CACHE_MODES; do
    for s in $IDENTITY_SUITES; do
      env=${s%%:*}; group=${s##*:}
      if ! suite_available "$group" > "$OUT/gpu2/suite_$group.log" 2>&1; then
        for a in $IDENTITY_ARCHS; do
          record gpu2 "identity_${mode}_D${a%%:*}W${a##*:}_$env" SKIPPED_UNAVAILABLE - - "$(date +%s)"
        done
        continue
      fi
      for a in $IDENTITY_ARCHS; do
        suite_run "$group" gpu2 "identity_${mode}_D${a%%:*}W${a##*:}_$env" identity step_identity \
          "${a%%:*}" "${a##*:}" "$env" "$group" "$mode"
      done
    done
  done
}

lane_gpu3() {
  run_step gpu3 null_dog_run null step_null
  local s env group arch  # per-suite training speed, one job, 4 cores (dog-run: the packing test's 1-job runs)
  for s in $SPEED_SUITES; do
    env=${s%%:*}; group=${s##*:}
    if ! suite_available "$group" > "$OUT/gpu3/suite_$group.log" 2>&1; then
      for arch in $SPEED_ARCHS; do record gpu3 "speed_${env}_$arch" SKIPPED_UNAVAILABLE - - "$(date +%s)"; done
      continue
    fi
    for arch in $SPEED_ARCHS; do
      suite_run "$group" gpu3 "speed_${env}_$arch" speed step_pack "$OUT/gpu3/speed/${env}_$arch" "$arch" 1 "$LANE_CPUS" "$env" "$group"
    done
  done
}

# ---- layout: GPUs by UUID, 4 CPU groups (16 each on a 64-CPU allocation) ----
CPUS=($(python -c "import os; print(' '.join(map(str, sorted(os.sched_getaffinity(0)))))"))
GPUS=()
if command -v nvidia-smi >/dev/null; then GPUS=($(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)); fi
per_lane=$(( ${#CPUS[@]} / 4 )); [ "$per_lane" -lt 1 ] && per_lane=1
lane_cpus() { local k=$1; local g=("${CPUS[@]:$(( (k * per_lane) % ${#CPUS[@]} )):$per_lane}"); (IFS=,; echo "${g[*]}"); }

dry_run() {
  local fail=0 f k
  echo "Block B dry run from $(pwd); output would go to $OUT"
  echo "ARM_M=$ARM_M  internal deadline ${BLOCKB_WALL}s  CPUs visible ${#CPUS[@]} (${per_lane} per lane)  GPUs visible ${#GPUS[@]}"
  if suite_available humanoid_bench >/dev/null 2>&1; then echo "HB_ENV=$HB_ENV: HumanoidBench and EGL load"
  else echo "HB_ENV=${HB_ENV:-<unset>}: HumanoidBench steps will be recorded SKIPPED_UNAVAILABLE"; fi
  for k in 0 1 2 3; do echo "  lane gpu$k: GPU ${GPUS[$k]:-<none visible here>}  CPUs $(lane_cpus $k)"; done
  echo; echo "settings:"
  for k in DEV_OVERRIDES DEV_CKPT_INTERVAL EXTRA_OVERRIDES PROBE_EXTRA PC_EXTRA PACK_CONFIGS PACK_STEPS PACK_WARMUP \
           PACK_CORES PACK_EXTRA TEST_PATTERN RUN_BREAK_CHECKS RANGE_ARCHS RANGE_POOLS NULL_ARCHS NULL_PAIRS \
           IDENTITY_ARCHS IDENTITY_SUITES IDENTITY_BUDGET IDENTITY_CACHE_MODES SPEED_SUITES SPEED_ARCHS HB_TESTS \
           PREFLIGHT_ARCHS PREFLIGHT_EXTRA LANES WATCH_POLL; do
    echo "  $k=${!k}"
  done
  echo; echo "timeouts (s):"; for k in "${!TIMEOUT[@]}"; do echo "  $k ${TIMEOUT[$k]}"; done | sort
  for f in step_dev_run step_positive_control step_preflight step_arm step_a1_followup step_tests step_break_checks \
           step_range_hopper step_identity step_null; do
    echo; echo "=== $f ==="; declare -f "$f" | sed '1,2d;$d'
  done
  echo; echo "=== step_pack: $PACK_CONFIGS; $PACK_STEPS timed steps after $PACK_WARMUP warm-up steps; $PACK_CORES cores per job ==="
  case "$ARM_M" in last|half|all|pc) ;; *) echo "ARM_M must be last|half|all|pc"; fail=1 ;; esac
  for f in run.py scripts/positive_control.py scripts/preflight_checkpoint_check.py scripts/probe_fresh_checks.py \
           scripts/compare_identity_fork.py scripts/exp12_reports.py scripts/collect_report.sh tests/exp12_break_checks.py \
           tests/test_exp12_injection.py tests/test_exp12_diagnostics.py experiments/exp12/precision.py \
           experiments/exp1.py experiments/exp2_arm.py experiments/exp12/trainer.py configs/base_exp12.yaml \
           configs/env/dmc_hard.yaml configs/env/dmc_medium.yaml configs/env/myosuite_simba.yaml configs/env/humanoid_bench.yaml; do
    [ -f "$f" ] || { echo "MISSING file: $f"; fail=1; }
  done
  for f in python timeout taskset; do command -v "$f" >/dev/null || { echo "MISSING command: $f"; fail=1; }; done
  command -v nvidia-smi >/dev/null || echo "note: nvidia-smi not found here (expected on a login node; needed on the GPU node)"
  bash -n "$SELF" || { echo "bash -n failed on $SELF"; fail=1; }
  [ "$fail" = 0 ] && echo "DRY RUN OK: all referenced files exist and $SELF parses." || echo "DRY RUN FAILED"
  return "$fail"
}

[ "$DRY_RUN" = 1 ] && { dry_run; exit $?; }

mkdir -p "$OUT"/{status,gpu0,gpu1,gpu2,gpu3}
{ echo "cpu_model	$(lscpu | sed -n 's/^Model name: *//p' | head -1)"
  echo "cpus_visible	${#CPUS[@]}"
  for k in 0 1 2 3; do echo "lane_gpu${k}	${GPUS[$k]:-none}	$(lane_cpus $k)"; done
  echo "arm_m	$ARM_M"; echo "hb_env	${HB_ENV:-unset}"; echo "started	$(date -Is)"; } > "$OUT/host.tsv"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv > "$OUT/gpus.csv" 2>&1
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null

write_report() { python scripts/exp12_reports.py blockB "$OUT" > /dev/null 2>"$OUT/report_errors.log"; }
trap 'echo "SIGTERM: writing the report"; write_report; exit 143' TERM

for k in $LANES; do
  ( export LANE_GPU="${GPUS[$k]:-}" LANE_CPUS="$(lane_cpus $k)"
    [ -n "$LANE_GPU" ] && export CUDA_VISIBLE_DEVICES="$LANE_GPU"
    "lane_gpu$k" ) > "$OUT/gpu$k/lane.log" 2>&1 &
done
wait
write_report
cat "$OUT/report.txt"
grep -q "^OVERALL: PASS" "$OUT/report.txt"
