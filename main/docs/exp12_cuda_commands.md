# Exp 1/2: commands that NEED CUDA VERIFICATION (run in this order)

Everything here has passed on CPU (Linux, jaxlib 0.4.34). These commands check
the same properties on the real Linux + CUDA stack. Run them from `main/` with
the project env activated, on one otherwise-idle GPU, and send me the logs and
JSON files.

Each step's pass/fail rule was pre-specified in `docs/exp12_decisions.md` on
2026-10-04, before any CUDA result existed. None of these commands launches or
schedules a grid run.

## 0. Setup
```bash
python -c "import jax; print(jax.devices()); import jaxlib; print(jaxlib.__version__)"
pip install -r requirements.txt                                     # adds rliable + pinned deps; no existing pin changes
bash scripts/install_humanoid_bench.sh /abs/path/to/humanoid-bench  # pinned commit, --no-deps, editable
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl                          # HumanoidBench builds an offscreen renderer
OUT=/abs/path/to/exp12_cuda_checks && mkdir -p $OUT
# float32 matmul precision on this GPU: Check 1 stops the injected arm if TF32 is detected (amendment (m))
python -c "import json; from experiments.exp12.fork import matmul_precision_report as r; print(json.dumps(r(), indent=2))" \
  | tee $OUT/0_matmul_precision.json
```
If `tf32_detected` is true, send me this file before anything else: the arms
would stop at Check 1.

## 1. Tests and break checks
Run the tests twice: once with default XLA flags and once with deterministic GPU
ops. If only the deterministic run passes, GPU reductions are non-deterministic
from run to run. That decides whether the identity fork can be bit-exact on
CUDA, and needs your decision on a tolerance; nothing is loosened silently.
```bash
python -m unittest discover -s tests -p "test_exp12_*.py" -t . -v 2>&1 | tee $OUT/1_tests_default.log
XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  python -m unittest discover -s tests -p "test_exp12_*.py" -t . -v 2>&1 | tee $OUT/1_tests_deterministic.log
python tests/exp12_break_checks.py 2>&1 | tee $OUT/1_break_checks.log
```

## 2. Fresh-critic dynamic range (fresh critics only)
Pre-specified rule: PASS if, at all three sizes, 0.1·b ≤ IQM(P) ≤ 0.9·b at the
configured pool (25,600). The round-to-round spread is always reported. The
verdict is written to `range_verdict.json`. If it fails, the fallback ladder is
a smaller pool first, then more probe steps; each step is proposed to you, not
applied.
```bash
python scripts/probe_fresh_checks.py --mode range --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --pools 1600 6400 25600 --out_dir $OUT/2_range_dog_run
python scripts/probe_fresh_checks.py --mode range --env hopper-hop --env_group dmc_medium \
  --archs D2W512 D4W1024 D6W1536 --pools 1600 6400 25600 --out_dir $OUT/2_range_hopper_hop
```

## 3. Fresh-pair null (at least 100 pairs per size)
Every pair's per-round P, b and L, its IQM and its bootstrap interval are
appended to `null_pairs_<arch>.jsonl` as each pair finishes. Re-running the
same command resumes. The summary reports the per-check fire rate and the 95th
percentile of L. Pre-specified: if the fire rate exceeds 5% at any size, you
decide whether to adopt a null-calibrated threshold; nothing is adopted
automatically.
```bash
python scripts/probe_fresh_checks.py --mode null --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --null_pairs 100 --out_dir $OUT/3_null_dog_run
```

## 4. Compute profile per critic size (probe, diagnostics, fork, post-fork evaluations)
- dog-run (1M env steps) also times angle_1 against exp1 with probes off.
- hopper-hop (500k env steps) is the worst case for probe overhead as a
  share of the run.
- myo-key-turn and h1-run-v0 give the post-fork evaluation cost for their
  suites.
```bash
python scripts/profile_exp12.py --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --train_steps 3000 --probe_repeats 2 \
  --angle1 --compare_steps 8000 --diagnostics_off --fork_timing --eval_cost --out $OUT/4_profile_dog_run.json
python scripts/profile_exp12.py --env hopper-hop --env_group dmc_medium \
  --archs D2W512 D4W1024 D6W1536 --train_steps 3000 --probe_repeats 2 --eval_cost --out $OUT/4_profile_hopper_hop.json
python scripts/profile_exp12.py --env myo-key-turn --env_group myosuite_simba \
  --archs D6W1536 --train_steps 3000 --probe_repeats 1 --eval_cost --out $OUT/4_profile_myo_key_turn.json
python scripts/profile_exp12.py --env h1-run-v0 --env_group humanoid_bench \
  --archs D6W1536 --train_steps 3000 --probe_repeats 1 --eval_cost --out $OUT/4_profile_h1_run.json
```
Key fields:
- `train_it_per_s_probes_off`, `probe_check_s`, `probe_overhead_pct_of_wallclock`
- `diagnostics_overhead_pct`: training it/s cost of the I1–I4 diagnostics
- `fork_save_s`, `fork_restore_s`, `fork_state_bytes`: complete state with
  the buffer at a 95%-of-budget fork
- `post_fork_eval_s`, `post_fork_eval_overhead_pct`: the F1 cost of 26
  evaluations against the arm's 25%-of-budget training
- `peak_device_bytes` and `recommended_jobs_per_gpu_upper_bound`: per
  process; set `XLA_PYTHON_CLIENT_PREALLOCATE=false` when running jobs
  concurrently
- `ratio_exp1_over_angle1`

The probe rule (stop and ask if the largest critic's overhead exceeds about
5%) applies to these numbers.

## 5. Identity-fork gate (D2): per forking architecture × suite, before any Exp 1 grid launch
Run this on the GPU model the grid will use (amendment (p)). Only D4W1024 and
D6W1536 fork. One environment per suite: dog-run, myo-key-turn and h1-run-v0.

Each parent is a dev run (seed 101, its own results root) whose f*_run is
forced at check 2 by the test-only hook: checks 1 and 2 fire, because 2
consecutive checks are required (amendment (l)). It forks there; the control
saves a snapshot 1,000 interaction steps after the fork and stops. The
identity arm restores the same fork state through the same path, saves its
snapshot at the same step, and stops.

`compare_identity_fork.py` then requires the following to be bit-identical:
every agent leaf, the optimizer state, keys, obs normalisation, the buffer,
the env RNG and replay record, the global RNGs, counters, meters, probe
records, post-fork evaluations, and the Check 1 panel. Exit code 0 = PASS.
If it passes only with deterministic ops (section 1), that decision comes to
you. Check 1 itself requires the identity arm to be bit-exact, and stops it
if TF32 is detected.
```bash
for ARCH in "4 1024" "6 1536"; do set -- $ARCH; D=$1; W=$2
for SUITE in "dog-run dmc_hard" "myo-key-turn myosuite_simba" "h1-run-v0 humanoid_bench"; do
  set -- $SUITE; ENV=$1; GROUP=$2; NAME=D${D}W${W}_${ENV}; BASE=$OUT/5_identity/$NAME
  COMMON=(--config_name base_exp12 --overrides env_name=$ENV --overrides env=$GROUP --overrides seed=101
          --overrides critic_num_blocks=$D --overrides critic_hidden_dim=$W --overrides run_role=dev
          --overrides testing.force_trigger_check=2 --overrides fork.identity_snapshot_steps=1000
          --overrides testing.stop_after_identity_snapshot=true --overrides results_root=$BASE/results)
  python run.py --experiment exp1 "${COMMON[@]}" --checkpoint_dir $BASE/parent 2>&1 | tee $BASE.parent.log
  python run.py --experiment exp2_arm "${COMMON[@]}" --overrides fork.source=$BASE/parent \
    --overrides fork.arm=identity --checkpoint_dir $BASE/identity 2>&1 | tee $BASE.identity.log
  python scripts/compare_identity_fork.py --run_dir $BASE/parent --arm_dir $BASE/identity \
    --out $OUT/5_identity/${NAME}.json
done; done
```
Cost: each parent trains to 10% of its budget plus 1,000 steps (dog-run D6W1536:
50,000 + 1,000 interaction steps), and each arm runs 1,000 steps. Section 4
gives the throughput needed to cost this.

## 6. Positive control and m-selection (after sections 1-5 pass)
A development run: D6W1536 on dog-run, seed 102 (outside 1-5), run_role=dev,
with the unchanged trigger (2 consecutive firing checks). It forks at its own
f*_run. Once `fork/FORK_READY` exists, the script does the following on the
trigger check's own pool:
- probes the fresh, degraded and injected critics (m = last, half, all);
- computes recovery = (L_trigger − L_injected) / L_trigger, the fresh critic
  being the healthy reference (amendment (q));
- applies the m rule (d);
- reports the one-time shared-offset check separately.

Exit 3 means STOP and consult: the run never triggered, L_trigger ≤ 0, or the
probe noise (pooled SD of per-round L / L_trigger) is ≥ 0.10.
```bash
PC=$OUT/6_positive_control
python run.py --experiment exp1 --config_name base_exp12 --overrides env_name=dog-run --overrides env=dmc_hard \
  --overrides seed=102 --overrides critic_num_blocks=6 --overrides critic_hidden_dim=1536 \
  --overrides run_role=dev --overrides results_root=$PC/results --checkpoint_dir $PC/dev_run 2>&1 | tee $PC.dev_run.log
python scripts/positive_control.py --run_dir $PC/dev_run --out_dir $PC/result 2>&1 | tee $PC.result.log
```
The chosen m is not applied automatically. You freeze it by setting
`injection.m` for the Exp 2 injected arms.

## 7. Preflight on the grid's GPU model (after sections 1-6, before the grid)
One smoke run per critic size on the node type the grid will use (tiny probe
settings, 4,000 env steps). The forking sizes also fork, run the injected arm
and require Check 1 to pass on that device.
```bash
for ARCH in "2 512" "4 1024" "6 1536"; do set -- $ARCH
  FORK=$([ "$1" = 2 ] && echo "" || echo "--with-fork")
  python scripts/preflight_checkpoint_check.py --experiment exp1 $FORK \
    --override critic_num_blocks=$1 --override critic_hidden_dim=$2 \
    --override env_name=dog-run --override env=dmc_hard --override seed=999 \
    --checkpoint_dir $OUT/7_preflight_D$1 2>&1 | tee $OUT/7_preflight_D$1.log
done
```

## 8. The grid (NOT run by me; for when you decide to launch)
```bash
GRID=/abs/path/exp12_grid; RESULTS=/abs/path/exp12_results
python generate_manifest.py --grid exp12 --ckpt-root $GRID --results-root $RESULTS   # 195 Exp 1 jobs
python scripts/claim_launcher.py --concurrency <from section 4> --num-gpus <n> --phase-files exp12_exp1_jobs.txt
# once the positive control froze m, and as forks complete; on the GPU model each file is named after:
python generate_manifest.py --grid exp12 --ckpt-root $GRID --results-root $RESULTS --injection-m <m>
python scripts/check_manifest_overlap.py exp12_exp1_jobs.txt exp2_arms_<device>.txt   # must print OK
python scripts/claim_launcher.py --concurrency <n> --num-gpus <n> --phase-files exp2_arms_<device>.txt
```
- Re-running generate_manifest.py is safe. Runs with DONE are skipped, and
  unfinished ones resume: from state/LATEST, from the saved fork state, or
  from scratch before their first save.
- Scaled runs (D4W1024, D6W1536) that fork continue as the control on their
  device model. A control resumed on another model refuses to run, so keep
  every D4/D6 Exp 1 job on the grid's GPU model.
