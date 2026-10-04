# Exp 1/2: commands that NEED CUDA VERIFICATION, in blocks

Everything here has passed on CPU (Linux, jaxlib 0.4.34). These commands check
the same properties on the real Linux + CUDA stack. Run them from `main/` with
the project env activated, and send me the logs and JSON files.

Each step's pass/fail rule was pre-specified in `docs/exp12_decisions.md` on
2026-10-04, before any CUDA result existed. None of these commands launches or
schedules a grid run.

- **Block A** (one A100, 4 CPUs, ≤ 30 min): smoke tests, fresh-critic range
  check, short profile.
- **Blocks B–F**: everything longer, in order. B: full tests and calibration.
  C: the identity-fork gate. D: the positive control. E: preflight. F: the
  grid, which I never run.

## Matmul precision: FP32 everywhere, no TF32 (decided 2026-10-04, amendment (v))

Every Exp 1/2 job runs with `export JAX_DEFAULT_MATMUL_PRECISION=highest`.
This covers training, probes, the range and null checks, Check 1 and 2, both
arms, the positive control, preflight and the grid. TF32 is not allowed
anywhere.

- **Why:** Check 1, the identity fork and amendment (m) assume FP32. The probe
  target sin(1e5·f(x)) amplifies matmul rounding by 1e5, so FP32 keeps it
  independent of the device's TF32 behaviour.
- **Cost:** an estimated ~3× slower training and probes for D6W1536 (table
  below).
- **How it is set:** JAX honours the variable (checked:
  `JAX_DEFAULT_MATMUL_PRECISION=highest` gives
  `jax.config.jax_default_matmul_precision == "highest"`), so no code change
  is needed. The code itself never sets the precision; without the variable,
  JAX's backend default applies, which on an A100 is believed to be TF32.
- **Check:** A0 verifies `tf32_detected: false` with the setting. Check 1
  also stops any injected or identity arm that runs with TF32. Training
  itself has no such check, so the variable must be exported for every job.

## How the runtime and memory estimates were made (no GPU measurement exists yet)

**Assumptions**

- **FLOPs per update:** ≈ 6 critic forward-equivalents × 2 × params × batch
  256. This covers the critic forward + backward, the target forward, and the
  actor update through the critic. UTD 2, so 2 updates per interaction step.
- **Critic parameters (dog-run):** D2W512 4.34M, D4W1024 33.9M, D6W1536 113.7M.
- **One probe check:** 2 critics × 5 rounds × (1,000 fit steps × 3
  forward-equivalents + 3 pool passes over 25,600 inputs) ≈ 33,000
  forward-equivalents at batch 256. A single-critic probe (range mode, the
  fresh check) costs half.
- **Sustained A100 throughput at batch 256:** FP32 ≈ 12 TFLOP/s (60% of the
  19.5 peak); TF32 ≈ 55 TFLOP/s (35% of 156).
- **Environment and loop overhead:** ≈ 6 ms per interaction step for dog-run,
  inferred from the CPU timing of a 10-episode evaluation (8 ms per step,
  including actor inference). Cheaper for hopper and MyoSuite. h1-run is
  ≈ 10 ms.
- **Compile:** ≈ 1–2 min per process for D6W1536 (update scan plus probe
  fit); less for smaller critics.
- **Uncertainty:** treat every number as ±50% until Block A measures it.
- **Which figures apply:** only the FP32 ones (amendment (v)). The TF32
  figures are kept for reference.

| Critic | Train it/s, dog-run (FP32 / TF32) | One probe check, 2 critics (FP32 / TF32) | Peak GPU memory per process* | Host memory per process |
|---|---|---|---|---|
| D2W512 | ~150 / ~150 (env-bound) | ~6 s / ~2 s | < 1 GB | ~3 GB + buffer |
| D4W1024 | ~42 / ~100 | ~48 s / ~11 s | ~2 GB | ~3 GB + buffer |
| D6W1536 | ~15 / ~50 | ~160 s / ~35 s | ~5 GB (training state 1.8 GB + probe copy with Adam 1.4 GB + gradients + workspace) | ~3 GB + buffer |

\* With `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Without it, XLA reserves 75%
of the GPU per process. Buffer host memory = filled transitions × bytes per
transition: dog-run 1,948 B, i.e. 0.97 GB at 500k transitions; hopper 148 B;
MyoSuite 832–1,088 B; HumanoidBench 496–544 B.

Forecast from these numbers: the probe takes ≈ 9% of a D6W1536 dog-run's
wall-clock (FP32), above the ~5% rule. A3 and B4 measure it.

## Setup (once per node; not timed in Block A)
```bash
python -c "import jax; print(jax.devices()); import jaxlib; print(jaxlib.__version__)"
pip install -r requirements.txt                                     # adds rliable + pinned deps; no existing pin changes
bash scripts/install_humanoid_bench.sh /abs/path/to/humanoid-bench  # pinned commit, --no-deps, editable
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl                          # HumanoidBench builds an offscreen renderer
export XLA_PYTHON_CLIENT_PREALLOCATE=false                          # several processes share the GPU in the tests
export JAX_DEFAULT_MATMUL_PRECISION=highest                         # FP32 everywhere, no TF32 (amendment (v))
OUT=/abs/path/to/exp12_cuda_checks && mkdir -p $OUT
```

## Block A: one A100, 4 CPUs, ≤ 30 min in total

Estimated total: ~28 min (FP32, as decided). Peak host memory
~4 GB; peak GPU memory ~5 GB. Run the steps in order. If A1 fails, stop and
send me the log.

### A0. Matmul precision report (< 1 min; host ~2 GB, GPU < 1 GB)
It records the configured precision, the device, and the measured float32
matmul error, both at the backend default and at `highest`.
```bash
for P in default highest; do
  ( [ $P = default ] && unset JAX_DEFAULT_MATMUL_PRECISION || export JAX_DEFAULT_MATMUL_PRECISION=highest
    python -c "import json; from experiments.exp12.fork import matmul_precision_report as r; print(json.dumps(r(), indent=2))" ) \
    | tee $OUT/A0_matmul_precision_$P.json
done
```
- The `highest` file must show `tf32_detected: false`. If it does not,
  stop and send me both files: the grid's precision setting is not taking
  effect.
- The `default` file is for the record only. It shows what the A100 does
  without the setting.

### A1. Smoke tests on the GPU (~10 min; host ~4 GB, GPU < 1 GB per process)
These are the tests whose result could differ on a GPU:
- probe isolation and pairing;
- injection Check 1 tolerances;
- diagnostics bit-identity;
- Check 1 and TF32 logic;
- post-fork evaluation isolation;
- the identity-validation procedure end to end.

The estimate assumes compile- and env-bound tiny networks at about 1.3× the
measured CPU time (7.4 min for these tests).
```bash
python -m unittest -v tests.test_exp12_probe tests.test_exp12_injection tests.test_exp12_diagnostics \
  tests.test_exp12_fork.ForkUnitTest tests.test_exp12_fork.IdentityValidationTest 2>&1 | tee $OUT/A1_smoke.log
```

### A2. Fresh-critic dynamic range, dog-run (~10 min FP32 / ~5 min TF32; host ~3 GB, GPU ~4 GB for D6W1536)
- **Rule (pre-specified):** PASS if, at all three sizes, 0.1·b ≤ IQM(P) ≤
  0.9·b at the configured pool (25,600). The round-to-round spread is always
  reported, and the verdict goes to `range_verdict.json`.
- **If it fails:** the fallback ladder is a smaller pool first, then more
  probe steps; each step is proposed to you, not applied.
- **Estimate:** per size, a 5,000-step random buffer fill (~0.5 min), plus
  compile, plus 3 pool sizes × one single-critic probe (5 rounds × 1,000
  steps; D6W1536 ≈ 80 s each in FP32).
```bash
python scripts/probe_fresh_checks.py --mode range --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --pools 1600 6400 25600 --out_dir $OUT/A2_range_dog_run
```

### A3. Short profile, D6W1536 on dog-run (~8 min FP32 / ~4 min TF32; host ~3 GB, GPU ~5 GB)
- **Measures:** training it/s, the time of one full probe check, the
  projected probe overhead for a whole run, and peak GPU memory, all for the
  critic that decides the 5% rule.
- **Steps timed:** 300 training steps after 100 warm-up steps; 1 probe check
  after a compile warm-up check.
- **Estimate:** 5,000 random steps (~0.5 min), compile (~1.5 min), 300 steps
  (~0.3 min), 2 probe checks (~5.5 min).
```bash
python scripts/profile_exp12.py --env dog-run --env_group dmc_hard --archs D6W1536 \
  --train_steps 300 --warmup_steps 100 --probe_repeats 1 --out $OUT/A3_profile_dog_run_D6.json
```
Key fields: `train_it_per_s_probes_off`, `probe_check_s`,
`probe_overhead_pct_of_wallclock`, `peak_device_bytes`. If the overhead
exceeds ~5%, I report it and ask; the probe is never reduced.

## Block B: full tests and calibration (~10 h FP32 / ~4 h TF32)

### B1. Full Exp 1/2 tests and break checks (~2 h for both passes; host ~6 GB, GPU < 1 GB per process)
- Run the tests twice: once with default XLA flags and once with
  deterministic GPU ops. If only the deterministic run passes, GPU reductions
  are non-deterministic from run to run. That decides whether the identity
  fork can be bit-exact on CUDA, and it needs your decision on a tolerance;
  nothing is loosened silently.
- **Estimate:** the same modules took 50 min on CPU, mostly env stepping and
  subprocess runs; ~60 min per pass on the GPU.
- The kill-matrix tests run a child process while the parent holds the GPU,
  so `XLA_PYTHON_CLIENT_PREALLOCATE=false` (in Setup) is required.
```bash
python -m unittest discover -s tests -p "test_exp12_*.py" -t . -v 2>&1 | tee $OUT/B1_tests_default.log
XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  python -m unittest discover -s tests -p "test_exp12_*.py" -t . -v 2>&1 | tee $OUT/B1_tests_deterministic.log
python tests/exp12_break_checks.py 2>&1 | tee $OUT/B1_break_checks.log        # ~6 min (4.5 min on CPU)
```

### B2. Fresh-critic dynamic range, hopper-hop (~8 min FP32 / ~4 min TF32; host ~3 GB, GPU ~4 GB)
The same rule as A2. hopper-hop has the 500k-step budget, so it is the worst
case for the overhead share.
```bash
python scripts/probe_fresh_checks.py --mode range --env hopper-hop --env_group dmc_medium \
  --archs D2W512 D4W1024 D6W1536 --pools 1600 6400 25600 --out_dir $OUT/B2_range_hopper_hop
```

### B3. Fresh-pair null, at least 100 pairs per size (~6 h FP32 / ~1.3 h TF32; host ~3 GB, GPU ~5 GB)
- **What it saves:** every pair's per-round P, b and L, its IQM and its
  bootstrap interval, appended to `null_pairs_<arch>.jsonl` as each pair
  finishes. Re-running the same command resumes.
- **Summary:** the per-check fire rate and the 95th percentile of L.
- **Rule (pre-specified):** if the fire rate exceeds 5% at any size, you
  decide whether to adopt a null-calibrated threshold; nothing is adopted
  automatically.
- **Estimate:** one full two-critic probe check per pair. D6W1536 100 ×
  ~160 s ≈ 4.4 h; D4W1024 ≈ 1.3 h; D2W512 ≈ 10 min (FP32).
```bash
python scripts/probe_fresh_checks.py --mode null --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --null_pairs 100 --out_dir $OUT/B3_null_dog_run
```

### B4. Full compute profile (~2.1 h FP32 / ~1 h TF32; host ~8 GB, GPU ~7 GB)
- **What each run adds:**
  - dog-run (1M env steps) also times angle_1 against exp1 with probes off;
  - hopper-hop (500k) is the worst case for the probe's share;
  - myo-key-turn and h1-run-v0 give the post-fork evaluation cost of their
    suites.
- **Steps timed:** 3,000 training steps per size; 2 probe checks (1 for the
  single-size runs); 8,000-step angle_1 vs exp1 comparisons on dog-run.
- **Memory:** host peaks during `--fork_timing`, which holds two
  475,000-transition dog-run buffers and reads a 1.8 GB agent. GPU peaks
  when two D6W1536 trainers are live.
- **Estimate (FP32):** dog-run ~64 min (D6W1536 ~41 min, of which the
  angle_1 comparison is ~18 min), hopper-hop ~28 min, MyoSuite ~16 min,
  h1-run ~18 min.
```bash
python scripts/profile_exp12.py --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --train_steps 3000 --probe_repeats 2 \
  --angle1 --compare_steps 8000 --diagnostics_off --fork_timing --eval_cost --out $OUT/B4_profile_dog_run.json
python scripts/profile_exp12.py --env hopper-hop --env_group dmc_medium \
  --archs D2W512 D4W1024 D6W1536 --train_steps 3000 --probe_repeats 2 --eval_cost --out $OUT/B4_profile_hopper_hop.json
python scripts/profile_exp12.py --env myo-key-turn --env_group myosuite_simba \
  --archs D6W1536 --train_steps 3000 --probe_repeats 1 --eval_cost --out $OUT/B4_profile_myo_key_turn.json
python scripts/profile_exp12.py --env h1-run-v0 --env_group humanoid_bench \
  --archs D6W1536 --train_steps 3000 --probe_repeats 1 --eval_cost --out $OUT/B4_profile_h1_run.json
```
Key fields:
- `train_it_per_s_probes_off`, `probe_check_s`, `probe_overhead_pct_of_wallclock`
- `diagnostics_overhead_pct`: training it/s cost of the I1–I4 diagnostics
- `fork_save_s`, `fork_restore_s`, `fork_state_bytes`: complete state with
  the buffer at a 95%-of-budget fork
- `post_fork_eval_s`, `post_fork_eval_overhead_pct`: the F1 cost of 26
  evaluations against the arm's 25%-of-budget training
- `peak_device_bytes` and `recommended_jobs_per_gpu_upper_bound`: per process
- `ratio_exp1_over_angle1`

The probe rule (stop and ask if the largest critic's overhead exceeds about
5%) applies to these numbers.

## Block C: identity-fork gate (D2), per forking architecture × suite, before any Exp 1 grid launch

**Setup of the test**
- Run this on the GPU model the grid will use (amendment (p)). Only D4W1024
  and D6W1536 fork. One environment per suite: dog-run, myo-key-turn and
  h1-run-v0.
- Each parent is a dev run (seed 101, its own results root). Its f*_run is
  forced at check 2 by the test-only hook: checks 1 and 2 fire, because 2
  consecutive checks are required (amendment (l)).
- It forks there; the control saves a snapshot 1,000 interaction steps after
  the fork and stops.
- The identity arm restores the same fork state through the same path,
  saves its snapshot at the same step, and stops.

**What must match:** `compare_identity_fork.py` requires the following to
be bit-identical: every agent leaf, the optimizer state, keys, obs
normalisation, the buffer, the env RNG and replay record, the global RNGs,
counters, meters, probe records, post-fork evaluations, and the Check 1
panel. Exit code 0 = PASS. If it passes only with deterministic ops (B1),
that decision comes to you. Check 1 itself requires the identity arm to be
bit-exact, and stops it if TF32 is detected.

**Reduced-budget option (equivalent; recommended).** Add `--overrides
num_env_steps=240000` to `COMMON`. The interaction budget becomes 120,000:
checks fall every 6,000 steps, and the fork is at 12,000 instead of 50,000
(dog-run, myo-key-turn) or 100,000 (h1-run-v0). Why this does not weaken
the test:
- **Same compiled programs.** No array shape depends on the budget: batch
  256, the probe pool of 25,600, the panel of 256, and buffer arrays
  allocated at `buffer.max_length` = 1,000,000 either way. XLA therefore
  compiles and runs the same programs, and a bit-exactness failure would
  show the same way.
- **The fork state is complete in both cases.** Its contents are the same in
  kind (trained agent with Adam state, a partly filled buffer, mid-episode
  env state, RNG streams, open logging window). The training state and RNG
  streams after 7,000 updates are as general as after 45,000.
- **The compared window holds the same operations.** The fork is on a
  logging-window boundary (12,000 = 6 × 2,000), as at full budget (50,000
  and 100,000 are also multiples of 2,000). Steps 1–1,000 after the fork
  then contain, at either budget:
  - the KL reference batch redrawn after the restore;
  - no logging flush and no regular evaluation;
  - no probe check (the next is 6,000, resp. 25,000/50,000 steps later);
  - no post-fork evaluation: evaluation 0 runs at the fork itself, in both
    arms, and the next comes 1,200, resp. 5,000/10,000 steps later.
  Checked by composing both configurations for all three suites: γ and
  `buffer.max_length` are unchanged.
- **The plan constraints still hold:** check 1 (6,000) comes after the
  5,000-step random warm-up; 30,000 is a multiple of the 1,200-step
  evaluation interval; and fork + 25% ≤ 120% of the budget.

What it does not cover: the buffer at the fork holds 12,000 transitions
instead of 50,000–100,000. Saving and restoring it is the same
size-independent code path (np.savez of the filled part). Full-size save and
restore is timed in B4 (`--fork_timing`).

**Runtime estimates (FP32 / TF32), for the six cells run sequentially:**
- **Full budget:** ~6 h / ~2 h. The D6W1536 parents dominate: ~45,000
  training steps (dog-run, myo-key-turn) or ~95,000 (h1-run-v0), plus the
  fresh probe and checks 1–2.
- **Reduced:** ~1.4 h / ~35 min. Per D6W1536 cell: 5,000 random steps,
  7,000 training steps (~8 min), the fresh probe and 2 checks (~7 min), and
  2 × 1,000 post-fork steps with the restore, Check 1 and evaluation 0
  (~5 min) ≈ 20 min. Per D4W1024 cell ≈ 8 min.

**Memory:** host ~4 GB, GPU ~5 GB (D6W1536). Disk ~4 GB per D6W1536 cell
(the fork state and two snapshots, each holding a 1.8 GB agent), ~22 GB for
all six.
```bash
BUDGET=()                                          # full budget
# BUDGET=(--overrides num_env_steps=240000)        # reduced, equivalent option (see above)
for ARCH in "4 1024" "6 1536"; do set -- $ARCH; D=$1; W=$2
for SUITE in "dog-run dmc_hard" "myo-key-turn myosuite_simba" "h1-run-v0 humanoid_bench"; do
  set -- $SUITE; ENV=$1; GROUP=$2; NAME=D${D}W${W}_${ENV}; BASE=$OUT/C_identity/$NAME
  COMMON=(--config_name base_exp12 --overrides env_name=$ENV --overrides env=$GROUP --overrides seed=101
          --overrides critic_num_blocks=$D --overrides critic_hidden_dim=$W --overrides run_role=dev
          --overrides testing.force_trigger_check=2 --overrides fork.identity_snapshot_steps=1000
          --overrides testing.stop_after_identity_snapshot=true --overrides results_root=$BASE/results "${BUDGET[@]}")
  python run.py --experiment exp1 "${COMMON[@]}" --checkpoint_dir $BASE/parent 2>&1 | tee $BASE.parent.log
  python run.py --experiment exp2_arm "${COMMON[@]}" --overrides fork.source=$BASE/parent \
    --overrides fork.arm=identity --checkpoint_dir $BASE/identity 2>&1 | tee $BASE.identity.log
  python scripts/compare_identity_fork.py --run_dir $BASE/parent --arm_dir $BASE/identity \
    --out $OUT/C_identity/${NAME}.json
done; done
```

## Block D: positive control and m-selection (after Blocks A–C pass)

**The run.** A development run: D6W1536 on dog-run, seed 102 (outside 1–5),
run_role=dev, with the unchanged trigger (2 consecutive firing checks). It
forks at its own f*_run.

**The script.** Once `fork/FORK_READY` exists, it does the following on the
trigger check's own pool:
- probes the fresh, degraded and injected critics (m = last, half, all);
- computes recovery = (L_trigger − L_injected) / L_trigger, the fresh critic
  being the healthy reference (amendment (q));
- applies the m rule (d);
- reports the one-time shared-offset check separately.

Exit 3 means STOP and consult: the run never triggered, L_trigger ≤ 0, or the
probe noise (pooled SD of per-round L / L_trigger) is ≥ 0.10.

**Estimates**
- **Dev run:** ~10.5 h FP32 / ~3.5 h TF32. 500,000 interaction steps at
  ~15 / ~50 it/s, plus 21 probes (~56 / ~12 min).
  - As launched, the run continues after its fork as the control, to the full
    budget. Only `FORK_READY` is needed for the script.
  - Memory: host ~4 GB (buffer up to 0.97 GB), GPU ~5 GB, disk ~6.2 GB.
- **Script:** ~15 min FP32 / ~4 min TF32. One probe with 5 critics (≈ 2.5
  checks) plus 3 shared-offset probes with 2 critics.
  - Memory: host ~4 GB, GPU ~10 GB (three injected critics of up to 3× the
    critic's parameters are held together).
```bash
PC=$OUT/D_positive_control
python run.py --experiment exp1 --config_name base_exp12 --overrides env_name=dog-run --overrides env=dmc_hard \
  --overrides seed=102 --overrides critic_num_blocks=6 --overrides critic_hidden_dim=1536 \
  --overrides run_role=dev --overrides results_root=$PC/results --checkpoint_dir $PC/dev_run 2>&1 | tee $PC.dev_run.log
python scripts/positive_control.py --run_dir $PC/dev_run --out_dir $PC/result 2>&1 | tee $PC.result.log
```
The chosen m is not applied automatically. You freeze it by setting
`injection.m` for the Exp 2 injected arms.

## Block E: preflight on the grid's GPU model (after Blocks A–D, before the grid)

One smoke run per critic size on the node type the grid will use (tiny probe
settings, 4,000 env steps). The forking sizes also fork, run the injected arm
and require Check 1 to pass on that device.

**Estimates (FP32):** D2W512 ~2 min, D4W1024 ~4 min, D6W1536 ~8 min (2,000
training steps, the injected arm's 500 steps, 2 × 26 one-episode
evaluations). Host ~3 GB; GPU ≤ ~5 GB.
```bash
for ARCH in "2 512" "4 1024" "6 1536"; do set -- $ARCH
  FORK=$([ "$1" = 2 ] && echo "" || echo "--with-fork")
  python scripts/preflight_checkpoint_check.py --experiment exp1 $FORK \
    --override critic_num_blocks=$1 --override critic_hidden_dim=$2 \
    --override env_name=dog-run --override env=dmc_hard --override seed=999 \
    --checkpoint_dir $OUT/E_preflight_D$1 2>&1 | tee $OUT/E_preflight_D$1.log
done
```

## Block F: the grid (NOT run by me; for when you decide to launch)
```bash
GRID=/abs/path/exp12_grid; RESULTS=/abs/path/exp12_results
python generate_manifest.py --grid exp12 --ckpt-root $GRID --results-root $RESULTS   # 195 Exp 1 jobs
python scripts/claim_launcher.py --concurrency <from B4> --num-gpus <n> --phase-files exp12_exp1_jobs.txt
# once the positive control froze m, and as forks complete; on the GPU model each file is named after:
python generate_manifest.py --grid exp12 --ckpt-root $GRID --results-root $RESULTS --injection-m <m>
python scripts/check_manifest_overlap.py exp12_exp1_jobs.txt exp2_arms_<device>.txt   # must print OK
python scripts/claim_launcher.py --concurrency <n> --num-gpus <n> --phase-files exp2_arms_<device>.txt
```
- Every grid job must run with the same `JAX_DEFAULT_MATMUL_PRECISION`
  setting and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Re-running generate_manifest.py is safe. Runs with DONE are skipped, and
  unfinished ones resume: from state/LATEST, from the saved fork state, or
  from scratch before their first save.
- Scaled runs (D4W1024, D6W1536) that fork continue as the control on their
  device model. A control resumed on another model refuses to run, so keep
  every D4/D6 Exp 1 job on the grid's GPU model (amendment (u)).
