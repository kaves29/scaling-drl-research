# Exp 1/2: commands that NEED CUDA VERIFICATION, in blocks

Everything here has passed on CPU (Linux, jaxlib 0.4.34). These commands check
the same properties on the real Linux + CUDA stack. Run them from `main/` with
the project env activated, and send me the logs and JSON files.

Each step's pass/fail rule was pre-specified in `docs/exp12_decisions.md` on
2026-10-04, before any CUDA result existed. None of these commands launches or
schedules a grid run.

- **Block A** (one A100, 4 CPUs, ≤ 30 min): smoke tests, fresh-critic range
  check, short profile, optional GPU trace (A4); run by scripts/exp12_blockA.sh.
- **Blocks B–F**: everything longer, in order. B: full tests and calibration.
  C: the identity-fork gate. D: the positive control. E: preflight. F: the
  grid, which I never run.

## Matmul precision: TF32 for every matmul that can use it (amendment (v), revised 2026-10-04)

- **Setting:** every Exp 1/2 GPU entry point (exp1, exp2_arm,
  probe_fresh_checks, profile_exp12, positive_control) calls
  `experiments/exp12/precision.set_matmul_precision()` first. On a GPU it sets
  `jax_default_matmul_precision = "tensorfloat32"`; on CPU it does nothing.
- **Why that value:** jax 0.4.34 documents `"tensorfloat32"` (=
  `Precision.HIGH`) as "On GPU: uses tensorfloat32 where available,
  otherwise float32". Its `DEFAULT` is documented the same way on GPU, but the
  explicit name keeps the choice independent of JAX's default.
- **What it covers:** training updates, action selection, probes, post-fork
  evaluation and diagnostics. All dtypes stay float32: there is no bf16 or
  fp16, and x64 is off.
- **Exception:** Check 1 computes its panel Q and dQ/da inside
  `jax.default_matmul_precision("highest")` (full FP32). Verified on CPU: all
  17 critic matmuls lower with `precision = [HIGH, HIGH]` under the run
  setting and `[HIGHEST, HIGHEST]` inside Check 1's context.
- **Removed:** the rule that stopped an arm when TF32 was detected. Each
  launch records its precision, GPU model and JAX/jaxlib/CUDA versions in
  `run_metadata.json`.
- **Do not set:** `JAX_DEFAULT_MATMUL_PRECISION` (the code sets the
  precision itself) or `NVIDIA_TF32_OVERRIDE=0` (it would disable TF32 inside
  cuBLAS).

## Measured on the A100 (Block A, job 22667743, 2026-10-04, commit 7812b17)

- **Node and stack:** NVIDIA A100-SXM4-40GB; jax 0.4.34, jaxlib 0.4.34; PJRT
  C API, CUDA 12030 (12.3).
- **A0:** float32 matmul max relative error 3.05e-4 at the run setting
  (tensorfloat32, so TF32 is active) and 2.16e-7 at highest (full FP32).
- **A3, D6W1536 dog-run, TF32, probes off:**
  - training 34.6 it/s (300 steps after 100 warm-up steps);
  - one two-critic probe check 50.0 s;
  - projected probe overhead 6.77% of a run (accepted, decision R2; the
    probe is unchanged);
  - peak GPU memory 4.01 GiB.
- **A4:** 31.8 it/s under the profiler (the trace adds overhead).
- **A2:** the old 10–90% rule failed at every size; replaced by amendment
  (w). See A2 below.
- **A1:** 34 tests, 8 failures (5 of them subtests of one test). See A1
  below and docs/exp12_decisions.md (2026-10-05).
- **Wall times:** A0 24 s, A2 238 s, A3 181 s, A1 300 s, A4 87 s.

**Still unmeasured:**
- D2W512 and D4W1024 training speed and probe time;
- MyoSuite and HumanoidBench speed;
- behaviour under deterministic ops;
- Delta's host overhead per step;
- post-fork evaluation cost on the GPU;
- slowdown when jobs share a GPU (the packing test, Block B).

## How the remaining estimates are made


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
- **Host overhead per training step** (measured 2026-10-05 on this machine's
  4-core Xeon at 2.8 GHz, with the device work excluded; Delta's CPUs may
  differ by ±30%). The GPU waits for it, because action selection needs the
  previous update (no overlap without changing the algorithm). It comprises:
  - the env step: dog-run 8.1 ms (MuJoCo 4.4 ms plus dm_control's Python),
    myo-key-turn 7.7 ms, h1-run 4.2 ms;
  - about 2.7 ms of loop work: sampling 2 batches, stacking, normalising,
    host-to-device copies, the buffer add and the action read-back.
  Total ≈ 11 ms (dog-run), ≈ 10.5 ms (MyoSuite), ≈ 7 ms (h1-run). The
  earlier assumption of 6 ms was too low.
- **Device time per training step** (2 updates, TF32): about 1.5 ms for
  D2W512 (launch-bound), 3.8 ms for D4W1024 and 12.7 ms for D6W1536. It adds
  to the host overhead.
- **Compile:** ≈ 1–2 min per process for D6W1536 (update scan plus probe
  fit); less for smaller critics.
- **Calibration to Block A:**
  - D6W1536 ran 28.9 ms per step against the 23.7 ms forecast. With ~11 ms
    of host overhead (assumed equal on Delta), the device took ~17.9 ms
    (≈ 39 TFLOP/s effective, not 55).
  - A probe check took 50 s against ~36 s, a factor of 1.4.
  - The unmeasured sizes are scaled by these two factors.
- **Uncertainty:** D6W1536's training speed and probe time are measured; the
  other sizes are ±50% until Block B's packing test and profile.
- **Which figures apply:** the TF32 ones (amendment (v), revised). The FP32
  figures are kept only for comparison.

| Critic | Train it/s, dog-run, TF32 | One probe check, 2 critics, TF32 | Peak GPU memory per process* | Host memory per process |
|---|---|---|---|---|
| D2W512 | ~76 (estimate; host-bound) | ~3 s (estimate) | < 1 GB (estimate) | ~3 GB + buffer |
| D4W1024 | ~61 (estimate) | ~16 s (estimate) | ~2 GB (estimate) | ~3 GB + buffer |
| D6W1536 | **34.6 (measured, A3)** | **50.0 s (measured, A3)** | **4.01 GiB (measured, A3)** | ~3 GB + buffer |

FP32 figures from the earlier plan (for comparison only): ~70, ~35 and ~15 it/s;
probe checks ~6, ~48 and ~160 s.

\* With `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Without it, XLA reserves 75%
of the GPU per process. Buffer host memory = filled transitions × bytes per
transition: dog-run 1,948 B, i.e. 0.97 GB at 500k transitions; hopper 148 B;
MyoSuite 832–1,088 B; HumanoidBench 496–544 B.

The probe takes 6.77% of a D6W1536 dog-run's wall-clock (A3, measured). It
is above the ~5% rule; the overhead was accepted (decision R2) and the probe
is unchanged.

**Whole-run forecast, dog-run (1M env steps), TF32, refreshed from Block A:**

| Critic | Training | Probes (21 checks) | Training evaluations | Saves | Total per run | Post-fork arm (25% of B) |
|---|---|---|---|---|---|---|
| D2W512 | 1.8 h (est.) | 1 min | 7 min | 1 min | ~2.0 h | — (does not fork) |
| D4W1024 | 2.3 h (est.) | 6 min | 7 min | 2 min | ~2.5 h | ~0.9 h |
| D6W1536 | 4.0 h (measured speed) | 18 min (measured) | 7 min | 3 min | ~4.5 h | ~1.4 h |

Training evaluations: 100 episodes × 500 steps × 8.7 ms (CPU-measured). A
post-fork arm is 125,000 steps plus 26 × 10 evaluation episodes (~19 min on
dog-run) plus 5 probe checks. A run that forks late (check 19) continues as
the control to fork + 25% of B, i.e. 120% of B: the D6W1536 control then
trains ~4.8 h plus its post-fork evaluations, ~5.8 h in all.

**What changed with the efficiency scan (2026-10-05):** the host overhead was
measured instead of assumed (11 ms, not 6 ms, on dog-run). D2W512 is
host-bound at ~80 it/s rather than ~150, D4W1024 ~68 rather than ~100, and
D6W1536 ~42 rather than ~50. No code change came out of the scan, so the
estimates move only because of the measurement.

**What changed with TF32:** every per-step and per-check estimate now uses the
TF32 column. The bigger critics get faster (D6W1536 training ~15 → ~50 it/s,
one probe check ~160 → ~35 s; D4W1024 ~42 → ~100 it/s). D2W512 is
environment-bound and unchanged. Block totals are refreshed below. Tiny-network
test runs (A1, B1) do not change.

## Setup (once per node; not timed in Block A)
```bash
python -c "import jax; print(jax.devices()); import jaxlib; print(jaxlib.__version__)"
pip install -r requirements.txt                                     # adds rliable + pinned deps; no existing pin changes
bash scripts/install_humanoid_bench.sh /abs/path/to/humanoid-bench  # pinned commit, --no-deps, editable
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl                          # HumanoidBench builds an offscreen renderer
export XLA_PYTHON_CLIENT_PREALLOCATE=false                          # several processes share the GPU in the tests
OUT=/abs/path/to/exp12_cuda_checks && mkdir -p $OUT
```

## Block A: one A100, 4 CPUs, ≤ 30 min in total

Block A runs unattended through `scripts/exp12_blockA.sh`, in this order:
A0, A2, A3, A1, then A4 only if `RUN_A4=1`. On Delta, submit
`scripts/sbatch_exp12_blockA.sh` from `main/`; `logs/` must exist first,
because Slurm opens `logs/%x_%j.out` before the job starts:
```bash
mkdir -p logs
bash scripts/exp12_blockA.sh --dry-run          # prints each command and timeout, checks every file; no jax
EXPECTED_COMMIT=<commit hash I give you> sbatch scripts/sbatch_exp12_blockA.sh   # sbatch exports the environment by default
# with the optional trace:  RUN_A4=1 EXPECTED_COMMIT=<hash> sbatch scripts/sbatch_exp12_blockA.sh
```
- **Where the output goes:** `$OUT` = `main/logs/blockA_${SLURM_JOB_ID:-local}/`.
  Each step writes `$OUT/<step>.log`. At the end, `$OUT/summary.txt` holds:
  - each step's result, exit code and wall time;
  - A0's matmul error at the run setting and at highest;
  - A2's fresh score, b, round-to-round spread of P and of the final loss
    per size and pool, the (w) verdict and the old rule (information only);
  - A3's training it/s, probe-check time, probe overhead % and peak GPU memory;
  - A1's pass/fail list. Every test is listed with its status, and each
    failing subtest separately. A status pushed onto a later line by other
    output is still attributed to its test; anything that cannot be
    reconciled with unittest's totals is flagged.

  Send me the `summary.txt` file and the logs.
- **Timeouts:** A0 5 min, A2 20, A3 15, A1 25, A4 10 (each killed with its
  process group).
- **Failures:**
  - A0 failing stops the runner, and the remaining steps are marked
    SKIPPED.
  - A2 fails (exit 3, `FAIL_RULE`) when amendment (w)'s criterion fails
    (P/b < 0.9 at the configured pool at any size), not only when it
    crashes. `probe_fresh_checks.py` itself exits 0 (its `range_verdict.json`
    still holds the old 10–90% rule). The runner applies (w) through
    `scripts/exp12_reports.py range-check`.
  - Any other failure or timeout is logged, and the runner continues.
  - The runner exits nonzero if any step failed.
- **Environment the runner sets:**
  - `WANDB_MODE=disabled`;
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false` unless already set;
  - `MUJOCO_GL=disable` unless already set (Block A renders nothing;
    HumanoidBench in later blocks still needs the Setup's `egl`);
  - it unsets `JAX_DEFAULT_MATMUL_PRECISION` and `NVIDIA_TF32_OVERRIDE`.

  Nothing in Block A sets a matmul precision except the code's own
  `set_matmul_precision()` (TF32 on GPU), plus the local "highest" context
  inside A0 and Check 1.
- **Unattended:** no step reads input, downloads, installs or needs the
  network:
  - wandb is disabled or stubbed;
  - dm_control's assets are in the package;
  - A1 uses DMC only (no MyoSuite, no HumanoidBench, no rliable).

  The Setup's `pip install` must have been done beforehand on a node with
  internet; Block A itself needs nothing beyond the pinned requirements.

Estimated total: ~20 min under TF32 (~24 min with A4); the timeouts add up
to 65 min (75 with A4). Peak host memory ~4 GB; peak GPU memory ~5 GB.

### A0. Matmul precision report (timeout 5 min; < 1 min; host ~2 GB, GPU < 1 GB) — NEEDS CUDA VERIFICATION
It measures the float32 matmul error (256×256, against float64) under the run
setting (what training uses) and under `highest` (what Check 1 uses), and
records the device and versions. Output: `$OUT/A0_matmul_precision.json`.
```bash
python - <<'PY' | tee $OUT/A0_matmul_precision.json
import json, jax
from experiments.exp12.precision import set_matmul_precision
from experiments.exp12.fork import matmul_precision_report as report
set_matmul_precision()
out = {"run_setting": report()}
with jax.default_matmul_precision("highest"):
    out["check1_highest"] = report()
print(json.dumps(out, indent=2))
PY
```
- **Pass rule (applied by the runner):** `run_setting.platform == "gpu"`,
  `run_setting.matmul_precision == "tensorfloat32"`, and a relative error
  ≥ 1e-5. TF32 is expected at about 1e-4 to 1e-3; full FP32 gives about
  1e-7 to 1e-6.
- **On failure:** if `run_setting` shows FP32-level error, TF32 is not
  active with `"tensorfloat32"` on this jaxlib. The runner stops; send me
  the file. The fallback is `"default"`.
- **Also expected:** `check1_highest` at about 1e-7 to 1e-6. If it is not,
  `summary.txt` prints a warning, because Check 1 relies on it.

### A2. Fresh-critic dynamic range, dog-run (timeout 20 min; ~4 min measured; host ~3 GB, GPU ~4 GB for D6W1536)
- **Criterion, amendment (w), from 2026-10-05:** PASS if P/b ≥ 0.9 at the
  configured pool (25,600) at every size. A failed criterion fails the step.
- **Superseded rule (information only):** 0.1·b ≤ IQM(P) ≤ 0.9·b. It failed
  at every size in Block A (P/b 0.990, 0.997 and 0.993 at 25,600), which is
  why (w) replaced it. Its fallback ladder (smaller pool, more steps) is
  withdrawn.
- **What it cannot show:** sensitivity. A fresh critic fitting the probe is
  the intended design (Lyle et al. 2023). Sensitivity comes from the positive
  control and the dev run, where a critic that has lost plasticity must
  score clearly lower; the fresh-pair null measures noise.
- **Reported:** per size and pool, P, b, P/b, the round-to-round SD and range
  of P, and the same for the per-round final loss. L = P(fresh) − P(current)
  is a difference of final losses on identical targets, so b cancels; the
  final-loss spread is the one that matters for L.
- **Probe settings:** unchanged (pool, steps, rounds, offsets).
```bash
python scripts/probe_fresh_checks.py --mode range --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --pools 1600 6400 25600 --out_dir "$OUT/A2_range_dog_run"
```

### A3. Short profile, D6W1536 on dog-run, TF32 setting only (timeout 15 min; ~4 min; host ~3 GB, GPU ~5 GB)
- **Measures:** training it/s, the time of one full probe check, the
  projected probe overhead for a whole run, and peak GPU memory, all for the
  critic that decides the 5% rule.
- **Steps timed:** 300 training steps after 100 warm-up steps; 1 probe check
  after a compile warm-up check.
- **Estimate:** 5,000 random steps (~0.75 min), compile (~1.5 min), 300
  steps (~0.1 min), 2 probe checks (~1.2 min). The script sets the TF32
  precision itself.
```bash
python scripts/profile_exp12.py --env dog-run --env_group dmc_hard --archs D6W1536 \
  --train_steps 300 --warmup_steps 100 --probe_repeats 1 --out "$OUT/A3_profile_dog_run_D6.json"
```
Key fields: `train_it_per_s_probes_off`, `probe_check_s`,
`probe_overhead_pct_of_wallclock`, `peak_device_bytes`. If the overhead
exceeds ~5%, I report it and ask; the probe is never reduced.

### A1. Smoke tests on the GPU (timeout 25 min; ~10 min; host ~4 GB, GPU < 1 GB)
These are the tests whose result could differ on a GPU:
- probe isolation and pairing;
- injection Check 1 tolerances;
- diagnostics bit-identity;
- Check 1 (including detection of a deliberately broken injection);
- post-fork evaluation isolation;
- the identity-validation procedure end to end.

The estimate assumes compile- and env-bound tiny networks at about 1.3× the
measured CPU time (7.4 min for these tests). Output: `$OUT/A1.log`.
```bash
python -m unittest -v tests.test_exp12_probe tests.test_exp12_injection \
  tests.test_exp12_diagnostics tests.test_exp12_fork.ForkUnitTest tests.test_exp12_fork.IdentityValidationTest
```
**Block A result:** 34 tests, 8 failures. Five are subtests of the injection
test; three are diagnostics tests. Details and classification:
docs/exp12_decisions.md (2026-10-05).

**Rerun only the four failing tests with deterministic ops** (exact command;
the injection test now computes Q and dQ/da under "highest", as Check 1 does):
```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true python -m unittest -v \
  tests.test_exp12_injection.InjectionInvariantsTest.test_predictions_unchanged_bit_for_bit_and_action_gradients_within_dtype_tolerance \
  tests.test_exp12_diagnostics.KnownAnswerTest.test_diagnostics_never_change_the_update \
  tests.test_exp12_diagnostics.KnownAnswerTest.test_policy_kl_is_the_closed_form_gaussian_kl_new_vs_old \
  tests.test_exp12_diagnostics.TrainingRunTest.test_training_is_identical_with_diagnostics_on_and_off \
  2>&1 | tee $OUT/A1_rerun_deterministic.log
```

### A4. GPU trace of one D6W1536 dog-run job (optional, RUN_A4=1, run last; timeout 10 min; ~4 min; host ~3 GB, GPU ~5 GB) — NEEDS CUDA VERIFICATION
- **What it gives:** a jax.profiler trace of 300 training steps after the
  5,000 random steps and 200 trained warm-up steps, under the TF32 setting
  (probes off). It shows the per-step split between the host (env step,
  sampling, transfers) and the GPU (the update scan), and any gap between
  them. Output: `$OUT/A4.log` (the traced it/s) and
  `$OUT/A4_trace_D6W1536_dog_run/`.
- **Open the trace:** `trace.json.gz` at https://ui.perfetto.dev, or the
  `.xplane.pb` with TensorBoard's profile plugin. "Can't import
  tensorflow.python.profiler.trace" is harmless.
- **What I need from it:**
  - per training step, the GPU busy time against the wall time;
  - the time between the action read-back and the next update launch (the
    host);
  - the host-to-device copies;
  - the flush of the update metrics every 2,000 steps (none falls inside
    this window unless you raise the step count).
- **Tested on CPU** with a tiny critic: the trace files are written and the
  it/s is printed. The GPU trace itself is untested.
```bash
python - <<'PY'
import os, random, tempfile, time
import jax, numpy as np
from experiments.exp12.precision import set_matmul_precision
set_matmul_precision()
from experiments.exp1 import compose_config
from experiments.exp12.trainer import Exp12Trainer
cfg = compose_config(os.path.abspath("configs"), "base_exp12", [
    "env_name=dog-run", "env=dmc_hard", "critic_num_blocks=6", "critic_hidden_dim=1536", "seed=990", "run_role=dev"])
np.random.seed(cfg.seed); random.seed(cfg.seed)
t = Exp12Trainer(cfg, tempfile.mkdtemp(prefix="trace_"))
t.start()
warm = int(cfg.buffer.min_length) + 200
t.train(warm); jax.block_until_ready(t._sac_agent.critic.params)
with jax.profiler.trace(os.environ["OUT"] + "/A4_trace_D6W1536_dog_run"):
    t0 = time.perf_counter()
    t.train(warm + 300); jax.block_until_ready(t._sac_agent.critic.params)
    dt = time.perf_counter() - t0
print({"it_per_s_traced": 300 / dt, "matmul_precision": jax.config.jax_default_matmul_precision})
t.close()
PY
```
Alternative with Nsight Systems (outside the runner; also NEEDS CUDA
VERIFICATION; the file covers the whole script, warm-up included): put the
same Python in `$OUT/a4.py` and run
`nsys profile --trace=cuda,nvtx,osrt --sample=none -o $OUT/A4_nsys python $OUT/a4.py`.

## Block B: full tests and calibration (~4.4 h TF32; was ~10 h FP32)

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

### B2. Fresh-critic dynamic range, hopper-hop (~4 min TF32; was ~8 min FP32; host ~3 GB, GPU ~4 GB)
The same rule as A2. hopper-hop has the 500k-step budget, so it is the worst
case for the overhead share.
```bash
python scripts/probe_fresh_checks.py --mode range --env hopper-hop --env_group dmc_medium \
  --archs D2W512 D4W1024 D6W1536 --pools 1600 6400 25600 --out_dir $OUT/B2_range_hopper_hop
```

### B3. Fresh-pair null, at least 100 pairs per size (~1.3 h TF32; was ~6 h FP32; host ~3 GB, GPU ~5 GB)
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

### B4. Full compute profile (~1 h TF32; was ~2.1 h FP32; host ~8 GB, GPU ~7 GB)
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

**Runtime estimates (TF32), for the six cells run sequentially:**
- **Full budget:** ~2 h (was ~6 h FP32). The D6W1536 parents dominate:
  ~45,000 training steps (dog-run, myo-key-turn) or ~95,000 (h1-run-v0), plus
  the fresh probe and checks 1–2.
- **Reduced:** ~35 min (was ~1.4 h FP32). Per D6W1536 cell: 5,000 random
  steps, 7,000 training steps (~2.5 min), the fresh probe and 2 checks
  (~1.5 min), and 2 × 1,000 post-fork steps with the restore, Check 1 (FP32)
  and evaluation 0 (~2 min), plus compile ≈ 8 min. Per D4W1024 cell ≈ 4 min.

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
- **Dev run:** ~3.7 h TF32 (was ~10.5 h FP32). 500,000 interaction steps at
  ~15 / ~42 it/s, plus 21 probes (~56 / ~13 min) and the training
  evaluations (~7 min).
  - As launched, the run continues after its fork as the control, to the full
    budget. Only `FORK_READY` is needed for the script.
  - Memory: host ~4 GB (buffer up to 0.97 GB), GPU ~5 GB, disk ~6.2 GB.
- **Script:** ~4 min TF32 (was ~15 min FP32). One probe with 5 critics (≈ 2.5
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

**Estimates (TF32; was 2/4/8 min under FP32):** D2W512 ~2 min, D4W1024 ~3 min, D6W1536 ~4 min (2,000
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
- Every grid job sets its own matmul precision (TF32 on GPU, amendment (v));
  do not export `JAX_DEFAULT_MATMUL_PRECISION` or `NVIDIA_TF32_OVERRIDE`.
  Run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Re-running generate_manifest.py is safe. Runs with DONE are skipped, and
  unfinished ones resume: from state/LATEST, from the saved fork state, or
  from scratch before their first save.
- Scaled runs (D4W1024, D6W1536) that fork continue as the control on their
  device model. A control resumed on another model refuses to run, so keep
  every D4/D6 Exp 1 job on the grid's GPU model (amendment (u)).
