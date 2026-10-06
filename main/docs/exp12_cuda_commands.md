# Exp 1/2: commands that NEED CUDA VERIFICATION, in blocks

Everything here has passed on CPU (Linux, jaxlib 0.4.34). These commands check
the same properties on the real Linux + CUDA stack. Run them from `main/` with
the project env activated, and send me the logs and JSON files.

Each step's pass/fail rule was pre-specified in `docs/exp12_decisions.md` on
2026-10-04, before any CUDA result existed. None of these commands launches or
schedules a grid run.

- **Block A** (one A100, 4 CPUs, ≤ 30 min): smoke tests, fresh-critic range
  check, short profile, optional GPU trace (A4); run by scripts/exp12_blockA.sh.
- **Block B** (one A100x4 node, unattended, 8 h limit, ~5.1 h critical path): the dev run, positive
  control and injected arm, preflight, packing test, GPU test suites, range
  check, identity forks and the fresh-pair null, one lane per GPU.
  Blocks C–E describe the same steps for manual reruns. F: the grid, which I
  never run.

## Matmul precision: TF32 for every matmul that can use it (amendment (v), revised 2026-10-04)

- **Setting:** every Exp 1/2 GPU entry point (exp1, exp2_arm,
  probe_fresh_checks, profile_exp12, positive_control) calls
  `experiments/exp12/precision.set_matmul_precision()` first. On a GPU it sets
  `jax_default_matmul_precision = "tensorfloat32"`; on CPU it does nothing.
- **Why that value:** jax 0.4.34 documents `"tensorfloat32"` (=
  `Precision.HIGH`) as "On GPU: uses tensorfloat32 where available,
  otherwise float32". Its `DEFAULT` is documented the same way on GPU, but the
  explicit name keeps the choice independent of JAX's default.
- **What it covers:** training updates, action selection, probes and post-fork
  evaluation (the actor diagnostics' forward passes use "highest", amendment (x)). All dtypes stay float32: there is no bf16 or
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
- **Critic parameters (dog-run, per Q network):** D2W512 4.34M, D4W1024 33.85M, D4W1536 75.95M
  (D6W1536, dropped by amendment (y): 113.72M).
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
  D2W512 (launch-bound), 3.8 ms for D4W1024 and 12.7 ms for D6W1536 (the
  pre-calibration forecast). It adds to the host overhead.
- **Compile:** ≈ 1–2 min per process for D6W1536 (update scan plus probe
  fit); less for smaller critics, D4W1536 included.
- **Calibration to Block A:**
  - D6W1536 ran 28.9 ms per step against the 23.7 ms forecast. With ~11 ms
    of host overhead (assumed equal on Delta), the device took ~17.9 ms
    (≈ 39 TFLOP/s effective, not 55).
  - A probe check took 50 s against ~36 s, a factor of 1.4.
  - The unmeasured sizes are scaled by these two factors.
- **D4W1536 (amendment (y)), all figures unmeasured estimates.** It has the
  same layer shapes as D6W1536 (width 1536) with 4 residual blocks instead
  of 6, so its device work is taken as proportional to the parameter count,
  75.95M / 113.72M = 0.668 on dog-run:
  - device time per step 17.9 ms × 0.668 = 12.0 ms; plus 11 ms of host
    overhead = 23.0 ms, i.e. **~43.6 it/s**;
  - one probe check 50.0 s × 0.668 = **~33.4 s**;
  - probe overhead 6.77% × (33.4 / 50.0) × (43.6 / 34.6) = **~5.7%** of
    a run;
  - peak GPU memory 4.01 GiB × 0.668 = **~2.7 GiB** (parameters, Adam
    state, target and probe copies all scale with the parameter count).
  Fixed costs (launches, fixed host work) do not shrink with the parameter
  count, so these may be slightly optimistic; Block B measures them.
- **Uncertainty:** D6W1536's training speed and probe time were measured (it
  is now dropped and serves only as the anchor); every other size,
  D4W1536 included, is ±50% until Block B's packing test and per-suite speed
  step.
- **Which figures apply:** the TF32 ones (amendment (v), revised). The FP32
  figures are kept only for comparison.

| Critic | Train it/s, dog-run, TF32 | One probe check, 2 critics, TF32 | Peak GPU memory per process* | Host memory per process |
|---|---|---|---|---|
| D2W512 | ~76 (estimate; host-bound) | ~3 s (estimate) | < 1 GB (estimate) | ~3 GB + buffer |
| D4W1024 | ~61 (estimate) | ~16 s (estimate) | ~2 GB (estimate) | ~3 GB + buffer |
| D4W1536 | ~43.6 (estimate, unmeasured) | ~33.4 s (estimate, unmeasured) | ~2.7 GiB (estimate, unmeasured) | ~3 GB + buffer |
| D6W1536 (dropped; anchor) | **34.6 (measured, A3)** | **50.0 s (measured, A3)** | **4.01 GiB (measured, A3)** | ~3 GB + buffer |

FP32 figures from the earlier plan (for comparison only): ~70, ~35 and ~15 it/s;
probe checks ~6, ~48 and ~160 s.

\* With `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Without it, XLA reserves 75%
of the GPU per process. Buffer host memory = filled transitions × bytes per
transition: dog-run 1,948 B, i.e. 0.97 GB at 500k transitions; hopper 148 B;
MyoSuite 832–1,088 B; HumanoidBench 496–544 B.

The probe took 6.77% of a D6W1536 dog-run's wall-clock (A3, measured). It
is above the ~5% rule; the overhead was accepted (decision R2) and the probe
is unchanged. For D4W1536 the estimate is ~5.7% (unmeasured).

**Whole-run forecast, dog-run (1M env steps), TF32, refreshed from Block A:**

| Critic | Training | Probes (21 checks) | Training evaluations | Saves | Total per run | Post-fork arm (25% of B) |
|---|---|---|---|---|---|---|
| D2W512 | 1.8 h (est.) | 1 min | 7 min | 1 min | ~2.0 h | — (does not fork) |
| D4W1024 | 2.3 h (est.) | 6 min | 7 min | 2 min | ~2.5 h | ~0.9 h |
| D4W1536 | 3.2 h (est., unmeasured) | 12 min (est., unmeasured) | 7 min | 2 min | ~3.6 h | ~1.2 h |
| D6W1536 (dropped; anchor) | 4.0 h (measured speed) | 18 min (measured) | 7 min | 3 min | ~4.5 h | ~1.4 h |

Training evaluations: 100 episodes × 500 steps × 8.7 ms (CPU-measured). A
post-fork arm is 125,000 steps plus 26 × 10 evaluation episodes (~19 min on
dog-run) plus 5 probe checks. A run that forks late (check 19) continues as
the control to fork + 25% of B, i.e. 120% of B: the D4W1536 control then
trains ~3.8 h plus its probes and evaluations, ~4.8 h in all (estimate; the
dropped D6W1536 was ~5.8 h).

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

## Setup (once; not timed in Block A)
```bash
python -c "import jax; print(jax.devices()); import jaxlib; print(jaxlib.__version__)"
pip install -r requirements.txt                                     # adds rliable + pinned deps; no existing pin changes
export XLA_PYTHON_CLIENT_PREALLOCATE=false                          # several processes share the GPU in the tests
OUT=/abs/path/to/exp12_cuda_checks && mkdir -p $OUT
```

**HumanoidBench environment (item 6; optional, for `HB_ENV`).** On a login
node (the clone and `git clone` need the network). This is a separate clone;
the main environment's pins are not touched:
```bash
module reset; source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
cd /work/hdd/biqc/skaveti1/exp12/main && git pull
conda activate scaling-drl-py31213                                   # 1. record the main environment
conda list --explicit --md5 > ~/py31213.conda.before.txt && pip freeze --all > ~/py31213.pip.before.txt
conda deactivate
conda create --yes --prefix /work/hdd/biqc/skaveti1/envs/exp12-hb --clone scaling-drl-py31213   # 2. clone it
conda activate /work/hdd/biqc/skaveti1/envs/exp12-hb                 # 3. install HumanoidBench into the clone only
python -c "import sys; print(sys.prefix)"; which pip                 #    both must point into .../envs/exp12-hb
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl bash scripts/install_humanoid_bench.sh /work/hdd/biqc/skaveti1/humanoid-bench
conda deactivate
conda activate scaling-drl-py31213                                   # 4. the main environment must be unchanged
conda list --explicit --md5 | diff ~/py31213.conda.before.txt - && pip freeze --all | diff ~/py31213.pip.before.txt - \
  && echo "MAIN ENVIRONMENT UNCHANGED"
python -c "import importlib.util; print('humanoid_bench importable in the main env:', importlib.util.find_spec('humanoid_bench') is not None)"
conda deactivate
```
- `install_humanoid_bench.sh` ends with a smoke test that builds the two
  tasks with EGL. Login nodes may have no EGL; if only that step fails, the
  `pip install` before it has finished, and Block B's HumanoidBench check
  runs on the GPU node.
- Step 4 must print `MAIN ENVIRONMENT UNCHANGED`, and `False` for
  `humanoid_bench` in the main environment.
- Submit Block B with `HB_ENV=/work/hdd/biqc/skaveti1/envs/exp12-hb`.

**Deterministic GPU ops (item 8d).** jaxlib 0.4.34 accepts both flag names,
tested on CPU with a tiny jit; an unknown name aborts with "Unknown flags in
XLA_FLAGS". Its built-in help text gives:
- `--xla_gpu_deterministic_ops`: "Guarantees run-to-run determinism on
  GPU". This is the one the sheet uses.
- `--xla_gpu_exclude_nondeterministic_ops`: "Excludes non-deterministic ops
  from compiled executables".

That the name is recognised on CPU does not show the flag works on the GPU:
**NEEDS CUDA VERIFICATION** (Block B's deterministic-ops pass).

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

### A2. Fresh-critic dynamic range, dog-run (timeout 20 min; ~4 min measured with D6W1536; host ~3 GB, GPU ≤ ~4 GB)
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
  --archs D2W512 D4W1024 D4W1536 --pools 1600 6400 25600 --out_dir "$OUT/A2_range_dog_run"
```
Block A (job 22667743) ran A2 to A4 with D6W1536, before amendment (y); the
commands now use D4W1536.

### A3. Short profile, D4W1536 on dog-run, TF32 setting only (timeout 15 min; ~4 min; host ~3 GB, GPU ~5 GB)
- **Measures:** training it/s, the time of one full probe check, the
  projected probe overhead for a whole run, and peak GPU memory, all for the
  critic that decides the 5% rule.
- **Steps timed:** 300 training steps after 100 warm-up steps; 1 probe check
  after a compile warm-up check.
- **Estimate:** 5,000 random steps (~0.75 min), compile (~1.5 min), 300
  steps (~0.1 min), 2 probe checks (~1.2 min). The script sets the TF32
  precision itself.
```bash
python scripts/profile_exp12.py --env dog-run --env_group dmc_hard --archs D4W1536 \
  --train_steps 300 --warmup_steps 100 --probe_repeats 1 --out "$OUT/A3_profile_dog_run_D4W1536.json"
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

### A4. GPU trace of one D4W1536 dog-run job (optional, RUN_A4=1, run last; timeout 10 min; ~4 min; host ~3 GB, GPU ~5 GB) — NEEDS CUDA VERIFICATION
- **What it gives:** a jax.profiler trace of 300 training steps after the
  5,000 random steps and 200 trained warm-up steps, under the TF32 setting
  (probes off). It shows the per-step split between the host (env step,
  sampling, transfers) and the GPU (the update scan), and any gap between
  them. Output: `$OUT/A4.log` (the traced it/s) and
  `$OUT/A4_trace_D4W1536_dog_run/`.
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
```
Alternative with Nsight Systems (outside the runner; also NEEDS CUDA
VERIFICATION; the file covers the whole script, warm-up included): put the
same Python in `$OUT/a4.py` and run
`nsys profile --trace=cuda,nvtx,osrt --sample=none -o $OUT/A4_nsys python $OUT/a4.py`.

## Block B: one A100x4 node, unattended (replaces the earlier single-GPU B1–B4 sequence)

Run with `scripts/sbatch_exp12_blockB.sh`, which calls the driver
`scripts/exp12_blockB.sh`. It needs one gpuA100x4 node (4 × A100 of one
model, 64 CPUs, 16 per GPU lane). Block B is a development job, not
confirmatory.
```bash
cd /work/hdd/biqc/skaveti1/exp12/main && git pull && mkdir -p logs
bash scripts/exp12_blockB.sh --dry-run                  # layout, commands, timeouts, files; no jax
EXPECTED_COMMIT=<hash I give you> sbatch scripts/sbatch_exp12_blockB.sh
# with HumanoidBench: HB_ENV=/work/hdd/biqc/skaveti1/envs/exp12-hb EXPECTED_COMMIT=<hash> sbatch scripts/sbatch_exp12_blockB.sh
bash scripts/collect_report.sh logs/blockB_<jobid>      # (also run at the end of the job) -> paste_me.txt
```

**Lanes** (each with its own GPU, by UUID, and its own 16 CPUs; nothing in
one lane stops another):

| Lane | Steps, in order (timeout) |
|---|---|
| GPU 0 | dev run: D4W1536 dog-run, seed 102, run_role=dev (6 h) → positive control (30 min; m candidates last / half / all = 1 / 2 / 4 blocks) → preflight D2W512, D4W1024, D4W1536 (20 min each) |
| GPU 1 | watcher: once `dev_run/fork/FORK_READY` exists, it runs the dev run's injected arm with `ARM_M=half` (the default; 3 h). The positive control chooses m separately and you freeze it. `ARM_M=pc` instead waits for the positive control's m |
| GPU 2 | packing test (15 min per configuration) → A1 follow-up measurements, default and deterministic ops (20 min each) → GPU test suite, default ops (2 h) → break checks (30 min) → GPU test suite, deterministic ops (2 h) → HumanoidBench tests under `HB_ENV` (30 min) → hopper-hop range check (30 min) → identity forks, reduced budget, D4W1024 and D4W1536 per suite that works on this install, first with a cold compilation cache per process and then with one shared warm cache (1 h per cell) |
| GPU 3 | fresh-pair null, 100 pairs per size, dog-run (4 h) → per-suite training speed: myo-key-turn and h1-run-v0 × D2W512, D4W1024, D4W1536, one job each, 4 cores, probes off, 600 timed steps (15 min each). Dog-run comes from the packing test's 1-job runs |

**Rules**
- Every step has its own log (`logs/blockB_<id>/<lane>/<step>.log`) and a
  status line (`status/<lane>.tsv`).
- A step that fails or times out does not stop the rest.
- Exit-3 stops skip only what depends on them:
  - **the dev run never triggers:** the positive control is skipped, and
    the watcher exits cleanly; both are recorded `SKIPPED_NO_TRIGGER`;
  - **the positive control stops** (`STOP_EXIT3`): with `ARM_M=pc` the arm
    is skipped. Preflight runs regardless.
- No network, installs or interactive input inside the job; wandb is
  disabled.
- **HumanoidBench** runs only through `HB_ENV`, a conda environment
  cloned from `scaling-drl-py31213` with HumanoidBench added (see Setup). Its
  steps run with that environment's Python and `MUJOCO_GL=egl`. Without
  `HB_ENV`, they are recorded `SKIPPED_UNAVAILABLE`, which does not fail
  OVERALL. MyoSuite uses an import test in the main environment.
- The driver keeps its own deadline (7.5 h of the 8 h limit) so that the
  report is always written. On SIGTERM it writes the report too.

**Persistent compilation cache (item 8a).** Every Exp 1/2 job enables JAX's
persistent compilation cache:
- one directory per GPU model, `main/jax_cache/<device kind>` (gitignored),
  with small compiles cached too;
- `run_metadata.json` records the path;
- each process prints its cache hits and misses at exit;
- `EXP12_JAX_CACHE_DIR` overrides the path (`off` disables it).

Why: separately compiled processes can pick different GEMM kernels.

**Identity forks, cold and warm (item 8b).**
- **Cold:** the parent and the identity arm each start from their own empty
  cache, so each compiles for itself.
- **Warm:** both use one shared cache, so the arm reuses the parent's
  executables.

Both comparisons are reported per cell, with each process's wall time and
cache hits; the arm's cold-vs-warm wall time is the cache's effect on
start-up (item 8c). This tests whether identical executables matter; it does
not assume they do.

**Packing test** (approved design):
- **Configurations:** D4W1536 at 1, 2 and 3 concurrent jobs; D4W1024 at
  1, 2 and 4; D2W512 at 1, 3 and 4. All run on GPU 2's A100.
- **Each job:** dog-run with probes off; 5,000 random steps and 100 trained
  warm-up steps, then a start barrier, then 600 timed steps.
- **CPUs:** `taskset` gives each job exactly 4 cores, with thread caps at 1.
- **Report:** per size and job count:
  - each job's it/s;
  - slowdown against 1 job;
  - total throughput;
  - per-job peak GPU memory and the GPU's total (nvidia-smi maximum);
  - the overlap flag;
  - cores per job and the host CPU model.
- **Scope:** dog-run with probes off only, so probe overhead and post-fork
  evaluation are not included. No threshold is chosen.

**GPU numerics in the test suite** (items 2 and 3).
- **Diagnostics isolation tests** (diagnostics on vs off; training
  identical): bit-exact on CPU, as before. On the GPU they follow PyTorch's
  TF32 on/off pattern:
  - in the deterministic-ops pass, both sides run under a local "highest",
    at a tight tolerance;
  - in the default pass, the side under test runs at the run setting (TF32)
    against a reference run under "highest", at a loose tolerance.
- **Tolerances:** each is set from a measured GPU deviation, at 10× the
  largest measured value. None is measured yet, so on the GPU these tests
  record the deviation and skip. The report lists every deviation next to
  its tolerance.
- **KL test:** uses a well-conditioned actor (σ 0.13–0.88, learning rate
  1e-3, KL ≈ 0.02) and a float64 numpy reference on FP32 forward passes, at
  rtol 1e-4. The near-deterministic actor (σ ≈ 5e-5) is reported, not gated.

**A1 follow-up measurements** (evidence for the A1 classification). Under
TF32 and under "highest", with default and with deterministic ops:
- injection Q bit-identity and dQ/da deviation;
- diagnostics on vs off, and off vs off;
- policy KL against the closed form.

**report.txt** opens with one line per gate (PASS, FAIL, NOT RUN, SKIPPED or
UNAVAILABLE) and an OVERALL line, which sets the driver's exit code:
- GPU test suite (default ops; deterministic ops);
- break checks;
- HumanoidBench tests;
- hopper-hop range, amendment (w);
- identity fork per cell and cache mode;
- fresh-pair null per size (fire rate ≤ 5%);
- dev run triggered;
- positive control m chosen;
- injected arm Check 1 and Check 2;
- preflight per size.

Then come the numbers:
- the packing table and per-suite speeds;
- the GPU numerics with their tolerances;
- the A1 follow-up;
- the dev run's checks (L IQM and interval, f*_run, the fork);
- the fork save and restore times, and the wall time per post-fork
  evaluation (dev run and arm);
- the positive control (L_trigger, recovery per m, noise, shared-offset
  check);
- the arm (Check 1, Check 2, last post-fork evaluation);
- the range table with final-loss spreads;
- the null per size;
- the identity cells (cold and warm) and the node.

**Time limit and resources.** Anchored on Block A's measured D6W1536 point
(34.6 it/s, 50.0 s per probe check); every D4W1536 figure is an unmeasured
estimate (~43.6 it/s, ~33.4 s per check; derivation under "How the remaining
estimates are made"):

| Item | Estimate |
|---|---|
| Dev run, fork late (check 19), control to 120% of B (the later of 100% of B and fork + 25%) | 600,000 steps / 43.6 = 3.8 h, + 25 probe checks × 33.4 s = 0.23 h, + training evaluations (200 episodes) 0.25 h, + 26 post-fork evaluations 0.33 h, + saves and compile 0.12 h ≈ **4.8 h** (≈ 3.7 h if it never forks) |
| Positive control | ≈ 0.15 h |
| Preflight, 3 sizes | ≈ 0.2 h |
| GPU 0 in all (critical path) | ≈ **5.1 h** |
| Injected arm (`ARM_M=half`, starts at the fork, ≤ ~3.4 h) | 125,000 / 43.6 = 0.8 h, + 26 evaluations 0.33 h, + 5 probe checks and restore 0.1 h ≈ 1.2 h, so it ends by **~4.6 h** |
| GPU 2 | packing ~0.7 h, follow-up 0.1 h, tests 2 × ~0.75 h, HumanoidBench tests 0.1 h, break checks 0.1 h, range 0.1 h, identity 2 × ~0.65 h ≈ **3.9 h** |
| GPU 3 | null 100 pairs × (3 + 16 + 33 s) + fills ≈ 1.6 h, + per-suite speed 6 × ~4 min ≈ **2.0 h** |

The request is `--time=08:00:00` (was 10 h for D6W1536's 6.4 h critical
path, ~55% over it). 8 h leaves ~57% over the 5.1 h critical path for what
is not measured: D4W1536 itself, Delta's host speed, GPU evaluations and
compile times. Step timeouts: dev run 6 h (~26% over its estimate, as
before), positive control 30 min, preflight 20 min per size; the driver's own
deadline is 7.5 h, so the report is always written. The rest of the request
is unchanged:
- `--gpus=4` and `--cpus-per-task=64` (16 per lane, 4 per packing job);
- `--mem=128G` (estimated peak ~30 GB);
- disk ~60 GB under `logs/blockB_<id>`: dev run ~5, arm ~2, identity forks
  2 × ~18, preflight ~8, and the compilation caches of the identity cells.

**Check the node before submitting** (no job is started):
```bash
sinfo -p gpuA100x4 -N -o "%N %c %m %G" | sort -u | head            # CPUs, memory (MB), GPUs per node
scontrol show node $(sinfo -p gpuA100x4 -h -N -o %N | head -1) | grep -E "CPUTot|RealMemory|Gres|CfgTRES|MemSpecLimit"
```
The request fits if `CPUTot` ≥ 64, `RealMemory` minus any `MemSpecLimit` ≥
131072 (MB), and `Gres` shows 4 A100s. My understanding is that Delta's
A100x4 nodes have 64 cores, ~256 GB and 4 × A100-40GB, but that is from
memory, not checked.

## Block C: identity-fork gate (D2), per forking architecture × suite, before any Exp 1 grid launch

**Now run by Block B's GPU 2 lane, with the reduced budget.** The commands below are for a manual rerun.

**Setup of the test**
- Run this on the GPU model the grid will use (amendment (p)). Only D4W1024
  and D4W1536 fork. One environment per suite: dog-run, myo-key-turn and
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
restore is timed on the dev run in Block B (its fork state save and restore, in report.txt).

**Runtime estimates (TF32), for the six cells run sequentially:**
- **Full budget:** ~1.5 h (estimate; ~2 h with the dropped D6W1536). The D4W1536 parents dominate:
  ~45,000 training steps (dog-run, myo-key-turn) or ~95,000 (h1-run-v0), plus
  the fresh probe and checks 1–2.
- **Reduced:** ~30 min (estimate). Per D4W1536 cell: 5,000 random
  steps, 7,000 training steps (~2.7 min), the fresh probe and 2 checks
  (~1 min), and 2 × 1,000 post-fork steps with the restore, Check 1 (FP32)
  and evaluation 0 (~2 min), plus compile ≈ 6–7 min. Per D4W1024 cell ≈ 4 min.

**Memory:** host ~4 GB, GPU ~3.5 GB (D4W1536, estimate). Disk ~3 GB per D4W1536 cell
(the fork state and two snapshots, each holding a ~1.2 GB agent), ~18 GB for
all six.
```bash
BUDGET=()                                          # full budget
# BUDGET=(--overrides num_env_steps=240000)        # reduced, equivalent option (see above)
for ARCH in "4 1024" "4 1536"; do set -- $ARCH; D=$1; W=$2
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

**Now run by Block B's GPU 0 lane** (dev run, then the positive control), with the dev run's injected arm on GPU 1. The commands below are for a manual rerun.

**The run.** A development run: D4W1536 on dog-run, seed 102 (outside 1–5),
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
- **Dev run:** ~3.6 h TF32 (estimate). 500,000 interaction steps at
  ~43.6 it/s (estimate), plus 21 probes (~12 min) and the training
  evaluations (~7 min). m candidates for depth 4: last = 1, half = 2,
  all = 4 blocks.
  - As launched, the run continues after its fork as the control, to the full
    budget. Only `FORK_READY` is needed for the script.
  - Memory: host ~4 GB (buffer up to 0.97 GB), GPU ~3.5 GB, disk ~4.2 GB (estimates).
- **Script:** ~4 min TF32 (was ~15 min FP32). One probe with 5 critics (≈ 2.5
  checks) plus 3 shared-offset probes with 2 critics.
  - Memory: host ~4 GB, GPU ~7 GB (estimate; three injected critics of up to 3× the
    critic's parameters are held together).
```bash
PC=$OUT/D_positive_control
python run.py --experiment exp1 --config_name base_exp12 --overrides env_name=dog-run --overrides env=dmc_hard \
  --overrides seed=102 --overrides critic_num_blocks=4 --overrides critic_hidden_dim=1536 \
  --overrides run_role=dev --overrides results_root=$PC/results --checkpoint_dir $PC/dev_run 2>&1 | tee $PC.dev_run.log
python scripts/positive_control.py --run_dir $PC/dev_run --out_dir $PC/result 2>&1 | tee $PC.result.log
```
The chosen m is not applied automatically. You freeze it by setting
`injection.m` for the Exp 2 injected arms.

## Block E: preflight on the grid's GPU model (after Blocks A–D, before the grid)

**Now run by Block B's GPU 0 lane after the positive control.** The commands below are for a manual rerun.

One smoke run per critic size on the node type the grid will use (tiny probe
settings, 4,000 env steps). The forking sizes also fork, run the injected arm
and require Check 1 to pass on that device.

**Estimates (TF32):** D2W512 ~2 min, D4W1024 ~3 min, D4W1536 ~3.5 min (2,000
training steps, the injected arm's 500 steps, 2 × 26 one-episode
evaluations). Host ~3 GB; GPU ≤ ~5 GB.
```bash
for ARCH in "2 512" "4 1024" "4 1536"; do set -- $ARCH
  FORK=$([ "$1" = 2 ] && echo "" || echo "--with-fork")
  python scripts/preflight_checkpoint_check.py --experiment exp1 $FORK \
    --override critic_num_blocks=$1 --override critic_hidden_dim=$2 \
    --override env_name=dog-run --override env=dmc_hard --override seed=999 \
    --checkpoint_dir $OUT/E_preflight_D$1W$2 2>&1 | tee $OUT/E_preflight_D$1W$2.log
done
```

## Block F: the grid (NOT run by me; for when you decide to launch)
```bash
GRID=/abs/path/exp12_grid; RESULTS=/abs/path/exp12_results
python generate_manifest.py --grid exp12 --ckpt-root $GRID --results-root $RESULTS   # 195 Exp 1 jobs
python scripts/claim_launcher.py --concurrency <from the packing test> --num-gpus <n> --phase-files exp12_exp1_jobs.txt
# once the positive control froze m, and as forks complete; on the GPU model each file is named after:
python generate_manifest.py --grid exp12 --ckpt-root $GRID --results-root $RESULTS --injection-m <m>
python scripts/check_manifest_overlap.py exp12_exp1_jobs.txt exp2_arms_<device>.txt   # must print OK
python scripts/claim_launcher.py --concurrency <n> --num-gpus <n> --phase-files exp2_arms_<device>.txt
```
- Every grid job sets its own matmul precision (TF32 on GPU, amendment (v));
  do not export `JAX_DEFAULT_MATMUL_PRECISION` or `NVIDIA_TF32_OVERRIDE`.
- GPU memory, for jobs that share a GPU: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  and no `XLA_PYTHON_CLIENT_MEM_FRACTION`. The old Angle 1 launch scripts
  carried `.10`, which is about 4.0 GiB of a 40 GB A100, while D6W1536 peaked
  at 4.01 GiB (A3; D4W1536 ~2.7 GiB, estimate). They now unset it (2026-10-05). Set the jobs per GPU
  from the packing test's measured per-job and total memory.
- Every job uses the persistent compilation cache under
  `main/jax_cache/<GPU model>` (shared by all jobs on that model).
- Re-running generate_manifest.py is safe. Runs with DONE are skipped, and
  unfinished ones resume: from state/LATEST, from the saved fork state, or
  from scratch before their first save.
- Scaled runs (D4W1024, D4W1536) that fork continue as the control on their
  device model. A control resumed on another model refuses to run, so keep
  every D4W1024 and D4W1536 Exp 1 job on the grid's GPU model (amendment (u)).
