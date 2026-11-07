# Method C v2: ten-target validation design (draft, 2026-10-15)

**Status:** historical design draft. The ten-target candidate is now implemented
in `configs/base_exp12_ten.yaml`; its official collection remains separately
gated. See [the current launch decision](exp96_launch_decision_2026-10-18.md)
for proposed dispositions. The seven-run pilot and its five-target Method C
source `f8619dd5f494399d9febc79fc9092010234c714d` remain immutable.
Lyle et al., *Understanding Plasticity in Neural Networks*, ICML 2023,
Appendix A.2 uses ten random target functions. Our sine target and score are
described in `method_c_construct_validation_controls.md`. The bootstrap,
longitudinal trigger, SAC replay sampling, and the following ten-target IQM
are project-specific decisions, not statistical guarantees supplied by that paper.

## CPU findings and required inputs

The archived pilot manifest identifies the exact pilot source. The read-only
`scripts/method_c_pilot_offline.py` checked every round at checks 0, 2, and 20
for seed-1 Myo Key Turn D2W512, D4W1024, and D4W1536: five finite target
fits per check, 2,000 recorded minibatch losses per fit, and equal fresh/trained
target variance within a round. At check 2, the mean final MSE/target-variance
ratios (fresh, trained) were respectively (0.075, 1.167), (0.097, 1.043),
and (0.149, 1.068); at check 20 they were (0.142, 1.121), (0.199, 1.061),
and (0.293, 1.474). These descriptive values do **not** isolate checkpoint age:
the pool, teacher, and normalizer also change. The fresh and trained mean
prediction offsets differ for all paired rounds, so their complete numerical
targets differ by a constant even though the randomized sine component is
shared. The archived minibatch curves are not initial *full-pool* errors and
do not contain gradients or AdamW state. No extension beyond step 2,000 can
be reconstructed from them.

The uploaded archive has no complete checkpoint, replay buffer, observation
normalizer, per-run `probes/check_02.npz`/`check_20.npz`, or RNG state. The
read-only checkpoint controls therefore **cannot execute in this CPU workspace**.
Required Delta inputs for run5 (D4W1024) and run7 (D4W1536) are:

- `fork/state/LATEST` at 50,000 interactions and `state/LATEST` at 500,000;
  their pointed-to `agent_ckpt`, `obs_rms.pkl`, `buffer.npz`, `buffer_meta.pkl`,
  `meta.pkl`; and `fork/FORK_READY` for the early state;
- `fresh_critic`, `run_metadata.json`, and `probes/check_02.npz` and
  `probes/check_20.npz` to require bitwise replay of the source five fits.

These live under `/work/hdd/biqc/skaveti1/exp12_prod_pilot7_20261014T213210Z/checkpoints/run{5,7}_D4W{1024,1536}_myo-key-turn_seed1`.
For run4 D2W512, the 500k state suffices for late-pool fresh/fresh controls;
an exact 50k checkpoint is necessary for an early-pool control. The original
five-function source is independently verified by focused CPU tests of target
keys, AdamW continuity, paired resampling, and state provenance. Those tests
validate the harness logic, not the unavailable pilot checkpoints.

The separate controls harness offers (1) fresh/fresh on early and late pools,
(2) the 50k/500k critic-by-pool crossover for D4 sizes, (3) identical full
targets anchored to current, fresh, or their mean, (4) initial-error/gradient/
fit-curve instrumentation, and (5) continuous 4,000-step exploratory fits.
All use saved original states and separate output directories. A source round
must replay bitwise before its crossover or five-versus-ten comparison is
accepted. Shared-target and longer-fit results never replace the original
pilot result. Offset-anchored comparisons can test whether differing complete
targets affect the observed gap. Conditioning traces are separately compiled,
so small numeric differences from the official probe must be disclosed.

## Proposed ten-function statistic

For function `r`, calculate `L_r = score(fresh,r) - score(trained,r)` using
matched replay indices, teacher draw, minibatch order, and the existing
network-specific offset convention. Positive means the trained critic fits
worse. Keep `g(x)=a+sin(10^5 f(x; omega_0))`, continuous 2,000-step AdamW,
full-pool final MSE, and `b=Var(target)` unchanged. Draw ten independently
initialized `omega_0` values using distinct source keys. The first five keys
of comparison set zero are exactly the original keys; the added five use
round indices 5–9. Further independent sets use disjoint round blocks.

**Draft estimator (owner review required):** sort ten paired losses as
`x1 <= ... <= x10`. IQM is the integral of the inverse empirical CDF over
quantile interval `[0.25,0.75]`, divided by 0.5:

`IQM10 = 0.1*x3 + 0.2*(x4+x5+x6+x7) + 0.1*x8`.

Each observation occupies an empirical-quantile bin of width 0.1. The 25th
and 75th percentiles cut bins 3 and 8 in half; there is no interpolation of
the observed values. This is a precise 25%-75% IQM, but it is **not** simply
calling SciPy `trim_mean(..., .25)` on ten values: that floors the trim count
to two per tail and averages the middle six. The owner should freeze which
ten-point convention is intended before an official study. The original
five-point middle-three result remains its own version and is never
reinterpreted using ten-point weights.

Resample ten **paired function-level losses** with replacement, 10,000 times;
calculate IQM10 in each sample. Use the same deterministic bootstrap seed
recipe `(run_seed, BOOT_STREAM, check_index)`. Take the 2.5th and 97.5th
percentiles using NumPy `method="linear"` (Hyndman–Fan type 7). A check fires
only when the lower bound is strictly above zero; the existing two-consecutive
and eligibility logic would apply after separate approval. This is an
unactivated reference in `scripts/method_c_ten_reference.py`.

Nominal 95% percentile-bootstrap coverage is **unestablished**. In a small
CPU simulation (500 independent synthetic checks; 10,000 resamples/check),
zero-mean IID Gaussian function losses fired 21/500 with ten observations,
while equicorrelated losses with correlation 0.4 fired 109/500. These are
illustrations of sensitivity to dependence, not estimates of A100 false
trigger rates. Calibrate on independent fresh/fresh pairs and preserve the
two-consecutive temporal rule when estimating longitudinal firing. Reversed
arms of one pair are dependent and count as one independent pair.

## Five-versus-ten comparison and minimum GPU work

Use the saved early/late Myo states, no SAC training. For each D4 size and
each state, use at least two independent ten-target sets. Within each set,
compare the first five under the original five-point IQM with all ten under
the draft IQM, and report paired estimates, between-function variance, CI
width, decisions, fit curves, and per-fit wall/peak memory. Set zero must
bitwise reproduce the original five source fits. On each saved early/late
pool, use four **independently initialized fresh pairs** per D4 size and four
on the D2 late pool as an initial exploratory negative control. Both directions
may be reported but are not independent. Larger independent samples would be
needed to establish a 5% per-check or longitudinal false-trigger bound.

Compare early/late trained states at source checks 2 and 20. Existing source
checks do not form a complete matched ten-target training trajectory, so
trigger *timing/stability* cannot be inferred from just those two points.
Replaying all saved scheduled states would be required; the archive contains
neither all states nor ten-target histories. Trained/fresh differences without
independent ground-truth plasticity-loss labels measure detector behavior,
not true-positive sensitivity. If the 4,000-step exploratory trace reduces
the trained/fresh gap, that indicates fitting-budget sensitivity, not a new
official Method C value.

The prepared (not authorized or submitted) wrapper is
`scripts/sbatch_method_c_ten_validation.sh`, one A100, 16 CPUs, 64 GiB,
8-hour Slurm maximum and a 7h40m internal cap. Expect roughly 3–6 hours,
bounded by 8 GPU-hours; this is a planning estimate because multi-arm
checkpoint restoration and extra fits are not measured on this exact job.
The wrapper saves source-specific condition outputs as they finish, requires
original five-function bitwise replay, and stops on missing/corrupt inputs.
Any smaller staged allocation should use measured fit times after the first
few conditions; do not infer results from an unsubmitted script.

After the exact harness commit is reviewed and published, the **proposed**
Delta invocation below remains subject to explicit GPU authorization and a
successful read-only inventory. It does not use or write the pilot checkout:

```bash
set -euo pipefail
HARNESS_SHA=REPLACE_WITH_REVIEWED_COMMIT
CHECKOUT=/work/hdd/biqc/skaveti1/exp12_method_c_ten/$HARNESS_SHA
PILOT_ROOT=/work/hdd/biqc/skaveti1/exp12_prod_pilot7_20261014T213210Z
PYTHON_BIN=/u/skaveti1/.conda/envs/scaling-drl-py31213/bin/python
EVIDENCE_ROOT=/work/hdd/biqc/skaveti1/exp12_method_c_ten/evidence/${HARNESS_SHA}_$(date -u +%Y%m%dT%H%M%SZ)
LOG_ROOT=/work/hdd/biqc/skaveti1/exp12_method_c_ten/logs/$HARNESS_SHA
test ! -e "$CHECKOUT" && test ! -e "$EVIDENCE_ROOT"
mkdir -p "$(dirname "$CHECKOUT")" "$LOG_ROOT"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$CHECKOUT"
git -C "$CHECKOUT" checkout --detach "$HARNESS_SHA"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$HARNESS_SHA"
test -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=all)"
export HARNESS_SHA PILOT_ROOT PYTHON_BIN EVIDENCE_ROOT
sbatch --parsable --output="$LOG_ROOT/%j.out" --error="$LOG_ROOT/%j.err" \
  "$CHECKOUT/main/scripts/sbatch_method_c_ten_validation.sh"
```

Preflight must additionally confirm that the checkpoint inventory lists both
D4 time points and the source check NPZ files. The wrapper performs this and
the exact-source/CUDA checks again inside the allocation. It requests one
A100 on `gpuA100x4`, account `biqc-delta-gpu`, 16 CPUs, 64 GiB, 8 hours.

## Versioned 96-parent implementation plan

After scientific review, add a new versioned config and output namespace;
keep the five-function pilot config, commit, and analysis executable. Matrix:
`{dog-run, humanoid-run, myo-key-turn, myo-pen-twirl} x
{D2W512,D4W1024,D4W1536} x 8 approved distinct seeds = 96` parent runs.
CPU registry checks resolved these to DMC `(dog,run)`, DMC `(humanoid,run)`,
`myoHandKeyTurnFixed-v0`, and `myoHandPenTwirlFixed-v0`, respectively. The
Myo pen task is marked held-out for an older Angle3 workflow; confirm its
new-study role explicitly before freezing the protocol.

Each parent has 500,000 agent interactions (1 million environment frames at
action repeat 2), ten targets and 2,000 fit steps at each of 50 scheduled
checks (every 10,000 interactions). Log function-level scores/curves, paired
IQM and CI, check-level and consecutive-trigger decisions, seed/teacher keys,
source/config hashes, backend/precision, W&B run ID, replay/normalizer/checkpoint
identity, failures, and resource usage. Define the cross-seed population,
estimator, interval, multiplicity handling, missing-run policy, and endpoint
*before* official launch. Preserve exact checkpoint/resume and durable queue
semantics. Experiment 2 arms need separate authorization and queue identities;
they are not among the 96 parents. A change from 20 to 50 check opportunities
can alter natural trigger timing. Its 95%-budget eligibility and post-fork
horizon must be explicitly versioned and reviewed, not inherited silently.

One reviewable cross-seed analysis proposal is to keep environments and critic
sizes as twelve separate cells, report the final (500k) plasticity-loss IQM
over eight seeds per cell plus a seed-resampled 95% interval, and report the
count and interaction time of natural triggers with non-triggered parents
censored at their last eligible check. With eight sorted seed outcomes,
empirical 25%-75% IQM averages ranks 3–6. Seed resampling must keep each
parent's entire check trajectory together; function draws within a parent
are not extra independent training seeds. Missing or incomplete parents must
be listed and excluded from a cell estimate until recovered, never silently
treated as non-triggers. Any family-wide claim across twelve cells needs an
explicit multiplicity rule or clearly descriptive status. This proposal is
**not frozen**; the owner must approve estimands, uncertainty, multiplicity,
and the handling of censored/missing trajectories before official reuse.

Initial packing proposals are 6 D2W512, 4 D4W1024, or 2 D4W1536 per A100,
with mixed-size fractional admission and immediate replacement after verified
completion. The current scheduler has ceilings 7/4/2 and weights 1/7,1/4,1/2;
a new version must use 6/4/2, CPU/RAM/VRAM admission, JAX non-preallocation,
atomic claims, and measured throughput before treating these as safe capacity.
No manifest or scheduler production configuration is changed in this branch.

**Cost arithmetic, excluding Exp2:** per single-critic parent, a fresh range
check plus 50 paired checks requires `10+50*20 = 1010` 2,000-step fits,
2,020,000 fitting updates. All four currently mapped env groups specify
`episodic:false`; a twin-critic change would increase this work and must be
costed separately. The old five-target/20-check pilot needed `5+20*10=205`
fits, so new fitting work is 4.927 times the old fitting work, not 4.927 times
the entire training runtime. Across 96 parents: 96,960 fits and 193.92 million
fitting updates. Prior A100 data measured *only* the second 1,000 steps at
1.848 s/fit (D4W1024) and 3.469 s/fit (D4W1536). Assuming, without proof,
the first 1,000 costs the same gives about 33.2 and 62.3 **unpacked** GPU-hours
for the 32 medium and 32 large parents' fits. D2 fit time is not measured here;
training, compilation, evaluation, checkpoints, packing contention, and
Experiment 2 all add cost. The seven-run pilot had concurrent wall runtimes
around 3.4–6.85 hours on two A100s with five targets/20 checks; those shared
GPU runtimes cannot be linearly extrapolated to the new workload. Freeze a
resource allocation only after a representative ten-target throughput sample.

## Decisions before official use

1. Approve the exact ten-point IQM convention and paired bootstrap rule only
   after empirical coverage and dependence checks; ten functions alone do
   not validate a nominal 95% interval.
2. Review shared-target and checkpoint-crossover evidence for construct
   validity, including whether the existing network-specific offset remains
   appropriate. No offset change is proposed here.
3. Select and freeze the eight seed IDs, endpoint/cross-seed analysis, and
   whether Myo pen twirl's older held-out designation has any bearing here.
4. Approve the 50-check schedule's effect on natural Experiment 2 eligibility
   and preserve the original post-fork horizon unless separately amended.
5. Authorize and resource-bound any A100 validity job and later the official
   study; this document grants neither authorization.
