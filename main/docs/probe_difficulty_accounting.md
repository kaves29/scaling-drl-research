# Probe difficulty, score components and DMC–Myo differences

Source-bound CPU analysis of the retained 103-record export. Original campaign:
`f597ee9898e4d401912092c336c3e5330cf2fda1`. Analysis branch
`codex/probe-difficulty-accounting`, based on PR14
`1d849c3b52f1b6bdc2514f8972a2388021b929c7`.

## Answers supported by the evidence

1. **The cause of fresh-score drift is not identified.** Its decomposition needs
   per-round target variance or final error over time. Only10 of48,340 round
   records have those components in the supplied original artifacts. All960
   run/round trajectories through70% admit both constant-variance and constant-
   fresh-MSE algebraic explanations. The latter reproduce falling fresh scores
   without worsening fresh fitting error. These are feasibility witnesses,
   **not measured/estimated components or certified teacher/fit histories**.
2. **The DMC–Myo difference survives the contemporary paired comparison.**
   Shared b cancels within paired rounds, so this is an identified difference
   in relative final fitting error, subject to source float32 subtraction.
   It is not solely an initial-reference or target-variance plotting artifact.
3. **Centered starting predictions do not yet explain the ordering.** One Myo
   case has very large trained initial minibatch residuals and much larger final
   pool error, but no initial full-pool predictions/covariances. Mean Q alone
   cannot supply them. No original DMC parameter/pool bundle is accessible here.
4. **DMC endpoint catch-up remains untested.** No original DMC fit curves or
   intermediate full-pool errors are present. The one Myo archive retains
   changing-minibatch curves, not intermediate full-pool curves.
5. **No normalization, routing, target or precision defect explaining the
   discrepancy was reproduced.** The known stale W&B fresh-summary field does
   not explain reconstructed round scores. Actual-offset rounding bounds in the
   one original Myo case are tiny for a fixed final predictor; sine-phase and
   optimization-path effects remain unmeasured.
6. **A conditional environment contrast remains after matching architecture,
   checkpoint, seed labels and contemporary reference.** Unmeasured replay,
   normalizer, teacher and starting-residual differences prevent a fully adjusted
   or causal environment effect. Two tasks/domain do not establish generalization
   to a population of environments.

No production code, historical results, optimizer, fitting, score, threshold or
scientific method was changed. No models were created/evaluated, no training or
new probe fits ran, and no GPUs were used. Only stored tensors/JSON, analytical
bounds and CPU statistics were evaluated. The earlier implementation controls
are cited as prior evidence, not rerun or presented as original checkpoints.

## What is actually retained

The comprehensive and metric ZIP SHA256 values remain
`f7337130705a59622e1b639449e0dad452317fa2a45b479760a92aa3d9b12a44` and
`f5b11ca7223dcb43f4cb6aa469aa9e4e03e52dfdb1d2aa06085b98dfce282505`.
All697 raw manifest hashes/history row counts and52 source snapshots are checked;
52,214 comparisons against supplied derived/supplement values pass. The reader
uses raw JSON, never supplied analysis scripts or `histories.pkl`.

| Evidence | Coverage and precise limitation |
| --- | --- |
|103 metadata/history records |96 canonical parents, seven non-independent pilot records,25,557 history rows. Pilot pairs are excluded from seed counts. |
|Raw probe scores |4,834 checks:96 fresh-only anchors and4,738 paired checks.48,340 round records total;960 fresh-only rounds,47,380 paired rounds. Source ten-round IQM and FP32 pair subtraction are preserved. |
|b/final pool MSE |Only pen-twirl/D2W512/seed5/check50:10 paired rounds,20 final errors.48,330 round records lack b; no hidden b=.5 replacement is made. |
|Original fit curves |Same one parent/check:40,000 pre-update changing-minibatch losses (2critics×10rounds×2,000). Only the20 final pool errors at update2,000 are available. |
|Pools, RMS, teacher, predictions |No original pool arrays, normalizer snapshots, teacher outputs/parameters, numeric targets, initial centered predictions or production parameter checkpoint bundle in the accessible evidence. |
|Artifacts/search |206 candidate containers/array/checkpoint files inventoried under artifact/upload/library roots, including member listings of nine ZIP/tar containers; synthetic/unbound files are not promoted to historical evidence. Additional repository/scratch/shared scans found no candidate arrays/checkpoint containers. |
|W&B references |File inventories contain19 `artifact/.../wandb_manifest.json` references with size0 and MD5null; their bodies are absent. These unresolved placeholders do not establish checkpoint contents or prove remote artifacts absent. No authenticated remote artifact/Delta inventory is claimed. |

The separately provided original archive is
`raw_archived/D2W512_myo-pen-twirl_seed5_check50.npz`, SHA256
`66a223890220be7aa9d022e9cdfbcd90909d6580132e20315492a0ff307839ca`.
Its recorded source is the original `campaign_owner_f597ee9_20261019T230748Z`
Delta tree, not the later retrospective runner. Run **63b389qb**, check50,
rounds0–9 have exact NPZ/W&B score binding. We verify that binding and supplied
provenance, not a new remote filesystem inspection.

Eight large-critic runs crashed. Every canonical run reaches check39/390,000
interactions (78%); after that coverage declines. Primary comparisons use the
previously selected check35/70%, with all eight seeds in every cell. Complete
observed trajectories and actual late counts remain in the plots/tables; no
missing check, crash or component is filled with zero or interpolated.

## Definitions, feasibility and bounds

The analysis plan is versioned at
`main/audits/probe_difficulty/analysis_plan.json`; its hash is in the receipt.
It fixed observable difficulty measures and statistical rules before this
script's trend/decomposition calculations. This is **exploratory, not a blinded
preregistration**: earlier reports and endpoints had already been reviewed.

Observable difficulty means fresh final pool MSE, MSE/b (undefined if b≤1e-12),
initial/intermediate pool MSE, or measured target distribution. Fresh score is
retained separately, not renamed difficulty. No epsilon is inserted into a
ratio. All20 retained final ratios have positive b well above the floor.

Source per round: `P_f=b−M_f`, `P_c=b−M_c`, `L=P_f−P_c`.
Source b is **variance of the unshifted sine base**, calculated before adding
own offsets; float32 materialized target variance need not be bitwise identical.
Source paired subtraction occurs in FP32 before float64 ten-round fractional IQM.
IQM does not generally distribute over subtraction; mean score-operand changes
are used for exact additive accounting, with source IQMs kept separately.

Because the sine base lies in[-1,1] and MSE is nonnegative, ideal-arithmetic
bounds are `0≤b≤1`, `b≥max(0,P_f,P_c)`. Thus for each critic:

`max(0,P_f,P_c)−P ≤ M ≤ 1−P`.

These are **structural bounds, not MSE estimates or confidence intervals**;
float32 effects qualify boundary precision. Per-round bounds and monotone-IQM
bounds are included. They require no assumption that every teacher produces
variance .5. The original archive's source arithmetic is independently checked
exactly: stored FP32 b minus stored final MSE reproduces all20 scores.
The saved paired score difference and arithmetic M_c−M_f can differ by rounding;
maximum observed discrepancy here is7.90e-7, not a reversed sign or failed fit.
No acceptance criterion was changed to accommodate it.

A trained score near its initial reference does not identify a small or stable
absolute MSE: even large-humanoid's score≈.488 permits final MSE between0 and
about .512 without observed b. A stable score can hide changes in both variance
and fitting error. The recorded score stability survives reconstruction; absolute
error stability or preserved same-task plasticity does not follow.

For each common-window run/round trajectory, choose constant b=.75; this yields
nonnegative varying final errors and reproduces both score streams algebraically.
Alternatively a constant fresh error m is feasible whenever

`max_t(0,−P_f,t,P_c,t−P_f,t) ≤ m ≤ min_t(1−P_f,t)`.

All960 intervals are nonempty. Choosing m inside each interval and setting
`b_t=P_f,t+m`, `M_c,t=b_t−P_c,t` gives the same joint scores with no fresh-MSE
increase. `ALGEBRAIC_WITNESSES_NOT_OBSERVED.csv` labels every witness explicitly.
They establish underidentification from reported scores/structural bounds,
not their likelihood under the actual random teachers, FP32 realizations or
AdamW trajectory. Reproducing those would require missing original state.

The one observed endpoint adds a small partial identification result: in round6
its b is at least .00440437 below the initial fresh score; in round9 at least
.00374886 below it. Since initial M_f≥0 implies b_0≥P_f,0, **variance decreased
between those two realizations**. This does not establish a stage trend or
variance dominance. All10 endpoint-anchored rounds still allow constant fresh
MSE over the initial/end comparison; their error-change bounds cross zero.
The table `archived_endpoint_anchor_bounds.csv` records every bound.

## Drift and condition accounting

All96 fresh IQMs are lower at70% than initially, but **none** decreases at every
adjacent check.93/96 OLS common-window slopes are negative. Exceptions with
positive slopes despite lower endpoints: dog-small seed3 **g3tcvyor**, dog-small
seed5 **7ic002ar**, and pen-large seed7 **6apfwv2e**. Endpoints therefore do not
establish monotonic worsening, even for the score itself.

| Environment | Critic | Δfresh IQM0→35 | Paired L35 | Mean fresh operand Δ1→35 | Mean trained operand Δ1→35 |
| --- | --- | ---: | ---: | ---: | ---: |
| dog-run | D2W512 | -0.006294 | 0.120598 | -0.005744 | -0.125915 |
| dog-run | D4W1024 | -0.014691 | -0.012484 | -0.010523 | -0.002361 |
| dog-run | D4W1536 | -0.032513 | -0.029829 | -0.025535 | -0.000303 |
| humanoid-run | D2W512 | -0.032170 | -0.022373 | -0.030981 | -0.009357 |
| humanoid-run | D4W1024 | -0.056003 | -0.051968 | -0.053712 | -0.002886 |
| humanoid-run | D4W1536 | -0.153446 | -0.151214 | -0.147257 | -0.003504 |
| myo-key-turn | D2W512 | -0.061934 | 0.592313 | -0.053704 | -0.241400 |
| myo-key-turn | D4W1024 | -0.070216 | 0.458922 | -0.060474 | -0.143241 |
| myo-key-turn | D4W1536 | -0.143149 | 0.464284 | -0.126032 | -0.195139 |
| myo-pen-twirl | D2W512 | -0.089573 | 7.433970 | -0.081507 | -8.532319 |
| myo-pen-twirl | D4W1024 | -0.157010 | 0.697953 | -0.152924 | -0.856457 |
| myo-pen-twirl | D4W1536 | -0.246451 | 2.846334 | -0.213204 | -3.025101 |

ΔP_f is initial→70% fresh-IQM change. L35 is source contemporaneous paired IQM.
The two mean-operand columns instead compare check1→35 using arithmetic means
of paired round scores. Their exact additive relationship is
`Δ(mean deficit)=Δmean(P_f)−Δmean(P_c)`; they are not differences of marginal IQMs
or a b-versus-MSE decomposition. For large humanoid, fresh operand changes
−.147257 while trained changes−.003504, giving deficit change−.143753. This is
consistent with either worsening fresh error at stable b **or** declining b
and improving trained error at stable fresh error. Pen-small instead has
trained change−8.532319 versus fresh−.081507; reference drift alone cannot
explain its large relative deficit.

### DMC versus Myo at a common checkpoint

The following difference is Myo minus DMC, with two tasks equally weighted
within each domain/architecture/seed.95% pointwise percentile intervals resample
**eight training seed labels**, preserving within-label cross-task pairing.
Ten probe rounds are not training seed replication. Intervals describe these
four tasks, not a causal/population environment contrast.

| Critic | Fresh-score contrast | Paired-deficit contrast | Initial-reference-gap contrast |
| --- | ---: | ---: | ---: |
| D2W512 | -0.055168 [-0.064772, -0.047396] | 3.964028 [2.351412, 5.395870] | 4.021421 [2.405395, 5.457073] |
| D4W1024 | -0.078458 [-0.099620, -0.057392] | 0.610663 [0.408549, 0.988119] | 0.689290 [0.502358, 1.049964] |
| D4W1536 | -0.102856 [-0.136635, -0.070221] | 1.745830 [0.745698, 2.755306] | 1.844716 [0.883863, 2.816823] |

The paired-deficit contrast remains positive for all three architectures, so
switching from initial to contemporary reference does not eliminate it. A
variance change **common to a paired round** cancels; unknown differences in
task complexity, inputs, optimizer conditioning and starting predictions do not.

There is also a useful absolute-error bound. At70%, mean over eight seeds of
source-defined IQM(trained MSE) for pen-small is at least7.43397; dog-small is
at most .632305 and humanoid-small at most .514814. Pen-medium is at least
.697953 (matched DMC upper bounds .507831/.510801); pen-large at least2.846334
(DMC upper bounds .507387/.512004). Therefore the pen-twirl absolute fitting
error contrast cannot be explained solely by its unknown target variance.
These deterministic bounds are far apart relative to FP32 roundoff. Key-turn
bounds overlap some DMC bounds; do not extend this absolute ordering to every
Myo condition. All per-condition bounds are in `condition_intervals.csv`.

## What changes in the recorded implementation

| Transition / exact original source | Verified behavior / retained data / discriminator |
| --- | --- |
|[Fresh capture and selection, run_probes.py:67–103](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp12/run_probes.py#L67) | Fresh parameters captured before first SAC update at warmup5,000; current live critic selected at later checks. Actual original parameter hashes absent. Earlier independent CPU routing tests are supporting fixtures, not production reconstruction. |
|[Replay pool, probe.py:103](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp12/probe.py#L103) | Uniform sampling with replacement from filled replay indices;25,600 entries per round. One processed obs/action pool passed to trained, fresh and teacher. Original pool arrays/index realizations absent. |
|[Observation normalization:11–46](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/agents/wrappers/normalization.py#L11) | Current RMS, `(obs−mean)/sqrt(var+1e-8)`, no clipping in this normalizer; pool normalized once without RMS update. Fresh parameters do not retain an initial normalizer. Count/mean/variance/distributions absent. |
|[Teacher, probe.py:65–69](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp12/probe.py#L65) | Same-architecture random critic initialized each round, then `sin(1e5*f)` on processed pool. New key each check/round; architecture changes teacher as well as learner. No saved teacher outputs/base vectors/quantiles. |
|[Critic and initializers](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/agents/sac/sac_network.py#L17), [residual block:31](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/networks/layers.py#L31), [head:13](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/networks/critics.py#L13) | Orthogonal input projection/head, He-normal residual affines, residual LayerNorm and final LayerNorm. Shapes depend on environment input/action dimension and architecture. No source-level DMC-only output bypass. Exact original initialization runtime/clean checkout not fully certified. |
|[Keys/index streams, probe.py:116–170](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp12/probe.py#L116) | Dedicated seed/stream/check/round keys; teacher/index split. First1,000 minibatches original stream, next1,000 folded stream; shared within pair. Environment/architecture are not additional key inputs; identical labels can share random streams while shapes/inputs differ. |
|[Own offsets/fit, probe.py:78–100](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp12/probe.py#L78) | Fixed own initial pool mean, shared base, distinct literal offset targets. AdamW1e-4/.01, fresh state each fit; one continuous2,000-step scan. Offset/targets do not update during fitting. No original initial centered/full-pool prediction arrays or optimizer states. |
|[Logging, run_probes.py:111–145](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp12/run_probes.py#L111) | NPZ retains b/MSE/curves locally; W&B logs round scores/deficits but not b/MSE components. Fresh IQM summary only initially; later IQMs reconstructed from round scores. New PR14 diagnostics were not present in this campaign. |

DMC/Myo share the generic pool/normalizer/teacher/fit/score paths, single dense
critic, float32 setting, action repeat2, reward_scale1 and fitting budgets.
They differ in underlying environment/wrappers/dimensions, intended horizon
(max_episode_steps1,000 versus100), discount .99 versus .95, and endogenous
visitation/value distributions. Source code differences do not measure their
causal contributions. Existing rank/dormancy/TD metrics are proxies on different
inputs/objectives; they do not supply missing centered pool predictions.

### A known change in sample-index support

Conditional on the recorded empty-start, single-env, n_step1 loop, replay
occupancy at interaction t is t through this budget (capacity1M is not reached).
This is **source-predicted occupancy, not a retrieved buffer snapshot**.
For M=25,600 uniform draws, expected distinct indices are
`N*(1−(1−1/N)^M)`. At warmup N=5,000, at most5,000 indices can be distinct:
at least80.46875% of draws repeat an index; expected distinct count≈4,970.
At70% N=350,000, expectation≈24,686, about3.57% repeated draws. Index support
therefore changes materially even though the nominal pool size stays fixed.

This is a concrete intended protocol difference that could alter effective task
support, not demonstrated fresh-MSE causation. It has the same schedule in all
12 conditions and cannot by itself explain environment/size rankings. Distinct
indices are not physical-state diversity: observations/actions may still be
concentrated. Missing pools prevent measuring that concentration. No sampling
rule is changed, and the predicted counts are explicitly labeled in the CSV.

### Timing, actor performance and metric associations

576 within-run associations are computed from raw data, not an imputed MSE:
fresh **score** versus Q mean, TD loss, training gradient norm, historical rank,
DRτ.2 and actor return. Structural/update metrics use an as-of observation at
most2,000 interactions old. Actor returns use only exact coincident evaluations
(checks5,10,…,35), never carried-forward sparse values. Raw and quadratic-time-
residual Spearman coefficients are retained; constant signals are undefined,
not forced to zero. No repeated-observation p-values are reported.

For example, the mean of eight large-critic Q/fresh-score residual correlations
is−.281 dog, +.364 humanoid, −.379 key-turn, −.420 pen-twirl. There is no universal
sign across tasks. These are descriptive coefficient means, not calibrated
causal effects or evidence that Q mean is initial centered variance. For actor return at the large architecture, coefficient means are approximately
.094/.040/.165/.330 for dog/humanoid/key/pen, based on only seven exact shared
evaluations per parent; these do not establish universal synchrony. They do
not identify b or fresh MSE. Matching architecture/stage and descriptive trend
removal cannot adjust for unmeasured input/teacher/starting-residual quantities.
`candidate_factor_accounting.csv` gives every parent/factor's observed quantity,
association or explicit missing-artifact status, without invented effect sizes.

## One original case: components, starting error and kinetics

For **63b389qb**, pen-small/seed5/check50, median fresh final MSE=.1422703,
trained=14.8014; median MSE/b≈.285551 and29.4504 respectively. b lies near .5
across these10 rounds and is shared within every pair. These observed endpoint
components identify that case's large trained fitting disadvantage; they do
not establish that b was stable over its training history or over DMC runs.

| Fitting updates completed | Fresh median stored loss | Trained median stored loss | Measurement domain |
| ---: | ---: | ---: | --- |
|0|.875613|164001.53|Initial minibatch, pre-update; not full pool|
|100|.492977|901.3618|Changing minibatch, pre-update|
|500|.459621|129.9041|Changing minibatch, pre-update|
|1,000|.334459|46.0071|Changing minibatch, pre-update|
|2,000|.142270|14.8014|Full pool, post-fit|

The arrays show substantial progress in sampled losses with a remaining final
trained error. They cannot yield valid initial-to-final **pool** improvement,
intermediate pool errors, or a fraction of final deficit attributable to initial
pool difficulty. Those columns remain null. No missing DMC trajectory is inferred
from this single Myo case. The initial own-offset residual identity is

`E[r²] = Var(Q_initial) + Var(z) + E[z]² − 2 Cov(Q_initial,z)`.

The stored offsets are the source-computed initial pool means: medians about
1,998.706 trained and−.428516 fresh. The own mean cancels algebraically; that
mean alone does not imply initial MSE≈164,000. A limited bound is possible:
since the sine base lies in[-1,1], the initial minibatch residual RMS minus
1 (and target-addition roundoff) lower-bounds the maximum absolute deviation
of an initial minibatch Q output from its stored pool offset. With conservative
normal-float32 reduction/product roundoff, these trained lower bounds are
362.612–450.845 across rounds, median403.897. Thus substantial initial
minibatch prediction deviation is established in this one case.

This is not an inferred full-pool variance, covariance or initial pool MSE.
The initial loss uses batch256 forwards whereas the offset uses chunked pool
forwards; original GPU batch/kernel differences are not qualified. The bound
cannot distinguish widespread variation from outliers, assign the final gap
to starting error versus optimization, or establish the DMC starting state.
Its derivation and round values are in `archived_precision_bounds.csv`.

### Actual-scale numerical check

Observed trained offsets are near2,000; fresh near−.43. For any stored float32
base z∈[-1,1], maximum target-addition rounding over the actual offset interval
is6.10352e-5 trained and5.96046e-8 fresh. Using stored final MSE plus a conservative
positive-sum float32 roundoff envelope, the fixed-prediction MSE perturbation
bound `2δ*sqrt(MSE_upper)+δ²` is at most **.0004942 trained** and **4.87e-8 fresh**.
Those direct-error bounds are orders below the observed≈14.66 paired error gap.
The detailed CSV records all20 bounds, including target-variance perturbation.

This rules against a **direct final target-addition rounding artifact** as an
explanation of that case's gap at fixed predictions. It does **not** bound
accumulated optimization-path sensitivity, original GPU/compiler differences,
or sine phase error. Teacher f values and `1e5*f` arrays are missing; neither
trained mean Q nor fresh offset is a substitute for teacher-output distribution.
No synthetic extreme-value fixture is used as evidence of a campaign defect.

## Ranked explanations, defects and remaining discrimination

| Rank / classification | Evidence and explanatory coverage | What is not established / smallest discriminator |
| --- | --- | --- |
|1 Identified conditional outcome|Myo relative final-error gap persists after paired variance cancellation and arch/stage matching; pen absolute-error bounds separate from DMC.|Not a cause. Recover original b/MSE time series to decompose fresh/absolute errors.|
|2 Intended protocol support change|Source-predicted distinct-index support grows markedly; current replay/RMS and teacher/check keys change.|Performance contribution unknown; common schedule alone cannot explain DMC–Myo ranking. Need original pools/index/RMS snapshots; no sampling change adopted.|
|3 Partially measured target variation|One endpoint forces some variance decrease in2of10 realizations, but all10 permit constant fresh error.|Not stage-general or dominance evidence. Recover original NPZ b arrays across checks.|
|4 Actual one-case starting/finite-budget mismatch|Huge trained initial minibatch loss and final14.8 pool error, versus fresh .14.|Centered Q/covariance/full-pool initial error absent; cannot assign fraction to initial difficulty versus fitting. Obtain initial predictions/base targets.|
|5 Plausible initialization/conditioning/offset interactions|Own offsets and fresh AdamW state are as designed; teacher architecture/input geometry change.|Not quantified mechanisms. Need original pool predictions/gradients/teacher; mean Q and rank proxies are insufficient.|
|6 Direct precision explanation disfavored in measured case|Actual-offset fixed-prediction bound<.0005 versus≈14.66 error gap; scores bind exactly.|Sine/optimization/GPU effects unqualified. Need actual teacher/target/prediction arrays, not an extreme synthetic bias.|
|7 Confirmed summary/reporting limitation, not explanation|Fresh summary stale; raw rounds reconstruct all drift. Missing component data makes “harder fit” wording unsupported.|Do not rewrite archives or treat omission as evidence MSE rose. Existing summary proposal remains separate/unapplied.|
|8 Endpoint catch-up plausible but untested in DMC|Final scores can conceal early fitting differences; source continuous2,000 steps correct.|No DMC curves/intermediate pool errors; one Myo curve cannot establish DMC catch-up. Recover existing artifacts first.|
|9 Unverified original runtime identities|Recorded source/arrays support intended routing, prior CPU fixtures pass.|No actual original checkpoint hashes/pools available. Obtain trained/fresh state+manifest; no invented reconstruction.|

No new confirmed implementation defect explaining the direction/magnitude of
the DMC–Myo contrast was found. Earlier retrospective wrong-optimizer and A100
segmentation failures are separate source/execution evidence; they are not
retroactively assigned to the original continuous campaign fits. No production
correction is made merely because proxy metrics and probe results disagree.

## Smallest recovery or future measurement

1. **First recover existing `probes/check_*.npz` files** from the original frozen
   campaign paths for each parent/check. The source already writes b and final
   fresh/current MSE. Three10-element arrays per check suffice for component
   accounting; no model checkpoint, GPU or new fit is needed for that first step.
   The supplied ZIP has only one of these files; remote retention is not verified.
2. Original input/normalizer/teacher/initial-prediction bundles are needed to
   distinguish visitation, normalization, target sensitivity and centered residuals.
   Source `LATEST` retention can remove older states, so do not assume an old
   check can be reconstructed from a final checkpoint. The unresolved19 manifest
   placeholders need actual contents/access before any checkpoint claim.
3. If never saved, the smallest future additions are per-round b/final MSE,
   input/index/normalizer identity and teacher/base/initial-prediction summaries.
   For early learning, retain full-pool errors at0,100,500,1,000 alongside2,000.
   Such observational additions must preserve fitting/RNG and pass unchanged
   equivalence criteria; compiler/GPU qualification needs separate authorization.
   No such fit, logging change or new scientific experiment is executed here.

## Reproduction and deliverables

CPU tools: `main/audits/probe_difficulty/accounting.py` and `figures.py`; usage and
output meanings are in the [README](../audits/probe_difficulty/README.md).
Full tables/plots/logs live outside Git at
`/workspace/artifacts/probe-difficulty-accounting/`. The [tracked validation receipt](../audits/probe_difficulty/validation_receipt.json)
binds input, plan and code hashes. Each round row identifies its run ID, check,
round and zero-based JSONL row index, so its raw line is index+1 in
`raw_api/<id>/history.jsonl`.

Outputs include all48,340 round rows;96 seed trend rows;12 condition trajectories
with eight-seed uncertainty and explicit absent components; DMC/Myo common-check
figures; original one-case fitting curves;1,537 parent/factor accounting rows;
576 descriptive stage-adjusted associations; structural component bounds;
actual-offset precision bounds; unresolved artifact-reference and workspace
inventories; and clearly separated algebraic witness tables.

Validation:10 deterministic CPU accounting tests,4,834 source-IQM checks,
52,214 derived/supplement comparisons,52 source snapshots and697 raw hashes pass.
Static checks and source-scope review confirm audit/docs/tests only. No broad
training/fit suite is rerun under this task's prohibition. Missing values and
untested causal paths remain explicit; no threshold is loosened or test disabled.
