> Historical report and evidence snapshot at `f9615de`. Current review and readiness: [independent Claude review](exp12_claude_review.md). Generated evidence links below resolve to the preserved Git snapshot.

> Historical investigation at HEAD `438a5b0`, before the subsequent approved implementation.
> Current implementation, decisions and gates: [approved implementation report](exp12_approved_implementation_report.md).

# Exp 1/2 follow-up investigation at 438a5b0

Investigation date: 2026-10-06. Audited HEAD: `438a5b049d897019e09407b2a40cf1da565e0d88`.
This report and its evidence are new documentation only. No existing code, configuration,
assertion, tolerance, measured quantity or experimental behavior was changed. No jobs were
submitted or launched, no Delta checkout was accessed or modified, and nothing was merged
or pushed. Every recommendation below is a proposal, not an adopted decision.

## Evidence and scope

Evidence is in [exp12_followup_evidence](https://github.com/kaves29/scaling-drl-research/tree/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence), including original test
logs, executable diagnostic harnesses, exact arrays, every differing element, synthetic
acceptance counterexamples and [SHA-256 fingerprints](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/sha256.json).
The diagnostic harnesses ran outside the repository under `/tmp`; the files here are copies.
They are not replacement production tests or research observations.

All executions used macOS arm64 CPU, Python 3.12.13, JAX/JAXlib 0.4.34, Flax 0.8.4,
Optax 0.2.3, Orbax 0.5.3, NumPy 1.26.4, pandas 2.1.4, MuJoCo 3.6.0,
dm_control 1.0.38 and MyoSuite 2.12.2. `rliable==1.2.0` and its temporary supporting
dependencies came from the prior audit's `/tmp/exp12-validation-deps`; no dependency was
installed or changed in this investigation. HB is unavailable locally. Full versions and
cross-process array equality are in [metadata.json](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/metadata.json).

Historical code was extracted with `git archive <commit> main` into `/tmp/exp12-history`.
All revisions used the same current dependencies. This isolates code changes; it does not
reconstruct a September machine or establish when a dependency upgrade first exposed a
failure. No full suite was rerun unnecessarily: this investigation reran the original named
failures and separate diagnostics. Prior full-suite results remain those in the coverage audit.

## 1. Twin compiled/eager panel failure

Original test: `tests.test_exp12_twin_critic.TwinCheck1Test.test_panel_values`.
The fixture uses seed 0, depth 2, width 8, 256 states, observation dimension 6,
action dimension 3 and two critics. Production `fork.panel_q_and_grad` is compiled;
the test's expected reference is eager, both under local `highest` precision.

The original test fails in two fresh processes at HEAD and at its introduction,
`2f67915cf9243d75f56299645ccd3961e9c0c44e`. Its exact Q assertion stops execution before
the gradient assertion. A separate diagnostic applies that later assertion unchanged and
also fails. [Original repeats and historical results](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/original_test_results.json),
[first-commit test log](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/twin_original_2f67915.log),
[later gradient assertion](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/twin_later_gradient_assertion.log).

Here and in the twin CSV, expected = eager, observed = production panel, and relative
difference = `abs(observed - expected) / abs(expected)`. Values are exact decimal
representations of the stored float32 values; array indices are zero-based.

| Quantity / element | Expected | Observed | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|
| Largest Q absolute difference, flat 354 = Q2, state 98 | -1.4848878383636475 | -1.4848871231079102 | 7.152557373046875e-7 | 4.81690077072018e-7 |
| Largest Q relative difference, flat 183 = Q1, state 183 | -0.0026113688945770264 | -0.002610921859741211 | 4.470348358154297e-7 | 1.711879301096744e-4 |
| Gradient [82,1], fails | 0.0029409804847091436 | 0.0029408172704279423 | 1.632142812013626e-7 | 5.549655363235235e-5 |
| Gradient [146,0], fails | -0.00685656676068902 | -0.006856751162558794 | 1.8440186977386475e-7 | 2.6894198833022682e-5 |
| Gradient [164,0], fails | 0.04219628870487213 | 0.04219687730073929 | 5.885958671569824e-7 | 1.3948996113703742e-5 |
| Largest gradient absolute difference [134,2], within bound | 2.8269433975219727 | 2.826944351196289 | 9.5367431640625e-7 | 3.3735175498816737e-7 |

Q: 67/512 elements differ; its criterion is exact equality. Gradients: 601/768 differ,
but only the three listed elements violate the existing `rtol=1e-5, atol=1e-7`.
Their existing bounds are approximately `1.2940981e-7`, `1.6856566e-7`, and
`5.2196287e-7`. A maximum absolute gradient difference does not identify the largest
criterion violation because the relative allowance changes with component magnitude.
All 668 nonidentical elements, exact hexadecimal values and existing bounds are in
[twin_differences.csv](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/twin_differences.csv); complete expected
and observed arrays are in [twin_arrays.npz](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/twin_current_1/twin_arrays.npz).

### Independent and historical comparisons

1. Both current independent processes produce identical arrays, including mismatch locations.
2. Separately compiled Q and min-Q gradient functions match the production panel **exactly**.
   This isolates eager-versus-compiled evaluation, but is not sufficient as an independent
   semantic oracle because it shares the network definition.
3. A second reference slices the two parameter trees and applies two independent single-Q
   networks before taking their minimum. Its different compiled graph also has small rounding
   differences; it is not bit-identical to the panel. Do not claim all compiled references agree.
4. A NumPy float64 forward pass and analytic action Jacobian, independently implemented from
   the same float32 inputs/parameters, give the following maximum absolute discrepancies:

   | Reference comparison | Q | min-Q action gradient |
   |---|---:|---:|
   | Production panel versus NumPy64 | 6.858193354331021e-7 | 1.1058659388218928e-6 |
   | Eager versus NumPy64 | 7.097325576688185e-7 | 1.1631862459893227e-6 |
   | Sliced-single compiled reference versus NumPy64 | 8.64633269759274e-7 | 8.080047978337168e-7 |

   Panel errors divided by the maximum reference magnitude are `2.9602114568606963e-7`
   for Q and `2.621923944303752e-7` for the gradient. The minimum Q1/Q2 separation is
   `0.002085272196446386`, and neither eager nor compiled evaluation changes which Q
   attains the minimum. Central differences of the independent NumPy function agree with
   its analytic gradient to `1.3124683562182327e-9` at h=1e-5 and `9.661249578130082e-10`
   at h=1e-6. At h=1e-4 the discrepancy is `0.005488494415589643`; finite-difference
   step size is sensitive to the piecewise network and cannot be used blindly as an oracle.
   These diagnostics do not establish a new pass threshold.
5. At `c293b9a226e6d12727151abecb3ee568fbdd0088`, the first working Stage 2 twin
   implementation, an equivalent compiled twin probe produces exactly the current eager
   and compiled arrays. The Exp12 fork helper at that commit is single-Q-only, so this is
   explicitly an adapted diagnostic, not a claim that the later test already existed.
6. Appropriate pre-Exp1/2 baseline: merge base `6ffccbaf11bd8f60ce663b59ac546e9ff7cffd9c`.
   Its original twin initializer fails with `ValueError: Expected None, got ...` under
   pinned JAX/Flax; an exact original twin-panel comparison is therefore unavailable.
   Feeding the same saved parameter slices to that baseline's two single-Q networks works:
   eager/JIT differ in 412 Q elements (max `1.1324882507324219e-6`) and 668 gradients
   (max `1.1920928955078125e-6`), with two gradient violations of the later criterion.
   Its compiled sliced-single arrays equal the current sliced-single reference exactly.
   See [pre-Exp12 probe](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/pre_exp12_twin_probe.json).

**Origin and classification.** The first commit containing the failing assertion is
`2f67915`; the same underlying twin eager/JIT behavior already exists in `c293b9a`.
The generic phenomenon exists in the pre-Exp12 single-Q implementation. There is no
meaningful passing original twin-panel predecessor on this dependency stack, so a more
specific first numerical-behavior commit cannot be assigned. The evidence supports normal
float32 compiler rounding plus an invalid exact cross-execution assumption and an unjustified
gradient reference bound for this fixture. It does not reveal an incorrect min, wrong Q axis,
or wrong action derivative. JAX explicitly documents that JIT optimizations can change exact
floating-point outputs: [official JAX FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs).
That explanation is consistent with, not a substitute for, the independent measurements.

**Impact on Exp1/2.** This test does not run inside the experiment. No result change is
demonstrated, but the unresolved semantic oracle is a validation blocker for HB's twin
panel and its min-gradient mutation. Correctness on one tiny CPU fixture does not establish
full-size CUDA behavior. This is separate from Check 1's exact original/control comparisons
and the approved injected-arm 64-eps bound, which remain binding.

**Recommended fix for approval.** Keep the experiment's dtype, compilation and precision
unchanged. Separate exact original/restored-control/identity requirements from the semantic
network oracle. Retain an eager comparison as diagnostic evidence, add the independently
sliced per-Q and independently implemented forward/analytic-gradient references, and use a
predeclared float32 error budget for the semantic reference. The actual budget and whether it
is componentwise mixed absolute/relative or normwise require lead approval; this tiny CPU
case is not enough to choose them. Make min-versus-Q1 and wrong-axis mutations fail the
semantic oracle, and require the restored test to pass under the approved rule. Replacing
the expected result with the same production JIT, removing the later gradient assertion, or
raising its tolerance until it passes would not establish independent correctness. No fix
has been made and the original gate remains red.

## 2. Older Baoding determinism failure

Original test:
`tests.test_angle2a_env_state_determinism_smoke.TestMyosuiteDeterminism.test_myo_baoding_p1`.
It captures after five warm-up actions, then restores and replays the same next five actions
twice. Every observation, reward, termination and truncation is required to match exactly.
There is no JAX/eager/JIT computation in this simulator-state test.

The original test fails in two independent current processes, on pre-Exp12 `6ffccba`,
and on `b554d453cf1cf4ad33cd7ecaeb06266d66a0e4ac` (2026-09-08), the commit that
introduced this permanent smoke test and MyoSuite support in the restore helper. The
diagnostic arrays/actions/rewards are identical across all four processes/revisions.
Its parent did not support MyoSuite capture, so `b554d45` is the first applicable bad
implementation/test commit under the current dependencies, not proof of a historical
September test run.

The original assertion stops at step zero: 12/86 observation elements differ. Diagnostic
continuation reveals the same 12 fields at every step: target1/target2 positions and errors.
First replay is called expected here; NumPy's `assert_array_equal(obs1, obs2)` calls the
second array its desired argument, which explains the different relative-error denominators.

| Step / quantity | Expected, first replay | Observed, second replay | Absolute difference | Relative to first / second |
|---|---:|---:|---:|---:|
| 0: observation[41], target1_err x; maximum abs and relative-to-second | -4.578214429784566e-5 | 0.0028999706264585257 | 0.0029457527707563713 | 64.3428309428265 / 1.0157871062141604 |
| 0: reward | -0.34023525454678505 | -0.32492551380196555 | 0.0153097407448195 | 0.044997514338168884 / 0.04711769342357777 |
| 1: reward | -0.3339130323580135 | -0.3193024843325843 | 0.014610548025429215 | 0.043755548929171884 / 0.045757702311551456 |
| 2: reward | -0.3212329301015351 | -0.30683245390011465 | 0.014400476201420431 | 0.044828767078358804 / 0.04693270225615808 |
| 3: reward | -0.3158220820408766 | -0.29964427712378633 | 0.016177804917090255 | 0.05122442614698606 / 0.05399003469172557 |
| 4: reward | -0.3485619821447117 | -0.32713577584805914 | 0.02142620629665254 | 0.06147029049128218 / 0.06549637147177725 |

Maximum observation absolute differences at steps 1–4 are
`0.0030585541389882565`, `0.003341197967529297`, `0.003563523292541504` and
`0.0037108297692611814`. Both rollouts have length five, with all termination/truncation
flags false. All 60 differing observation values, exact encodings and both relative errors
are in [baoding_differences.csv](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/baoding_differences.csv);
[baoding_exact_summary.json](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/baoding_exact_summary.json) includes
each step's extrema and all rewards. Full arrays/actions are retained in the NPZ files.

**Cause and classification.** The capture contains physics, solver and wrapper state and
an optional `steps` attribute, but omits Baoding's distinct `counter`. It is 5 at capture;
the first restore leaves it at 5, the first rollout advances it to 10, and the second restore
leaves it at 10, so the second rollout indexes goals 10–14 instead of 5–9 and ends at 15.
Installed MyoSuite's `baoding_v1.py:step` reads `goal[self.counter]`, moves target sites,
then increments the counter. Isolated diagnostic restoration of that one omitted counter
makes observations, rewards and flags exact for all five steps, including exact agreement
with the original first replay, on all four current/historical probes. No production helper
was edited. The helper's existing documentation explicitly describes the omission.

This is a real incomplete-state-restore defect in the older helper, previously accepted
outside Angle 2A's supported task scope. The permanent test's all-environment claim conflicts
with that known limitation; exact equality itself is appropriate if claiming Baoding support.
The differences are not floating-point noise, and this is not evidence of a new Exp12 or
MyoSuite dependency regression.

**Impact on Exp1/2.** Baoding is not in the 13-environment grid, and Exp12 uses
`experiments/exp12/envs.py:ExactRestore`, not the older capture helper. It resets from the
saved simulator RNG state and replays actions, recreating task counters. An independent
five-step continuation into freshly constructed environments matched exactly for all four
selected Exp12 MyoSuite tasks, and Baoding as an extra diagnostic, on this CPU.
[Restore diagnostic](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/exp12_restore.json).
No direct Exp12 result path through the bad helper was found. These short CPU probes do
not establish CUDA/full-run replay fidelity. The old defect would affect exact-state Baoding
rollout estimates if that helper were used for them, including a future Angle 3 use.

**Recommended fix for approval.** Capture/restore the Baoding counter alongside `steps`,
with a regression checking the counter and exact observations/rewards through repeated
restore and a fresh-instance case. Preserve the original exact assertion. Because this
changes replayed observations/rewards for users of the older helper, obtain lead approval
before implementing it. Alternatively explicitly retain unsupported Baoding status and the
red test pending a broader restore design; do not label it a passing determinism guarantee,
silently skip it, or loosen equality.

## 3. Analysis completeness: confirmed current behavior

These are acceptance counterexamples on synthetic files, not allegations that production
results are corrupt. [Acceptance probe and results](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/analysis_acceptance.json).
Endpoint probes use 20 bootstrap replicates solely to test code acceptance cheaply; neither
production defaults nor statistical criteria were modified.

### Exp1 currently accepts

`ledger.load` requires the exact run/check CSV column schema and `experiment=exp1`,
then defaults to `run_role=confirmatory, status=complete`. It does not know the expected
population. `score_matrix` pivots observed seed/environment identities, keeps the first
duplicate, and rejects holes only inside the observed dimensions. Primary analysis reads
`run.csv.final_loss_iqm`, not a validated check-20 record. It computes the two unpaired
scaled-minus-D2 IQM comparisons on whatever matrices survive.

| Synthetic case | Current result |
|---|---|
| All 195 intended rows | Accepted, 5×13 per architecture |
| Seed 5 absent everywhere | Accepted, 4×13 per architecture |
| Seed 5 absent only in D4W1536 | Accepted, D4W1536 4×13 versus D2 5×13 |
| h1-reach absent everywhere | Accepted, 5×12 per architecture |
| One cell absent while its seed/environment remain observed | Rejected |
| Duplicate identity with conflicting endpoint | Accepted; first is used |
| Extra seed 6 or extra environment | Accepted; 6×13 or 5×14 matrices |
| An infinite endpoint | Accepted by matrix/primary code; no explicit finiteness gate |
| Empty checks.csv with correct headers and stale finite run.csv endpoint | Accepted by loader and primary estimator |

The last case was tested through actual CSV loading for three architectures:
[exp1_ledger_acceptance.json](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/exp1_ledger_acceptance.json).
An entirely absent run directory is invisible. A present malformed directory may raise a
file/schema error; that accidental failure is not a complete population validator. Whole
dimensions filtered out as unfinished can disappear the same way as absent dimensions.
NaN endpoints can similarly disappear when an entire pivot dimension has no usable values.

### Exp2 currently accepts

`load` includes non-complete confirmatory parents, discovers only existing Exp2 directories,
silently skips a directory with no matching parent or no `fork.json`, and trusts that JSON's
own `horizon_steps // eval_every_steps + 1`. Each arm is called complete when it has that
many distinct evaluation indices. It does not enforce their values, times, episode IDs,
number of episodes, uniqueness, plan agreement with methodology, finite returns,
Check 1 success, completion markers, stored checkpoint state, or the full eligible census.

| Synthetic case | Current result |
|---|---|
| 26 indices × 10 episodes × 2 arms | Complete; 26 paired rows |
| Only one episode at each of 26 indices | Complete; 26 paired rows |
| Indices 100–125 instead of 0–25 | Complete; 26 paired rows |
| Injected times shifted by one interaction step | Complete; 52 unpaired rows with NaN differences |
| Duplicate episode with conflicting return | Complete; contributes extra weight to the mean |
| One NaN episode return | Complete; group mean silently uses remaining episodes |
| One infinite return | Complete; nonfinite paired difference enters output |
| Missing evaluation, or entire injected arm | Incomplete; silently removed from primary pairing |
| Missing or failed Check 1 | Complete; enters primary pairing |
| Missing Check 2 | Complete; enters primary; absent from success-only secondary |
| Failed Check 2 | Complete; enters primary, correctly excluded from success-only secondary |
| Running parent | Complete; enters primary |
| Missing fork.json or entire fork directory | Invisible to the fork census |
| Plan changed to require three evaluations, three supplied | Complete; enters primary |
| D2 parent with an Exp2 directory | Accepted into paired tables; default plots only show scaled architectures |

`paired_returns` averages available episode returns per arm/evaluation/time and subtracts
injected minus control. Primary inclusion uses `complete`, not Check 1 or Check 2. The
secondary additionally uses Check 2 success. There is no global fail-closed validation
before exports; files are written progressively. If every observed fork is incomplete,
the current analysis can emit empty primary tables rather than fail. If there are no
observed forks at all, it raises, including a scientifically valid all-null situation.
The API also permits `statistic=mean` and a normalization callback; defaults are the
approved IQM and raw returns. `--include-dev` can mix development runs into ordinary outputs.

## 4. Proposed rule for lead approval — not implemented

**Recommendation:** confirmatory publication outputs require a complete, independently
specified population and all required observations for the output being published. No
available-case seed, arm, checkpoint or episode deletion, imputation, or shortened horizon.
Fail the analysis request with an explicit missing/conflicting-data report; do not silently
change the scientific sample. Keep incomplete data available for operational inspection.

### Expected population

Derive expectations from a frozen, approved study manifest, not from discovered files and
not solely from each output's self-declared plan. Record manifest/config/source/dependency
hashes and a snapshot digest in the validated analysis artifact.

**Exp1:** 3 architectures (`D2W512`, `D4W1024`, `D4W1536`) × seeds 1–5 × these 13 tasks
= 195 unique runs, 65 per architecture:

- DMC: dog-run, dog-trot, humanoid-run, humanoid-walk, humanoid-stand,
  swimmer-swimmer15, hopper-hop.
- MyoSuite: myo-key-turn, myo-pen-twirl, myo-pose-hard (PoseRandom), myo-reach.
- HB: h1-run-v0, h1-reach-v0, using twins; DMC/MyoSuite remain single-Q.

Nominal raw budgets are 500k for swimmer/hopper, 1M for other DMC and MyoSuite, 2M for HB.
Action repeat is 2; interaction budgets N are 250k, 500k, 1M respectively. Each run has
fresh check 0 after 5,000 replay transitions, before training, plus checks k=1…20 at kN/20,
five probe rounds/check with the approved 1,000-update probe settings. The primary endpoint
is check 20 at nominal N, even when the control continues beyond N. Exactly 4,095 required
nominal check records exist across 195 runs (195×21); any legitimate post-N checks are
additional labeled observations, not replacements for check 20.

**Exp2:** the complete candidate census is the two scaled architectures × 13 tasks × 5 seeds
= 130 parents. Each must be resolved as eligible, valid no-trigger, or unresolved. Compute
eligibility from the saved checks under the existing rule: the first two consecutive firing
checks, strict lower CI > 0, completion check k in 2…19. Do not infer no-trigger from a
missing fork directory. A valid no-trigger parent is retained in the census, not assigned a
zero return difference or counted as an intervention outcome.

Let F be the data-dependent eligible subset (0…130). Require **every** eligible parent,
both control and injected arms, horizon 0.25N, evaluations j=0…25 at j×0.01N after the fork,
ten episodes with IDs 0…9 at each evaluation. For each row, absolute interaction step must
equal f*+j×0.01N. This is 260 unique episode rows/arm, 520/fork, at most 67,600 rows.
The two arms must use the same seeded evaluation protocol, source fork and frozen m.
Injected ends at f*+0.25N; control ends at max(N,f*+0.25N), never beyond 1.2N.
The primary curve concerns eligible triggered runs; it is not an effect over all 130 parents.
Report candidate/eligible/null counts by architecture and task.

Five parent seeds do **not** imply five eligible forks in every task. Requiring five actual
forks or discarding a task with valid null seeds would change the methodology. When F is
empty, propose a validated all-null census and explicit 'no eligible intervention outcomes',
with no fabricated curve. For one eligible seed, current bootstrap bands collapse to the
single trajectory; recommend showing the line and labeling the lack of estimable seed
uncertainty. Omitting that band is a reporting-policy choice requiring approval; do not
silently invent a minimum eligible-seed threshold or drop the group.

### Conditions that should fail confirmatory analysis

| Boundary | Proposed validation / failure condition |
|---|---|
| Population | Any expected run missing, unresolved or unfinished; unexpected confirmatory seed/task/architecture; duplicate identity/run key; folder/row identity mismatch; missing candidate census entry |
| Provenance | Incompatible/unapproved config or source revision; wrong critic count, budgets, precision, GPU model or frozen m; enabled testing hooks; missing required metadata; files change while validation/analysis reads them |
| Exp1 endpoint | Missing/duplicate check 20; wrong scheduled step; endpoint not finite/valid; run.csv endpoint conflicts with recorded check 20; missing required five-round data; missing fresh reference or invalid initial probe |
| Scheduled checks | Missing, duplicate or mis-timed required check; invalid check falsely marked valid/firing; stored f* disagrees with existing trigger rule; loss/score/IQM fields inconsistent with saved per-round values under their declared representation |
| Eligible fork | Missing fork plan/readiness/state evidence or source fresh critic; missing/failed original-control or injected Check 1; mismatched panel/source/device; plan inconsistent with independent nominal N/f*; missing or unfinished required arm |
| Evaluations | Index set differs from exactly 0…25; wrong since-fork/absolute time; arm grids differ; any missing, extra or duplicate (arm,j,episode); episode IDs differ from 0…9; nonfinite return or impossible recorded length; wrong row arm label |
| Required reporting | Missing Check 2 record or required per-Q records/learning curves/diagnostic observations for the requested report; missing post-fork probes on the existing schedule; missing nominal descriptive evaluations or used diagnostic logging windows |
| Artifact publication | Any validation error; missing expected validation case/result; stale/mixed output from an earlier failed analysis; no validated manifest/snapshot digest matching the consumed data |

Finite numerical comparisons must use existing recorded quantities and their serialization
rules. Do not introduce a new roundoff tolerance for recomputation without review. Invalid
checks are distinct from absent checks: approved decision B5 already permits nonfinite probe
checks to be recorded as invalid and never trigger. Preserve such intermediate records and
show the gap; do not exclude their entire run merely because an intermediate check is invalid.
An invalid final endpoint cannot supply the prescribed finite primary statistic and must
block that estimate pending a lead decision. Check 0 legitimately lacks current/loss/CI fields.
Likewise unavailable success labels, pre-update actor diagnostics and the documented
injected structural-metric NaNs are not corrupt data. The applicability schema must make
these explicit rather than requiring every CSV value to be finite.

Check 2 false (or explicitly recorded invalid rescue measurement) must **not** remove an
otherwise valid fork from the primary return analysis. Report it, and apply the approved
success-only restriction only to the secondary. A missing record is an incomplete report,
not a scientifically negative rescue result. Check 1 failure is an unresolved implementation
boundary and blocks the confirmatory analysis request; do not solve it by dropping that seed.

Early episode termination/truncation is a legitimate complete episode. Do not require fixed
episode length, success or positive reward; require the actual completed episode record and
a length consistent with the registered/wrapper limits. Myo PenTwirl's shorter inner limit
is intentional. Store the same ten episode identities in both arms without equating outcomes.

Do not require 20 retained checkpoint directories: amendment (t) intentionally deletes old
routine states. Require a verifiable terminal/latest state at the correct end, completion
evidence, preserved fresh reference, and immutable fork state/panel/provenance where applicable.
Checks/evaluations must survive pruning. The parent control need not stop at nominal N.
Supplementary arm probes include the saved fork check plus subsequent scheduled checks;
the inherited fork row must not be reinterpreted as the post-injection Check 2 measurement.

The full-report contract includes the methodology's descriptive outputs. A separately
validated endpoint-only product could, if explicitly approved and labeled, remain valid when
an unrelated descriptive artifact is missing; it must not be labeled the complete study.
Legacy unused metric columns should not become new scientific inclusion criteria. Pin
confirmatory IQM/raw-return analysis; any development, mean or normalized variant gets an
explicit exploratory product identity.

### Architecture and partial-data treatment

Proposed stages: frozen expected manifest → read immutable completed-data snapshot →
validate population, provenance and rows → reconcile eligibility and required observations →
produce a machine-readable validation report → pass a validated dataset object to the
existing estimators → write all outputs to a staging directory → publish atomically with
the successful validation digest. A failed attempt may publish an error inventory, never
new primary tables/figures. Consumers must verify the digest, preventing use of old files
left behind by a failed rerun. No fallback to observed dimensions or automatic deduplication.

Partial data may be useful for progress monitoring and explicitly requested exploratory work
because jobs finish at different times. Recommend a separate exploratory mode with a visible
incomplete label, explicit missingness inventory and different output names/directories.
It must not emit validated confirmatory artifacts or silently activate when strict validation
fails. Do not impute, average over surviving episodes, shorten horizons, or select completed
forks for the confirmatory primary analysis. Approving available-case inference instead would
require an explicit estimand, missingness assumptions, sampling/inclusion rule and sensitivity
analysis; it would be a new methodological decision.

### Tests proposed after approval

1. A complete 195-run/130-candidate fixture passes; exact 65-run matrices and both unpaired
   IQM comparisons reproduce the unchanged estimator.
2. Remove one cell, a whole seed across all architectures, one architecture's seed, one whole
   task across all architectures, or a whole architecture: each fails with exact missing keys.
3. Extra seed/task/architecture, duplicate identities, conflicting duplicate endpoint rows,
   mismatched folder/CSV identities and dev rows cannot enter confirmatory outputs.
4. Empty/missing check file, missing/duplicate/wrong-time check 20, stale endpoint, NaN/Inf final
   loss, wrong round count, or truncated curves/required diagnostics fails the relevant report.
5. Explicit invalid intermediate check is retained, cannot fire or bridge a consecutive pair,
   and is visibly reported; invalid final endpoint fails. Legitimate check-0 blanks pass.
6. Recompute eligibility: first eligible crossing, late-only crossing and complete no-crossing
   cases reconcile; incomplete history is unresolved, not no-trigger. Zero eligible forks
   produce a validated null census; one eligible seed follows the approved reporting rule.
7. Remove an entire eligible fork directory, fork.json, one arm, readiness/provenance or
   terminal-state evidence: fail, rather than silently reducing the sample.
8. Correct 26×10 arm grids pass; 26 wrong indices, wrong times, 25/27 evaluations, nine/eleven
   episodes, wrong episode IDs and misaligned arm grids fail with offending keys.
9. Duplicate episode records fail even if byte-identical; conflicting duplicates cannot
   alter episode weighting. NaN/Inf returns fail rather than being silently skipped.
10. Missing/failed Check 1 blocks; original/control common drift blocks. Missing Check 2
    blocks the complete report, while recorded Check 2 failure remains in primary and is
    absent only from the success-only secondary. Verify primary values stay identical.
11. Wrong self-declared horizon/spacing/episode count cannot redefine expected completeness;
    wrong m, device, critic count, precision or forced confirmatory trigger fails provenance.
12. Early-terminated episodes, shorter PenTwirl limits, unavailable success, intentional metric
    NaNs and pruned routine checkpoint histories pass their declared applicability rules.
13. Nominal endpoint remains check 20 despite extra control checks/steps; post-fork pairing
    remains exactly bounded to 0.25N and uses the existing per-evaluation ten-episode means.
14. A file changed during validation invalidates the snapshot. Any failure creates no new
    confirmatory tables/figures; stale outputs lack a valid matching publication digest.
15. Exploratory partial output is visibly labeled and cannot be passed off as validated
    confirmatory output, including through `include_dev`, mean or normalization options.

**Approval sought before implementation:** the strict population/grid contract, treatment of
invalid endpoints versus intermediate invalid checks, missing descriptive artifacts, valid
null/one-eligible-seed presentation, and the separate explicitly exploratory pathway. None
has been silently adopted in code.

## 5. Three remaining decision items

| Decision | What it means and why the lead owns it | Options and scientific effects | Recommendation |
|---|---|---|---|
| Twin reporting | Amendment (z) uses mean per-round L for the main trigger/endpoint while retaining each Q's behavior. Canonical Exp1 reindexing drops per-Q fields; twin NPZ has Q-specific curves but the plotting helper expects absent aggregate loss curves. Choosing an aggregate curve is a new descriptive representation. | (A) Canonical per-Q records and separate Q1/Q2 current/fresh curves; preserves heterogeneity. (B) Add a precisely defined average curve alongside individual curves; can conceal Q imbalance and its order of averaging must be declared. (C) Recover from local records manually and explicitly accept the canonical plotting gap; fragile. Selecting a favorable Q or changing the trigger is not authorized. | A. Preserve existing per-Q measurements in canonical outputs and plot their saved curves. Keep the mean-of-two per-round L, IQM/bootstrap and trigger unchanged. No new aggregate learning curve is necessary. |
| Numerical-reference criteria | A semantic reference across execution graphs is distinct from exact fork fidelity. Changing its error metric/bound changes what defects the validation can miss, so it cannot be chosen just to remove a red test. | (A) Keep the gate red pending an independently justified oracle/bound. (B) Independently implemented forward/analytic derivative plus sliced per-Q JIT references with approved float32 error budget. (C) Change experiment execution/dtype to force eager equality; changes numerical behavior and cost. A production-JIT expected value alone risks a shared-bug oracle. | Investigative option A now, then B after approval and CUDA characterization. Preserve all exact original/control/identity requirements and the existing injected 64-eps rule. Do not infer a universal bound from this tiny CPU example. |
| Measured CUDA bounds | Four diagnostics on/off isolation thresholds are None: update/training × TF32/highest-deterministic. For each supplied array pair, the helper converts to float64, divides max absolute difference by that reference array's max magnitude (zero-scale fallback is absolute difference), then takes the maximum across pairs. Integer/RNG leaves are included after conversion. This is not componentwise relative error, a return-effect threshold or Check 1's bound. | (A) Approve/reject each measured value under a declared rule. The log's 10× maximum, one-significant-digit rule was agent-proposed, not lead-approved. (B) Require diagnostics to preserve the same update graph/exact trajectory and investigate discrepancies instead of permitting a bound. (C) Defer the grid. Loose bounds admit more diagnostic perturbation; zero bounds can reject benign compiler rounding. Mode changes could alter experiment behavior and need a separate decision. | Obtain final-tree, both-critic-path, repeated same-GPU measurements with differences identified. Review magnitude, mechanism and the helper's scope limitations below, then approve explicit bounds/rule and rerun gates. No threshold values can responsibly be recommended without those measurements. Keep experiment TF32 and local Check 1 FP32 fixed. None/skip remains unresolved. |

CPU semantic-reference errors cannot set A100 TF32 diagnostic bounds. A bound for tiny
diagnostic tests cannot be asserted to prove full-run actor/return equivalence. If final-tree
measurements exceed an approved bound, investigate and consult rather than recomputing a
larger bound automatically. If only deterministic ops pass, the run-mode decision belongs
to the lead and must be propagated to the protocol and identity validation.

**Additional measured-bound scope finding.** The helper is not a complete state-validity
gate. Update tests pass agent/optimizer/RNG leaves and common info arrays; the GPU training
test measures only the saved agent tree and reports a count of other state differences as
context, without asserting that count is zero. The helper casts before checking dtype, so
integer leaves are normalized too. NaNs can hide discrepancies: an unchanged call to the
helper returns 0 for `[2, NaN]` versus `[1, NaN]`, and for `[NaN]` versus `[1]`, because
the NaN maximum never increases the initialized worst value. A CPU-only synthetic probe
confirms these cases and returns 0.1 for integer `[11]` versus `[10]`:
[cuda_metric_scope.json](https://github.com/kaves29/scaling-drl-research/blob/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs/exp12_followup_evidence/cuda_metric_scope.json).
Some update info intentionally contains cadence NaNs, so requiring all values finite would
also be incorrect. Before treating a measured bound as acceptance, recommend an explicitly
reviewed applicability/NaN-mask check and exact discrete-state checks, with the scalar float
metric's scope stated separately. That changes validation criteria and needs lead approval;
neither the helper nor any assertion was changed here. The finding does not establish any
actual CUDA state corruption. The first helper probe accidentally omitted explicit CPU
selection and aborted in the installed experimental Metal plugin; the successful rerun
explicitly selected CPU and is the sole evidence for these helper values.

## 6. Updated gates and CUDA work

| Gate | Current evidence/status | Remaining condition |
|---|---|---|
| Audit and approved original/control repair | Complete; 203 mapped methodology items; guard/regressions and fixture isolation in HEAD; ratios for both HB tasks recorded | Preserve in the validated CUDA revision |
| Exp12 CPU mechanics | Prior full run: 149 tests, 146 pass, 1 fail, 2 HB skips; tiny single/twin pipeline, restore, probe/injection/optimizer mechanics verified within audited scope | Resolve twin oracle under approved rule; missing HB coverage stays explicit |
| Twin panel / later gradient | RED; two fresh original repeats, historical introduction repeat, all exact differences recorded; evidence supports rounding/reference assumption | Lead-approved semantic oracle/criterion; then restored mutant must pass |
| Older Baoding exact restore | RED; historical real omission, counter-reset causal diagnostic exact; separate Exp12 restore short probes exact | Lead decision before replay-changing fix; not a direct selected Exp12 task/code-path blocker |
| Mutation checks | Prior run: 72 cases, 71 OK, one restored twin min-gradient case PROBLEM | Genuine semantic failure for mutation and passing restored case; final-tree CUDA checks |
| Analysis completeness | HIGH-severity validation gap confirmed by synthetic counterexamples; design only | Approve rule, implement validator and adversarial tests; no silent available-case analysis |
| Twin canonical reporting / descriptive curves | Incomplete; raw Q-specific data retained | Lead chooses representation; approved repair and tests |
| HB environment / twins on CUDA | Not established locally; historical Block B predates twins | Real HB GPU integration, Reach RNG/eval restoration, twin correctness and full-size gates |
| Exact identity and full-size Check 1 | Tiny CPU boundary evidence; no final-tree CUDA evidence | Both scaled sizes × all suites on grid model; original/control exact, identity exact for 1,000 steps; injection within existing bound |
| Diagnostic CUDA bounds / precision | Four None thresholds; historical A0 only | Final-tree measured deviations, approved bounds, passing reruns without hidden skips; expected FP32/TF32 IR and measured modes |
| Positive control / m / calibration | Mechanics CPU-tested; Block B outcomes not inspected | Natural D4W1536 dog-run trigger at dev seed outside 1–5, prescribed noise/recovery/sensitivity, m frozen by lead; range/null review |
| Cache, job status, GPU model and resources | Cache risk audited; HB aggregate exit gate absent; global one-model check absent; performance/disk projections not final measurements | Qualified cache layout preserving identity scenarios; all statuses fail closed; approved concurrency/time/disk and one-model manifests |

### What Block B has and has not established

The repo documents submission `22706349` at `b4a90cb`, before `c293b9a` twin support
and before the approved original/control guard. No raw Block B status/results were available
locally or fetched during this investigation. Scheduler completion, positive-control outcome,
chosen m, numerical deviations, null/range/identity results and packing measurements are
therefore **unverified here**. A documented submission or reported cache warning is not a
scientific PASS. Historical Block A's A100 precision/range numbers are separate evidence;
its largest architecture was D6W1536 and cannot validate current D4W1536/twins.

After raw-result review, B could establish development calibration, single-Q DMC/MyoSuite
CUDA mechanics, contemporaneous cold/warm identity, packing, speed, range/null and diagnostic
measurements for that old revision. It cannot establish final-tree twin/HB correctness,
the new boundary guard on CUDA, complete final analysis protection, or approved final
diagnostic bounds merely because it recorded numbers. Its `m=half` arm is development,
not evidence that the confirmatory m was selected/frozen. Leave its running/submitted checkout
untouched; final-tree evidence must come from a separate approved checkout/job.

### What Block HB must establish, and present script limits

Required HB qualification: actual CUDA backend in the cloned compatible HB environment,
correct EGL integration, HB pipeline and Reach goal/evaluation RNG restoration without
skips, twin SAC/probe/injection/Check 1 correctness, both scaled sizes' exact 1,000-step
HB identity forks, and measured twin speed/probe cost/peak memory for all three sizes.
Final-tree boundary guard and metadata must be exercised. HB isolation measurements must
be reviewed alongside the four diagnostic tolerances, not counted as PASS when skipped.

The current `sbatch_exp12_hb.sh` runs HB pipeline/Reach tests plus the twin test module,
six identity cells (two scaled sizes × dog-run, myo-key-turn, h1-run-v0), and h1-run-v0
profiling at three sizes. Limits:

- No explicit cold/warm identity split or deterministic-mode suite.
- No h1-reach full-size identity cell; Reach has a dedicated seeding test.
- Full-size identity is not full-size **injected** Check 1; injection tests in the module
  use tiny networks.
- It does not run the whole Exp12 CUDA suite/break checks or close all Block B gates.
- Step failures accumulate in status.tsv, but the final tee pipeline can still exit zero.
- The four-hour budget and approximately 1.5-hour estimate are not measured validation.

Recommend explicit complementary CUDA coverage for missing modes/full-size injection,
and h1-reach identity as additional protection for the second HB task. Expanding the
HB measurement/cache scenarios or adding new inclusion gates needs the lead's approved
plan; no such extension was implemented. Cold/warm cases required by the existing B
validation plan still need final-tree evidence, whether in HB or a complementary job.

### Exact conditions before recommending Block HB submission

1. Lead approves the CPU numerical-reference resolution, or explicitly directs a
   characterization-only job with the known red gate recorded and no validation-PASS claim.
   Default recommendation is to resolve the relevant CPU oracle and restored mutation first.
   The unrelated older Baoding failure remains disclosed; repairing it is not necessary to
   run the selected Exp12 HB path.
2. Pin a separate, reviewable validated revision containing the boundary guard and approved
   changes. Supply EXPECTED_COMMIT; verify the intended checkout, no testing-hook overrides,
   dependency/driver/GPU identity and cloned HB environment. Main environment remains unchanged;
   the submitted B checkout stays untouched.
3. Review and approve a concrete job coverage plan: required HB tests/identity/profiling,
   which missing full-size injected/mode/cache/task cases are included or scheduled separately,
   and whether this is measurement-only for unset bounds. Do not count None/skips as passes.
4. Make result accounting fail closed: nonzero for any failed/timed-out required step,
   reject missing status/identity JSON/backend evidence and unapproved skips, and inspect
   per-step logs even when Slurm reports COMPLETED. Operational shell repair/tests can be
   made in a future authorized implementation turn; none was made now.
5. Qualify cache configuration with isolated active writers while preserving the approved
   cold/warm identity measurements. Review per-step timeout/resource feasibility, reserve
   appropriate A100 resources/disk, and retain complete logs/metadata. A cache-warning fallback
   is not a correctness pass or a justification to alter the identity experiment silently.
6. Explicit user authorization to submit the reviewed job. There is none in this turn.

HB is a qualification/measurement job: approving final CUDA diagnostic numbers is not
logically possible before collecting them. Likewise a frozen PC m is not required for
forced-trigger identity validation. It **is** required before confirmatory runs; if injected
full-size checks use a provisional m, validate the eventual frozen choice before the grid.
B raw results should be reviewed for relevant environmental failures; B's natural trigger
is not a logical prerequisite for an HB identity-only characterization job.

### Exact conditions before recommending the full Experiment 1/2 grid

All of the following are conjunctive, with any existing stop/consult rule obeyed:

1. Lead approves the inclusion/reporting/reference policies; the approved completeness
   validator, canonical twin reporting and oracle repairs are implemented and their meaningful
   tests pass. The Exp12 CPU suite and mutation restores pass, with missing HB cases supplied
   by actual HB qualification. The unrelated older Baoding red status is explicitly scoped,
   never presented as a clean whole-repository suite.
2. Review final-tree CUDA tests in default TF32 and highest/deterministic diagnostic modes,
   with actual GPU evidence distinguished from CPU-forced subprocesses. Every relevant
   assertion/mutation passes; no missing HB or None-tolerance skip substitutes for acceptance.
   Approve all four measured diagnostic bounds and rerun them; resolve the helper's NaN-mask,
   discrete-state and state-coverage limitations under an approved validation contract, and
   consult on any GPU-only failure.
3. Both scaled architectures × all three suites pass exact original/control and 1,000-step
   identity gates on the selected GPU model, with required cold/warm cache cases accounted
   for. Full-size single/twin injected Q and min-action gradients pass the existing 64-eps
   Check 1 rule in local highest precision for the frozen m; optimizer/target invariants pass.
   Any identity or deterministic-only outcome requiring a decision is resolved by the lead.
4. Review full-settings natural D4W1536 dog-run positive control at a development seed outside
   1–5: L_trigger>0, noise<0.10, recoveries for last/half/all and shared-offset sensitivity
   reported. Apply the prescribed smallest-within-0.10 rule and have the lead freeze m before
   the first confirmatory Exp1 run. No trigger or stop outcome means consult, not retune.
5. Complete current-size configured-pool range and fresh-pair null evidence: P/b>=0.9 under
   the existing range rule, all planned 100-pair/size null records, and lead decision if the
   per-check null firing rate exceeds 5% at any size. Never automatically adopt the candidate
   p95 threshold. Historical D6 figures do not cover D4; absent twin-null coverage stays
   explicit and any auxiliary extension requires an approved plan.
6. Direct production-size CUDA checkpoint/resume/preflight passes at all three sizes,
   including scaled forks; HB stack, both task RNG/evaluation behavior and twin resource
   measurements are qualified. TF32 run setting, highest Check 1, no unintended mixed
   precision/x64 and correct versions/device provenance are verified.
7. Use measured packing, speed, probe overhead, fork save/restore/evaluation and disk sizes
   to approve concurrency, time limits and storage for current architectures/suites/twins.
   Keep PREALLOCATE=false and no memory fraction; never shrink prescribed probes to meet
   resources. Qualify cache/multiprocess/restart behavior on the final execution configuration.
8. Freeze the 195-run manifest and 130-parent eligibility census, study config/source and
   dependency provenance, chosen GPU model and frozen m. Enforce one model grid-wide and
   same-model source/arms; verify absolute paths, disjoint run directories, no duplicate
   launch/resume claims, unique job logs, overlap check and credentials/available disk.
9. Fail-closed job/result accounting and analysis publication are operational; outstanding
   relevant Block B/HB failures/skips have documented dispositions and required reruns.
10. Lead explicitly authorizes the full reviewed grid submission. No such authorization is
    inferred from permission to audit or to submit a separate validation job.

**Current recommendation:** do not submit Block HB yet under the default qualification path,
and do not launch the full grid. This investigation is complete; code fixes, methodology
adoption and all job submissions remain pending lead decisions/authorization.
