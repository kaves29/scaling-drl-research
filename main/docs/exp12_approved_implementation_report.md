# Exp1/2 approved implementation and remaining gates

2026-10-06, following the lead's attached decisions. This report supersedes the
implementation/status portions of [the preceding investigation](exp12_followup_investigation.md).
The historical failure measurements and baseline investigation there remain evidence.
No Delta checkout was modified, no job was submitted, and nothing was merged or pushed.

## Implemented scope

- **Angle 2A only:** capture and restore MyoSuite's optional Baoding `counter`;
  reject an older capture missing that required counter before modifying physics.
  Original exact assertions remain. Added zero-counter, repeated replay and seeded
  fresh-instance replay coverage. Exp12's reset/action-replay implementation is unchanged.
- **Confirmatory publication:** both analysis entry points now require an independent
  frozen study manifest, validate the prescribed population and saved evidence, write
  into a temporary output directory, recheck input hashes/directory listings, and publish
  outputs plus `validation.json` atomically. Existing nonempty output directories are
  refused. Invalid/incomplete inputs produce a failure report with census/errors and no
  validated output directory. The estimators, bootstrap procedures, primary inclusion
  of Check 2 failures, episode means, trigger and training execution are unchanged.
- **Twin reporting:** canonical Exp1 checks retain all existing Q1/Q2 scores and losses;
  legacy CSVs remain readable as uncertified progress data. Learning-curve plots show
  separate current/fresh Q1/Q2 curves. Exp2 retains per-Q fields and additionally plots
  per-Q post-fork plasticity trajectories where present. The approved mean-per-round L
  and its trigger remain unchanged.
- **Independent numerical characterization:** test-only NumPy float64 forward and
  analytic action Jacobian, with primitive/finite-difference arithmetic checks and a
  standalone characterization script. No existing numerical assertion or tolerance
  was changed; exact identities and injected Check 1 are untouched.
- **HB hardening:** explicit clean revision and GPU-model requirements; fail on setup,
  activation, backend, cloned-package or pinned HB-source mismatch; enforce EGL settings;
  prohibit local hooks in Slurm jobs; isolate ordinary test/profile caches under the job;
  reject reused status files; collect every required step; require matching outputs and
  1,000-step identity endpoints; reject failures, timeouts, missing data and unresolved
  skips through the final exit code. No CUDA workload or identity cache scenario was added.

## Exact completeness contract

Let N = raw environment budget / 2 interaction steps. Budgets are 500,000 raw steps
for hopper-hop/swimmer-swimmer15, 1,000,000 for other DMC/MyoSuite tasks, and 2,000,000
for HumanoidBench. The independent architecture set is D2W512, D4W1024, D4W1536;
seeds are exactly 1–5. Environments are exactly:

dog-run, dog-trot, humanoid-run, humanoid-walk, humanoid-stand, swimmer-swimmer15,
hopper-hop, myo-key-turn, myo-pen-twirl, myo-pose-hard, myo-reach, h1-run-v0, h1-reach-v0.

**Exp1:** all 195 parents, exactly one canonical run record each, complete nominal
budget, fresh check at interaction step 5,000, scheduled checks 1–20 at kN/20,
all required five-round evidence and 5×1,000 learning curves, nominal evaluations,
and logging windows including step 0. The primary endpoint must equal check 20,
even if a late fork extends the control beyond N. Terminal source records must
agree exactly with the ledger under its unchanged CSV serialization/parser;
initial score, round loss, IQMs, intervals and f* are recomputed from saved records.
Required fresh-reference/terminal state must exist. Eligible scaled parents also
need the immutable fork state and original/restored-control Check 1 evidence.
Exp1 certification does not require injected arms to have completed.

**Exp2:** independently account for all 130 scaled candidate parents, without
inferring candidates from existing fork directories. Each is `eligible_trigger`,
`valid_no_trigger` or `invalid_incomplete`; invalid/incomplete prevents publication.
Every eligible parent requires both control and injected arms, a consistent
fork plan/state, finite FP32 panel evidence and recomputed existing Check 1 criteria,
Check 2 scores/curves/interval record, required terminal/probe/logging evidence,
and exactly 26 evaluations × 10 episodes per arm. Evaluation j is at f* + jN/100,
j=0…25, episode IDs 0…9. Duplicate/missing/extra rows, wrong indices/times/arms,
nonfinite required returns or incompatible provenance fail. Both arms' exported
scientific records must match their terminal saved evidence. A Check 2 **failure**
is retained in primary analysis; its success-only subset remains secondary.

`candidate_census.csv` preserves all 130 candidates. `eligibility_by_environment.csv`
shows all 26 scaled architecture/environment groups and explicit `zero_eligible`;
zero forks produces empty paired tables and a census, rather than disappearing.
A one-seed primary or success-only secondary group currently stops publication
with `ReportingDecisionRequired`, pending the lead's uncertainty convention.
This is a pending reporting decision, not scientific exclusion of that seed.
Recommendation for approval: keep its trajectory, label n=1, omit the bootstrap
band and state that between-seed uncertainty is not estimable. Alternatives are
an explicitly labeled degenerate bootstrap band or leaving that report unpublished.
Apply the chosen convention consistently to primary and secondary groups.

**Preserved applicability:** recorded invalid intermediate probe checks remain under
B5 and cannot fire the trigger. They are different from absent checks. Required
initial/final endpoints must be finite. Early episode termination is retained;
positive integer episode lengths up to the existing task limit are accepted.
Routine checkpoint history is deliberately not required: terminal, fresh-reference,
and applicable immutable fork evidence suffice. No shortened horizon, available-case
analysis, episode deletion or imputation is used for certification.

**Provenance:** the separately frozen manifest records source revision, checkpoint
root, GPU model, frozen m, per-suite simulator/JAX/CUDA-stack versions and all 39
resolved configuration templates. Their key population and settings are checked
against the approved grid builder; observed outputs cannot shrink that expectation.
Each run/resume must match it, including TF32 launch precision and single/twin count.
Metadata hashes and source pointers are checked. Freeze/retain this manifest before
the grid; it is a reviewed study contract, not something to reconstruct from whichever
jobs finished. This is a completeness/provenance check, not authentication against a
malicious manifest author or reexecution of every saved scientific computation.

Required state payloads are checked for existence; the certificate hashes the
scientific records/evidence it consumes, not every multi-GB replay/Orbax payload.
Physical state fidelity is established by the separate restoration/identity gates.

**Progress:** `--exploratory` is the separate, explicit path for partial/dev/custom
summaries. Its marker is `EXPLORATORY.json`, `validated_confirmatory=false`; it never
writes `validation.json`. Raw loaders and private writers are uncertified utilities,
not a way to obtain a certificate. Do not present progress figures as confirmatory.

Example interface (do not execute until the study contract is reviewed):

```bash
python scripts/freeze_exp12_analysis_manifest.py \
  --checkpoint-root /abs/grid --expected-commit REVIEWED_FINAL_HASH \
  --gpu-model 'REVIEWED_DEVICE_KIND' --injection-m FROZEN_M \
  --runtime-json /abs/independent_runtime_by_suite.json --out /abs/study.json
python -m analysis.exp1_analysis --results-root /abs/results \
  --study-manifest /abs/study.json --out /abs/new_exp1_analysis
python -m analysis.exp2_analysis --results-root /abs/results \
  --study-manifest /abs/study.json --out /abs/new_exp2_analysis
```

The runtime JSON has dmc/myosuite/humanoid_bench keys, each with
`simulator_versions`, `jax`, `jaxlib`, `backend_platform_version`,
`jax_cuda12_plugin`, and `jax_cuda12_pjrt`, from the intended independently
qualified stacks. m/GPU model/runtime values are explicit inputs, not inferred
from incomplete results or chosen by this implementation.

Regression coverage includes whole missing seeds/environments/architecture,
manifest shrinking and changed probe settings, stale endpoints, duplicate records,
missing checks/windows/source evidence, mismatched runtime, changed original panel,
incomplete/wrong per-Q curves, missing forks/arms/Check 2 evidence, bad episode
counts/IDs/times/returns/lengths, source-record disagreement, input changes during
publication, explicit exploratory markers, zero eligibility, n=1 stop, Check 2
failure retained in primary, legitimate B5 invalids/early termination/pruning,
and the last eligible check 19 keeping a full 25%-budget horizon beyond N.

## Twin numerical proposal: approval required

Original failure values/baselines remain in the preceding investigation. The Q
assertion requires exact compiled/eager equality and fails; its later gradient
assertion also fails independently at three entries under unchanged rtol=1e-5,
atol=1e-7. These are graph-dependent float32 rounding differences in the observed
CPU case, not evidence that the panel computes the wrong Q/min gradient.
No change to the criterion has been made.

The [new characterization](exp12_approved_evidence/twin_numpy_characterization.json)
uses seeds 0–4 at D2W8, D1W16 and D4W32, with 256 states, 6 observation inputs,
3 actions uniform in [-1,1], including the exact original failing fixture.
Parameters/inputs are the same stored float32 values, evaluated independently in
NumPy float64. It reports full error quantiles, absolute and relative differences,
eager/JIT differences, per-Q errors, minimum Q gaps and semantic mutations.

| Quantity | Maximum observed error against independent reference |
|---|---:|
| Q, eager or JIT, whole-panel normalized | 4.276683 eps |
| Q, eager or JIT, per-network normalized | 5.999878 eps |
| min-action gradient, eager | 8.433310 eps; 3.173065e-6 absolute |
| min-action gradient, JIT | 7.799646 eps; 2.934646e-6 absolute |
| Independent analytic Jacobian vs centered FD, h=1e-6 | 1.740040e-9 absolute |
| Minimum independent Q1/Q2 gap | 8.906708e-4; zero min-selection disagreements |

Maximum eager/JIT difference across cases is 1.251698e-6 for Q and 1.432375e-6
for the action gradient. Elementwise relative error near zero can be much larger
than whole-panel normalized error; it is reported rather than hidden.

**Proposed CPU unit-test criterion, not adopted:** for each Q separately and the
min-action-gradient array separately, require finite arrays and

`max(abs(float64(got) - reference64)) <= 64 * eps_float32 * max(abs(reference64))`.

A zero reference scale requires exact zero error. Compare both eager and production
JIT to this independent semantic oracle; retain eager/JIT differences as diagnostics.
Do not mask or discard inconvenient states/rounds. Keep exact original/control and
identity comparisons exact and injected Check 1 at its existing 64-eps rule.

Derivation: maximum legitimate observed normalized error = 8.43331 eps;
four times that is 33.73324 eps; next power of two is 64. The resulting empirical
headroom is 7.589× the maximum, not a formal worst-case bound. Its numerical value
happens to equal injected Check 1's bound; the derivation and purpose are separate.
This proposal applies to these CPU unit-test fixtures, **not** to A100 diagnostic
bounds or arbitrary full-size/trained networks. CUDA semantic-oracle behavior must
be characterized and reviewed; do not automatically widen the bound on a failure.
Near a ReLU/min boundary, investigate branch changes rather than excluding data.

Minimum semantic mutation margins, each over all 15 cases, divided by this proposed
bound (Q margins conservatively use the whole-panel scale):

| Mutation | Smallest exceedance of proposed bound |
|---|---:|
| Q1 action gradient replacing min gradient (existing break) | 45,080× |
| Q2 gradient replacing min gradient | 42,731× |
| max gradient | 122,924× |
| mean gradient | 61,462× |
| zero gradient | 131,072× |
| swapped Q order | 116,371× |
| duplicated Q1 | 116,371× |

These separations support the proposal against the tested semantic defects; they
do not prove detection of every possible bug. The original numerical test and its
restored mutation therefore remain red until the lead approves a criterion.
The harness's mutated failure alone does not establish min-gradient detection:
its first Q assertion already fails in the restored implementation. The independent
characterization above supplies the separate gradient-mutation evidence.

## CUDA helper correction: design only, approval required

The current helper casts every leaf to float64 before checking dtype and uses zip.
It therefore also normalizes integer/RNG state, can truncate mismatched leaf lists,
and lets a NaN maximum mask actual differences. Examples `[2,NaN]` versus `[1,NaN]`
and `[NaN]` versus `[1]` both return zero. No helper body, measurement or GPU bound
was changed in this work.

Proposed correction for review:

1. Match tree paths/structure, leaf counts, shapes and original dtypes first; refuse
   mismatch before arithmetic. Preserve leaf-level names in the report.
2. Compare integer/bool/counter/RNG/discrete state exactly. Never place keys or
   optimizer integer counters inside a floating tolerance.
3. Only floating leaves receive a normalized finite-value deviation, retaining
   the existing per-leaf reference-max/zero-reference absolute fallback unless a
   different metric is explicitly approved. Report max absolute error too.
4. Permit matching NaN masks only in an explicit whitelist of existing cadence/
   unavailable-diagnostic outputs (e.g. unsampled actor cosine). Compare all finite
   entries even when an allowed NaN exists elsewhere. A mismatched mask, unexpected
   NaN in a learned state/required finite quantity, or infinity fails independently
   of the scalar deviation. Do not use blanket nanmax or blanket equal_nan.
5. Treat continuous drift and discrete/invalid-state failures as separate results;
   a small drift scalar cannot override a discrete failure.

The whitelist and any coverage expansion change acceptance and need lead approval.
Tests should include masked finite changes, one-sided/all-NaN masks, finite↔Inf,
wrong structure/shape/dtype, extra leaves, one-bit RNG/counter changes, allowed
cadence NaNs with equal finite entries, and a zero reference scale.

| Existing validation | Compared | Important omissions/limits |
|---|---|---|
| Single/twin diagnostic update | Actor, critic, target, temperature TrainStates (weights/optimizer/counts), agent key, common info arrays; four updates | Current scalar mishandles NaNs/discrete state; paths/list lengths not checked; no environments/replay/obs_rms |
| Diagnostic training | Agent checkpoint tree, 400 interaction steps; existing metadata-difference count is context | Non-agent differences are not acceptance failures in the GPU return path; no continuous bound on replay, obs_rms, physics, meters/logs |
| 1,000-step identity-fork script | Agent leaves; obs_rms; filled replay arrays; train/eval reset RNG/action-replay state, observations/timestep, global RNGs, update/interaction counters, meters/media/logs, actor diagnostics/window buffers; selected extra_state; exact Check 1 panels | `buffer_meta.pkl` (replay cursor/count/n-step queue) is not compared; same-length renamed agent paths and extra right-side dictionary keys can escape structural comparison; unrecognized extra_state keys ignored. WandB ID intentionally differs |

Exp12 restores simulator state by resetting with saved RNG and replaying actions,
not by serializing every live MuJoCo datum. The prescribed 1,000-step gate is a
finite trajectory test; it is not a proof of all future trajectories, driver
behavior, every task/seed or every possible state. Expand exact state accounting
(including buffer metadata and symmetric paths/keys) only after the lead approves
this correction. Do not describe the current comparator as exhaustive state identity.

The four unset bounds are `diagnostics_update/tf32`,
`diagnostics_update/highest_deterministic`, `diagnostics_training/tf32`, and
`diagnostics_training/highest_deterministic`. In the current tests TF32 compares
**diagnostics-on TF32 against diagnostics-off highest**, mixing precision and
instrumentation differences. Highest/deterministic compares both at highest.
A small TF32 deviation is not an isolated causal estimate of diagnostic influence.
The comment's historical 10×/one-significant-digit rule remains unapproved.

Recommendation: approve the helper/mask/discrete-state contract first; collect
repeated final-tree A100 measurements in both modes, single and twin update paths,
then approve four explicit bounds or a declared rule. Include twin **training**
isolation, which is currently absent. If pure diagnostic isolation is intended,
add same-precision on/off comparisons and report cross-precision behavior separately;
that changes the measured comparison and requires approval. Size/seed/length scope
must be explicit; the current tiny four-update/400-step fixtures cannot establish
universal full-size/long-horizon bounds. No numbers can be selected from CPU evidence.

## Block HB: exact current coverage and omissions

The hardened script still runs the existing workloads:

| Required step | Coverage |
|---|---|
| HB tests, HB cloned Python | Real h1-reach pipeline; dedicated h1-reach evaluation/global NumPy/Python/agent RNG save/restore test; complete twin test module, including tiny twin SAC, probes, injection, Checks 1/2 and identity |
| Six identity cells | D4W1024 and D4W1536 × dog-run/main env, myo-key-turn/main env, h1-run-v0/HB env; seed 101, reduced 240,000 raw-step budget, forced development trigger at check 2; exact prescribed 1,000 further interaction steps |
| Three speed/memory records | h1-run-v0 twin D2W512, D4W1024, D4W1536; 600 timed training steps, 100 warm-up steps, one probe repeat; time/probe overhead/peak GPU bytes |

Before workloads: clean literal EXPECTED_COMMIT, explicit EXPECTED_GPU_MODEL checked
against JAX GPU devices in both envs, independent cloned prefixes, all main package
versions preserved in HB, pinned clean HB commit cb1189039151c8aadaaa987b442da54383c87fab,
EGL variables and captured package/device provenance. Checks run after actual activation;
no environment was modified by this local work. Cluster values remain unverified until
an authorized job checks them. The script resolves its own checkout, so a separate
reviewed checkout need not redirect into the submitted Block B checkout.

Ordinary test and profile processes have private job/step caches. **Identity cache
behavior is intentionally unchanged**: it still inherits the caller/default cache
setting. Thus the complete HB cache plan is not yet qualified. Recommend the already
prescribed explicit cold-per-process and warm sequentially shared identity scenarios
in an isolated job/cell namespace, with hit/miss evidence and no concurrent writable
cache sharing. Do not silently substitute this plan or change the original B job.

Omissions that still require an approved CUDA plan:

- Explicit cold/warm identity split and highest/deterministic suite; inherited XLA_FLAGS
  must be reviewed rather than accidentally selecting a mode.
- Full-size **injected** Check 1 for both scaled sizes on single/twin paths (and eventual
  frozen m); identity-only full-size runs cannot establish injection preservation.
- h1-reach full-size identity; present Reach pipeline/RNG coverage is tiny. Recommend
  this complementary task case because its goal randomization differs from h1-run.
- Single-Q diagnostic tests in HB's selected test step, diagnostic training measurements,
  twin training isolation, repeated measurements and approved NaN/discrete-state comparator.
  HB's twin update test can record the default-mode update quantity only, and skips when
  its bound is unset. No four-bound qualification follows from it.
- Full Exp12 CUDA suite and mutation run, current-size range/null/positive-control,
  packing/concurrency/disk and per-suite cost/resume preflight remain separate gates.

No expensive case was added. Needed repetitions/modes/full-size injections are
additional CUDA work to approve, not results established by this script review.
The four-hour request and performance projections remain estimates.
With current numerical red/unset bounds the script cannot truthfully return a
qualification PASS, even if all other workloads succeed; a measurement-only job
would need explicit lead direction and a separate interpretation.

## Updated gates and launch conditions

| Gate | Current status |
|---|---|
| Older Baoding restore | Fixed and CPU verified, original assertions exact; all 262 older tests pass |
| Completeness/twin reporting | Implemented; synthetic full-population and break coverage CPU verified; n=1 uncertainty pending |
| Twin semantic numerical gate | RED under unchanged existing criterion; independent CPU evidence complete, proposed 64-eps CPU criterion pending |
| Exp12 CPU/mutations | See exact validation counts below; known restored twin numerical failure remains red; HB skips remain explicit |
| CUDA diagnostic helper/bounds | RED/unresolved: design only; four bounds unset, masks/discrete scope needs lead decision |
| Final-tree CUDA exact identities/injected Check 1 | Still needs CUDA at both scaled sizes/all suites, prescribed modes/cache cases and selected model |
| HB environment/twin resources | Script hardened and CPU bookkeeping tested; actual cluster integration/cost/memory still unverified |
| Block B | Submission 22706349 at b4a90cb documented; raw status/results not reviewed here; cannot certify final twins/guard/analysis or freeze m |
| Positive control/range/null/packing/grid preflight | Mechanics tested; final-tree accepted evidence and frozen m still required |
| Submission authorization | None; HB and full grid remain unsubmitted |

Before recommending HB submission, conjunctively require: reviewed numerical criterion
and restored test/mutation passing **or explicit measurement-only authorization**;
a clean pinned separate final checkout with intended main/HB stacks and exact GPU model;
lead-approved disposition of missing injected/task/mode/cache cases; approved isolated
cold/warm identity plan; honest failure/skip/output accounting; feasible resources/storage;
and explicit authorization to submit. Actual GPU numbers cannot be approved before
measurement. Positive-control m need not be frozen for identity-only qualification,
but any provisional injected-m validation must be repeated for the eventual frozen choice.

Before the full grid, additionally require all relevant CPU/CUDA gates and mutations
passing with no unresolved skips; approved helper and four CUDA bounds with passing
reruns; exact original/control and 1,000-step identities for both scaled sizes × all
suites on the selected model, including the required cold/warm cases; full-size injected
Check 1 for frozen m; accepted natural D4W1536 dog-run positive control at a development
seed outside 1–5, prescribed recovery/noise/shared-offset checks and lead-frozen m;
current-size range/null evidence under existing stop rules; direct production-size
checkpoint/resume/preflight; measured cost/memory/packing/time/storage; reviewed independent
195-run manifest and 130-parent census/provenance; one GPU model and disjoint paths/claims;
operational result accounting; reviewed disposition of relevant B/HB failures; resolved
n=1 reporting; and explicit full-grid launch authorization. No observed failure permits
automatic retuning, a new inclusion rule or a larger bound.

## Validation evidence and commits

| Validation | Exact final result |
|---|---|
| Older suite, real guarded runner | 262 run, 262 pass, 0 fail/error/skip; 292.717 s |
| Exp12 full CPU suite, committed implementation | 180 run, 177 pass, 1 failure, 0 errors, 2 HB skips; 621.143 s |
| New Exp12 tests within that run | 31 pass: 19 completeness, 8 HB plumbing, 2 independent-oracle arithmetic, 2 twin reporting |
| Existing mutation harness | 72 checks: 71 OK, 1 PROBLEM; mutated failure and restored failure for twin min-gradient panel |
| Numerical characterization | 15 panels; both execution modes; independent reference, seven mutations/case; no adopted pass criterion |
| Formatting/syntax | Black check for 11 new Python files, bash syntax and git whitespace checks pass |

The sole Exp12 failure is `TwinCheck1Test.test_panel_values`, unchanged exact Q
comparison: 67/512 differing elements, max absolute difference 7.152557e-7.
The later gradient assertion's independently established three violations remain
in the historical evidence; no pass is inferred from its being reached after
an earlier assertion. The two skips require the actual HumanoidBench environment:
the real HB pipeline and Reach RNG/evaluation test. No CUDA execution occurred.
Some negative-fixture logs deliberately print FAIL; the unittest summary above
records actual test outcomes.

Raw logs, characterization and [per-commit file inventory](exp12_approved_evidence/commit_files.json)
are in [the approved evidence directory](exp12_approved_evidence/).
SHA256 inventories preserve both the 55 historical evidence files and the new
validation records. The 195/130 study tests use explicitly synthetic provenance;
they prove accounting behavior, not real CUDA production of a complete study.

Two implementation-stage errors were corrected without loosening a criterion:
zero-eligible empty-frame handling, and a toy development analysis call needing the
explicit exploratory path. Mutation string anchors were updated after the wrapper/body
split and empty-frame correction; all mutation meanings/assertions were retained.
The first older-suite runner used stdin and its spawned workers failed to reopen
`<stdin>`; rerunning from a guarded real file produced the clean 262-test result.
Black was installed into `/tmp/exp12-format` after sandbox DNS prevented downloading;
the research virtual environment and dependency pins were not modified.

| Commit | Purpose and files |
|---|---|
| `4e9ad64` | Angle2A `env_state.py` and its unit/exact determinism regression files |
| `7e482f0` | Independent NumPy oracle, arithmetic tests, characterization script |
| `fabd2c0` | HB Slurm driver, environment/status checkers and CPU qualification tests |
| `515a238` | Per-Q/eager-JIT characterization and preserved JSON evidence |
| `08f7474` | Canonical twin fields, individual learning/plasticity plots and reporting tests |
| `2dc8734` | Strict analysis validator/entry points, manifest freezer, source pointers, synthetic full-study fixture, 19 completeness tests and explicit exploratory toy fixtures; existing mutation anchors only |
| `8d4cf46` | Historical investigation, original failure/baseline arrays/logs and 55-file SHA256 inventory |
| Final documentation commit | This report and final CPU logs/inventories; no implementation changes |

The reviewed executable/test tree is `2dc8734fc79c04d44d5072644e9eb45f8389f90f`.
Subsequent report/evidence commits do not change experiment, analysis, validation or configuration source.

## Submission command for review only

No command below has been executed. It must run from the separate final clean checkout,
after the conditions above and explicit authorization. The known cloned-env path is
from repository documentation; its actual existence/stack remains to be verified on Delta.
GPU model is a required independent study choice, not auto-detected into acceptance.

```bash
cd /work/hdd/biqc/skaveti1/exp12-hb-final/main
mkdir -p logs
HB_ENV=/work/hdd/biqc/skaveti1/envs/exp12-hb \
EXPECTED_COMMIT=2dc8734fc79c04d44d5072644e9eb45f8389f90f \
EXPECTED_GPU_MODEL='NVIDIA A100-SXM4-40GB' \
sbatch scripts/sbatch_exp12_hb.sh
```

This pins the complete executable/test implementation commit. Subsequent audit/report
commits contain documentation/reproduction evidence only. The separate checkout path is proposed and
has not been created here. The GPU literal follows the documented Block A node model;
confirm that this is the intended grid model and exact JAX device_kind before authorizing
submission. A mismatch must stop, not auto-select another model.

Pending decisions to return to the lead: (1) n=1 uncertainty convention for primary
and secondary groups; (2) proposed CPU independent-reference criterion; (3) NaN/discrete/
state-coverage correction and whether to separate same-precision diagnostic isolation;
(4) concrete additional CUDA/cache/mode coverage; (5) four measured CUDA bounds after
actual measurement. No requested job submission is inferred from approval of code work.
