# EchoCritic forensic probe audit — 24 October 2026

Published as [draft PR13](https://github.com/kaves29/scaling-drl-research/pull/13).
Audit implementation/test commit: `30a9f6663ec21a55eca3266160418f2e2cf3983b`;
resolve the live PR head for documentation-only followups.

## Conclusions

**The large-critic DMC stability survives independent reconstruction.** Raw round
values reproduce both the contemporary comparison and the initial-reference
comparison. This establishes the observed measurement, not its scientific cause
or a general claim about retained plasticity. The available evidence does not
establish a campaign implementation defect that explains the pattern.

- **Network identities:** archived source selects the live trained critic and
  separately persisted fresh parameters. Nine synthetic end-to-end routes
  (three seeds at campaign, pilot and current revisions) preserve deliberately
  distinguishable predictions, checkpoint optimizer state, normalizer state and
  fresh-file hashes. Original production parameter checkpoints are absent;
  their hashes, LATEST selections and predictions cannot be certified here.
- **Tasks/preprocessing:** source and CPU controls agree on shared sampled
  inputs, current observation statistics, one normalization, teacher, base
  targets and minibatch indices within a pair. The protocol gives each critic
  its **own offset**, so final targets and initial difficulty need not match.
  Across checkpoints replay, normalization, keys and teacher change. Across
  architectures the teacher also changes. These are measurement confounds,
  not newly established coding errors.
- **Fitting:** original source/configuration uses AdamW LR `1e-4`, decay `.01`,
  fresh optimizer state per fit and one continuous 2,000-update scan. Independent
  one-update and CPU serialized-continuation controls pass. The later wrong
  retrospective optimizer is a separate defect already addressed in PR #8.
  The original A100 segmentation failure remains unresolved and unqualified.
- **Scores/plots:** all 4,834 raw probe points, including 96 fresh-only anchors,
  and 52,214 comparisons to supplied derived tables agree within explicit
  floating-point reference-arithmetic bounds. No score clipping or reversed
  paired subtraction was found in those paths. Bootstrap checks also agree.
- **Fresh deterioration:** at the common 70% checkpoint all 96 fresh scores are
  lower than their own initial scores. Source nonstationarity permits drift,
  but does not explain its consistent direction or size. The cause is
  **unresolved**, not demonstrated to be fresh-parameter training or a bug.
- **Larger DMC critics:** their trained scores are numerically closer to their
  initial anchors, while their contemporary fresh scores fall more. This
  separates two observed contributions. Capacity, centered output variation,
  changing task difficulty, conditioning and finite-budget optimization cannot
  be causally separated without the original inputs/parameters.

No production or frozen implementation, setting, numerical threshold or archived
output was changed. All candidate corrections below remain unapplied. No GPU
jobs, new parent training, production probe reruns, messages to other agents or
merges were performed. Synthetic fits are explicitly separate from original data.

## Evidence, provenance and limits

Audit base: PR #11 documentation head
`5880ec010e6ff185fcd8a9313d695ec29dda1305`; its tested production implementation
is `f8ad82f064547c76e12539804c98be75b32656d8`. Live GitHub inspection preserved
#8 `868cab4` → #9 `55166a6` → #10 `c05d77c` → #11 `5880ec0`.
PR #12 `66d4fe3` is a sibling of #11; it was not merged or overwritten.

| Input | SHA256 |
| --- | --- |
| `echocritic-comprehensive-20261024.zip` | `f7337130705a59622e1b639449e0dad452317fa2a45b479760a92aa3d9b12a44` |
| `critic_metric_comparison.zip` | `f5b11ca7223dcb43f4cb6aa469aa9e4e03e52dfdb1d2aa06085b98dfce282505` |
| Original Myo NPZ, small critic, seed 5, check 50 | `66a223890220be7aa9d022e9cdfbcd90909d6580132e20315492a0ff307839ca` |

The ZIPs were checked for unsafe member paths/symlinks and extracted outside Git.
Provided reports were treated as claims; supplied Python analysis and pickle
were not executed or deserialized. Independent readers use raw JSON and NPZ
`allow_pickle=False`. All **697** raw-file manifest hashes and **52** supplied
source snapshots match; the latter also match immutable Git blobs. The metric ZIP's additional seven source snapshots also match their hashes and Git blobs.

There are **103 unique run IDs**, 25,557 raw history rows, 96 campaign conditions
(4 environments × 3 architectures × 8 seeds), and seven pilot records. Recorded
commits agree with the source manifests: campaign
`f597ee9898e4d401912092c336c3e5330cf2fda1`, pilot
`f8619dd5f494399d9febc79fc9092010234c714d`. Seven duplicate display-name pairs
have distinct IDs but **7,989 identical shared finite training/evaluation
values each**. They are not independent seed replication; the audit excludes
pilot records from campaign aggregation. Identical observed trajectories alone
do not prove whether a record was rerun or reexported.

Eight campaign records crashed at fractions .78–.92. All 96 reach check 35/.70,
so the common70 comparison has eight seeds in every condition. Sixty-two late
scheduled checks are missing; no interpolation or completion was invented.
Recorded row counts equal metadata history counts. Explicit nulls are treated
as missing, not as numeric measurements. Pilot histories contain no probe
measurements, so historical five-round semantics are checked against source and
fixtures, not inferred from absent pilot probe arrays.

All records list the same `requirements.txt` size (2,350 bytes) and MD5
`F5Ehgu0UZUaIpzgvYIK0wA==`, but its bodies are absent. W&B metadata reports
CPython 3.12.13 and A100-SXM4-40GB; dirty-worktree status and complete installed
package versions are unrecorded. Matching Git snapshots do not prove a clean
runtime checkout. CPU controls use Python 3.12.14, JAX/JAXLIB .4.34,
Optax .2.3, float32 and source matmul settings. These are not exact GPU-runtime
reconstructions.

The requested `initial_reference_comparison/` directory and its PDF are absent
from both ZIPs. Its original script/PDF therefore cannot be audited directly.
The anchor and all available scalar comparisons are independently reconstructible
from raw histories and the supplied metric-alignment CSV; that is the evidence
used here. Original checkpoints, pools, normalizer snapshots, teacher weights,
indices and intermediate full-pool losses are absent.

See the [source-linked measurement map and metric definitions](forensic_probe_dataflow.md) for every transition and the DMC/Myo difference matrix.

## Reconstructed observations

Means over the same eight seeds at check 35 (70%); numbers rounded only for
this display. `P0` is each run's initial fresh score, not a same-task control.
`L` is the source IQM of paired round differences, averaged across seeds;
it is not generally equal to the difference between marginal score IQMs.

| Environment | Architecture | P0 | Trained | Fresh | Trained − P0 | L |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dog-run | D2W512 | .495211 | .367695 | .488917 | −.127517 | .120598 |
| dog-run | D4W1024 | .494277 | .492169 | .479586 | −.002107 | −.012484 |
| dog-run | D4W1536 | .495087 | .492613 | .462574 | −.002473 | −.029829 |
| humanoid-run | D2W512 | .494839 | .485186 | .462669 | −.009653 | −.022373 |
| humanoid-run | D4W1024 | .492836 | .489199 | .436833 | −.003637 | −.051968 |
| humanoid-run | D4W1536 | .490453 | .487996 | .337007 | −.002457 | −.151214 |
| myo-key-turn | D2W512 | .496346 | −.157765 | .434413 | −.654111 | .592313 |
| myo-key-turn | D4W1024 | .494356 | −.034953 | .424140 | −.529309 | .458922 |
| myo-key-turn | D4W1536 | .493662 | −.114481 | .350514 | −.608143 | .464284 |
| myo-pen-twirl | D2W512 | .496410 | −7.029489 | .406838 | −7.525899 | 7.433970 |
| myo-pen-twirl | D4W1024 | .492373 | −.362643 | .335362 | −.855016 | .697953 |
| myo-pen-twirl | D4W1536 | .489806 | −2.596413 | .243355 | −3.086220 | 2.846334 |

Machine-readable per-seed values retain full precision. Raw reconstruction and
linear-scale plots establish that symlog alone does not generate the DMC
pattern. Scores have a per-round upper bound `b` because MSE is nonnegative;
near-ceiling values compress differences. This is a property of the score,
not evidence of clipping. Different architectures need not have identical `b`.

The archived Myo seed-5/check-50 NPZ contains ten 2,000-point **pre-update
minibatch** loss curves and final **post-fit full-pool** MSEs. Scores are exactly
stored float32 `b − final_loss` and exactly match W&B round scores. Fresh/current
`b` arrays are identical. Median fresh/current offsets are −.428516/1998.706421;
median first minibatch losses .875613/164001.53125; median final pool losses
.142270/14.801420. A last minibatch loss is not the final score operand.
These arrays establish one real measurement's arithmetic and severe finite-budget
error difference. They cannot identify the fitted parameter hashes or reconstruct
initial/intermediate full-pool error, gradients, conditioning or GPU execution.

## Ranked issue ledger

IDs below are also the anomaly mapping. “Ruled out” always refers to the stated
source, records or fixtures, not every unobserved production execution.

| ID / classification | Claim, evidence and counterevidence | Affected scope / anomaly relevance | Smallest discriminator / result |
| --- | --- | --- | --- |
| L1 Confirmed logging defect | Contemporary `probe/score_fresh_iqm` is logged only at check 0; later round fields change. Supplied analysis reconstructs fresh IQM from ten rounds correctly. | Misuse of W&B summary can hide A; does **not** explain this analysis's A–F. | Reconstruct all 4,834 points: pass. Future logging proposal only; do not backfill archives. |
| L2 Confirmed analysis/reporting defect | Metric architecture summaries report nominal `n_seeds=8`, while late finite-value windows have effective n=7 for nine metrics in dog-large and key-large (18 rows). Their estimators filter missing values. | Late metric uncertainty/labels; cannot explain common70 B/D or paired-score plots. | Independently count effective seeds: 18 mismatched nominal labels. Report effective n per metric/window in an unapplied analysis patch. |
| L3 Confirmed future ingestion defect | PR11 grouping removes only top-level seed; actual configs also vary in coupled `agent.seed`/`env.seed`. Once provenance is adapted, eight replicates would remain separate groups. | Future PR11 aggregation, **not** supplied campaign scripts or A–F. | Two-record original-copy groups 2×n1; proposed-copy groups 1×n2; regression passes. |
| L4 Integration blocker | Native real export uses `attrs.commit`, `_step` and probe counters; PR11 expects a normalized source/campaign/counter schema. Missing dirty status cannot be fabricated. | Real CLI preserves 103 records but yields zero qualified groups; synthetic compatibility never established real compatibility. | Direct original CLI ingestion completed; reviewed adapter and explicit provenance policy remain required. |
| N1 Numerical discrepancy, relevance unverified | Float32 materialization `offset + base` can erase variation. Synthetic constant offset 2^24 yields source score .0625 versus centered-residual candidate 0 at zero LR. | Extreme-offset fixture only. Available actual offset ≈2,000; no evidence this explains A–F. | Separate candidate-copy regression reproduces artifact and correction. Need actual target/residual ULP measurements before adoption. |
| M1 Intended protocol / confound | Replay, normalizer, teacher/key change across checks, despite fixed fresh weights; teacher architecture changes across sizes. | Can permit A; confounds A/B/D/E/F. Does not predict direction of all 96 declines. | Same-check repeat bitwise; changed check differs. Original pools/teacher/norm needed to decompose cause. |
| M2 Intended optimizer / confound | Own offsets and AdamW decay make translated fits non-invariant even when initial errors match. | Plausible contribution to trained/fresh asymmetry C/E, not established DMC cause. | Zero versus 2,000 bias scalar controls have same initial error but different final errors under correct AdamW. No optimizer change proposed. |
| M3 Measurement interpretation | Historical DR aggregates Dense preactivations (excluding predictor), not every post-ReLU activation. Synthetic DR=0% coexists with expansion ReLU inactivity=100%. | Invalidates inference that low logged DR rules out dead nonlinear branches; does not establish their actual prevalence or explain A–F. | Original activations/parameters needed; source fixture passes at all three revisions. |
| M4 Measurement interpretation | SRank is uncentered 99%-singular-value-sum count on a batch, capped by batch 256, not stable rank; feature norm is after final LayerNorm. | Cross-width rank/norm interpretations require care, not direct probe corruption. | Definitions inspected; obtain pre/post-layer activations before mechanistic conclusions. |
| R1 Hypothesis contradicted in audited path | Wrong fresh parameters substituted for trained, dropped learned head, overwritten fresh or accidental probe mutation. Distinguishable outputs ≈3/5 versus 1 survive full route and restore; state hashes stable. | Does not explain source/fixture behavior. Actual production checkpoint identity remains unverified. | Nine routes pass; obtain production manifest/parameters for decisive campaign check. |
| R2 Hypothesis contradicted in audited arithmetic | Wrong sign, marginal-IQM subtraction, clipping, wrong repetition axis or stale-summary use generated plotted scores. | Does not explain reconstructed A–F. | 52,214 point comparisons pass; paired-vs-marginal nonadditivity exists (max .05249) but supplied analysis uses paired values correctly. |
| R3 Hypothesis contradicted | Wrong retrospective optimizer caused original campaign. Recorded original config/source is 1e-4/.01; later 4f10ca9/descendants use 3e-4/0. | Retrospective evidence only; does not establish original A–F cause. | Source/config provenance verified. Existing PR8 correction preserved. |
| R4 Hypothesis contradicted in source/fixtures | Double normalization, replay mutation, shared training RNG consumption, reset between original fit halves, zero LR/stale fitted closure. | Cannot explain the tested routes. Not a certificate for missing original inputs. | Raw replay warmup fixture, normalization/hash checks, one-update reference and 2,000-step continuation pass. |
| U1 Plausible untested mechanisms | Capacity, initial centered output variation/covariance, teacher difficulty, conditioning, AdamW offset effects and sine sensitivity. | B/D/E and possibly A; no identified mechanism predicts full pattern from available evidence. | Recover same-check pools, normalizer, teacher, current/fresh weights and indices; inspect predictions/residuals/gradients before new fits. |
| U2 GPU execution discrepancy | Earlier A100 segmented-versus-frozen final error .00405173 exceeds unchanged guard .000751385. Both internal paths used the same wrong retrospective optimizer. | Later reconstruction; original campaign fit is continuous, so causality for A–F unproven. | CPU continuation passes, GPU remains failed/unresolved; separately authorized matched-input A100 qualification required. |

Minimal unapplied `contemporary_score_logging.patch` and `late_effective_n.patch` proposals are provided for L1/L2, `coupled_seed_grouping.patch`
for L3, and `extreme_offset_centered_residual.patch` for N1. L1's minimal proposal
is adding contemporaneous marginal IQM log fields alongside existing round fields;
no historical rewrite. L4 requires a reviewed native-schema adapter that preserves
unknown provenance, not a permissive replacement of “unknown” with “clean”.

## Anomaly-specific interpretation and consequences

| Anomaly | Established evidence | What remains unresolved / consequences of proposals |
| --- | --- | --- |
| A Fresh declines | 96/96 at common70; fresh params unchanged in source fixtures; actual tasks change. | Task nonstationarity alone does not explain direction. L1 correction changes future logs, not reconstructed trend. N1 impact unknown. |
| B DMC trained near P0 | Raw and derived values agree, including all eight seeds at70. | P0 is cross-task, so this is numerical stability relative to an anchor, not matched-task invariance. |
| C Trained above contemporary fresh | Negative paired L is a valid conditional advantage on that checkpoint's protocol. | Not proof of general retained plasticity. Need task/difficulty decomposition, not sign correction. |
| D Larger DMC less deterioration | Trained scores closer to P0; contemporary fresh drift increases with size. | A pure moving-baseline explanation is insufficient because trained-to-P0 also differs. Changing teacher architecture and endogenous replay prevent causal size attribution. |
| E Myo deterioration | Reproduced; one original small-pen archive confirms large final trained MSE. | No monotone architecture rule; offset/centered-variance/conditioning hypotheses unresolved. One seed/check cannot establish all Myo dynamics. |
| F Initial anchor changes meaning | Distinct operands answer distinct comparisons; initial anchor is correctly reconstructed. | Neither anchor removes nonstationarity. Preserve both with paired uncertainty and observed-seed masks. |

L2 corrects labels/effective n, not the already finite-filtered estimators or
common70 results. L3 changes future PR11 group membership/CIs once ingestion
provenance is resolved; it does not alter the supplied independent analysis.
N1 changes floating-point residual evaluation and potentially historical numerical
comparability; it requires owner approval and GPU qualification and is **not**
a recommendation to eliminate the observed phenomenon. Existing PR10 terminal
success attribution changes success labels, not these score operands. PR12's
sparse-mask restore correction is not silently integrated; campaign configs have
zero sparsity, so that defect is not implicated by these records.

The strongest supported conclusion is that the unexpected score patterns are
real in the recorded measurement stream and are not explained by the tested
bookkeeping hypotheses. Their scientific interpretation remains conditional on
changing diagnostic tasks, finite fitting budgets and missing production state.
This audit neither proves general plasticity preservation nor proves a hidden
campaign implementation bug.

## Validation and reproducibility

Independent tools live under `main/audits/echocritic_forensic/`; production imports
none of them. Commands, archive identity and source snapshots are described in
[the audit README](../audits/echocritic_forensic/README.md). Outputs, original
ZIPs, copied Git snapshots and logs are outside Git under
`/workspace/artifacts/forensic-probe-audit-20261024/`.

- Raw reconstruction: **4,834** source-IQM checks; **52,214** derived-point checks;
  no mismatches; max absolute differences 1.78e-15 and 3.55e-15 respectively.
- Statistical reconstruction: **9,476** operational CI endpoint checks,
  **1,188** seed-aggregate checks, **3,456** metric-window checks; no mismatches.
  Trigger decisions use the exact logged strict lower-bound rule. Inferred
  consecutive-check schedules are not a reconstruction of unavailable disk ledgers.
- Focused deterministic pytest: **15 passed** (new audit controls plus existing
  task-bank identity tests). Proposal tests execute original and candidate code
  copies separately, demonstrate the original problem, and leave source unchanged.
- Original-source controls: campaign, pilot and current each pass scalar,
  deterministic fit, dormancy and three-seed end-to-end routing controls (nine
  routes total). Campaign final rerun also includes raw replay insertion through
  four synthetic warmup transitions with **zero SAC updates**.
- Scope static checks: Black, Pyflakes and compilation for added Python files;
  proposed patches checked for applicability only, never applied.
- PR11 production validation reused, not relabeled as a new audit execution:
  901 CPU tests/103 isolated modules, mutation74/74 twice, simulator65/65,
  supplementary pytest51 passed/one unavailable historical pilot-archive skip.
  The new uploaded NPZ is not that different archived replay fixture. No
  production bytes differ from the tested PR11 base in this audit branch.

New reader reference checks use a declared bound of 32 float64 epsilons times
value scale for different reduction arithmetic. They do not modify any source
scientific tolerance, trigger criterion or failed GPU acceptance test.

## Minimal next diagnostics

1. Obtain existing production trained/fresh checkpoint manifests and parameters,
   replay sample and normalizer snapshots for matched small/large DMC checks
   (plus one Myo comparison); hash run/seed/check/layer trees and evaluate both
   critics on exactly the stored inputs. This resolves identity, preprocessing
   and initial centered variation before proposing new fitting runs.
2. Recover original teacher/key/indices or saved target arrays and compare base
   targets, own offsets, literal targets, variance and initial full-pool residuals.
   Evaluate ULP loss at actual offsets. High `sin(1e5*f)` sensitivity is a
   property to measure, not permission to change frequency or precision.
3. Use existing curves/parameters to separate gradient directions, first update
   displacement, decay and fitting-budget exhaustion. Do not infer full-pool
   convergence from noisy changing-minibatch curves or mean Q alone:
   initial residual energy depends on centered Q variance and covariance with
   the base target, while the own mean cancels algebraically.
4. Resolve PR11 native export schema/provenance and coupled-seed grouping through
   separately reviewed changes. Preserve the seven non-independent pilot pairs,
   all crashes, missing metrics and unknown dirty status.
5. Only after separate authorization, run matched-source/input A100 equivalence
   with the existing acceptance criteria. CPU passes do not qualify TF32/GPU.
