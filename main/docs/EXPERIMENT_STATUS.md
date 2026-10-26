# Experiment status by source

## Opt-in sequential alternative diagnostic (26 October 2026)

Independent branch `codex/sequential-alternative-probe`, based on pilot
`57d2beb78be7cf0e1145f4a9479082a07924cd16`. Existing training/probe/config code
and active Kaggle execution are untouched. New8-task sequential A/B/C diagnostic
uses one environment/seed fixed pool/teacher bank, frozen architecture-specific
fresh-derived offset, fresh AdamW state per target and2,000 updates/phase.
It is a new extension inspired by Lyle2023, not paper-validated methodology.

49 CPU tests pass (32 alternative,17 existing regressions); tiny real residual
SAC critics and a standalone synthetic CPU example cover all K=1..8, immutable
sharing, carry/reset identities, original-disabled outputs and verified phase
resume. Full CPU logs/arrays are outside Git under
`/workspace/artifacts/sequential-alternative-probe/`; receipt under
`main/diagnostics/sequential_validation_receipt.json`.

See [contract and usage](sequential_alternative_probe.md). Instance-local hooks
are explicit opt-in; no automatic entrypoint/approved manifest change. Draft
30-parent pilot (two proposed tasks ×three sizes ×five seeds) and separate
five-job smoke are unapproved/non-executable. Task eligibility/removal, fixed
teacher choice, bank producer, cadence, storage/runtime and qualification need
owner review. No GPU work, RL training, job submission/cancellation or merge.

## Ten-run simultaneous diagnostic pilot revision (25 October 2026)

Updates [PR16](https://github.com/kaves29/scaling-drl-research/pull/16) in place.
Tested implementation `9efae21`; resolve the live PR head for documentation
followups.
Exactly ten executions: five matched condition/seed pairs, one of each on GPU0
and GPU1, five processes/device, one all-ten barrier, no waves/replacement/retry.
All four tasks and three sizes represented; five unique scientific configs,
not ten independent seeds. Pen-small/seed5 anchors the original NPZ case.

New manifest v2, source config bindings, guarded launcher/sbatch wrapper and
[review checklist](diagnostic_pilot_10_review_checklist.md) preserve the original
science/fit procedure. Original own-offset literal targets intentionally differ
by a constant; shared inputs/base/indices remain paired and both arrays/hashes
must be retained. See [updated report](source_bound_probe_discrepancy.md).

72 CPU tests pass, including synthetic gate rejection and all-ten barrier tests;
no actual GPU/Delta qualification is implied. Proposal: one gpuA100x4 node,
two40GB A100s, one task,32CPUs/128GiB/48h; measured packing feasibility unknown.
Cloud has no Delta access/Slurm/GPU tools. Adapter is unimplemented/unreviewed;
source freeze, smoke, representative five/five profiling and owner/allocation
gates remain blocked. Draft launch_approved/executable/packing_qualified=false.
No Slurm submission, training, probe fit, scientific change or merge occurred.

## Source-bound paired discrepancy and pilot design (24 October 2026)

[Draft PR16](https://github.com/kaves29/scaling-drl-research/pull/16), tested
implementation `13948eb`. Task branch
`codex/source-bound-discrepancy-pilot`, based on PR15
`822c2e5aae1aa99b970fdce8e0b4bff94e7d4724`. See
[diagnosis and launch gates](source_bound_probe_discrepancy.md) and
[previous pilot draft (superseded by v2 above)](../audits/probe_discrepancy/pilot_manifest.DRAFT.json).
At check35, all32 medium/large DMC parents beat fresh, all32 medium/large Myo
parents lose. Shared b cancels; mechanism remains unidentified. Only one original
component/curve NPZ is accessible. Recovery inventories48 early/late targets,
207 accessible candidate files/10 containers; Delta TCP proxy refused and W&B
HTTPS proxy403 prevent live remote inventory. No remote-absence claim.

22 CPU tests pass, including six source guard mutations and ten prior accounting
tests; nine source guards and original-NPZ arithmetic pass. New original-source
routing controls use stand-ins and perform no numerical fits or model forwards.
No production fixes justified, training, GPUs/Slurm, historical changes or merge.
Pilot adapter is not implemented/frozen/qualified; original full recipe stays
unchanged. Selected W&B runtime sum206.882h is only a planning proxy; resources,
source/runtime/equivalence/restoration, owner approvals and actual launch remain
blocked. No prior2/4h budget transfers and no main experiment is queued.

## Retained-data probe difficulty accounting (24 October 2026)

[Draft PR15](https://github.com/kaves29/scaling-drl-research/pull/15), tested
implementation `26299ae`. Analysis-only branch
`codex/probe-difficulty-accounting`, based on PR14
`1d849c3b52f1b6bdc2514f8972a2388021b929c7`. See
[report](probe_difficulty_accounting.md) and
[validation receipt](../audits/probe_difficulty/validation_receipt.json).
All103 records inspected;96 canonical parents,4,834 checks and48,340 rounds.
697 raw hashes,52 source snapshots,52,214 supplied-value comparisons and
4,834 source IQMs pass;10 deterministic CPU accounting tests pass.

Only one original parent/check has b/MSE/curves. Fresh-score drift cannot be
assigned to target variance versus fresh fitting error; algebraic witnesses
are explicitly not measured components. The contemporary paired Myo–DMC
contrast survives eight-seed comparisons at common checkpoints. One pen-small
case establishes large initial minibatch Q deviations from its stored offset,
but original DMC curves, inputs/RMS/teachers and initial pool predictions are
missing. Nineteen listed W&B manifest bodies remain unavailable; no remote
absence claim. No production changes, model forwards, fits, training, GPUs,
scientific-setting changes or merges. Published after validation.

## Automatic W&B critic diagnostic transport (24 October 2026)

PR14 follow-up on `codex/reference-critic-diagnostics-20261024`. Once the
observer is enabled, versioned scalars now enter the existing W&B history row
at the original raw-environment step. Each measured observation queues a per-run
mask/receipt artifact, including frozen inputs/session metadata when present.
No extra `wandb.log` call, replay sampling, model operation or RNG draw is added.
Skipped observations do not repeat old values; disabled W&B keeps local files,
and offline W&B spools locally. Diagnostics/SVDs remain opt-in.

Validation:28 CPU tests passed, including22 original diagnostics tests and six
transport checks with real W&B offline SDK/history read-back and artifact
manifest verification. No authenticated live run, training, probe fit, GPU job
or merge. See [documentation](reference_critic_diagnostics.md) and
[transport receipt](critic_diagnostics_wandb_validation.json).

## Reference-backed critic diagnostics v1 (24 October 2026)

[Draft PR14](https://github.com/kaves29/scaling-drl-research/pull/14); tested
implementation `f08c5b189e2b53d36b22d06cf165b8ede1f77e15`. New branch
`codex/reference-critic-diagnostics-20261024`, based on audit PR13
`5179c43b4a45f6d9032a385adbf72713e2456373`. Historical metric calculations/keys,
optimizer, architecture parameters, probe, schedules and RNG streams preserved.
Opt-in sidecars add actual post-ReLU and separate affine dormancy, per-layer
and two named aggregates, saved per-unit masks/new transition diagnostics,
conventional stable rank and reference-code entropy/threshold ranks. Current
preprocessed replay inputs are reused; optional immutable frozen inputs are
separate. Rank SVDs are independently optional. Raw per-layer gradient-array
adapter is explicit about ownership/timing; original global norm is aliased.

Authors' code pinned and inspected: SparseNetwork4DRL613a2b9,
Dopamine/ReDo5873f54, loss-of-plasticitya6b7958. Shared scaling dormancy/rank
semantics are inherited, not a newly established EchoCritic bug. Paper sites
returned403; paper-to-code agreement remains unverified. Validation:22 tiny
CPU tests passed; no-training methodology reference195 commands/39 configs/
130 eligibility candidates. No RL training, scientific probe fits, GPU jobs or
merges. Single plain SACCritic supported; twin/injected adapters and GPU
qualification remain outstanding. Historical scalar data cannot reconstruct
new activation/rank/persistence measurements. See
[reference comparison and usage](reference_critic_diagnostics.md).

## Diagnostic/probe disagreement follow-up (24 October 2026)

Continuing draft PR13 on its existing branch. Read-only code/artifact review
adds [a focused report](diagnostic_probe_disagreement_audit_20261024.md) and
an audit reader; no production changes, new fits, training, forwards, gradients
or GPU execution. Independent raw checks:2,784 per-seed values match supplement,
23,512 exact single-critic q1/q2 matches, no checked metric invariant failures.
Fourteen campaign/current source paths compared;13 are byte-identical, with
trainer differences limited to prior success correction/provenance observations.
All48 Myo rank declines and large-humanoid8/8 rank declines reproduce; no
campaign defect explaining disagreement is established. Prior synthetic fit
controls are cited as prior evidence and were not rerun under this restriction.
Remaining need: existing original parameter/pool/normalizer/target artifacts;
future diagnostic or measurement changes require separate authorization.

## Forensic probe audit (24 October 2026)

[Draft PR13](https://github.com/kaves29/scaling-drl-research/pull/13), tested audit
implementation `30a9f6663ec21a55eca3266160418f2e2cf3983b`. Separate task
branch `codex/forensic-probe-audit-20261024`, based on PR11
`5880ec010e6ff185fcd8a9313d695ec29dda1305`. Production/frozen code unchanged;
all four proposed corrections remain unapplied. Both real ZIPs are available
and hash-verified:103 records,96 canonical/seven non-independent pilot records,
eight crashes; independent reconstruction4,834 points/52,214 table comparisons,
operational CI9,476 checks, seed aggregates1,188 and metric windows3,456, all pass.
Focused pytest15 pass; nine synthetic source-bound parameter routes pass across
campaign/pilot/current. This is CPU evidence, not production checkpoint identity
or GPU qualification. DMC stability and fresh drift survive raw reconstruction;
causes remain unresolved. Native PR11 ingestion/provenance and seed grouping
remain integration work requiring separate reviewed changes. No GPU jobs,
new parent training, production reruns, external-agent contact or merges.

See [audit report](forensic_probe_audit_20261024.md),
[source-linked dataflow](forensic_probe_dataflow.md) and
[validation receipt](../audits/echocritic_forensic/validation_receipt.json).
Production PR11 validation is reused explicitly, not claimed as newly rerun.
Historical sections below retain their original availability/status snapshots.

## PR #11 integration finalization (24 October 2026)

PR #11 continues b061 on `codex/echocritic-batch2-20261024`, preserving #8 →
#9 → #10. Tested implementation `f8ad82f064547c76e12539804c98be75b32656d8`:
901 CPU tests across103 modules; complete mutation74/74 twice; supplementary
pytest51 passed/one unavailable-pilot-archive skip; simulator65/65. Focused79
overlaps the full suite. Source stayed unchanged and no threshold changed.
Opt-in runner lifecycle/provenance is wired to Exp1, Exp2 and fixed-task results;
failed-setup cleanup, source-change guards and order-dependent test mocks are fixed.

The supplied real ZIP was processed:103 unique IDs,96 campaign/seven pilot,
eight crashes,25,557 preserved history rows,697 raw hashes and52 source snapshots
verified. Native schema/counter/provenance mapping still prevents qualified
aggregation in the existing CLI. Historical twin-failure attribution and A100
segmentation equivalence remain unverified. PR stays draft; no GPU job or merge.
See [finalization report](pr11_finalization.md). The owner's subsequent forensic
probe audit is separate and keeps production/frozen code unchanged; new corrections
are proposals requiring owner adoption.


## Batch 2 offline campaign infrastructure (24 October 2026)

Codex branch `codex/echocritic-batch2-20261024` starts at PR #10 head
`c05d77cc9899e50e857c170faf9937e5635c3663`, preserving the #8 → #9 → #10
stack. Additive offline ingestion, validation, statistics, plotting and opt-in
diagnostic APIs are documented in [the usage guide](echocritic_batch2.md).
Focused CPU suite: 81 tests passed with an isolated pinned NumPy overlay.
Final mutation/check receipts are in the delivery. No real 103-record campaign
export was processed, no historical artifact or training kernel was changed,
and the original A100 segmentation-equivalence failure remains unresolved.

## Pilot root-cause corrections (23 October 2026)

Codex's dedicated `codex/pilot-root-cause-corrections-20261023` branch starts
at PR #9 `55166a643c979240510f3bdb76484c9777d72fee`, preserving PR #8.
Implementation `89cb004` corrects terminal success attribution after auto-reset
and rejects invalid completed horizons in the retrospective pairing API;
`51f33a3` adds the configured-decay known answer and final CPU evidence fields.
See [audit and correction report](pilot_root_cause_corrections.md) and
[PR #10](https://github.com/kaves29/scaling-drl-research/pull/10).

The specified f861 pilot source uses the intended AdamW1e-4/.01. Its actual
seven runtime records and `/workspace/echo_pilot_local_partial.zip` are absent,
and the later seven-parent scheduler names087 as its training source: reconcile
this provenance before certifying actual run settings. Six small residual CPU
cases match frozen continuous and serialized segmented fitting bitwise. The
old A100 discrepancy remains failed/unresolved. Synthetic high-output-bias
sensitivity under approved decay is a mechanism control, not a causal pilot
finding. No GPU job, scientific definition, optimizer or tolerance changed.


## Exp96 optimizer validity audit (23 October 2026)

Codex audited the frozen f597 source and PR #8 without GPU execution or changes
to the original training/probe implementation. All 96 canonical configurations
resolve to critic AdamW 1e-4/.01; the selected actual pen-seed5 metadata hash
matches. The initial f297 diagnostic draft used those values; regression4f10ca9
introduced generic3e-4/0. Historical64/64 and96/96 findings are documented as
original-archive statistics, not results of the failed wrong-config runner. Their
exact counts and all96 raw runtime records were not independently reprocessed
in this Cloud audit. Preserve that evidence limitation.

Review: [PR #9](https://github.com/kaves29/scaling-drl-research/pull/9),
Codex-owned audit checkpoint `b78cfb07010c5ecaf5f736739d475416c6fdf9ae`.

[Optimizer validity audit](exp96_optimizer_validity_audit.md) contains immutable
source/line citations, executed scope, validity classifications, controlled CPU
results and a minimal proposed GPU sensitivity screen.36 small CPU fits and
their exact repeat demonstrate configuration sensitivity, not a production-scale
mechanism. Job23120792's old segmentation discrepancy remains unresolved. No
scientific conclusion, new metric, GPU allocation or experiment is approved by
this technical audit; the owner retains scientific authority.

## Exp96 implementation status: CPU validated, GPU qualification pending (23 October 2026)

Owner-authorized implementation branch `codex/exp96-diagnostic-engineering-20261023`
starts at handoff `6af1bb463eba6d8856606bc040d003d1cb4df2be`. The constructor
regression reproduced the wrong generic optimizer; correction `b98e5ed` binds
AdamW to validated archived parent metadata. The delivered implementation adds
first-update function-space diagnostics, explicit retrospective plans, durable
partial results, allocation/accounting preflight and compact evidence export.
Review: [PR #8](https://github.com/kaves29/scaling-drl-research/pull/8),
owned by Codex; implementation/test checkpoint
`ac360d9a07f3632a25a8f6401fcf950549f9208d`. Later publication edits are
documentation only. All 36 targeted CPU tests passed with no skips.

See [implementation and execution guide](exp96_diagnostic_engineering.md) for
files, schemas, CPU validation and source-bound Delta commands. Resolve the final
implementation SHA and PR from Git/the delivery; this file cannot contain its own
final commit hash.

The authenticated Mac worker reconciled jobs 23120674/23120792: 181 cumulative
physical A100-seconds, no other investigation allocations/restarts, 14219 seconds
remaining at that audit. Historical remote budget.json is stale and preserved.
The old GPU segmentation error remains unresolved and its guard is unchanged;
new corrected-source A100/TF32 reconstruction, equivalence and full-size runtime
validation have not run. No GPU job or scientific-methodology change is authorized
by this implementation/status update. The next experiment/allocation requires
owner approval after fresh accounting.

Updated 17 October 2026. This is a status index, not methodology or approval.
The [dated status history](history/EXPERIMENT_STATUS_2026-10-17.md) preserves
earlier results and evidence references. Read [CURRENT_PROTOCOLS.md](CURRENT_PROTOCOLS.md)
to choose the intended version.

| Version | Source and population | Status |
| --- | --- | --- |
| Approved five-target Exp1/Exp2 | Frozen baseline `087feab341b9be1f377697668f2d1c750ec0467f`; 195 parents, 13 environments × 3 critics × 5 seeds | Method C (2,000 continuous fit steps, five randomized targets) and amendment (ab) are recorded in the [methodology](../.claude/methodology-exp1-exp2.md) and [decisions](exp12_decisions.md). Exp1 and Exp2 retain separate source-bound launch gates. No consolidation commit qualifies a new GPU source. |
| Versioned ten-target candidate | Exp96 lineage ending at `be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574`; 96 proposed parents, four environments × 3 critics × 8 seeds | Engineering CPU checks exist. Scientific calibration, source-matched A100 smoke, resource approval, and official study approval remain separate prerequisites. See [launch specification](exp96_launch.md) and [pending gate](exp96_pending_queue.md). |

## Deployment status

- Owner-reported Delta job `22945337` at `bd568d482c261f067add0b1d44bb57a6aceb52b1`
  started 17 October at 14:24:46, failed `3:0` after 45 seconds. Its
  startup receipt, Slurm logs, and parent ledger have not been inspected in
  this workspace. A fail-closed BLOCKED result is plausible, not confirmed.
- Owner-reported A100 smoke job `22952754` at `be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574`
  remains pending. Its future receipt cannot satisfy the existing allocation
  gate at a different source SHA without separately reviewed source-bound
  approval. Kaggle T4 development coverage does not qualify A100 packing.
- Existing Delta checkouts, queues, evidence, and job records are immutable
  with respect to this repository consolidation.

## Evidence and unresolved decisions

The [original five-target evidence](history/EXPERIMENT_STATUS_2026-10-17.md)
records limited-scope Method C fresh-null 5/100 and D4W1536 hopper ratios
0.9467, 0.9567, 0.9552. These do not establish trained-state sensitivity or
longitudinal false-trigger calibration. Exp2 natural-trigger positive control,
injection magnitude, and intervention approval remain Exp2-only requirements
under amendment (ab). The ten-target study requires its own protocol freeze,
scientific validity and approval; it is not silently adopted for the
195-parent experiment.

CPU validation on the consolidation candidate recorded 788 `unittest` tests
run (787 passed, one optional skip) and, separately, 46 passed/one
pilot-archive-path skip in six pytest-style ten-target/Exp96 modules. The
single skipped curve-integrity test then passed with the original uploaded
archive, giving 47/47 effective pytest coverage. All 74 mutation checks, 65
environment restoration cells and the independent 195-parent reference
passed. These establish engineering coverage only. Read
[BRANCH_CONSOLIDATION_REVIEW.md](BRANCH_CONSOLIDATION_REVIEW.md) for the exact
source and validation scope.

## Multi-GPU successor engineering (18 October 2026)

Owner requested8×A10040GB on gpuA100x8,7/5/4 packing, fixed96-parent science and
600weightedSU including≤2physicalGPUh targeted qualification. The separate
launcher preserves existing scientific/result/source gates and adds measured
packing/orchestration evidence plus atomic cumulative spending reservations.
The original smoke is inherited only for byte-matched scientific code, not
relabeled as multi-GPU evidence. No qualification PASS or campaign submission
is claimed by this source snapshot; inspect the durable Delta receipts.

## Qualification continuation (19 October 2026)

Delta22990946 FAILED3:0 after108s: packed-child test used batched environment
shape[0] instead of the single-environment dimensions. This is not measured
packing failure or qualification PASS. GPU assignment on eight devices and
one-winner ownership race passed independently. The successor branch fixes the
shape check before restore/work and binds reuse of those unchanged components.
Five focused CPU regression tests pass locally; no broad suite is repeated.
A13minute8GPU continuation requests at most1.733333physicalGPUh; together with
0.24actualGPUh previously charged, the maximum is1.973333GPUh, below2. No new
engineering PASS is asserted by this document. Old source/queue/job remain frozen.

## One-GPU qualification correction (19 October 2026)

23020460 CANCELLED pending, elapsed0/noAllocTRES. Owner requires one-GPU test
allocations. Successor requests one A100/16CPU/60GiB for30minutes on gpuA100x4,
three sequential homogeneous phases; optional mixed packing disabled. Seven
focused new/affected CPU checks passed. No GPU result exists yet. New reservation
maximum0.5SU/0.5GPUh; including prior0.24GPUh, qualification maximum0.74GPUh.
The earlier13minute/eight-GPU proposal is historical and superseded.

## Final owner-authorized qualification retry preparation (19 October2026)

23020573 FAILED3:0 after17s, no children: invalid global-GPU-token/NVML-index
comparison. Real consumption0.004722222SU and physicalGPUh. This was not memory
qualification. Driver-identity successor preserves science and all gates; new
identity/diagnostic tests pass. Real CPU paired ten-target2000-step fitting and
bitwise checkpoint restore/next-update contracts passed in50.374s, explicitly
using small synthetic fixtures. No new GPU PASS asserted. One final30min/oneGPU
retry authorized; cumulative qualification maximum0.744722222physicalGPUh.

## Qualification remains incomplete (19 October2026)

Final authorized retry23021182 FAILED3:0 after14s, before any workers, because
an expanded nvidia-smi query returned exit2. RecordedGPUvars2(global)/0(CUDA)
confirm distinct numbering namespaces. CPU mocks had not verified optional CLI
columns. Portable-query correction prepared and focused tests pass; not GPU-run.
Actual finalretry0.003888889SU/GPUh; cumulativetargetedqualification0.248611111
physicalGPUh. No further retry submitted. Originalsmokeevidence/queues preserved;
old96-parentcampaign22996973 remainsheld/blocked. No7/5/4memoryPASS exists.

## Minimal startup-first qualification preparation (19 October2026)

Latest owner direction permits renewed minimal qualification after relevant
preparation. Startup gate inside the one-GPU job must actually execute a finite
FP32 GPU kernel on the correct UUID/CPUs with observable telemetry before any
fitting starts. Six new/affected local gate/controller checks pass. CPU numerical
fixture is explicitly not GPU evidence. Existing fitting/checkpoint CPU contracts,
portable-query tests and passed historical components are reused. No scientific
settings change. Required7/5/4 memory and full checkpoints remain unqualified.

## Owner accepts partial engineering qualification (19 October2026)

23021908 FAILED3:0 after28m51s at28min workload cap. Actual GPU startup and
7small/5medium full paired ten-target fitting, checkpoint round trips and phase
memory headroom checks passed. Fourlarge workers reached fitting but produced no
completed curves/checkpoint receipts. Source-bound owner disposition records this
as unverified and profiles as planning reservations. No fabricated PASS. Explicit
owner instruction now authorizes campaign submission under that disclosed risk.
See exp96_owner_disposition.md; inspect actual submission receipts for live status.
