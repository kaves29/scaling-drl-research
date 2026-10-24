# Experiment status by source

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

## Recovered Batch1 correctness audit (24 October 2026)

The interrupted branch and exact705490f commit were recovered locally with
original receipts. `codex/overnight-batch1-correctness-20261023` preserves
#8→#9→#10 and is a sibling of offline-analysis PR #11. Recovery core
`e26e37f3cb353e212be02c151fd2d84aded63f53` rejects changed static-mask seeds
for sparse checkpoint restores while retaining dense seed portability. Same
masks, moments, counters, RNG and next updates are checked exactly. See the
[correctness report](overnight_batch1_correctness.md).

Fresh checks:77 focused tests,51 pytest passes/one unavailable-archive skip,
4 broader restore/isolation tests,74 mutation checks,65 simulator cells and
19 isolated PR #11 compatibility tests pass. Counts overlap. The recovered
889-test run at705 had two stale infrastructure fixture failures, corrected
without weakening production gates. Results retain their actual source scopes.
The A100 segmentation failure remains unresolved; no GPU jobs or merges occurred.

Review: [PR #12](https://github.com/kaves29/scaling-drl-research/pull/12);
PR #11 remains unchanged and unmerged.
