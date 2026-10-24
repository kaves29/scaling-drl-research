# Current agent handoff

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

## Exp96 diagnostic engineering implementation (23 October 2026)

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

Updated 17 October 2026. Resolve live refs and job state before acting. This
page is a navigation handoff, not a scientific amendment or launch approval.
The [dated handoff history](history/AGENT_HANDOFF_2026-10-17.md) preserves
earlier branch and validation records.

## Shared source

- Default `main` is the approved five-target Exp1/Exp2 baseline at
  `087feab341b9be1f377697668f2d1c750ec0467f`.
- `codex/scaling-drl-consolidation` is the review branch. It retains the
  195-parent protocol, the separately gated ten-target 96-parent candidate,
  the seven-parent pilot tools, and the numerical-isolation harness. Its tip
  must be resolved from Git; this file cannot contain its enclosing commit.
- Read [CURRENT_PROTOCOLS.md](CURRENT_PROTOCOLS.md) before selecting a
  configuration. The five-target methodology and amendments in
  [`.claude/methodology-exp1-exp2.md`](../.claude/methodology-exp1-exp2.md)
  remain authoritative for the approved 195-parent version. The 96-parent
  candidate does not supersede it.

## Frozen Delta jobs

| Job | Source | Last owner-reported state |
| --- | --- | --- |
| Exp96 allocation `22945337` | `bd568d482c261f067add0b1d44bb57a6aceb52b1` | FAILED `3:0`, 45 seconds, 17 October; startup receipt and logs not available in this workspace, so parent-claim outcome is unverified. |
| A100 engineering smoke `22952754` | `be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574` | PENDING at last report; no PASS evidence. |

Do not retarget either job or its checkout, queue, approvals, checkpoints, or
evidence. The smoke and allocation have different source SHAs; a smoke PASS
does not satisfy the unchanged allocation's source-bound startup gate.

## Ownership and next actions

- Codex: engineering, runtime, infrastructure, and integration.
- Claude: scientific investigation and validation.
- Research owner: methodology, scientific dispositions, compute spending,
  and production submission.
- Review this branch's diff and [branch inventory](BRANCH_CONSOLIDATION_REVIEW.md).
  Use a scoped branch and PR for new work; do not edit another agent's branch.
  Keep archival refs for unique branch tips before any later deletion.
- Obtain `startup_gate_22945337.json` and Slurm stdout/stderr to determine
  whether the failed allocation was the expected fail-closed BLOCKED case.
  Inspect the ledger before asserting that no parent was claimed.
- The current scientific and qualification limits are in
  [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md). Neither this handoff nor a
  successful CPU suite authorizes official training.

## Owner-authorized multi-GPU successor (18 October 2026)

The task branch `codex/exp96-multigpu-754` adds a separate8GPU launcher and
source-bound targeted qualification; see [operational design](exp96_multigpu_754.md).
It preserves scientific rows/validators and old queues. Owner authorizes600SU
including≤2physicalGPUh qualification; actual engineering PASS is external and
must be inspected before approving/submitting the campaign. Do not infer PASS
from this implementation or from the original one-GPU smoke alone.

## Qualification continuation (19 October 2026)

`codex/exp96-qualification-shapes` is based on the frozen multi-GPU revision
`a5c376d740b072d5211e55d94860a61dbd054fec`. Job22990946 failed after108s
because its test read the vector batch dimension as the dog input width.
Passed production-assignment and claim-race components are reused only with
artifact hashes and unchanged implementation checks. Full packed fitting,
memory and checkpoint qualification remain unverified until the new actual
receipt passes. See [continuation scope](exp96_qualification_continuation.md).
Do not release/retarget historical held campaign22996973 on this basis.

## One-GPU correction (19 October 2026)

Owner limited qualification allocations to one GPU. Eight-GPU23020460 was
cancelled pending, with zero runtime/no allocation. `codex/exp96-single-gpu-qualification`
serializes homogeneous7/5/4 checks on one A100, disables unmeasured mixtures,
and preserves source/result/spending gates. See [scope](exp96_single_gpu_qualification.md).
Old campaign22996973 remains untouched and blocked; do not release it on a new
source's receipt. No new GPU PASS is asserted by this source snapshot.

## Driver-identity repair (19 October2026)

OneGPU23020573 failed17s before workers: a Slurm global GPU token was compared
to NVML ordinal. Exact raw values were not saved. Owner authorized one final
retry after deep analysis. Successor `codex/exp96-qualification-driver-identity`
uses inherited CUDA visibility and CUDA UUID/PCI-to-NVML matching, with durable
raw diagnostics and per-child stages. See [failure analysis](exp96_qualification_failure_analysis.md).
Two real pinned-stack CPU fitting/checkpoint contracts passed; five new/affected
identity/controller tests passed. GPU packing remains unverified until actual
artifacts pass. Keep one GPU/30min, no automatic further retry, old campaign frozen.

## Final retry failed; no further submission (19 October2026)

23021182 FAILED14s/no children because the expanded NVIDIA inventory command
returned exit2. Optional PCI/minor diagnostic columns were unverified; exact
CLI error text was not preserved. `codex/exp96-portable-discovery-query` restores
the previously successful four-field inventory query, retains CUDA UUID identity,
and records any future CLI stdout/stderr. Four identity tests plus the affected
mapping test pass. No GPU job was submitted for this correction. The final retry
authorization is consumed; all actual packing/UUID GPU qualification remains
unverified. Preserve old campaigns and require explicit authorization before any
further GPU retry. See the actual-outcome addendum in the failure analysis.

## Renewed minimal qualification direction (19 October2026)

Owner now requests minimal required checks only, with extensive relevant
preparation and real GPU execution verification before fitting. Separate branch
`codex/exp96-gpu-startup-gate` adds a90s real GPU startup stage inside the same
one-GPU allocation, then admits only unfinished homogeneous7/5/4 phases after
identity/kernel/CPU/telemetry success. No repeated broad tests or separate GPU
diagnostic job. See [minimal scope](exp96_minimal_gpu_startup_gate.md). Preserve
old jobs/evidence, all scientific bytes and spending/source gates; no GPU PASS
is asserted by this source snapshot.

## Explicit owner launch disposition (19 October2026)

Owner accepts unverified4large packing/recovery after23021908 timed out. New
`codex/exp96-owner-disposition` supports an explicit OWNER_WAIVED engineering
admission, never engineering PASS. See exp96_owner_disposition.md. Preserve all
scientific bytes, original evidence, source/queue integrity, physical admission
and shared600SU accounting. Only the obsolete held allocation is superseded;
new immutable source/queue must be verified before replacement submission.
