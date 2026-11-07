# Current agent handoff

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
