# Current agent handoff

## Delta Exp2 integration preparation (19 October 2026 UTC)

`codex/exp2-delta-integration` descends from Cloud's immutable
`defd636e68554d60dc07ab6a30ce6ee5be60d516`. See
[Delta readiness](EXP96_EXP2_DELTA_READINESS.md). It adapts parent approval
to the real multi-GPU schema, isolates arm data, fixes the impossible256GiB
A100x4 request, pins the Slurm environment and requires separate Exp2 funding.
No Exp2 GPU work or m approval is authorized. Frozen Exp1 source/jobs remain
unchanged. Refer to external Delta preparation receipts for the final SHA and
targeted integration results; do not repeat Cloud's completed broad CPU suites.

## Exp2 successor (19 October 2026 UTC)

`codex/exp2-successor` is an isolated branch based on Exp1 multi-GPU source
`a5c376d740b072d5211e55d94860a61dbd054fec`. The owner retired historical
Angle 1/2A/2B/2C workflows. Current Exp2 is the natural-trigger
control/injection intervention; see [successor handoff](EXP96_EXP2_SUCCESSOR.md).
No frozen Exp1 queue, checkout, checkpoint, or Delta job is changed here.
Separate same-source positive control, m selection, measured arm resources,
A100 restore qualification and ten-target analysis certification remain.

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
