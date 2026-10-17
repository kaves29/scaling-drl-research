# Current Exp1/Exp2 protocol and source index

Updated 17 October 2026. This is a navigation and status document, not a new
scientific amendment or a launch approval. Resolve live Git refs and Delta job
state before execution.

## Scientific authority

- The approved five-target Exp1/Exp2 protocol is the chronological
  [methodology](../.claude/methodology-exp1-exp2.md), including amendments (aa)
  and (ab), with the [decision log](exp12_decisions.md). Its implementation is
  `experiments/exp1.py`, `experiments/exp2_arm.py`, `experiments/exp12/`, and
  `configs/base_exp12.yaml`. The original population is 195 parents across
  13 tasks, three critics and five seeds. The natural-trigger Exp2 arms retain
  their separate positive-control, injection and approval gates.
- The ten-target Method C, four-task, eight-seed, 96-parent version is a
  **separate candidate study**, not an amendment to the archived five-target
  pilot. Read [its design](method_c_v2_ten_target_design.md),
  [launch specification](exp96_launch.md), and
  [pending-allocation gate](exp96_pending_queue.md). Its implementation uses
  `configs/base_exp12_ten.yaml`, `scripts/exp96_campaign.py`, and the shared
  Exp1 trainer. CPU implementation checks do not establish statistical
  calibration, A100 concurrency, or permission to train official parents.
- The [Exp1 seven-parent pilot](exp1_seven_parent_pilot.md) and numerical
  [isolation harness](../scripts/exp12_diagnostic_isolation.py) are separate
  source-bound operational tools. They do not change the scientific authority
  or retroactively qualify a frozen source.

## Deployment provenance that must remain distinct

| Item | Frozen source | Status known from owner-provided Delta report |
| --- | --- | --- |
| Original five-target baseline | `087feab341b9be1f377697668f2d1c750ec0467f` | Historical qualification source; retain history. |
| Exp96 pending allocation, job `22945337` | `bd568d482c261f067add0b1d44bb57a6aceb52b1` | Pending at last report; startup exits before parent training if its own approval gate fails. |
| Exp96 A100 smoke, job `22952754` | `be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574` | Pending at last report; no PASS artifact established here. |

The `be1ffae` smoke receipt cannot satisfy the unchanged `bd568d` campaign
source check. No consolidation commit changes those jobs or their existing
checkouts, queue, manifest, approvals, or evidence. Reconcile source-bound
approval separately before any official 96-parent training.

## Where to work

| Purpose | Active paths | Historical material |
| --- | --- | --- |
| SAC and shared infrastructure | `scale_rl/`, `experiments/exp12/`, `experiments/exp1.py`, `experiments/exp2_arm.py` | `experiments/angle_*` are imported by current code or tests; retain. |
| Five-target launch | `configs/base_exp12.yaml`, `generate_manifest.py`, `scripts/exp12_dynamic_scheduler.py` | Older launch reports and diagnostics in `docs/` remain provenance, not current commands. |
| Ten-target 96-parent launch candidate | `configs/base_exp12_ten.yaml`, `scripts/exp96_campaign.py`, `scripts/sbatch_exp96_allocation.sh` | Kaggle notebook and runner are development evidence only. |
| Tests and dependencies | `tests/`, `requirements.txt` | Keep pinned dependencies; do not infer a package is unused from import search alone. |

Before launching, read the applicable version's approval and qualification
requirements. Use a clean checkout at the approved full SHA and write results
outside Git. [Agent handoff](AGENT_HANDOFF.md) contains historical task records;
its older branch commands are not standing authorization.
