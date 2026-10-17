# Experiment status by source

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
run (787 passed, one optional skip) and, separately, 46 passed/one skipped in
six pytest-style ten-target/Exp96 modules. All 74 mutation checks, 65
environment restoration cells and the independent 195-parent reference
passed. These establish engineering coverage only. Read
[BRANCH_CONSOLIDATION_REVIEW.md](BRANCH_CONSOLIDATION_REVIEW.md) for the exact
source and validation scope.
