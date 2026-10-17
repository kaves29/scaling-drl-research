# Script entrypoints by protocol

Read [CURRENT_PROTOCOLS.md](../docs/CURRENT_PROTOCOLS.md) and the matching
scientific approval before running any experiment. A script's presence is
not authorization for a GPU job.

| Scope | Current entrypoints | Notes |
| --- | --- | --- |
| Approved five-target Exp1/Exp2 | `exp12_dynamic_scheduler.py`, `sbatch_exp12_dynamic_allocation.sh`, `exp12_launch_plan.py`, `positive_control.py` | The 195-parent manifest and natural-trigger Exp2 arms retain separate source-bound approvals. |
| Versioned ten-target 96-parent candidate | `exp96_campaign.py`, `sbatch_exp96_allocation.sh` | Separate queue, source-bound startup gate and scientific approval; not a replacement for the five-target protocol. |
| Ten-target engineering smoke | `exp96_gpu_smoke.py`, `sbatch_exp96_gpu_smoke.sh`, `exp96_kaggle_t4_smoke.py` | Development qualification only. T4 evidence does not establish A100 packing. |
| CPU and qualification checks | `run_exp12_cpu_validation.py`, `check_exp12_cpu_environments.py`, `methodology_cpu_reference.py`, `exp12_fullwidth_gate.py` | Preserve the evidence source SHA and use the corresponding specification. The full-width comparator supports audited retries and writes its report outside immutable evidence. |

Other scripts here are dated diagnostic, validation, and Angle research
entrypoints. Several are called by current tests or referenced by historical
evidence. They remain available to reproduce those source revisions; do not
mistake them for current production launch commands. See
[dated records](../docs/history/README.md) and
[the branch archive plan](../docs/BRANCH_ARCHIVE_PLAN.md).

The isolated CPU runner discovers `unittest` tests. Run the six pytest-style
ten-target and Exp96 modules separately using
[the CPU validation instructions](../docs/exp12_cpu_validation_setup.md);
an imported module with zero tests is not a pass for its assertions.
