# Seven-parent Experiment 1 pilot worker

This scheduler-only revision executes seven unchanged `parents`-stage commands
from a **separate clean checkout** of scientific commit
`087feab341b9be1f377697668f2d1c750ec0467f`. Its own commit is recorded
in the immutable pilot specification. It uses the approved 195-parent plan and
shared durable queue; the existing ledger atomically claims rows and certifies
completion using `validate_done`. A failed or walltime-stopped attempt remains
visible and needs the existing source-bound `review-retry` decision before its
next allocation. No Exp2 arms are selected. The source plan requires all
parent approval gates, including the unresolved numerical gate.

The requested placement is two D4W1536 jobs on logical GPU 0, and one D2W512
plus four D4W1024 jobs on logical GPU 1. It exceeds the ordinary scheduler's
nominal GPU-1 weight (1.143). The pilot worker therefore uses explicit
reviewed per-parent reservations instead of that nominal weight. The included
resource-estimate file is a **draft**, not a measurement: 32 CPUs, 128 GiB
host RAM, 14/30 GiB expected device use with 8 GiB device headroom on each
GPU. DMC/Myo historical short profiles support lower observed VRAM for those
architectures; HumanoidBench and mixed packing remain unmeasured. The worker
checks live memory before dispatch and samples device utilization/memory,
host free RAM, per-process RSS/CPU use and checkpoint-derived rates every 15
seconds. It also polls checkpoint directory creation and committed pointer
publication once per second, recording an approximate lower bound on save
publication time; it is not a trainer-internal save-duration measurement.
At 36 GiB device use or low host RAM it stops processes; the last committed
checkpoint remains and the failure needs `review-retry`. At the ten-hour limit,
Slurm ends still-running processes; their orphaned claims can be recovered
only after Slurm confirms that allocation ended, and resume validates the
last complete checkpoint. A fast transient spike between samples cannot be
ruled out; the source-bound ledger never marks an incomplete run complete.

Training uses frozen `WANDB_MODE=online` and its existing `EchoCritic-exp12`
project/run ID. A separate post-run publisher reads validated complete probe
CSV rows and publishes the requested IQMs and degradation percentage to
`Scaling-DRL-Experiment-1`, with raw environment steps on the x-axis. A zero
or near-zero baseline (`|baseline| <= 1e-12 * max(1, observed score scale)`)
leaves the percentage undefined; scores remain unchanged. The publisher uses
an atomic once-only local claim. If it is interrupted, review W&B and the
local CSV before retrying; it never replays history automatically.

After source-bound parent approval and review of the draft resource estimates,
prepare from the clean scientific checkout's `main/` directory:

```bash
python scripts/exp12_launch_plan.py prepare --stage parents --root "$CAMPAIGN_ROOT" --commit 087feab341b9be1f377697668f2d1c750ec0467f --approval "$PARENT_APPROVAL" --out "$PARENT_PLAN"
python scripts/exp12_dynamic_scheduler.py prepare --plan "$PARENT_PLAN" --queue "$EXP12_QUEUE"
```

From the clean scheduler checkout's `main/` directory, create the immutable
pilot specification and print the one-job command:

```bash
python scripts/exp12_seven_parent_pilot.py prepare --queue "$EXP12_QUEUE" --training-checkout "$FROZEN_CHECKOUT" --out "$PILOT_SPEC" --cpus 32 --memory-gb 128 --estimates configs/launch_readiness/exp1_seven_parent_resource_estimates.draft.json
python scripts/exp12_seven_parent_pilot.py command --spec "$PILOT_SPEC"
```

The printed command requests one node, two A100s, 32 CPUs, 128 GiB and ten
hours. Set `EXP12_HB_PYTHON` to the absolute pinned HumanoidBench interpreter
before submission. Slurm resource/account availability must be checked on
Delta. After completion, publish already validated parents with:

```bash
python scripts/exp12_pilot_plasticity_wandb.py --queue "$EXP12_QUEUE"
```

Do not run the printed `sbatch` until the remaining parent approval gate and
the owner's submission authorization are complete.
