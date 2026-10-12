# Exp1/Exp2 persistent allocation launch

Prepared, not authorized. This replaces the historical 25-wave commands for
the campaign; the queued pilot 22800749 is untouched.

| Allocation | Slurm shape | Duration | Physical capacity |
|---|---|---:|---:|
| A | one node × four A100 | 30 h | 120 A100-h |
| B | two nodes × four A100 | 48 h | 384 A100-h |

Both can queue or run independently. If they overlap, up to 12 A100s are in
use across three nodes. One persistent worker per node draws from the same
SHA-bound 195-parent queue. A free slot immediately draws the next compatible
run. Nominal GPU weights are D2W512=1/7, D4W1024=1/4 and D4W1536=1/2,
with same-size ceilings 7, 4 and 2. Mixed sizes are allowed. CPU and host
RAM tokens, live free host RAM and GPU memory with explicit headroom, and
observed checkpoint throughput further restrict starts. A second process
on a GPU waits until the first has two real checkpoint observations. Slower
than the approved throughput floor stops further placements, without
interrupting scientific training. XLA preallocation is disabled. Each
attempt has its own cache, W&B directory and training log outside Git.
The allocation approval also caps all processes per GPU, defaulting to one;
the owner can raise this only after reviewing actual packing evidence. The
7/4/2 architecture ceilings do not themselves authorize those occupancies.

The approval template leaves per-node CPU/RAM, per-architecture CPU/RAM/VRAM,
suite parent/arm runtime estimates, minimum throughput and headroom blank. The owner
must populate these from measured or explicitly reviewed evidence. The
approved HumanoidBench Python path is also required; its jobs use EGL.

## Durable identity and recovery

Parent IDs hash experiment kind, environment, architecture, seed, resolved
configuration hash and exact source commit. An injected-arm ID additionally
hashes parent ID and frozen m. The existing parent-plan validator establishes
the complete 195-row manifest before queue creation. Both allocations read
that immutable manifest. Atomic directory claims, protected by per-row
transition guards, prevent simultaneous duplicate starts. A stale claim is
reclaimed only after squeue confirms its allocation ended. Failed squeue,
partial claims or orphan guards block recovery for manual review. Retired
claims and every attempt log remain intact.

Completion requires the existing parent complete-state postvalidator or
the injected-arm endpoint/source/Check-1/artifact validator. A valid DONE
can be reconciled after a crash; a bare DONE never counts. An immutable
completion receipt hashes the validator output, DONE, metadata and terminal
LATEST pointer. Changed evidence blocks the row. A failed attempt blocks
immediate retries. A run interrupted by allocation loss resumes only from a
same-source, same-config checkpoint with restorable agent, replay and
normalization state. A run without a committed checkpoint needs manual
recovery; no completed result is overwritten.

A recorded nonzero failure stays blocked. After reviewing its attempt, an
owner may fill a new external
`configs/launch_readiness/dynamic_retry_decision.template.json` with this
row's exact ID, source commit, current `blocked.json` SHA-256 and decision
reference. The CPU-only command below rechecks checkpoint restoration, records
the decision and retires the old block under the row guard; it never
removes failed evidence or launches training:

```bash
python scripts/exp12_dynamic_scheduler.py review-retry \
  --queue "$DURABLE_QUEUE" --id "$REVIEWED_ID" \
  --decision "$OWNER_RETRY_DECISION"
```

Exp2 discovery is off unless the allocation approval contains a natural,
same-source positive-control result, matching owner-frozen m and a separate
Exp2 decision. The existing natural-trigger verifier checks each fork before
publishing its injected-arm manifest. An approved arm can start while its
Exp1 control continues, on the A100 model recorded by the fork. D2 and
forced-development forks are excluded. Check 2 failures remain recorded.

The campaign mount must support atomic mkdir, hard-link publication and rename
across nodes. Confirm this on the actual mount before launch. Confirm
Slurm's per-node GPU binding and CUDA_VISIBLE_DEVICES in a minimal allocation
preflight. CPU tests cannot establish either Delta property. Orphan
transition guards require manual recovery with all owners stopped.

## Capacity assessment

The combined reservation is 504 physical A100-h. The short D4W1536 dog-run
A100 profile measured 42.44 warm interactions/s and projected 3.27 h
training only for that cell. Applying that rate to all 105 million parent
interactions yields about 687.2 A100-h, a homogeneous illustration rather
than a measured grid forecast. Maximum control-extension and injected-arm
illustrations add about 91.6 and 114.5 h if all scaled parents trigger.
Thus 504 h would need at least 1.36× effective throughput improvement for
parents alone, or 1.77× for that all-trigger illustration, before compilation,
checks, evaluations, saves, queue delay or failures. D2 may be faster and
HumanoidBench slower; process sharing may improve or reduce total throughput.
There is no evidence yet that A+B can finish both experiments. Use the
full-length pilot, per-suite speeds and actual fork count to size any further
allocation; do not treat 7/4/2 as measured speedups.

## Exact preparation; submission requires separate authorization

Use a new clean detached checkout at the final reviewed SHA. All capitalized
output paths are new absolute paths outside Git on the shared filesystem.

~~~bash
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$FINAL_CHECKOUT"
cd "$FINAL_CHECKOUT"
git fetch origin --tags
git checkout --detach "$FINAL_SHA"
test "$(git rev-parse HEAD)" = "$FINAL_SHA"
test -z "$(git status --porcelain --untracked-files=all)"
cd main
WANDB_MODE=online python -c 'import wandb; assert wandb.Api(timeout=15).viewer'
cp configs/launch_readiness/parents_approval.template.json "$PARENT_APPROVAL"
# STOP: owner fills genuine final-source scientific evidence and approval.
python scripts/exp12_launch_plan.py prepare --stage parents \
  --root "$CAMPAIGN_ROOT" --commit "$FINAL_SHA" \
  --approval "$PARENT_APPROVAL" --out "$PARENT_PLAN"
python scripts/exp12_dynamic_scheduler.py prepare \
  --plan "$PARENT_PLAN" --queue "$DURABLE_QUEUE"
sha256sum "$DURABLE_QUEUE/queue.json"
cp configs/launch_readiness/dynamic_allocation_approval.template.json "$APPROVAL_A"
cp configs/launch_readiness/dynamic_allocation_approval.template.json "$APPROVAL_B"
# STOP: separately fill A/B evidence, resources and owner decisions.
python scripts/exp12_dynamic_scheduler.py allocation-command \
  --queue "$DURABLE_QUEUE" --approval "$APPROVAL_A" --allocation A
python scripts/exp12_dynamic_scheduler.py allocation-command \
  --queue "$DURABLE_QUEUE" --approval "$APPROVAL_B" --allocation B
~~~

Each command prints one SHA-bound sbatch; it does not execute it. After
authorization, create the external queue/slurm and queue/submissions/A or
queue/submissions/B directories exactly once before executing only the
approved printed command. Existing reservation directories are a stop signal.
Preserve job-id.txt, submission.stderr, Slurm logs, attempts/, validation/,
ledger/ and arms/. Neither allocation submits the other. The old fixed-wave
commands are historical and must not be used for this campaign.

## Current gates

Scientific launch remains blocked by D4W1024 fresh null 13/100 and 6/100
against the approved ≤5/100 rule, D4W1536 hopper-hop P/b=.602 against .9,
six final-source CUDA identity cells, final-source natural positive control
and frozen m, entropy and CUDA numerical decisions, resource measurements
and explicit campaign/allocation approval. See exp12_48h_launch_plan.md for
evidence and decision alternatives. This scheduler changes no thresholds,
SAC settings, architectures, precision, training budgets or statistics.
