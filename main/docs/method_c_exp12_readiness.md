# Method C Exp1/Exp2 readiness review (12 October 2026, Eastern)

**Production verdict: NO-GO for both experiments.** The owner authorized
Method C implementation in this task, not job submission or a waiver of the
remaining launch approvals. Resolve the exact commit from
`codex/exp12-method-c-production`; this report cannot contain its own final
commit hash. The development pilot 22800749 and pending Method C validation
pilot are untouched.

## Implemented scientific delta

`configs/base_exp12.yaml` changes only `probe.steps: 1000` to `2000`.
`experiments/exp12/probe.py` retains the original first 1,000 minibatch
indices for every check/round and obtains the second 1,000 from
`fold_in(idx_key, 1)`, exactly as the A/B/C exploratory diagnostic did. The
same five `probe_key(seed, check, round)` teacher initializations, replay
sampling, target scale, per-critic offsets, and current/fresh pairing remain.
One `_fit` scan carries each critic's parameters and AdamW state through all
2,000 updates. The five-round IQM, zero threshold, 10,000-resample 95%
percentile bootstrap, strict positive lower bound, and two consecutive
scheduled checks remain. All SAC, architecture, precision, optimizer,
injection, evaluation, budget, and training settings are unchanged. The
owner-authorized amendment is appended to the chronological methodology.

## Evidence and its limits

The uploaded A/B/C job 22852240 archive (SHA-256
`294ccf90a6b904b8cf775132bec173ff71a15a32a0b362dae694b1334467faed`)
records clean exploratory source `18c76da70c7e7c53bb01cae0afec8b8ae4dfb3af`
on an A100. At 2,000 steps, D4W1536 Hopper-Hop configured-pool P/b was
0.9466655, 0.9566678, and 0.9552110 for seeds 990, 992, 993, each above
the 0.90 gate. The D4W1024 seed-991 fresh-pair null had 5/100 single-check
firings; the exact-binomial 95% interval is 1.64%–11.28%. All 100 saved
five-round bootstrap intervals recomputed exactly, and their first-1,000
losses/CIs matched the historical seed-991 file. The seed-990 Hopper-Hop
first-1,000 curves were **not bitwise identical** to the historical capture;
the new ratio was 0.6043159 versus the earlier 0.6023976. The within-job
1,000/2,000 comparison is valid, but exact historical reproduction is not
established. These observations come from an exploratory revision, not this
new production source.

The 5/100 null result is a per-check fresh-pair rate, not a longitudinal
two-check false-trigger rate. No trained Hopper-Hop critic was present in the
capture, so Method C's sensitivity and natural trigger timing are unknown.
The prior full-width dog-run bitwise gate passed on an older source, and does
not certify this changed probe. CPU tests cannot replace final-source CUDA
identity, backend, or precision evidence.

## Engineering readiness

The canonical generator still enumerates 195 unique parents (13 environments
× 3 critics × 5 seeds), 65 per architecture and 15 per environment. It
preserves seed assignments and all raw-step budgets. The production scheduler
uses source/config-bound identities, shared atomic claims, immutable completion
receipts, guarded recovery from complete checkpoints, preserved attempts, and
separate naturally eligible Exp2 arm IDs. It continuously replaces a finished
compatible run. Allocation A requests four A100s for 30 hours; B requests
eight A100s for 48 hours. They are independently submitted; overlap and more
than one process per GPU require separate approval and measured admission
data. The conservative default is one process per GPU. The actual Delta
shared-mount claim and GPU binding preflight has not passed on this source.

The reviewed Slurm worker pins a clean commit and environment, starts W&B
online, records attempts, and never submits another allocation itself.
Approval templates remain `approved: false` with resource and evidence fields
blank. No automatic parent, fork arm, or grid submission is authorized.
Source/config-aware restoration prevents Method A checkpoints from being
silently resumed as Method C; a parent interrupted before its first committed
checkpoint still needs manual review.

The A/B/C job measured the **incremental second 1,000 fitting steps**, not
whole-campaign speed: mean 1.848 s per D4W1024 fit (1,000 fits) and 3.469 s
per D4W1536 fresh fit (15 fits). Extrapolating 205 single-critic fits per
parent (one fresh check plus 20 paired checks) gives about 6.8 additional
A100-hours over 65 D4W1024 parents and 12.8 over 65 D4W1536 parents.
D2W512, HumanoidBench twins, Exp2 extensions, CUDA compilation and load
interference are not measured by that extrapolation. The old 42.44
interactions/s dog-run D4W1536 short window implies about 687 parent
training-only A100-hours if incorrectly applied uniformly; it is a capacity
illustration, not a grid estimate. A+B supply only 504 physical A100-hours.
The owner's approximately 900-hour capacity may still be insufficient for
all parents, control extensions and injected arms after Method C overhead,
evaluation, checkpoints and retries. The full-length pilot and suite-specific
throughput must bound allocation admissions and total capacity.

## Remaining launch blockers

| Scope | Required result or decision |
|---|---|
| Exp1 scientific interpretation | Method C itself is owner-authorized. Its 5/100 single-check null sample is at the empirical boundary with wide uncertainty; no two-check calibration or trained-critic sensitivity has been measured. Record those limitations rather than relabeling the sample as a production false-trigger rate. The new source has no CUDA scientific qualification. |
| Exp1 GPU engineering | Six source-matched scaled-critic CUDA identity cases and the actual Delta two-node claim/GPU-binding preflight remain unqualified. Review source-matched precision/Check 1 numerical evidence before filling the approval gate. |
| Exp1 pilot/resources | Read the actual completed pilot 22800749 artifacts, storage and W&B results; approve measured or conservative per-suite CPU/RAM/VRAM/walltime profiles, total A100 capacity and allocation A/B separately. The pilot used an older source. |
| Exp1 launch integrity | Fill a new parent approval with genuine evidence and final SHA; create a fresh external plan and durable queue. Template values are not approvals. |
| Exp2 | A natural final-source eligible fork, positive-control/injection-magnitude result, owner-frozen `m`, source-matched injection/restore identity, separate Exp2 approval and arm resource estimates are outstanding. No forced trigger or automatic injected-arm launch. |

Packing above one process per GPU and warm-cache repetitions can wait until
after a conservative one-process/GPU launch **if** the owner separately
approves that launch and the resulting throughput/capacity plan. Trained
detector sensitivity is scientifically unresolved; it cannot be inferred
from fresh-null or range tests.

## Guarded launch preparation (commands do not submit jobs)

Use a new detached checkout of the published final commit, never pilot
22800749's checkout. All paths below must be new, absolute and on the
approved shared Delta filesystem. The validation and identity commands in
`exp12_oct12_execution_handoff.md` also need the new final SHA; their old
source evidence cannot be relabeled. After reviewing those gates and filling
genuine owner approvals:

```bash
set -euo pipefail
FINAL_SHA=REPLACE_WITH_PUBLISHED_METHOD_C_COMMIT
FINAL_CHECKOUT=/work/hdd/biqc/skaveti1/exp12_final_checkouts/$FINAL_SHA
CAMPAIGN_ROOT=/work/hdd/biqc/skaveti1/exp12_campaign_method_c
PARENT_APPROVAL=/work/hdd/biqc/skaveti1/exp12_approvals/$FINAL_SHA/parents.json
PARENT_PLAN=/work/hdd/biqc/skaveti1/exp12_approvals/$FINAL_SHA/parents_plan.json
DURABLE_QUEUE=$CAMPAIGN_ROOT/queue
APPROVAL_A=/work/hdd/biqc/skaveti1/exp12_approvals/$FINAL_SHA/allocation_A.json
APPROVAL_B=/work/hdd/biqc/skaveti1/exp12_approvals/$FINAL_SHA/allocation_B.json
test ! -e "$FINAL_CHECKOUT" && test ! -e "$DURABLE_QUEUE"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$FINAL_CHECKOUT"
git -C "$FINAL_CHECKOUT" checkout --detach "$FINAL_SHA"
test "$(git -C "$FINAL_CHECKOUT" rev-parse HEAD)" = "$FINAL_SHA"
test -z "$(git -C "$FINAL_CHECKOUT" status --porcelain --untracked-files=all)"
cd "$FINAL_CHECKOUT/main"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
WANDB_MODE=online python -c 'import wandb; assert wandb.Api(timeout=15).viewer'
mkdir -p "$(dirname "$PARENT_APPROVAL")"
cp configs/launch_readiness/parents_approval.template.json "$PARENT_APPROVAL"
cp configs/launch_readiness/dynamic_allocation_approval.template.json "$APPROVAL_A"
cp configs/launch_readiness/dynamic_allocation_approval.template.json "$APPROVAL_B"
# STOP: owner fills and approves each file using source-matched evidence.
python scripts/exp12_launch_plan.py prepare --stage parents \
  --root "$CAMPAIGN_ROOT" --commit "$FINAL_SHA" \
  --approval "$PARENT_APPROVAL" --out "$PARENT_PLAN"
python scripts/exp12_dynamic_scheduler.py prepare \
  --plan "$PARENT_PLAN" --queue "$DURABLE_QUEUE"
python scripts/exp12_dynamic_scheduler.py allocation-command \
  --queue "$DURABLE_QUEUE" --approval "$APPROVAL_A" --allocation A
python scripts/exp12_dynamic_scheduler.py allocation-command \
  --queue "$DURABLE_QUEUE" --approval "$APPROVAL_B" --allocation B
```

The last two commands print exact SHA- and queue-bound `sbatch` submissions
for separate authorization. Do not execute them until the printed resource
requests, source SHA, evidence, shared mount and owner approvals pass review.
Exp2 injected arms require their separate approved queue settings and frozen
`m`; preparing parents does not approve them.
