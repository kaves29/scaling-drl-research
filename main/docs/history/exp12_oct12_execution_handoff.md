# Exp1/Exp2 execution handoff (11 October 2026, Eastern time)

**Production NO-GO.** This document authorizes no GPU job or scientific change.
Pilot 22800749 and its checkout remain untouched. Resolve the final reviewed
source SHA from `origin/codex/exp12-production-readiness` before qualification.

## Engineering and launch population

The existing dynamic scheduler has one shared, source-bound 195-parent queue,
atomic cross-allocation claims, immutable completion receipts, preserved attempt
logs, source-checked checkpoint recovery and continuous replacement of a finished
run on a free compatible GPU. Allocation A is four A100s for 30 hours; B is eight
A100s for 48 hours. The safe default is one process per GPU. The nominal 7/4/2
architecture ceilings require GPU packing evidence before raising that cap.
Shared-mount atomicity and Slurm GPU binding still require Delta validation.

The canonical CPU generator was enumerated: 195 distinct run keys, checkpoint
paths and configuration hashes; 65 runs per architecture, 15 per environment,
39 per seed. Budgets: 135 at 1M raw steps, 30 at 500k, 30 at 2M. The resolved
role is `confirmatory`. **All 195 are BLOCKED** pending the owner scientific
decisions and final-source GPU qualifications below. A generated review snapshot
is not an approved durable queue; the latter is created from an approved plan.
Generate a current, source-bound **review manifest** without owner approvals or
training side effects by calling `cells()` on a proposed external campaign root;
do not confuse its `BLOCKED` status with a future execution receipt:

```bash
export FINAL_SHA=FULL_REVIEWED_40_CHARACTER_SHA
export CAMPAIGN_ROOT=/work/hdd/biqc/skaveti1/exp12_campaign_PROPOSED
export PREVIEW_MANIFEST=/absolute/new/external/exp12_prelaunch_manifest.json
test "$(git rev-parse HEAD)" = "$FINAL_SHA"
test -z "$(git status --porcelain --untracked-files=all)"
test ! -e "$PREVIEW_MANIFEST"
python - <<'PY'
import json, os
from collections import Counter
from pathlib import Path
from scripts.exp12_launch_plan import cells

rows = cells(Path(os.environ['CAMPAIGN_ROOT']), 'parents')
for field in ('run_key', 'checkpoint_dir', 'config_hash'):
    assert len({row[field] for row in rows}) == 195, field
assert len(rows) == 195
assert Counter(row['architecture'] for row in rows) == {
    'D2W512': 65, 'D4W1024': 65, 'D4W1536': 65}
for row in rows:
    row['qualification'] = 'BLOCKED'
    row['scientific_blockers'] = (
        ['D4W1024_fresh_null'] if row['architecture'] == 'D4W1024' else []) + (
        ['D4W1536_hopper_range'] if row['architecture'] == 'D4W1536'
        and row['environment'] == 'hopper-hop' else [])
manifest = {
    'source_commit': os.environ['FINAL_SHA'],
    'qualification': 'BLOCKED',
    'qualified_count': 0,
    'blocked_count': 195,
    'global_blockers': ['owner_scientific_disposition',
                        'final_source_cuda_identity', 'final_source_positive_control',
                        'resource_and_campaign_approval'],
    'rows': rows,
}
with Path(os.environ['PREVIEW_MANIFEST']).open('x') as stream:
    json.dump(manifest, stream, indent=2)
PY
sha256sum "$PREVIEW_MANIFEST"
```

## Prepared GPU commands; submit only after authorization

Use a fresh clean detached checkout. `FINAL_SHA` must be the final published
40-character commit; none of these paths is pilot 22800749's checkout.

```bash
set -euo pipefail
FINAL_SHA=FULL_REVIEWED_40_CHARACTER_SHA
FINAL_CHECKOUT=/work/hdd/biqc/skaveti1/exp12_final_checkouts/$FINAL_SHA
QUAL_ROOT=/work/hdd/biqc/skaveti1/exp12_qualification/$FINAL_SHA
test ! -e "$FINAL_CHECKOUT"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$FINAL_CHECKOUT"
git -C "$FINAL_CHECKOUT" checkout --detach "$FINAL_SHA"
test "$(git -C "$FINAL_CHECKOUT" rev-parse HEAD)" = "$FINAL_SHA"
test -z "$(git -C "$FINAL_CHECKOUT" status --porcelain --untracked-files=all)"
cd "$FINAL_CHECKOUT/main"
mkdir -p "$QUAL_ROOT/slurm" "$QUAL_ROOT/identity"
```

Six independent *cold* identity cases can run concurrently if available Slurm
resources and a separate approval allow it. Each requests one A100, 16 CPUs,
64 GB and four hours; these are caps, not measured runtime guarantees. Each
case gets a unique output path. Stop on a failed source/backend/Check 1/identity
certificate or timeout and inspect the evidence before any retry. Warm-cache
repetitions are conditional on the approved cache policy and cold results.

```bash
for case_id in D4W1024_dog-run D4W1536_dog-run \
               D4W1024_myo-key-turn D4W1536_myo-key-turn \
               D4W1024_h1-run-v0 D4W1536_h1-run-v0; do
  identity_root="$QUAL_ROOT/identity/${case_id}_cold_01"
  test ! -e "$identity_root"
  EXPECTED_COMMIT="$FINAL_SHA" IDENTITY_CASE="$case_id" \
  IDENTITY_CACHE_MODE=cold IDENTITY_ROOT="$identity_root" HB_ENV="$APPROVED_HB_ENV" \
    sbatch --export=ALL --cpus-per-task=16 --mem=64G --time=04:00:00 \
      --output="$QUAL_ROOT/slurm/identity_%j.out" \
      --error="$QUAL_ROOT/slurm/identity_%j.err" \
      scripts/sbatch_exp12_identity_case.sh
done
```

Prepare **exactly one** representative live Delta shared-mount/binding/claim
preflight before A or B: two nodes, eight A100s, 32 CPUs total, 128 GB total,
10-minute cap (at most 1.33 A100-hours). It checks one atomic claim winner,
hard-link publication, GPU/CPU binding and preserved stale-claim evidence on
the actual mount. It does not test full training dispatch or scientific results.
It cannot overlap the six identity cases under an eight-GPU ceiling.

```bash
preflight_root="$QUAL_ROOT/fsbind_01"
test ! -e "$preflight_root"
EXPECTED_COMMIT="$FINAL_SHA" EXP12_PREFLIGHT_ROOT="$preflight_root" \
  sbatch --export=ALL --nodes=2 --ntasks-per-node=1 \
    --gpus-per-node=4 --cpus-per-task=16 --mem=64G --time=00:10:00 \
    --output="$QUAL_ROOT/slurm/fsbind_%j.out" \
    --error="$QUAL_ROOT/slurm/fsbind_%j.err" \
    scripts/sbatch_exp12_dynamic_delta_preflight.sh
```

PASS requires exit 0, `fsbind_01/verdict.json` PASS, two distinct nodes,
four A100s/node, CPU backends, one claim/link winner and a retired claim.
FAIL/TIMEOUT stops allocation launch. Multi-process packing requires the
separate existing packing profile and review; it is optional for one/GPU.

Exp2 positive control requires a full-setting **natural development fork on
the same final source**. Pilot 22800749 is on an older source and cannot be
relabeled. Once a qualifying fork exists, the new source-bound wrapper runs the
canonical positive control in a separately approved allocation and preserves
logs, results, hashes and exit status. Its 55,000 probe-fitting updates lack a
reliable GPU runtime estimate, so the walltime, CPU and host RAM values must
be reviewed before submission. It requests one A100; `PC_OUTPUT_ROOT` must
be new and external. Exit 0 recommends m; the owner must freeze it. Exit 3,
forced eligibility or source mismatch is STOP.

```bash
PC_OUTPUT_ROOT="$QUAL_ROOT/positive_control_01"
test ! -e "$PC_OUTPUT_ROOT"
EXPECTED_COMMIT="$FINAL_SHA" \
PC_RUN_DIR="$FINAL_SOURCE_NATURAL_D4W1536_DOG_RUN_DEV_FORK" \
PC_OUTPUT_ROOT="$PC_OUTPUT_ROOT" \
  sbatch --export=ALL --cpus-per-task="$APPROVED_PC_CPUS" \
    --mem="${APPROVED_PC_RAM_GB}G" --time="$APPROVED_PC_WALLTIME" \
    --output="$QUAL_ROOT/slurm/positive_control_%j.out" \
    --error="$QUAL_ROOT/slurm/positive_control_%j.err" \
    scripts/sbatch_exp12_positive_control.sh
```

Only after m is frozen and Exp2 is separately approved may the scheduler
discover and start same-source naturally eligible injected arms while Exp1
controls continue. No forced trigger or automatic submission is authorized.

## Scientific decisions and capacity

| Gate | Established | Owner action |
|---|---|---|
| D4W1024 fresh null | Valid original 13/100 and seed991 6/100; both fail approved ≤5/100. No arithmetic defect confirmed. | Retain and hold D4W1024, or approve a prespecified threshold-calibration amendment with independent held-out null and positive-control sensitivity. No value may be selected from the failed datasets; a nonzero threshold changes comparability. |
| D4W1536 hopper-hop range | Job 22817406 at `3d75a05` reproduced P/b=0.6023976348. All five 1,000-step curves still improve; replay, targets, normalization and arithmetic check out. | Retain and hold this cell, explicitly exclude it, or authorize a fitting-budget amendment. A concrete unapproved candidate is 2,000 steps at unchanged pool/optimizer/targets, matching Lyle's original budget; it requires all-architecture range/null/sensitivity requalification. More steps are not proven to meet 0.9. |
| Positive control/m | Pilot result unavailable here; final source differs. | Obtain a qualifying natural final-source development fork, review positive control, and explicitly freeze m. |

A partial grid excluding failed cells changes approved architecture/environment
coverage and needs owner approval. No CPU test resolves these gates.

Roughly 900 A100-hours may be available. A+B reserve 504, leaving 396 for
later approval. The short dog-run D4W1536 profile measured 42.44 interactions/s,
projecting 3.27 hours *training only* for 500k interactions. Applying that rate
uniformly gives illustrative 687 parent + at most 92 control + 115 injected
training A100-hours, about 894 **before** compile, evaluation, checkpoint,
W&B, recovery or environment differences. No full pilot throughput is available
in this checkout. Review pilot 22800749 before setting CPU/RAM/VRAM/runtimes.
If A+B overlap perfectly, 504 A100-hours need 48 calendar hours; another 396
need at least 49.5 hours at eight GPUs, excluding queue and qualification.

## Production sequence after owner approval

Use fresh external approval, plan, queue and log paths. The templates start
unapproved. These commands create the approved 195-parent queue and **print**
source-bound A/B submission commands; they do not submit jobs:

```bash
cp configs/launch_readiness/parents_approval.template.json "$PARENT_APPROVAL"
# Owner fills true gate evidence SHAs, scientific decisions and resources.
python scripts/exp12_launch_plan.py prepare --stage parents \
  --root "$CAMPAIGN_ROOT" --commit "$FINAL_SHA" \
  --approval "$PARENT_APPROVAL" --out "$PARENT_PLAN"
python scripts/exp12_dynamic_scheduler.py prepare \
  --plan "$PARENT_PLAN" --queue "$DURABLE_QUEUE"
cp configs/launch_readiness/dynamic_allocation_approval.template.json "$APPROVAL_A"
cp configs/launch_readiness/dynamic_allocation_approval.template.json "$APPROVAL_B"
# Owner separately approves each allocation and measured resource profiles.
python scripts/exp12_dynamic_scheduler.py allocation-command \
  --queue "$DURABLE_QUEUE" --approval "$APPROVAL_A" --allocation A
python scripts/exp12_dynamic_scheduler.py allocation-command \
  --queue "$DURABLE_QUEUE" --approval "$APPROVAL_B" --allocation B
```

Preserve approvals, plan, queue ledger, attempts, results and Slurm logs on the
shared mount. The historical fixed-wave scripts are not the campaign path.
