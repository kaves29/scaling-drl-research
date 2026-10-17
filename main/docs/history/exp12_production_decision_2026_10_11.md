# Exp1/Exp2 production decision, 2026-10-11

Source baseline: `7d0883cfa99d27fed062d77a68cbbfa27e560061`. This
review branch adds only an Exp2 source-provenance preflight and its CPU
regression. Pilot Slurm job 22800749, its checkout, and its evidence are
untouched. The approved methodology and all experiment configurations are
unchanged. The October 18 target is a planning target, not qualification.

## Decision ledger

| Blocker | Established evidence | Minimum next action |
|---|---|---|
| D4W1024 fresh-null | Block B reports 13/100; owner reports independent seed991 6/100. Both exceed the approved maximum 5/100. The seed991 raw rows and provenance have not been supplied to this checkout. The five-round paired IQM percentile bootstrap can have conditional initialization effects; counts alone do not identify a software defect or the longitudinal two-check false-fork rate. | Retrieve both complete `null_pairs_D4W1024.jsonl`, summaries, config/source/backend records and checksums; independently recompute each pairing/CI/firing. Owner must disposition a verified failed gate. Keep threshold zero and the existing two-check rule unless explicitly amended. See `exp12_null_threshold_decision.md` and `plasticity_validation_investigation.md`. |
| D4W1536 hopper-hop range | Approved configured-pool ratio is 0.6023976348, below 0.9; all five rounds fall below 0.9. The routine fits an untrained fresh critic, so this fails measurement qualification but does not establish degradation of a trained critic. Generic CPU fitter and ratio arithmetic checks passed historically; finite-budget conditioning is plausible, not proved. | Inspect the already designed same-settings curve/teacher-pool capture (`scripts/capture_exp12_range_evidence.py`) on an authorized A100; owner decides disposition after source-aware evidence. Do not use smaller-pool results as a waiver or alter fitting. |
| Positive control / frozen m | A genuine natural dog-run D4W1536 seed102 development fork is required before the Exp1 grid; m is not frozen. Pilot 22800749 is queued, outcome unknown. | Review pilot's natural trigger, full probe/recovery/noise evidence, source and checkpoint. If no eligible trigger, stop and consult. Owner freezes m and any unresolved success semantics before injected Exp2 arms. |
| Six CUDA identity cases | Required combinations are D4W1024 and D4W1536 in DMC, MyoSuite and HumanoidBench. The older D4W1536 dog-run full-width engineering PASS checks restore, cache, Check1 and bitwise state at its source, but is not a natural-fork identity receipt for all six cases. No complete six-case case-specific GPU certificate is available here. | Qualify each actual scaled architecture/suite case on the final source with the approved identity/Check1 path and retain exact GPU placement, checkpoint, source and result receipts. Select representative tasks/staging and resource limits with owner approval. |
| Exp2 arm source provenance | Previously arm startup compared config and GPU model but did not compare parent checkpoint code to the running arm revision. This branch rejects source commit/dirty-state mismatch before arm directory creation or restoration; valid matching-source operation is unchanged. | Merge only after review; qualify actual GPU fork/injection/identity path. Keep parent and arm on the same frozen source and GPU model. |
| CUDA numerical gates | The four diagnostic CUDA bounds remain unset; no owner-approved replacements are present. | Measure on final source, obtain explicit bounds decision, then run the existing mutation/qualification checks. No CPU result sets CUDA tolerance. |
| Resources and complete population | The immutable plan enumerates 195 unique parent cells, requires exact source/evidence/approval, one GPU per job and explicit allocation/concurrency. DMC dog-run D4W1536 short-profile training rate is 42.4411 interactions/s and allocator peak 3.764 GB; MyoSuite/HumanoidBench and full-run peaks are unmeasured. | Use pilot and representative suite/size measurements to approve per-cell or uniform worst-case wall/memory, quota, storage and concurrency. No 195-run submission yet. |
| Exp2 scheduling | Exp1 automatically detects a natural eligible trigger, saves the complete fork and continues its control arm. It does **not** automatically submit injected arms. `generate_manifest.py --grid exp12 --injection-m <frozen m>` later emits candidate arm commands from `FORK_READY`; it trusts markers and is not the approval-gated launch plan. | After owner freezes m, independently census eligible forks and validate full checkpoint/panel/metadata/parent completion, source/device/config and duplicates; review a separate arm launch allocation. Do not pipe generated commands directly to Slurm. |
| W&B local artifact retention | Prepared parent launcher sets `WANDB_MODE=online`. `wandb_dir()` defaults to the ignored `main/wandb/` tree unless `WANDB_DIR` is supplied; campaign checkpoint/result paths do not themselves archive that local W&B tree. | Verify live authentication and remote run IDs; preserve the ignored local W&B files separately with source/job metadata or approve a future external `WANDB_DIR` engineering change. Do not infer pilot 22800749's logging outcome while it is queued. |

The original full-width engineering PASS evidence and the queued W&B-online
pilot belong to different source revisions; neither may be relabeled as
final-grid qualification. The pilot is a development/positive-control attempt,
not a confirmatory seed. The old null and hopper failures are not optional test
suggestions; entropy wording alone is not a reason to change the implemented
approved `-0.5` target. Production remains **NO-GO**.
If queued pilot 22800749 creates a fork at `7d0883c`, an injected/identity
arm restoring that fork must run on that exact pilot source. The new guard
rejects using this review branch's changed code for that fork. Pilot-derived
positive-control evidence may still be reviewed as historical development
evidence; its relevance to a later grid source requires explicit source-aware
owner review, not a relabeled checkpoint or silent cross-revision restore.

**Owner decisions after evidence review:** For the null gate, the conservative
decision is to hold D4W1024 confirmatory use, verify the raw original and
seed991 rows, then preregister one unchanged final-source 100-pair calibration.
If it still exceeds 5/100, the owner must explicitly choose a methodology
amendment, a scientifically justified population exclusion, or continued hold;
none is implied by a 6/100 result. For hopper-hop, first capture the existing
full-pool fitting curves/targets and check actual arithmetic and convergence.
If the unchanged configured-pool ratio remains below 0.9 without a proved
implementation defect, the owner must likewise choose hold, exclusion, or an
explicit approved change. CPU tests or a smaller-pool pass cannot make either
failed gate PASS.

Evidence locations: the original full-width archive supplied for this review
is `/workspace/attachments/4adebe09-4602-47b2-b25e-3eec2fe4f120/exp12_gate_small.tar.gz`
(`exp12_gate_small/validation.json` records PASS at `7d21eaa`; original Delta
root `/work/hdd/biqc/skaveti1/exp12_validation/7d21eaa106a336168464cd5418a59136b6936a64_fullwidth_20261010T163647Z`).
Null raw rows requested from the original job are
`logs/blockB_22706349/gpu3/null_dog_run/null_pairs_D4W1024.jsonl`
and its summary; seed991 raw paths/checksums have not been supplied. Range
arithmetic and the missing curve requirements are in
`docs/plasticity_validation_investigation.md`. No six-case CUDA receipt is
available in this checkout. No artifact from pending job 22800749 was read.

## Injection and restoration review

`experiments/exp12/injection.py` splits the online and target critic separately
at the selected last/half/all head boundary. Online/target each retain their own
old head and trunk; both receive identical fresh `new`/`copy` values. The
`old + (new - copy)` grouping preserves initial predictions; frozen old/copy
receive zero parameter updates, while gradients reach the trunk. Online trunk
AdamW moments/count carry forward; new-head moments start fresh; target has no
optimizer. Subsequent SAC target Polyak updates traverse the full injected
tree. HumanoidBench twins get independent head keys and per-network storage.
The trainable count is preserved while total stored parameters grow by two
heads. `exp2_arm.py` restores the full parent fork, checks the fixed panel,
injects only the injected arm, and saves all state for continuation. Existing
single/twin injection, target/Polyak, Check1, checkpoint and fork CPU tests
exercise these rules. CUDA execution of all six cases remains unverified.

Focused CPU results on this branch: `test_exp12_manifest`,
`test_exp12_injection`, and twin injection/Check1 classes **22/22 passed**;
real `ForkEndToEndTest.test_identity_fork_is_bit_identical` **1/1 passed**;
`test_exp12_launch_preparation` **23/23 passed**. These cover the new
cross-source refusal, same-source parent/arm startup, 195-cell composition,
the existing injection invariants and fail-closed launch approval. No GPU
qualification or scientific-gate calibration was run.

## Allocation request, with measurement limits

The manifest specifies 210 million raw Exp1 steps = 105 million interactions
before any control extension: 195 parents (13 tasks x 3 sizes x 5 seeds).
Scaled controls can add **at most 14 million interactions** under the existing
20% total-extension cap. At most 130 scaled natural forks could create injected
arms; their fixed 25% horizons total **at most 17.5 million interactions**.
Actual eligible counts are unknown. No fork is forced.

The *measured short-window* D4W1536 dog-run rate was 42.4411 interactions/s;
its one 500,000-interaction training-only extrapolation is 3.2725 h. If one
incorrectly assigned that rate to every task/architecture, Exp1 would be
687.2 GPU-h training-only, maximum control extension 91.6 GPU-h, and maximum
injected arms 114.5 GPU-h. These are **homogeneous-rate reference calculations,
not campaign forecasts or resource approvals**. Compilation, probes,
evaluation, saving, W&B, full-length drift and simulator differences add
unknown cost. The queued pilot's approved eight-hour envelope is one measured
opportunity to update the dog-run estimate, not a uniform cap for 195 jobs.
Historical dog-shaped final-state estimates (`docs/exp12_overnight_readiness.md`)
are approximately 1.043/1.516/2.189 GB for D2W512/D4W1024/D4W1536,
including a full 500k dog replay. Multiplying those *illustrative* sizes by
65 cells per architecture gives about 309 GB for one state per cell, before
fork copies, atomic-save overlap, other environment dimensions, results, W&B
and backup. It is not a validated disk quota.

Request a *staged* one-A100-per-job allocation: first the existing pilot and
representative DMC/MyoSuite/HumanoidBench size/identity measurements; then a
quota decision for the 195 parents, with concurrency selected from real
quota/storage/throughput data. Four simultaneous A100 slots are a **planning
request, not an approved launch setting**. Exp2 injected-arm GPU-h cannot be
reliably estimated until eligible fork count and injected-critic speed are
measured. One Exp3 development pilot likewise has no selected source cell,
approved protocol or A100 runtime; request one separate A100 measurement slot
only after those prerequisites, not a fabricated GPU-hour total. Stage-specific
GPU-hour requests should use `sum(measured cell runtime + measured overhead)`
and the observed eligible-arm count.

## Exact next commands, prepared only

For the grid, start in a **new clean checkout at the final approved source**;
never reuse pilot 22800749's checkout. The template remains unapproved until
the owner supplies all eight parent-gate references and SHA256-bound PASS
receipts (`configs/launch_readiness/parents_approval.template.json`). Then:

```bash
export EXPECTED_COMMIT=FULL_APPROVED_40_HEX_SOURCE_SHA
export CHECKOUT=/absolute/new/clean-exp12-grid-checkout
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$CHECKOUT"
git -C "$CHECKOUT" checkout --detach "$EXPECTED_COMMIT"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=all)"
cd "$CHECKOUT/main"
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
python scripts/exp12_launch_plan.py prepare --stage parents \
  --root /absolute/new/confirmatory-parents --commit "$EXPECTED_COMMIT" \
  --approval /absolute/owner/approved-parents.json \
  --out /absolute/new/parents-plan.json
mkdir -p /absolute/new/confirmatory-parents/slurm
python scripts/exp12_launch_plan.py submission-command \
  --plan /absolute/new/parents-plan.json
```

`submission-command` prints a command; it does not submit. Do not execute that
command until each production gate and allocation is approved. After a genuine
natural fork and frozen m, use a separate fresh directory to *generate* candidate
Exp2 arms, then validate every row and create a separately reviewed submission:

```bash
mkdir /absolute/new/exp2-manifests
(cd /absolute/new/exp2-manifests && \
 python "$CHECKOUT/main/generate_manifest.py" --grid exp12 \
 --ckpt-root /absolute/new/confirmatory-parents/checkpoints \
 --results-root /absolute/new/confirmatory-parents/results \
 --injection-m OWNER_FROZEN_LAST_HALF_OR_ALL)
```

For each completed parent, `python scripts/exp12_launch_plan.py validate-run
--run /absolute/parent --commit "$EXPECTED_COMMIT" --out
/absolute/new/parent-validation.json` produces an engineering completeness
receipt; canonical study analysis still requires its frozen manifest and all
eligible runs/arms. No command in this report submits a job or selects m.
