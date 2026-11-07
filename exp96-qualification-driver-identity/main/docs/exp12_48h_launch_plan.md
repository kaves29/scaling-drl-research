# Exp1/Exp2 eight-A100 launch decision sheet (2026-10-11)

Source under preparation: `codex/exp12-production-readiness`, based on `273c53b285b1a026ba7fd01679f12e55f41bf55b`. Pilot job 22800749 is independent and untouched. This sheet grants no campaign or GPU approval.

## Decisions that cannot be inferred from existing results

| Gate | Established evidence | Minimum defensible decision or evidence |
|---|---|---|
| D4W1024 fresh null | The archived original Block B rows were independently integrity-checked in `null_followup_original_results.md`: 100 unique pairs, exact recomputation of all firing flags, 13/100. The approved ≤5/100 criterion **fails**. A separate seed-991 summary reports 6/100; its raw rows are unavailable here, so row-level integrity is unverified. The original prespecified pair-effect W p=.069 did not detect an effect at .01. Exploratory within-pair shuffle p=.003 suggests pair structure on the original replay; this is not a replacement qualification test. | Obtain the seed-991 raw archive/manifest and verify its 100 rows, keys, CI and provenance. Then the owner must decide whether to retain the approved gate (D4W1024 remains NO-GO), commission a scientifically reviewed calibration study, or explicitly amend the methodology. Neither 13 nor 6 is a pass under the current rule. The production two-consecutive-check false-trigger rate is not identified by either 100-check experiment. |
| D4W1536 hopper-hop range | Block B configured-pool IQM P=.30084511637687683, IQM baseline=.4994128445784251, ratio=.6023976348; five reported round ratios .558653/.613904/.649450/.586019/.601805, all below approved .9. The ratio arithmetic is consistent. At 6,400 samples the same reported ratio was .8451; at 25,600 it was .6024 and final fit losses increased from roughly .03–.12 to .18–.22 while the baseline stayed near .5. This supports a pool/fit-capacity interaction but does not identify its cause. Per-step fit curves, teacher targets, replay-pool distributions and raw profile JSONs are unavailable here. | Recover those exact artifacts and examine fit convergence and target/baseline consistency at the **same approved** pool and budget. A fresh critic failing to fit establishes failure of the measurement gate, not trained-critic plasticity loss. If the existing artifacts cannot distinguish a correctness defect from finite-budget underfitting, request one same-settings instrumented replay, without tuning or excluding hopper. Owner decides any scientific exclusion or revised gate. |
| Six CUDA identity cells | Original Block B D4W1024/1536 × dog-run/myo-key-turn cold/warm attempts timed out or were skipped; h1-run-v0 unavailable. D4W1536 dog-run full-width engineering gate passed at `7d21eaa`, but did not yield the canonical identity certificate on the final source. `certify-identity` requires full canonical configuration/budget, GPU source and exact state comparison, so reduced-budget Block B artifacts cannot certify these cells. | On the final approved commit, obtain at least a cold certificate for each of six architecture × suite cells. Warm-cache runs are separately needed to establish the cache policy; disagreement requires an owner decision. Six cold and up to six warm mode-specific runs are pending, stopping at the identity snapshot; no full 195-run training is needed for this gate. Stop on any mismatch, backend fallback, timeout, or unavailable HB environment. |
| Positive control and injection m | CPU pathways and the natural trigger are implemented; no final-source, full-setting CUDA positive-control qualification/frozen m is established. | Owner reviews final CUDA PC evidence and approves m/interpretation before injected arms. Do not force eligible parents or pick m from a diagnostic default. |
| Entropy notation/sign | `exp12_launch_readiness.md` records an authoritative conflict while source behavior is fixed. | Owner must settle the scientific authority before confirmatory launch; no code change here. |

Under a hypothetical independent 5% per-check null, a count ≥13/100 has binomial probability .001464, while ≥6/100 has probability .384. These numbers do not account for shared initialization/replay dependence and cannot replace the approved deterministic gate. The 6/100 replication is compatible with ordinary sampling variation under that hypothetical model but still misses the required ≤5 count.

## Exact identity qualification preparation; no submission authorized

Use a new clean detached checkout of the **eventual approved final SHA**. The new `scripts/sbatch_exp12_identity_case.sh` runs one cold or warm case on one A100, uses canonical full-budget settings with only the documented development trigger/snapshot stop, writes outside the checkout, and calls the source-bound `certify-identity`. It needs explicit resource review because reduced-budget Block B attempts exceeded their former allocation. Four hours, 16 CPUs and 64 GB are the existing HB job request, **not** a measured sufficiency guarantee for all six cases. Each job needs a distinct absolute `IDENTITY_ROOT`; warm uses a shared cache between parent and arm within that job. First run one case; stop and re-estimate on timeout rather than retrying all twelve.

```bash
cd "$FINAL_CHECKOUT/main"
test "$(git rev-parse HEAD)" = "$FINAL_SHA"
test -z "$(git status --porcelain --untracked-files=all)"
case_id=D4W1024_dog-run                 # choose one of six listed below
cache_mode=cold                       # repeat as warm after cold passes
identity_root="/work/hdd/biqc/skaveti1/exp12_identity/${FINAL_SHA}/${case_id}_${cache_mode}_attempt01"
log_root="/work/hdd/biqc/skaveti1/exp12_identity/${FINAL_SHA}/slurm"
mkdir -p "$log_root"                    # external to the clean checkout
EXPECTED_COMMIT="$FINAL_SHA" IDENTITY_CASE="$case_id" IDENTITY_CACHE_MODE="$cache_mode" \
 IDENTITY_ROOT="$identity_root" HB_ENV="$APPROVED_HB_ENV" \
 sbatch --export=ALL --cpus-per-task=16 --mem=64G --time=04:00:00 \
 --output="$log_root/%x_%j.out" --error="$log_root/%x_%j.err" \
 scripts/sbatch_exp12_identity_case.sh
```

Cases: `D4W1024_dog-run`, `D4W1536_dog-run`, `D4W1024_myo-key-turn`, `D4W1536_myo-key-turn`, `D4W1024_h1-run-v0`, `D4W1536_h1-run-v0`. Submit only after each resource request, source SHA and HB environment are reviewed and GPU qualification is authorized. The certificate, two logs, submission metadata, GPU metadata, dependencies, exit code and SHA256SUMS appear under `identity_root`; Slurm logs are under `log_root`. A certificate alone is not a PASS unless source, CUDA placement, Check 1, comparison and all recorded artifacts validate.

## Production wave plan and timing

The approved parent manifest has 195 unique entries (39 architecture/environment cells × five seeds). With eight concurrent A100s, `prepare-batches` partitions it into 25 contiguous waves: 24 × 8 plus 1 × 3, with exact index bounds verified at execution. Each parent job is limited to ≤48 hours and ≤8 concurrency by the approval validator. **A Slurm array with `--array=...%8` does not impose a collective 48-hour wall from the first start:** queue delays can stagger starts. If “48-hour batch” means strict elapsed wall for the entire wave, request a co-scheduled/held allocation or another approved scheduling mechanism; do not claim this script enforces that interpretation. If it means each job's wall, an explicitly approved full array can be printed with `--duration-policy per-job`. Two overlapping waves mean up to 16 concurrent GPUs and require separate authorization.

For one-wave-at-a-time planning, 195/8 implies ~25 × the slowest job duration per wave, plus queue, compile, checkpoint, evaluation and recovery. Finishing in seven days needs a mean wave duration below 6.72 h **before** overhead/queue. Two authorized overlapping waves require ~13 wave durations, below 12.9 h before overhead/queue. At 8 h per wave the lower bounds are 200 h versus 104 h; at the 48 h cap they are 1,200 h versus 624 h. These are scenarios, not measured campaign predictions. The dog-run A100 diagnostic's 3.27 h training projection excludes full checkpoint/evaluation and other environments. Pilot 22800749 and per-suite measurements must set the actual resource plan. Control extension and injected arms consume additional A100 time only for naturally eligible scaled parents (up to 130); queue/resume overhead is unknown. Historical dog-shaped estimates of ~687 A100-h parent training, ≤91.6 h controls and ≤114.5 h injected arms are illustrative, not a capacity reservation. An Exp3 pilot is separate and needs its own approval/estimate.

## Source-bound preparation, inventory and Exp2 readiness

Use a new checkout; never repoint the queued pilot. All paths for approval, plan, schedule and campaign outputs must be outside the clean checkout. The template's evidence fields must be filled from actual final-source artifacts and signed by the owner; the script checks them, and neither this document nor a historical SHA supplies approval. Set resources to one GPU/job, ≤8 concurrency, ≤2880 minutes/job and the reviewed CPUs/RAM. Choose the wave path unless the owner explicitly confirms per-job interpretation.

```bash
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$FINAL_CHECKOUT"
cd "$FINAL_CHECKOUT"
git fetch origin --tags
git checkout --detach "$FINAL_SHA"
test "$(git rev-parse HEAD)" = "$FINAL_SHA" && test -z "$(git status --porcelain --untracked-files=all)"
cd main
python scripts/check_exp12_campaign_inventory.py --checkpoint-root "$CAMPAIGN_CHECKPOINTS" \
 --expected-commit "$FINAL_SHA" --out "$NEW_INVENTORY_JSON"
cp configs/launch_readiness/parents_approval.template.json "$OWNER_APPROVAL_JSON"
# STOP: owner fills evidence, source, scientific decisions and actual allocation in external approval.
python scripts/exp12_launch_plan.py prepare --stage parents --root "$CAMPAIGN_ROOT" \
 --commit "$FINAL_SHA" --approval "$OWNER_APPROVAL_JSON" --out "$NEW_PLAN_JSON"
python scripts/exp12_launch_plan.py prepare-batches --plan "$NEW_PLAN_JSON" --out "$NEW_SCHEDULE_JSON"
python scripts/exp12_launch_plan.py batch-command --schedule "$NEW_SCHEDULE_JSON" --number 0
# Printed command only. Do not execute it without explicit production authorization.
```

After authorization, the operator must create the external campaign `slurm/` directory and the new `"$NEW_SCHEDULE_JSON.batch-00.submission-reservation"` directory **once** before executing the printed command; an existing reservation is a stop signal. Check the resulting job ID and retain the submission stderr. Review each completed wave and inventory before requesting the next; never submit a second wave by default.

The prepared parent launcher uses `WANDB_MODE=online`; the worker now stores local W&B files inside the external per-attempt directory. Confirm authentication in the final environment without creating a run (`python -c 'import wandb; print(bool(wandb.Api(timeout=15).viewer))'`) and retain stdout/stderr; no secret should enter job metadata. Inspect the pilot's W&B, checkpoint, throughput and numerical logs before populating approval. If inventory reports stale/invalid/incomplete artifacts, stop and resolve the actual records. On restart, use only the source-bound `--resume` path and keep the same index, approved plan, GPU model, settings and original attempt evidence.

For Exp2, run the inventory guard and `validate-run` on every completed scaled parent before reading its natural `FORK_READY`; independently inspect the fork's source commit, exact state, trigger history, GPU provenance and positive-control m. After the owner freezes m, the following prepares a queue without submitting it:

```bash
python scripts/exp12_launch_plan.py validate-run --run "$NATURALLY_ELIGIBLE_PARENT" \
 --commit "$FINAL_SHA" --out "$NEW_PARENT_VERDICT_JSON"  # repeat for every candidate
mkdir "$FRESH_EXTERNAL_ARM_MANIFEST_DIR" && cd "$FRESH_EXTERNAL_ARM_MANIFEST_DIR"
python "$FINAL_CHECKOUT/main/generate_manifest.py" --grid exp12 \
 --ckpt-root "$CAMPAIGN_CHECKPOINTS" --results-root "$CAMPAIGN_RESULTS" \
 --injection-m "$OWNER_APPROVED_M"
```

Cross-check each `exp2_arms_<device>.txt` source against its validated parent and frozen study manifest; the generator keys only on `FORK_READY`, so its output alone is **not** a scientific eligibility certificate. Only natural eligible forks are allowed, never D2 or forced-development triggers. Review each queue and resource reservation before any submission. There is no automatic arm launch in this plan.

**Current GO/NO-GO:** engineering wave/identity preparation is CPU tested; campaign remains NO-GO pending the owner scientific decisions above, final-source CUDA identity/PC qualifications, measured pilot/per-suite allocation and explicit production approval. Checkpoint restoration and full-width bitwise gate PASS are supporting evidence, not substitutes for missing scientific gates.
