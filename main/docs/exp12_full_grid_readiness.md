# Exp1/Exp2 staged launch readiness

2026-10-10, America/New_York. Approved base:
`844e3c0cc24998b25c3d1e667bcbfe056b1b336c`. Task branch:
`codex/exp12-launch-readiness`; obtain its enclosing final SHA from Git/the
delivery. No integration/main/Exp3 branch changes or Delta commands occurred.
Canonical methodology and scientific configurations remain unchanged.

## Evidence ledger and launch blockers

| Item | Evidence/status | Class / disposition |
|---|---|---|
| Tiny GPU cross-process resume | Owner reports22783981 PASS at844e3c0: references identical, crash95 restored60/102, endpoint300/582, GPU training/save/restore, no differences. Actual new artifacts unavailable here. | GPU evidence reported, independently UNVERIFIED; not full-width qualification |
| Combined A100 profile | Uploaded22765261 archive/profile/validation re-read:9d6a82d,6061/2124,42.4411 iterations/s,3.2725 training-only hours,35.8204 s/check,6.074 s/save,2.512 s/restore,22.834 s/10-episode evaluation,3.764 GB allocator peak. Prior full trace audit retained. | Verified short diagnostic; full-budget prediction UNVERIFIED |
| Cache/metric-transfer/comparator engineering | Already in approved base; new scripts do not edit them. | CPU regression and new full-width qualification preparation |
| Full-width cold/warm, restore, optimizer placement, Check1 | New staged harness uses unchanged D4W1536 config and actual trainer/save/restore/fork/representation-exact comparator. No GPU execution. | Separately authorized GPU gate required before pilot |
| Null calibration | Reported13/100 and seed9916/100 both fail≤5/100. Threshold remains0. Raw seed991 evidence absent. | Scientific owner disposition; read-only memo and raw-file follow-up prepared |
| Hopper range | D4W1536 P/b=.602 fails≥.9; fitting-transient evidence is a hypothesis, not a waiver. | Scientific decision/final-source GPU evidence |
| Positive control / m | Natural dog-run D4W1536 dev fork, healthy reference, shared-offset sensitivity, noise and three candidate recoveries must qualify; m remains unset. | Positive control mandated BEFORE Exp1 grid; frozen m required for injected Exp2 arms, not read by parent training |
| Identity before grid | Amendment(b) requires identity BEFORE the grid; later(p) explicitly lists D4W1024 and D4W1536 in each suite. Existing identity_warm has~329 s subtotal before omitted work, beyond300 s. | GPU staging/owner allocation decision; tiny resume does not satisfy six scaled-size/suite cases |
| GPU numeric bounds | Four diagnostic-isolation bounds remain unset. | Measure separately and obtain owner approval; no agent-selected tolerance |
| Entropy wording | G1 explicitly approved−.5|A|; equation reconciliation already documented. | Nonblocking for unchanged engineering/pilot; wording clarification before study freeze |
| Exp2 reporting / missing invalid runs | Canonical validators reject incomplete/duplicate/missing records; one-eligible-seed and invalid-null-pair dispositions remain owner issues. | Exp2/reporting decisions; not a reason to block independent CPU engineering |
| Suite resources, full-budget runtime/storage, allocation/quota | DMC short profile only; Myo/HB full-width runtime/peak/restore coverage and concurrency capacity unmeasured. | GPU/deployment approval before grid; no packing or resource ceilings inferred |
| Duplicate/stale markers/submission | Existing generator trusts DONE; new approval-gated immutable plans enumerate ALL195, compare exact commands/configs and use atomic never-stolen claims. | Engineering mitigation implemented; claim recovery and retries require explicit review |

No confirmed production-source defect was established in this sprint; new
engineering work removes launch-preparation/evidence hazards without changing
learning. Scientific failures are not repaired by test-rate tuning.

The new GPU resume artifacts, if supplied, are expected under
`/work/hdd/biqc/skaveti1/exp12_validation/844e3c0cc24998b25c3d1e667bcbfe056b1b336c_20261010T030038Z/gpu_resume_95_22783981/resume_probe/`.
Need JSON verdict, both reference and resumed state/checkpoint evidence, backend
and stage receipts, stdout/stderr, source/config and checksum. No Delta access
exists here and no reported Slurm state is independently inferred.

## Engineering interfaces

- `scripts/exp12_fullwidth_gate.py`: separate cold reference, warm reference,
  abrupt crash, resume, CPU comparison. Uses the canonical D4W1536 architecture,
  actor D1W128, original environment/training budgets/warmup/batch/UTD/probes in
  its resolved config. The training-only qualification stops at explicitly
  supplied diagnostic steps AFTER real warmup; it does not run/qualify probes,
  alter the scientific horizon or synthesize a natural trigger.
- Placement snapshots include every actor/critic/target/temperature parameter
  tree and floating optimizer moments. Target optimizer is genuinely absent.
  CPU initialization is preserved; actual training/restore placement is observed,
  never forced with device_put. Backend, dtype policy and pinned JAX are checked.
- Cold-cache directory must not exist; warm children share it. Record actual
  persistent hits/misses, backend compilation seconds, synchronized stage times,
  allocator memory stats and bytes. Cold process can reuse its own newly written
  entries; a cold-start assertion does not require zero lifetime cache hits.
- Full online AND target Check1 is representation-exact on the original256-pair
  replay panel. Twin-Q dispatch uses the existing panel implementation when the
  chosen canonical task is HB. Diagnostic fork construction is labeled
  `qualification_only`, never writes FORK_READY, does not force the scientific
  trigger, and is checked against unchanged complete parent state. Real natural
  fork/Exp2 entry-point qualification remains distinct.
- Full final states compare with both references, including optimizer, JAX and
  NumPy/Python RNG, replay/order/n-step queue, normalization, counters, both env
  states, targets, diagnostics and logging windows. Only existing wandb ID
  exclusion is used. Complete source/spec, actual exits, endpoint/update counts,
  cold/warm cache evidence and nonempty placement records are required.
- Verdicts: PASS, RESUME_DEFECT, NONDETERMINISTIC_BACKEND, INCOMPLETE,
  SCIENTIFIC_INVARIANT_FAILURE, INFRASTRUCTURE_FAILURE. External timeout is
  separately recorded; absent receipts never qualify. Reference differences do
  not prove a backend cause and do not authorize deterministic flags/tolerances.
- `scripts/exp12_launch_plan.py`: explicit owner approval plus SHA-bound GPU
  evidence; immutable population/config/command plan; no sbatch call. Pilot and
  parent gates are distinct. Preparation does not choose scientific settings.
- `validate-run`: reads full payloads/probes and Check1, checks endpoint/update
  counters and canonical config, refuses partial/malformed output. COMPLETE is
  engineering completeness; DEV_EXCLUDED or CANONICAL_CERTIFICATION_REQUIRED is
  separately reported. It never labels a complete dev run scientifically valid.
- Existing `analysis.exp1_analysis` / `analysis.exp2_analysis` and frozen study
  manifest remain the scientific certificate path. Do not use --exploratory to
  bypass qualification. Existing freezer requires frozen m/runtime provenance;
  that analysis-schema requirement must not be confused with a runtime m
  dependency in Exp1 training. Freeze analysis before confirmatory reporting.
- `certify-identity` wraps the existing native complete-state identity comparator,
  rechecks original pre-fork control Check1, full approved settings, source and
  actual GPU launch metadata, and pinned Orbax parameter/AdamW-moment saved
  sharding. It produces SHA256-bound case coverage receipts. Generic PASS, tiny
  GPU resume, CPU snapshots, reduced settings or only one suite cannot satisfy
  the six-case production identity gate. This converter is CPU-side/read-only;
  its successful path needs real GPU identity artifacts, currently unavailable.

## Minimum remaining jobs and resource estimates

1. **Full-width dog-run qualification:** four separately authorized stages, each
   one A100-SXM4-40GB,4 CPUs,32 GB,7 minutes/300-second cap. This retains the
   existing envelope; it is NOT an authorization. Candidate diagnostic extent
  6000 checkpoint /6001 crash /6061 end matches the measured profile window but
   must be explicitly approved/populated; templates keep extents null. Cold stage
   omits expensive full-setting probes/post-fork evaluations and includes fork
   roundtrip. Historical entire profile was282 s, so feasibility is plausible,
   not measured. Warm/reference/crash/resume costs are unknown. Maximum allocation
  4×7min=.467 A100-hours; capped workload4×300s=.333 hours. Stop on any timeout;
   do not silently extend. CPU comparison requires no new GPU allocation.
2. **Full-length dog-run D4W1536 seed102 dev/positive-control attempt:** only after
   gate1 and owner natural-trigger/role/allocation approval. Existing report's
   proposed16 CPUs/64 GB/8 h is UNAPPROVED, not a new default. Short-profile
   extrapolation~3.7 h no fork/~4.6 h late fork includes probe/eval/save estimates;
   only training projection3.2725 h is directly in profile.json. The60-step rate
   omits later bulk flush, long-term changes and full-run peaks. Historical>6 h
   timeout concern is mitigated, not proven resolved. The pilot measures it.
3. **Current-source suite/architecture identity and numerical qualification:**
   both scaled sizes in all three suites required by amendment(p); prioritize
   HB twins and Myo restoration/runtime after DMC. Existing warm identity cap
   is infeasible by the prior subtotal; owner must approve revised staging or
   allocation. No new timeout or numerical bound is selected. Full-width harness
   also accepts a canonical HB/Myo environment for D4W1536, but their duration
   under300 s is UNVERIFIED. D4W1024 identity remains separate coverage. D2W512
   does not fork in production; its checkpoint/runtime qualification is separate
   from the scaled identity scope explicitly named by the later amendment.
4. **Positive-control sensitivity / m:** reuse the natural dev fork if eligible;
   no new parent just for calibration. Additional candidate probe/arm GPU cost
   depends on approved fork/m steps; not estimated from missing results. If no
   natural eligible fork occurs, stop and consult, never force/loosen it.
5. **Null/range decisions:** existing raw seed991 analysis needs no new GPU.
   If new qualification experiments are chosen, preregister them independently;
   scientific settings and allocations require separate approval.

These are ordered prerequisites, not a request to launch all jobs automatically.
Elapsed queued time cannot be estimated from Cloud. Do not interfere with queued
jobs22783980/22783981 or ongoing Claude Exp3 work.

## Separate GO / NO-GO lists

**One development pilot:** exact clean published source; pinned CPU+CUDA/JAX and
actual A100; full-width gate PASS and intact state/cache/Check1 evidence; owner
approval for dev role/seed102, natural zero-threshold trigger and allocation;
fresh external plan/checkpoints/results/logs; storage capacity; reviewed resume
procedure. Status **NO-GO pending full-width evidence and approvals**. Scientific
null/range failures need not prevent this explicitly approved development study,
but may not be represented as passed or as confirmatory evidence.

**195 Exp1 parents:**195 exact fingerprints and independent manifest; canonical
CUDA identity for both scaled sizes in every suite (later amendment(p)); approved disposition/qualification
of null and hopper; qualified positive control before main Exp1; frozen source,
suite runtime/storage/resource evidence, quota/concurrency approval and provenance;
all mandatory decisions/evidence in approval file; role=confirmatory, seeds1–5;
no duplicates/foreign/partial DONE accepted; study freeze before reporting.
Status **NO-GO**. Positive-control and identity requirements are explicit in the
methodology, not avoidable Exp2 dependencies. Injection m is not used by parent
runtime. Parent probes can be collected with current0, but scaled parents also
naturally fork: changing the threshold later can require unavailable historical
fork states. Do not disable forks or silently treat such runs as unchanged.

**Exp2 injected forks:** eligible scaled natural completed two-check trigger by
check19; compatible immutable full fork/panel/control proof; exact same GPU model,
source/config; lead-frozen m and qualified noise/sensitivity; complete restore,
original online/target Check1 and appropriate injected64-eps rule; horizon25%
within1.2B, paired evaluation cadence/10 episodes; Check2 failures retained;
complete eligible seed census and reporting convention; strict certificates.
Status **NO-GO**. Parent training does not wait for separate injected arm jobs;
arm execution/reporting depends on valid parent forks, not the reverse.

## Array / recovery / cost policy

Plan entries reuse generate_manifest.exp12_overrides and original save interval
N/20. There are13 tasks×3 sizes×5 seeds=195 unique parents, total210M raw steps
or105M interactions before control extension. Max20% extension on the two scaled
architectures adds at most14M interactions. Actual eligible forks are unknown.
Aggregate GPU-hours = sum per-cell(interactions / measured suite/size throughput
+ probe checks×measured check cost + eval/save/compile cost), plus actual control
extension and separately approved injected arms. There is no credible measured
aggregate yet. Applying dog-run D4W1536 rate42.44 to105M gives~687.2 training-only
hours, a deliberately invalid homogeneous illustration, NOT a campaign forecast.

One run per GPU; no memory-based packing heuristic is qualified. New array uses
one explicitly approved uniform worst-case resource ceiling across all entries;
each architecture/environment's needs are currently UNVERIFIED. Approve that
ceiling only after DMC/Myo/HB measurements; heterogeneous resource groups require
separately reviewed plans. Concurrency is an explicit quota/storage-approved
integer, never derived from GPU memory alone. A cap does not change budgets.

Claims use atomic mkdir and are retained after success/failure. Duplicate array
submissions refuse claimed work; they can still waste allocations, so use the
prepared submission reservation below and never reuse it for a second sbatch.
Scheduler uncertainty/orphan reservation is a manual reconciliation task, not
automatic retry. No stale claim stealing, destructive repair or automatic retry.
After owner reviews exit/checkpoint/metadata, confirms the old job ended and
reconciles artifacts, recover a claim manually and resubmit only the explicit
failed index with --resume. Command/source/config stay identical. DONE alone is
not certification and never triggers automatic deletion or restart. Pilot/parent
roots, results and plans are separate; dev is excluded from confirmation.

## Future commands, execute only after stage-specific authorization

Use final delivered readiness SHA, not the original844e3c0. Fresh detached source:

```bash
export EXPECTED_COMMIT=FULL_DELIVERED_READINESS_SHA
export CHECKOUT=/absolute/new/readiness-checkout
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$CHECKOUT"
git -C "$CHECKOUT" fetch origin codex/exp12-launch-readiness
git -C "$CHECKOUT" checkout --detach "$EXPECTED_COMMIT"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=all)"
cd "$CHECKOUT/main"
```

For gate1, fill a COPY of `configs/launch_readiness/fullwidth_spec.template.json`
outside Git with exact source and explicitly approved diagnostic extents. The
wrapper verifies the pinned scaling-drl-py31213 environment and account/partition
are declared biqc-delta-gpu/gpuA100x4; owner must verify account membership and
partition availability locally. Precreate external Slurm log directory only.

```bash
export GATE_SPEC=/absolute/approved/fullwidth-spec.json
export GATE_ROOT=/absolute/new/fullwidth-evidence
mkdir -p /absolute/external/slurm-logs
# Submit ONLY the currently authorized stage; review it before requesting the next.
GATE_STAGE=reference_cold sbatch --export=ALL \
  --output=/absolute/external/slurm-logs/%x_%j.out \
  --error=/absolute/external/slurm-logs/%x_%j.err scripts/sbatch_exp12_fullwidth_gate.sh
# Later separate authorizations, same spec/source/root:
GATE_STAGE=reference_warm sbatch --export=ALL \
  --output=/absolute/external/slurm-logs/%x_%j.out \
  --error=/absolute/external/slurm-logs/%x_%j.err scripts/sbatch_exp12_fullwidth_gate.sh
GATE_STAGE=crash sbatch --export=ALL \
  --output=/absolute/external/slurm-logs/%x_%j.out \
  --error=/absolute/external/slurm-logs/%x_%j.err scripts/sbatch_exp12_fullwidth_gate.sh
GATE_STAGE=resume sbatch --export=ALL \
  --output=/absolute/external/slurm-logs/%x_%j.out \
  --error=/absolute/external/slurm-logs/%x_%j.err scripts/sbatch_exp12_fullwidth_gate.sh
# After all four completed artifacts are verified: CPU-only exact comparison.
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python scripts/exp12_fullwidth_gate.py \
  --spec "$GATE_SPEC" --out "$GATE_ROOT" --stage compare
# Seal the final comparison too, excluding the cache and checksum files.
(cd "$GATE_ROOT"; find . -type f ! -path './persistent_cache/*' \
  ! -name SHA256SUMS ! -name FINAL_SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > FINAL_SHA256SUMS)
```

Crash stage must exit3 and leave the declared checkpoint, not DONE/final state.
The other stages must exit0;124 is timeout. Expected evidence: spec.json,
attempts/<stage>_<job>/workload.log/process.json/exit_status.txt/SHA256SUMS,
dependencies/GPU metadata, each stage receipt/config/state, cold fork/panel/online
and target Check1/roundtrip, persistent cache, final validation.json. Preserve
Slurm allocation/stdout/stderr too. Timeout/incomplete/error never PASS.

For pilot or parents, fill the corresponding owner approval template COPY:
decision references, allocation numbers, final source and SHA256-bound gate PASS
JSONs. Production templates remain unapproved. `APPROVED` in a local file is the
owner's documented decision; the tool does not authenticate or invent approval.
Source/runtime actual checks still run inside each job.

```bash
python scripts/exp12_launch_plan.py prepare --stage pilot \
  --root /absolute/new/dev-pilot --commit "$EXPECTED_COMMIT" \
  --approval /absolute/owner/pilot-approval.json --out /absolute/new/pilot-plan.json
# Or later, only after ALL parent-grid gates:
python scripts/exp12_launch_plan.py prepare --stage parents \
  --root /absolute/new/confirmatory-parents --commit "$EXPECTED_COMMIT" \
  --approval /absolute/owner/parents-approval.json --out /absolute/new/parents-plan.json
```

Use allocation values EXACTLY from the approved plan; no defaults are supplied.
Prepare parent directories/Slurm logs externally. Reserve once before sbatch:

```bash
export LAUNCH_PLAN=/absolute/approved/pilot-plan.json  # or parents-plan.json
mkdir "$LAUNCH_PLAN.submission-reservation"  # refuses duplicate invocation
# Create PLAN_ROOT/slurm outside Git. This prints one exact command and DOES NOT execute it.
# It obtains CPU/RAM/walltime/concurrency from the approved plan and includes
# the correct0-194 array only for parents; no manual numeric substitution.
python scripts/exp12_launch_plan.py submission-command --plan "$LAUNCH_PLAN"
# Review/copy the printed command only after explicit submission authorization.
python scripts/exp12_launch_plan.py validate-run --run /absolute/completed/parent \
  --commit "$EXPECTED_COMMIT" --out /absolute/new/engineering-verdict.json
python scripts/exp12_launch_plan.py certify-identity --run /absolute/gpu/parent \
  --arm /absolute/gpu/identity-arm --commit "$EXPECTED_COMMIT" \
  --out /absolute/new/identity-case-receipt.json
python -m analysis.exp1_analysis --results-root /absolute/confirmatory/results \
  --study-manifest /absolute/frozen/study.json --out /absolute/new/exp1-analysis
# Exp2 is a later, separately authorized stage using the existing manifest/arm entry point.
python -m analysis.exp2_analysis --results-root /absolute/confirmatory/results \
  --study-manifest /absolute/frozen/study.json --out /absolute/new/exp2-analysis
```

Do not fill unresolved allocation values with arbitrary numbers. Scientific
certificate requires frozen study provenance, all required runs/arms and unchanged
canonical validation, independently of training exit status. No command here is
authorization to submit a job.

For later Exp2 preparation, first qualify the actual eligible natural forks,
same-device identity/Check1/Check2 behavior, positive control and frozen m. Use
the frozen owner decision, never a value selected by this tool. The existing
generator only writes commands and does not submit them:

```bash
export APPROVED_FROZEN_M=OWNER_APPROVED_LAST_HALF_OR_ALL
export CONFIRMATORY_CHECKPOINT_ROOT=/absolute/confirmatory/checkpoints
export CONFIRMATORY_RESULTS_ROOT=/absolute/confirmatory/results
export ARM_MANIFEST_DIR=/absolute/new/exp2-manifests
mkdir "$ARM_MANIFEST_DIR"
(cd "$ARM_MANIFEST_DIR"; python "$CHECKOUT/main/generate_manifest.py" \
  --grid exp12 --ckpt-root "$CONFIRMATORY_CHECKPOINT_ROOT" \
  --results-root "$CONFIRMATORY_RESULTS_ROOT" --injection-m "$APPROVED_FROZEN_M")
```

Inspect every generated `exp2_arms_NVIDIA_A100_SXM4_40GB.txt` row against the
validated FORK_READY parent, frozen source/configuration and approved arm
allocation. Preserve/hash manifests and metadata; refuse unknown/mismatched
devices, duplicate arms or unvalidated DONE markers. These legacy command lists
are not approval-gated launch plans: do not pipe them into a scheduler. A
separately reviewed arm submission is still required. Individual scientifically
eligible forks do not depend on every other parent finishing; final canonical
study reporting requires the declared complete population and missing-run rules.

CPU test receipts and exact verification scope are in
`docs/exp12_launch_validation_report.md`.
