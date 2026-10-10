# Full-length D4W1536 dog-run development pilot

2026-10-10, America/New_York. **Preparation only; no submission authorization.**
This report makes no code, methodology, tolerance, trigger, environment or budget
change. No jobs were submitted, branches merged or production grid approved.

## Source, budget and evidence disposition

Use the exact GPU-qualified source:
**`7d21eaa106a336168464cd5418a59136b6936a64`**,
`codex/exp12-launch-readiness`. Live remote verified unchanged. The corrected CPU
comparator is `07721bf27055c2c407f635149bd9bbf54e97e989`; it is not substituted
as the source that generated the GPU evidence. This report branch is
`codex/exp12-full-length-pilot-plan`; its enclosing commit is documentation only.

**Budget discrepancy requiring clarification:** the owner requested “1-million
interaction-step” while also requiring the existing methodology and unchanged
budgets. Authority `.claude/methodology-exp1-exp2.md:82`,
`generate_manifest.py:219`, `configs/env/dmc_hard.yaml` and
`configs/base_exp12.yaml:35` specify **1,000,000 raw environment steps = 500,000
trainer interactions** at action repeat 2 and one training environment. This is
also the pilot budget in `docs/exp1_pilot_readiness.md`. The commands below prepare
that canonical run; they do not satisfy a literal 1,000,000-interaction request.
Such a run would require an explicitly approved separate development budget and
new plan/source qualification. None is selected here. Clarify the unit before
submission; no `num_env_steps`/`num_interaction_steps` override is introduced.

The owner officially reports the four-stage gate PASS, comparator exit 0 and all
reference/resume/round-trip differences empty. Reported jobs: cold 22791700;
failed warm 22791701 retained; successful warm 22793638 with 172 GPU cache hits;
expected crash 22793941; resume 22795539. The complete new receipts, payloads,
logs and corrected JSON are **not accessible in this Cloud workspace**. Their
Delta location is known, but no Delta access was attempted:
`/work/hdd/biqc/skaveti1/exp12_validation/7d21eaa106a336168464cd5418a59136b6936a64_fullwidth_20261010T163647Z`.
The final corrected comparator output path was not supplied. Thus PASS is
owner-reported, not an independently repeated artifact audit in this task.

Original gate provenance/attempt-accounting implementation and launch-readiness
reports were reviewed. Before launch, retain the real corrected PASS JSON at its
original external path and hash-bind it in the owner approval. It must say
`expected_commit=7d21eaa...`, `comparator_commit=07721bf...`,
`qualification=fullwidth_resume_identity`, cover dog-run/D4W1536/DMC, identify the
accepted jobs and preserve the failed warm attempt. Review stage timings,
compilation totals, allocator peaks, host RSS, actual allocations and checkpoint
bytes from each receipt. Missing evidence is not filled with inferred values.

Accessible older evidence: job 22765261 at `9d6a82d4aef81299901177865e383640439cde2a`.
Original ZIP SHA-256 reverified:
`ceea3feb7d1b123d6ea68bee9351eae3f890b32bcd58ee1d26bd2ba82e78c720`.
Its actual profile, validation, log and complete trace were reread. It is a short
runtime diagnostic, not the latest four-stage gate or a full-run measurement.

## Exact scientific run and proposed allocation

| Item | Unchanged setting / proposal |
|---|---|
| Role and source cell | Exp1, `run_role=dev`, dog-run, D4W1536, seed 102; excluded from confirmatory analyses |
| Actor/critic | D1W128 actor; D4W1536 critic; single critic for DMC |
| Nominal budget | 1,000,000 raw steps / 500,000 interactions; action repeat 2, UTD 2 |
| Probe | Enabled; 5 rounds, 1,000 fitting steps, 25,600 pool, 256 batch, 20 scheduled checks plus initial fresh check |
| Natural trigger | Unchanged lower CI bound > 0, two consecutive checks, eligibility through check 19; no forced trigger |
| Fork/control | Existing natural fork; 125,000-interaction post-fork horizon; control ends at max(500,000, fork+125,000), at most 600,000 |
| Checkpoints | Every 25,000 interactions from fraction 0; final state; latest complete state retained, previous routine state pruned |
| Precision | Existing initialization, FP32 arrays, production matmul policy and Check 1 policy preserved |
| Allocation proposal | One A100-SXM4-40GB, 16 CPUs, 64 GB RAM, 08:00:00 wall time; one non-array job |
| Scientific exclusions | No injected/identity job, no m selection, no Exp3 recording claim, no production qualification |

The Exp1 process itself continues its approved control pathway if it naturally
forks; this is not a separately submitted Exp2 arm. Seed 102 is not a selected
confirmatory Exp3 source. No recording selection decision is made here.

## Runtime, memory and storage: measured versus estimated

Actual accessible job 22765261 measured 42.4411 interactions/s over only 60 warm
interactions with probes off, 35.8204 s per warmed paired probe check, 6.0740 s per
save at synthetic 475,000 replay transitions, 2.5124 s per restore and 22.8342 s
per ten-episode post-fork evaluation. The trace records 305 backend compilations
summing to 29.8581 s, 16.638 s runtime import, 12.429 s first trainer initialization,
15.465 s first structural metric call and a 1.1887 s grouped flush of 1,001 groups.
Compilation intervals overlap enclosing initialization/training/probe intervals;
do not add every interval as independent overhead.

| Component | Canonical no-fork model | Latest eligible fork/control model | Limitations |
|---|---:|---:|---|
| Training alone | 500k / 42.4411 = **3.2725 h** | 600k / 42.4411 = **3.9270 h** | Short window; warmup and later sustained rate differ |
| Probes | 21 x 35.8204 s = **12.54 min** | 25 x 35.8204 s = **14.93 min** | Fresh check is different; full-setting later-pool cost unmeasured |
| Routine saves | 20 x 6.074 s = **2.02 min** | Up to 24 routine saves, plus fork save/rebuild | State/replay size and filesystem load vary |
| Ordinary evaluations | 100 episodes x 2.2834 s = **3.81 min** | 200 episodes = **7.61 min** | Derived from actual trainer schedule; source multiplies final/extended 50k evaluations to 50 episodes |
| Post-fork evaluations | None | 26 x 22.8342 s = **9.89 min** | Separate from ordinary evaluations; early forks also incur these |
| Startup/recompilation | Cold import/init/compilation/first structural work | Additional restore/rebuild may recompile | Latest cold/warm gate receipts needed; no fixed measured full-run total |
| Metric/logging windows | 250 windows; costs additional to the 60-step rate | Up to 300 windows | 2,000-interaction groups larger than the measured 1,001-group flush; scaling is a hypothesis |

Existing readiness planning gives about **3.7 h without a fork / 4.6 h with a late
fork**. Treat roughly 3.7–5 h as planning scenarios, not confidence intervals or a
completion guarantee. The pre-mitigation Block B timeout beyond six hours remains
historical caution; the new short gate does not prove full-run sustained runtime.
Eight hours is a reasonable provisional envelope for the **canonical** run, with
about 1.6–2.2x scenario margin. It is not yet independently verified against the
latest gate's timing/RSS receipts or Delta's partition/account limits. A literal
1M-interaction run would already project 6.55 h of training alone; these commands
and estimates must not be reused to justify that unapproved doubled budget.

Old diagnostic allocator peak: **3,763,942,656 bytes = 3.51 GiB**, not total GPU
reservation or a full-run bound. One 40-GB A100 is plausible; no packing is proposed.
The older snapshot was 2,035,145,345 bytes at 475k transitions. Replay capacity is
1M transitions but checkpoint files contain only populated transitions. Host peak
at full probes/control/restore is unmeasured. The passed short stages' 32-GB
requests, without their RSS records, do not prove a 32-GB full-run requirement.
64 GB is provisional headroom. Sixteen CPUs are requested capacity, not 16
parallel environments; preserve the wrapper's existing thread/environment setup.

Existing storage estimate is about 5 GB routine state or 10 GB including a fork,
plus cache/probes/logs. Atomic saving temporarily keeps old and new complete
states; a recovery archive adds another copy. Plan at least about 20 GB owned
scratch headroom plus measured cache/log/archive growth, not a certified maximum.
Do not retain all 20 routine snapshots by changing pruning. Verify actual quota,
space, inodes and backup destination before authorization.

## Preparation and exact future submission

The commands below are **not executed by this review**. They stop on an existing
checkout/plan/reservation and require a real owner approval file. They do not
create approval or fabricate a PASS receipt. Use the actual corrected JSON path
and SHA-256 in a copy of `configs/launch_readiness/pilot_approval.template.json`.
That file must remain unapproved until the owner approves the dev-only natural
zero-trigger disposition and the exact allocation. Resource fields then are
`gpus:1, cpus:16, memory_gb:64, wall_minutes:480, concurrency:1`. Its
`expected_commit` is the original GPU source, not the comparator/report revision.
The approval gate evidence is `fullwidth_resume_identity`; all decision references
must name real owner decisions, not this report as automatic approval.

```bash
set -euo pipefail
export EXPECTED_COMMIT=7d21eaa106a336168464cd5418a59136b6936a64
export CHECKOUT=/work/hdd/biqc/skaveti1/exp12_sources/7d21eaa106a336168464cd5418a59136b6936a64_pilot01
export PILOT_ROOT=/work/hdd/biqc/skaveti1/exp12_dev/dog_run_D4W1536_seed102_7d21eaa_pilot01
export PILOT_ADMIN=/work/hdd/biqc/skaveti1/exp12_pilot_plans/dog_run_D4W1536_seed102_7d21eaa_pilot01
export LAUNCH_PLAN=$PILOT_ADMIN/pilot-plan.json
export PILOT_APPROVAL=$PILOT_ADMIN/pilot-approval.json
mkdir -p "$(dirname "$CHECKOUT")" "$PILOT_ADMIN"
test ! -e "$CHECKOUT"
test ! -e "$PILOT_ROOT"
test ! -e "$LAUNCH_PLAN"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$CHECKOUT"
git -C "$CHECKOUT" checkout --detach "$EXPECTED_COMMIT"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=all)"
cd "$CHECKOUT/main"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/exp12_launch_plan.py run.py experiments/exp1.py configs/base_exp12.yaml
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1
python -B -c 'import sys; from scripts.exp12_launch_plan import require_pinned_stack; assert sys.version_info[:3] == (3,12,13); require_pinned_stack()'
# Owner supplies the real, reviewed approval and SHA-bound comparator receipt.
test -f "$PILOT_APPROVAL"
scontrol show partition gpuA100x4 > "$PILOT_ADMIN/partition.txt"
sinfo -p gpuA100x4 -o '%P %l %G %c %m %a' > "$PILOT_ADMIN/partition-capacity.txt"
sacctmgr -nP show associations where user="$USER" account=biqc-delta-gpu > "$PILOT_ADMIN/account-associations.txt"
df -h "$PILOT_ADMIN" > "$PILOT_ADMIN/filesystem-space.txt"
df -i "$PILOT_ADMIN" > "$PILOT_ADMIN/filesystem-inodes.txt"
# Inspect these plus site quota output: confirm authorized account, A100 model,
# one GPU/16 CPU/64 GB availability, >=8h MaxTime and sufficient owned storage.
# If association details are restricted, obtain site/account-owner confirmation.
python -B scripts/exp12_launch_plan.py prepare --stage pilot \
  --root "$PILOT_ROOT" --commit "$EXPECTED_COMMIT" \
  --approval "$PILOT_APPROVAL" --out "$LAUNCH_PLAN"
# Verify one entry, seed102/dev, canonical 1M raw/500k interaction config.
# Config hash: 7efc91337a1b200b8c437dbea5f5f0f1d251df056cb8bbb5dc0e0fd790ee924a.
mkdir -p "$PILOT_ROOT/slurm"
mkdir "$LAUNCH_PLAN.submission-reservation"
python -B scripts/exp12_launch_plan.py submission-command --plan "$LAUNCH_PLAN" \
  > "$PILOT_ADMIN/reviewed-submission-command.txt"
```

The exact command expected from that plan, **only after explicit submission
authorization**, is:

```bash
env EXPECTED_COMMIT=7d21eaa106a336168464cd5418a59136b6936a64 \
  LAUNCH_PLAN=/work/hdd/biqc/skaveti1/exp12_pilot_plans/dog_run_D4W1536_seed102_7d21eaa_pilot01/pilot-plan.json \
  sbatch --parsable --export=ALL --cpus-per-task=16 --mem=64G --time=480 \
  --output=/work/hdd/biqc/skaveti1/exp12_dev/dog_run_D4W1536_seed102_7d21eaa_pilot01/slurm/%x_%A_%a.out \
  --error=/work/hdd/biqc/skaveti1/exp12_dev/dog_run_D4W1536_seed102_7d21eaa_pilot01/slurm/%x_%A_%a.err \
  scripts/sbatch_exp12_prepared_parent.sh \
  > "$LAUNCH_PLAN.submission-reservation/job-id.txt" \
  2> "$LAUNCH_PLAN.submission-reservation/submission.stderr"
```

Run from the checked-out `main/` directory. The unmodified script requests account
`biqc-delta-gpu`, partition `gpuA100x4`, one node, one task and one GPU. No array is
specified. `--time=480` is 08:00:00. The printed plan command must match; otherwise
stop. Never execute both the printed command and the shown command as two jobs.
Do not repeat a submission with a reservation/job ID uncertainty: reconcile with
Slurm first. After a successful single submission, record `scontrol show job`
and `sacct` in PILOT_ADMIN and confirm the effective allocation; do not silently
accept a different CPU/RAM/time allocation or GPU model.

Inside the allocation the existing script reinstates `cuda,cpu`, clears the CPU
platform-name override, checks pinned Python/JAX and actual A100/CPU availability,
validates clean source, approval hash, config/command, allocation and no-array
identity, and atomically claims the run. It uses a job-private cache. It launches
the unchanged full Exp1 entrypoint, with no 300-second diagnostic timeout. This is
a proposed new full-run allocation, not a modification to historical gate caps.

## Expected artifacts and recovery

`PILOT_ROOT/checkpoints/exp1/D4W1536/dog-run/seed_102` is the run directory.
Expect run_metadata.json (resolved config, protocol/source/launch provenance),
state/LATEST and its complete agent/replay/RNG/env/normalization/counter state,
fresh_critic, probes/probe_checks.csv and check_00.npz through check_20.npz,
logs/* metric/evaluation CSVs, and DONE only after final save/ledger publication.
An extension adds checks through floor(endpoint/25,000), up to check24. If an
eligible natural fork exists, expect fork/fork.json, fork/FORK_READY, immutable
fork state, panel and original/control Check 1 files. No trigger means no required
fork: it is a valid null, not permission to force one.

Results are under PILOT_ROOT/results/exp12/exp1/runs/
`exp1_D4W1536_dog-run_seed102` (run/checks/metrics CSVs and probe_curves.npz); a
natural fork also produces results/exp12/exp2/<run_key>/fork.json and arm_control
checks/eval/metrics/source files. Attempts are under
PILOT_ROOT/attempts/<run_key>/<jobid>_0, containing training.log, exit_status.txt
and jax_cache; claims/<run_key>/owner.json is retained on success/failure.
Slurm logs live under PILOT_ROOT/slurm; a single-job `%a` can be Slurm's non-array
sentinel. Abrupt Slurm termination can prevent exit_status.txt: scheduler status
then matters and its absence never proves success. Preserve approval, plan,
submission record, actual allocation, accounting including .batch MaxRSS, logs
and checksums outside the source checkout. The unmodified pilot does not emit a
full per-stage profiler or a device-memory high-water report; do not invent these.

Routine checkpoints are every 25k interactions (about 9.8 training-only minutes
at the old measured rate). Inspect LATEST/meta/probe ledgers for progress, not
only stdout: the disabled WANDB logger need not print each progress window.
Eight-hour TIMEOUT, OOM, node failure, exception, disk-full, invalid/nonfinite
payload, source/config/backend mismatch or failed Check 1 means stop and preserve
all evidence. Do not shorten training, weaken criteria, change threads/precision,
force a fork, choose m, change partition or automatically resubmit. A return drop
alone is a scientific observation, not an automatic numerical-error verdict.

For recovery, first confirm the old job is terminal and no writer remains.
Archive/checksum the current complete state and metadata before starting another
writer. Inspect state/LATEST with latest_state_dir/load_meta and compare metadata
source/config/role; ignore uncommitted orphan directories, never repoint LATEST
to a newer-looking incomplete directory. Review any FORK_READY/fork state with
its Check 1 and same GPU model; entrypoint logic restores the appropriate fork
state and continues control when required. If there is no valid routine checkpoint,
claim/fork-only ambiguity or corruption, stop for recovery engineering review.

Only after separate recovery/submission authorization, preserve the old claim by
moving it to a fresh claim-history location outside claims/<run_key>; do not delete
its owner record or steal a live claim. Reuse the identical plan, approval,
checkpoint/results roots and exact source. Reserve a distinct recovery submission
record, retain the original reservation, and submit the same resource/script
command with `EXPLICIT_RESUME=true` in the env prefix. That flag makes the existing
wrapper pass `execute --resume`, which validates the original config/source and
LATEST before claiming. Do not edit the plan or raw evidence to clear a failure.
A new attempt/cache/log directory is created; prior attempts remain. The restarted
trajectory may replay work since the last committed save; it does not add a new
scientific horizon. An eight-hour overrun does not authorize a longer allocation.

## Post-run acceptance and next qualification

Use an external fresh CPU receipt path and the exact original source:

```bash
export RUN=$PILOT_ROOT/checkpoints/exp1/D4W1536/dog-run/seed_102
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python -B scripts/exp12_launch_plan.py \
  validate-run --run "$RUN" --commit "$EXPECTED_COMMIT" \
  --out "$PILOT_ADMIN/post-run-engineering-verdict.json"
# This command writes a verdict; independently assert it, not only its exit code.
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python -B - <<'PY'
import json, os
from pathlib import Path
from experiments.exp12.state import latest_state_dir
from scripts.exp12_launch_plan import verify_saved_gpu_sharding
report = json.loads((Path(os.environ['PILOT_ADMIN']) / 'post-run-engineering-verdict.json').read_text())
assert report['engineering_verdict'] == 'COMPLETE', report
assert report['scientific_verdict'] == 'DEV_EXCLUDED', report
verify_saved_gpu_sharding(latest_state_dir(Path(os.environ['RUN']) / 'state'))
PY
```

- Slurm COMPLETED/exit0, training exit0, no OOM/traceback, final save + DONE;
  consistent job ID and one owner across attempts. Failed/recovered attempts retained.
- Source 7d21eaa, clean config hash above, seed102/dev; pinned stack and actual A100;
  no testing flags, budget override, metric/probe/precision change or injected arm.
- Nominal endpoint500k and **990,002 updates** without an extended control;
  with a fork, endpoint=max(500k,fork+125k), at most600k/**1,190,002 updates**;
  generally updates=2*(endpoint-4999), update_counter0, restored counters coherent.
- All scheduled fresh/paired checks and arrays present, valid and finite; complete
  replay/normalization/RNG/environment/optimizer/targets/log-window state; ledgers,
  eval episodes, metrics and DONE agree. Check numerical trajectories independently,
  not just final-state finiteness. No score cutoff is newly introduced.
- Natural trigger recomputable under unchanged rules; fork proof/control Check 1
  bit-exact, 26 post-fork evaluations present when eligible, full horizon retained.
  No natural fork is valid null evidence but leaves fork/positive-control work open.
- Record full elapsed time, sustained checkpoint-to-checkpoint progress, saves,
  evaluation/probe pauses if logged, accounting RSS and disk/cache peaks actually
  available. Distinguish measured total from modeled per-stage costs; absent GPU
  peak/phase timing remains unmeasured. Investigate regressions before scaling.
- Archive original artifacts and checksums; dev ledger excluded from confirmation.
  Do not label COMPLETE or this development pilot as scientific qualification.

A clean complete pilot supports broader **qualification**, not the 195-parent
grid. Remaining independent gates include scaled-size/suite identity (both widths
in DMC/Myo/HB), twin-critic/full-width restore/runtime coverage, fresh-null and
hopper disposition, positive-control sensitivity and owner-frozen m, approved
numeric bounds where needed, suite storage/runtime allocations and study freeze.
An eligible dev fork can supply later separately authorized identity/positive-
control experiments; no fork means consult rather than force or create an
unapproved source. Exp2 injected jobs and final reporting remain separately gated.

## Preparation receipt / handoff

CPU-only synthetic plan/approval fixtures verified one canonical pilot entry,
config hash above, 16/64/480 allocation formatting and actual-allocation validator,
no array and no run-root creation. Synthetic fixtures are not GPU evidence or
owner approvals. Launch/config/entrypoint files were byte-compared against 7d21eaa;
unchanged. Bash/Python snippets are syntax-checked; no Slurm command executed.
Only this report is committed on the isolated branch. Generated preparation
receipt stays outside Git at /workspace/scratch/exp12-full-length-pilot-preparation.json.

Before submission: clarify raw-versus-interaction budget wording; make the real
new gate receipts/comparator JSON available for independent timing/integrity
review; confirm Delta account/partition/quota suitability; approve the dev-only
natural-trigger disposition, allocation and one actual submission. None is
silently inferred from the gate's engineering PASS.
