# Exp96 diagnostic engineering implementation

Implementation branch: `codex/exp96-diagnostic-engineering-20261023`, based on diagnostic handoff `6af1bb463eba6d8856606bc040d003d1cb4df2be`. Resolve the delivered final SHA from Git. The first independently reproduced/fixed defect is committed at `b98e5ed`; subsequent commits complete instrumentation, explicit plans, execution preparation and export. No original Exp96/SAC/probe implementation or frozen scientific settings were changed.

## Correctness and scope

The old constructor selected generic `sac.yaml` (critic LR `3e-4`, decay `0`). A constructor-level regression failed with `0.0003 != 0.0001` before correction. The constructor now reads all canonical queue parents' `run_metadata.json`, validates exact frozen source/clean launches, logical identities, resolved config hashes, supported single residual critic dimensions, finite optimizer values and agreement between parents. It builds one shared equivalent `critic_optimizer` transformation using metadata-derived settings, without a generic-config fallback. Missing/conflicting metadata stops before fitting. Results include every original metadata/config hash and AdamW numerical defaults. Canonical population duplication is rejected instead of being silently collapsed by a dictionary.

`diagnostic_artifacts.state_dir` confines LATEST to an existing snapshot beneath its own state root. CPU Orbax fixtures verify exact fresh, latest and early-fork critic leaves, read-only replay, unchanged checkpoint bytes and parameter-tree identities. Actual production checkpoint reconstruction on the corrected final source remains a Delta validation step; fixture success is not production qualification.

Fixed numeric tasks remain immutable, checksummed and reloaded before training. Common zero/shared targets, inputs and minibatch streams retain identical task IDs across learners. Own-offset and incremental targets are explicitly labelled learner-specific; they must not be reported as identical common tasks. Incremental construction saves starting predictions, not merely a target recipe. Original first-2k index construction and the production `probe._fit` are unchanged. Continuous fitting carries parameters and AdamW state through every 1k segment and emits 2k/4k/8k measurements without restarting the optimizer.

The known failed GPU job `23120792` used old source `bcf6cf3c90726c77f998e5795fedf0d62c5f11e2`: 181 seconds on one A100; the cancelled job `23120674` used zero. The Mac worker independently reconciled all investigation allocations and restarts. Cumulative recorded use is 181/3600 = 0.05027778 physical A100-hours; 14219 GPU-seconds remain before new allocations. Remote historical budget.json is stale and must remain preserved as original evidence.

The old segmentation final error `0.00405173` exceeded the existing guard `0.000751385`. Its cause is unresolved. The corrected runner preserves that guard, retains both full loss curves, numerical differences, exact-repeat observations and state hashes, and stops on failure. No tolerance was loosened and no GPU equivalence claim is made. Within-process fixed-task CPU segmented/monolithic tests cover a small real SACCritic as well as Tiny fixtures. Cross-allocation optimizer checkpoint/resume is not implemented; a saved loss curve is not a resumable optimizer checkpoint.

## Implemented measurements and schemas

`fixed_task.first_update` performs an isolated functional AdamW update from copied starting state. It records minibatch MSE, gradient L2, full-pool prediction-change RMS/mean and full-pool before/after MSE. This is a separate diagnostic kernel, not proof that every compiled GPU scan has bitwise-identical first-step behavior. Input parameters/optimizer state remain unchanged.

`results.json` retains `source`, `frozen_source`, `optimizer`, `optimizer_sha256`, the full original `inputs` provenance, `plan`, `plan_sha256`, condition statuses, fit/reconstruction rows, synchronized timings and JAX memory stats. Each controlled fit row includes:

- Identity: name/environment/architecture/seed/kind/offset mode, immutable task ID, starting-parameter and final-state hashes, optimizer hash and starting-state preservation.
- Initial discrepancy: full-pool residual RMS, mean error, centered error variance, Q mean/variance, target mean/variance, and isolated first-update measurements.
- Trajectory: full-pool MSE/decomposition every 1k updates, optimizer count, reduction in MSE per update over each segment, and pre-update minibatch loss curves.
- Early amplification: initial loss, first-at-most-100 mean/peak, and ratios to initial loss. Ratios are null when the initial loss is zero; there is no arbitrary epsilon or scientific cutoff.

Per-fit `progress.json` is atomic. Numeric NPZ outputs are published atomically; `numeric_task/` retains the immutable manifest/arrays. Every completed segment persists `curve_through_NNNN.npz` and updated progress, even if later work fails. Fitting, explicit lowering/compilation, input transfer and other synchronized calls have separate labels. First-use metric/target kernels can include compilation; their timing is not advertised as steady-state throughput. Runtime.json and process-specific telemetry retain precision/device/source bindings and memory observations. Serialization/host-copy overhead belongs to the recorded allocation and overall elapsed time.

Root output must be new. `status.json` distinguishes `FAILED_PREFLIGHT`, `FAILED`, `PARTIAL_BUDGET` and `COMPLETED_BUDGETED_STAGE`. Partial plans exit 3; failures are nonzero. Complete bounded execution is not scientific qualification. Budget omissions remain explicit in condition/fit records; zero executed conditions cannot become a completed plan.

## Reusable experimental configurations

All templates are under `main/configs/diagnostics/`. They prepare experiments; they neither authorize them nor submit anything. A plan strictly selects explicit environment, architecture(s), seed, target round, learners, target constructions, horizons and replay/normalization settings. Unknown fields are rejected. No LR/decay sweep or implicit full matrix exists.

| Template | Prepared contrast |
|---|---|
| reconstruction.json | Original pen-twirl D2W512 seed 5 and humanoid D4W1536 seed 1 reconstruction, repeat and segmentation guard |
| architectures.json | Fresh small/medium/large learners on a common large-teacher numerical task |
| training_states.json | Fresh/early/latest learners on a shared numerical task |
| pen_targets.json | Pen-twirl seeds 5/7, own/shared-reference/incremental targets |
| normalization.json | Early/latest-normalizer, early/early-normalizer input-only crossover, and restricted late replay |
| extended_replication.json | Pen seeds 5/7, continuous 2k/4k/8k fitting |

`shared_reference` explicitly names the learner whose pool mean supplies the shared scalar offset. The teacher, target scale and keyed minibatch construction remain fixed. `target_normalizer` declares whether labels are regenerated on each input condition or held to another normalization coordinate system. Pools with reconstructed initial normalization are labelled retrospective. Indexed support restriction is not a claim that sampled supports are nested or that geometric diversity is equal. Templates select round 0 screens; they are not ten-target IQM10 or scientific seed qualification. Add explicit condition IDs/rounds only as part of a reviewed scientific plan.

## CPU validation

Use the already prepared Python environment; do not reinstall Delta packages:

```bash
source /workspace/setup/scaling-drl/activate.sh
export MUJOCO_GL=disable JAX_ENABLE_X64=false
python -B -m unittest -v \
  tests.test_fixed_task tests.test_probe_tasks \
  tests.test_fixed_task_bank_integration \
  tests.test_exp96_diagnostic_engineering tests.test_exp96_diagnostic_execution
```

The implementation tests exercise metadata mismatch refusal, real AdamW one-step measurements against an independent reference, real small-residual 2k/4k/8k continuation, task reuse, actual fit/progress integration, exact checkpoint reads, budget omissions, old GPU guard rejection, accounting, live Slurm preflight with mocked scheduler responses, and partial evidence export/checksums. These do not fit production-size critics or use GPU hardware. Raw before-fix and subsequent test logs are outside Git at `/workspace/artifacts/echocritic-access/` in the implementation Cloud task. Refer to the delivery for the final test count and source revision; do not substitute older setup-suite results.

## Delta execution preparation (not executed)

The next scientific experiment and its allocation require owner approval. The following commands are for the authenticated execution agent after that approval; the owner should not act as a command runner. Cloud-to-Delta dispatch is not established by these scripts. Preserve all old checkouts and evidence. Stage the reviewed final Git SHA in a new clean detached checkout under a new durable investigation location, using HTTPS Git or an approved source bundle; never update a checkout used by an old job.

From the new checkout's `main/`, with the existing Delta Conda Python available, prepare fresh read-only accounting and a plan-bound receipt. Replace the example paths/SHA with the reviewed delivery and fresh durable paths:

```bash
export EXPECTED_COMMIT=FULL_REVIEWED_40_CHARACTER_SHA
export FROZEN_CAMPAIGN=/projects/biqc/skaveti1/exp96/campaign_owner_f597ee9_20261019T230748Z
export DIAGNOSTIC_ROOT=/projects/biqc/skaveti1/exp96/NEW_DIAGNOSTIC_OUTPUT
export DIAGNOSTIC_PLAN="$PWD/configs/diagnostics/reconstruction.json"
export DIAGNOSTIC_BUDGET_RECEIPT=/projects/biqc/skaveti1/exp96/NEW_BUDGET_RECEIPT.json
export DIAGNOSTIC_CAP_SECONDS=3000
TZ=UTC sacct -D -nP -u skaveti1 --starttime 2026-10-22T22:00:00 \
  --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,Restarts,Partition \
  > /projects/biqc/skaveti1/exp96/NEW_ACCOUNTING.txt
/u/skaveti1/.conda/envs/scaling-drl-py31213/bin/python -B \
  scripts/check_exp96_diagnostic_budget.py \
  --accounting /projects/biqc/skaveti1/exp96/NEW_ACCOUNTING.txt \
  --jobs 23120674 23120792 --plan "$DIAGNOSTIC_PLAN" \
  --commit "$EXPECTED_COMMIT" --allocation-seconds 3600 \
  --out "$DIAGNOSTIC_BUDGET_RECEIPT"
```

Any new investigation allocation must join that inventory. Unknown allocations, pending/running jobs, duplicate records, missing rows and restarts block admission rather than being silently omitted. This conservative account-wide query also blocks on unrelated overlapping jobs: investigate and obtain a scoped inventory decision rather than filtering them silently. Requested time is a reservation ceiling, not actual used GPU-hours. Receipt generation is preparation, not scientific approval or job submission.

Only after the separate owner approval, the execution agent may use this command shape:

```bash
sbatch --parsable --no-requeue --job-name=exp96_corrected_reconstruction \
  --account=biqc-delta-gpu --partition=gpuA100x8-interactive \
  --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=60G \
  --time=01:00:00 --constraint=projects --export=ALL \
  --output=/projects/biqc/skaveti1/exp96/NEW_SLURM_%j.out \
  --error=/projects/biqc/skaveti1/exp96/NEW_SLURM_%j.err \
  scripts/sbatch_exp96_fixed_task_diagnostics.sh
```

The wrapper validates exact clean source, one GPU, the explicit plan, and a budget receipt. Before fitting it re-queries account-wide sacct/squeue and the actual Slurm allocation, rejects requeue/competing jobs and oversized walltime, and checks used + outstanding reserved + full new allocated walltime <=14400 GPU-seconds. The 3000-second work cap, 300-second timeout margin and 60-second termination grace fit the one-hour Slurm limit. The existing environment, TF32 policy and 16 CPU/60 GiB allocation are retained. No package upgrade or automatic retry occurs.

This is a single owner-controlled allocation workflow, not an atomic multi-controller reservation service. Do not run concurrent submitters or admit successors automatically. A later submission can race any snapshot; a general autonomous scheduler requires a separately reviewed locked ledger/controller. Reconcile failed and idle time after each actual allocation. Inspect reconstruction and GPU segmentation evidence before selecting broader contrasts; a bounded exit/status cannot certify historical target identity.

## Compact evidence export

On a machine that actually has the evidence and NumPy installed:

```bash
python -B scripts/export_exp96_diagnostics.py \
  --root "$DIAGNOSTIC_ROOT" --out "${DIAGNOSTIC_ROOT}-export"
```

Default export contains runtime/status/results, reconstruction/segmentation receipts, fit progress, loss curves, a flattened `measurements.csv`, source-file hashes, and `SHA256SUMS.txt` for the review archive/CSV/manifest. Failed and partial status stays explicit. `--include-tasks` adds potentially large numeric tasks and pools for reproducible reanalysis; no replay buffers or critic checkpoints are included. Symlinks and overwriting an export directory are refused. A missing final status is labelled, not promoted to success; failed-preflight runtime/status can still be exported.

Transfer the archive through the authorized local executor or approved artifact storage, verify checksums after transfer, and provide CSV/manifest to the independent scientific analyst. The exporter does not transfer credentials or implement a Cloud/local messaging channel. Nonfinite numeric evidence is retained and labelled; corrupt NPZ files fail integrity inspection and must be preserved separately for forensic review.

## Remaining GPU-only and scientific decisions

No new GPU job ran in this implementation task. Required next evidence is corrected-source reconstruction, original/segmented comparison and repeatability on A100/TF32, actual production-size memory/timing, and metadata validation against the full real selected checkpoint set. The known optimizer correction must not be presented as the proven cause of the old segmentation failure. If divergence persists, preserve both curves and diagnose arithmetic/compilation/state boundaries without relaxing guards.

The owner chooses the next contrast, replication scope and allocation. Named shared/incremental/normalization tasks are retrospective diagnostic alternatives; they do not replace the official detector. CPU success, a lower fitting loss, or a completed bounded stage grants no scientific validation or production permission.

## Delivery validation and changed files

The final targeted run executed **36 tests: 36 passed, zero failures/errors/skips** (13 existing contracts and 23 new tests). Python exit status was 0. Pyflakes, Bash syntax, CLI help and Git whitespace checks also passed. GPU/Slurm operations in entrypoint/budget tests are mocked; they are not GPU qualification.

```text
main/configs/diagnostics/architectures.json
main/configs/diagnostics/extended_replication.json
main/configs/diagnostics/normalization.json
main/configs/diagnostics/pen_targets.json
main/configs/diagnostics/reconstruction.json
main/configs/diagnostics/training_states.json
main/docs/AGENT_HANDOFF.md
main/docs/EXPERIMENT_STATUS.md
main/docs/exp96_diagnostic_engineering.md
main/experiments/exp12/diagnostic_artifacts.py
main/experiments/exp12/diagnostic_budget.py
main/experiments/exp12/diagnostic_plan.py
main/experiments/exp12/fixed_task.py
main/scripts/check_exp96_diagnostic_budget.py
main/scripts/exp96_fixed_task_diagnostics.py
main/scripts/export_exp96_diagnostics.py
main/scripts/sbatch_exp96_fixed_task_diagnostics.sh
main/tests/test_exp96_diagnostic_engineering.py
main/tests/test_exp96_diagnostic_execution.py
```

Reviewable delivery: [PR #8](https://github.com/kaves29/scaling-drl-research/pull/8), targeting the diagnostic branch. Implementation/test checkpoint: `ac360d9a07f3632a25a8f6401fcf950549f9208d`; later publication edits are documentation only. Full commit history is retained in the PR.
