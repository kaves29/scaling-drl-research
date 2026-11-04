# Official Exp1 Delta launch handoff — 4 November 2026

This packet prepares the owner-requested launch; it is not evidence of a submitted
job or passed GPU/resource gates. The exact source SHA is supplied in the final
Cloud validation receipt/prompt, not inferred from a moving branch. Parent for
this change is58e6a97e0096db330e52a503b32c856df342f929. Candidate branch:
`codex/exp1-missing-run-manifest-20261104` (PR25, on PR24). Do not substitute
integration844 or cherry-pick PR19/20 twice. No merge is necessary to execute a
separately owner-selected pinned candidate, but that selection must be recorded.

## Owner choices and fixed science

- Eleven tasks; ten seeds1–10 explicitly frozen, including development-exposed9.
- Small D2W512 / medium D4W1024 / large D4W1536; fixed actor D1W128.
- Ten randomized sine targets,2,000 continuous updates,50 checks plus the
  existing fresh-only baseline. Preserve source-matched first/next1,000 streams,
  AdamW, teacher, normalization/offsets, precision, RNG, scores and statistics.
- Legacy1M raw steps except500k swimmer-swimmer15/hopper-hop; action repeat2,
  UTD2,warmup5,000. Keep all checkpoint/evaluation/diagnostic settings and inherited
  natural-fork/control behavior; choose no m and launch no injected arm.
- Use88 completed historical cells only as scheduling exclusions. The242 remaining
  cells are78/78/86. Eight old partial cells start fresh; their old files are not
  deleted. Historical source/admission remains distinct from new-source admission.

## Layout and conditional reference

|Critic|Partition class|GPUs|Processes/GPU|Reference allocations|Starts after|
|---|---|---:|---:|---|---|
|Small|gpuA100x4|4|10|14h|Initial|
|Medium|gpuA100x4|4|6|34h+34h|All required small cells verify complete|
|Large|gpuA100x8|8|3|48h+48h|Initial, parallel with small→medium|

Verify actual account/partition/time limits, node topology and resources on Delta;
these class names are not proof of live entitlement. Peak12 GPUs across separate
allocations. Concurrent-stream reference96h, sequential178h; total reference
allocated ceiling1,096GPUh. These use maximum historical elapsed+assumed20%
allowance rounded to7/17/24h per run, not measured new packing throughput. Queues,
new tasks, diagnostics and recovery can change completion/cost. The shared total
owner budget is4,000GPUh across all four experiments; it is not an unlimited Exp1
spending authorization. Reserve exact submitted maximums; no automatic extra
allocations beyond the listed first-pass plan when measured runtime is worse.
Record physical GPU-hours and Delta's weighted billing/service units separately;
an A100x8 allocation need not have the same service-unit rate per GPU as A100x4.

`generate_remaining_exp1_manifest.py --architecture-dir` writes three command
lists and `allocation_reference.json` with false approval/execution/qualification
flags. These are selection/planning outputs, not Slurm scripts. Fresh root and
explicit seeds1–10 required. W&B location/settings are already wired.

## Work for Delta Codex

1. Verify the exact supplied SHA, clean checkout and unchanged scientific files.
   Read AGENTS, methodology/decisions, official integration/remaining-run/resource
   reports. Record source/config hashes and prior test receipts. Implement only
   Delta Slurm orchestration, resource/binding, durable receipts and safe existing
   checkpoint continuation on a separate execution branch. No scientific retuning,
   merge, force-push or modification/cancellation of other campaigns.
2. Generate fresh durable campaign manifests/lists. Inspect the official runner's
   argument parser and existing launcher/checkpoint protocols before invoking;
   do not guess a resume flag or execute an old pilot launcher. One claimant per
   cell; unique output/checkpoint/W&B identity, no duplicate parent starts.
3. Resolve CPU binding from actual allocation. The historical4GPU node had64
   CPUs;40 small workers×2 disjoint CPUs would require80. Do not oversubscribe
   or silently copy old2/3/4-core profiles into the new10/6/3 layout. Record any
   feasible CPU assignment and resulting unverified throughput assumptions. GPU
   binding uses allocated physical UUIDs translated to child-local device0, not
   global GPU indices; preallocation remains disabled as in source.
4. Use existing source-matched qualification evidence if it actually covers the
   new workload/layout. Never manufacture passed flags from estimates or CPU
   tests. Keep required preflight fail-closed for wrong source/config, imports,
   incompatible JAX/CUDA, invalid scheduler request, binding, known OOM/CPU/RAM
   insufficiency, inaccessible/full durable storage and broken checkpoint/receipt
   recovery. A node with eight GPUs does not pool VRAM. Record unmeasured risks
   explicitly. If an existing hard gate lacks evidence or a required choice is
   unresolved, report that blocker rather than submitting a known-invalid job.
5. Verify current paths and Python import roots, not old failed launch assumptions.
   Retain the fixes for missing `audits` imports and incompatible JAX backends;
   record actual installed versions/backend and executable paths without changing
   scientific dependency/precision pins to make startup pass. Keep child tracebacks
   and cleanup receipts available on home storage if /projects fails.
6. Online W&B: entity `s_kaveti23-seckinger`, project `echocritic_official_exp1`.
   The project exists (103 records verified read-only). Use WANDB_MODE=online,
   WANDB_ENTITY, and current valid Delta credentials via WANDB_API_KEY; never
   print keys. Config project_name already selects the project; do not rely on
   WANDB_PROJECT to override an explicit wandb.init(project=...). Set WANDB_DIR
   per child under the durable campaign folder; preserve raw arrays, logs,
   checkpoints and receipts locally—W&B is not their substitute. Upload all existing
   logger metrics/diagnostics. Preserve checkpointed W&B IDs on legitimate resume;
   new cells receive unique IDs. Maintain a cohort manifest linking old88 IDs to
   new242; don't copy/relabel historical source as new-source data. Do not delete
   historical records under the ambiguous label "unused"; list exclusion IDs for
   later cleanup and do not block launch on cosmetic project cleanup.
7. Run focused CPU launcher/selection/binding/recovery/logging checks and actual
   cluster preflight. Preserve approval/executable flags in reviewed templates;
   record owner authorization and exact execution commit/diff separately. If all
   required launch gates pass, submit small and the first large allocation; run
   large continuation only after checking its preceding terminal receipt and
   ledger. After small cells verify complete, submit medium with the same checked
   boundary handling. Never use afterany or a blind retry chain. Reconcile partial
   work, natural-fork continuations and durable restore state before another
   allocation. Missing completion is not permission to omit a cell or shorten it.
8. Return actual JobIDs/states, exact sbatch commands/resources/dependencies,
   source and infrastructure SHAs/hashes, CPU/GPU map, submitted maximum GPU-hours,
   output/checkpoint paths, W&B links, admitted/reused/missing counts and remaining
   risks. If blocked, return the specific failed check and evidence. Do not claim
   launched/completed until scheduler evidence exists. Do not run an additional
   broad audit, choose m, change active pilots or spend beyond the first-pass plan.

## Cloud validation scope

CPU tests cover official config/population, missing-cell selection, disjoint
lists,4/4/8 allocation geometry, stream order and raw logging/resume regressions.
The official config project matches the live existing W&B project. Read-only API
inspection made no run edits/deletions. No Delta authentication, live allocation,
new-layout GPU test, Slurm submission or remote launcher qualification occurred
from Cloud. Final tested SHA and commands are external in
`/workspace/artifacts/exp1-launch-handoff-20261104/VALIDATION.json`.
