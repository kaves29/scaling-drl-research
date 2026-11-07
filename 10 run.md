# EchoCritic ten-run diagnostic pilot

**Report status:** corrected replacement job 23257841 is PENDING in Delta; original job 23173276 failed before any pilot execution.  
**Latest scheduler snapshot:** 2026-10-27 23:57:36 UTC / 2026-10-27 18:57:36 CDT (Delta local).  
**Scientific conclusion:** none can be drawn; the ten diagnostic executions have not started.

**Time-zone correction:** Delta's Slurm `SubmitTime`, `LastSchedEval`, `StartTime` and `EndTime` fields are in Central Daylight Time (CDT, UTC−5) for these checks. Earlier versions of this report compared those local timestamps directly with UTC query timestamps and incorrectly called future backfill estimates “past” or “stale.” Those interpretations were wrong; corrected conversions are reported below. The job is now terminal, so no current start estimate exists. The remote `date -u` timestamps remain UTC.


## Corrected resubmission (2026-10-27)

The owner authorized a corrected resubmission after job `23173276` failed its GPU visibility check. PR #16 now points to source `5ecca42cc820f936782f92f708896abcfbe651a0`. The wrapper changed from `srun --gpus-per-task=2 --gpu-bind=none` to explicit `--gpu-bind=per_task:2` for its single task. The controller now durably writes `startup_gpu_evidence.json` with raw Slurm/CUDA GPU environment variables and NVIDIA UUID inventory before validating two distinct GPU tokens. This addresses the identified launcher assumption and preserves evidence if Delta still exposes a different mapping. The old job did not save the actual `CUDA_VISIBLE_DEVICES` value, so this source change has not yet been validated on an allocated GPU. It does not change the scientific conditions, probe, targets, optimizer, or ten-run schedule.

A clean isolated Delta checkout at `/u/skaveti1/echocritic-diagnostic-pr16-resubmit-20261027` has the published PR head. The fresh campaign root is `/projects/biqc/skaveti1/echocritic_diagnostic_pilot_20261027T1925Z_retry1`, with a new W&B namespace `ecpilot1027retry1skaveti1`. The ten pilot IDs still comprise five matched configurations, each assigned once to GPU 0 and once to GPU 1: **ten concurrent child processes, five per GPU, one wave**. The new manifest and launch config bind the exact source and fresh root. The repository preflight returned `PASS`; Bash syntax, Python compilation and Git whitespace checks passed. The pinned Delta Python lacks `pytest`, so the existing focused pytest suite was not rerun for this two-file change. No five/five GPU qualification receipt exists; that earlier owner-authorized bypass remains explicitly recorded.

Slurm accepted the repository-generated exact `sbatch` command as job **23257841** at 2026-10-27 19:26 UTC. The requested resources are one `gpuA100x4` node, two A100 GPUs, 32 CPUs, 128 GiB RAM, and **299 minutes**. The failed job used 9 seconds × 2 GPUs = 18 GPU-seconds. The new maximum is 299 minutes × 2 GPUs = 35,880 GPU-seconds; combined maximum is 35,898 seconds = **9.9717 GPU-hours**, below the authorized 10 GPU-hour cap. The exact repository-generated submission command was:

```sh
sbatch --parsable --account=biqc-delta-gpu --partition=gpuA100x4 --nodes=1 --ntasks=1 --gres=gpu:2 --cpus-per-task=32 --mem=128G --time=299 --job-name=echocritic-diagnostic-10 --chdir=/u/skaveti1/echocritic-diagnostic-pr16-resubmit-20261027/main --output=/projects/biqc/skaveti1/echocritic_diagnostic_pilot_20261027T1925Z_retry1/slurm-%j.out --error=/projects/biqc/skaveti1/echocritic_diagnostic_pilot_20261027T1925Z_retry1/slurm-%j.err /u/skaveti1/echocritic-diagnostic-pr16-resubmit-20261027/main/scripts/sbatch_echocritic_diagnostic_pilot.sh /u/skaveti1/.conda/envs/scaling-drl-py31213/bin/python /projects/biqc/skaveti1/echocritic_diagnostic_pilot_20261027T1925Z_retry1/pilot_manifest.json /projects/biqc/skaveti1/echocritic_diagnostic_pilot_20261027T1925Z_retry1/unqualified_launch_config.json 6b748e70389b604eaf538692fe47123720be1e42475b251d623459ba6074ae88 0a9ea04f9bbdcf50ef07e08419aebc58ffef962e36b868771ab2dc38a7a6b2a8
```

At the latest **2026-10-27 23:57:36 UTC** check, the new job was `PENDING` with `ElapsedRaw=0` and no node assigned. `squeue --start` returned **N/A** and accounting returned `Start=Unknown`. The earlier 2026-10-28 16:34:32 EDT estimate is no longer displayed by Slurm and must not be treated as current. No pilot outputs or results exist yet. The new job must be monitored through actual allocation, startup and per-run receipts before any result or successful binding can be claimed.

## What this pilot is testing

The diagnostic pilot investigates a specific discrepancy in the adaptation probe: trained DMC critics have often performed better than fresh critics, while MyoSuite critics have underperformed. The cause is not identified. This is an observational diagnostic pilot; it is not evidence that a bug exists, and no expected direction of effect is assumed.

The ten executions are five matched configurations replicated on two GPUs. The GPU replicas use the same task, architecture, and seed. They are engineering/device replications, not ten independent random seeds. The pilot has five paired configuration comparisons, with one execution of each configuration assigned to each GPU.

## Requested and submitted run matrix

| Pair | Environment | Critic | Seed | GPU 0 identity | GPU 1 identity |
|---|---|---:|---:|---|---|
| C01 | `dog-run` | D4W1536 | 3 | `C01G0` / `echocritic-c01g0` | `C01G1` / `echocritic-c01g1` |
| C02 | `humanoid-run` | D4W1024 | 5 | `C02G0` / `echocritic-c02g0` | `C02G1` / `echocritic-c02g1` |
| C03 | `myo-key-turn` | D4W1536 | 2 | `C03G0` / `echocritic-c03g0` | `C03G1` / `echocritic-c03g1` |
| C04 | `myo-pen-twirl` | D4W1024 | 6 | `C04G0` / `echocritic-c04g0` | `C04G1` / `echocritic-c04g1` |
| C05 | `myo-pen-twirl` | D2W512 | 5 | `C05G0` / `echocritic-c05g0` | `C05G1` / `echocritic-c05g1` |

Every execution was assigned a distinct output directory and W&B run ID. The planned output directories are `runs/C01G0` through `runs/C05G1` below the campaign root. The controller was designed to start all ten children behind one readiness barrier, with five bound to each allocated GPU. There were to be no waves, replacements, or automatic retries. Whether all ten actually started and completed must be established from the Slurm and per-run receipts on Delta.

## Submission and current evidence

The recorded campaign root is:

```text
/projects/biqc/skaveti1/echocritic_diagnostic_pilot_20261025T0419Z
```

The recorded Slurm job ID is **23173276**. It was submitted for a single-node, two-A100 allocation with a five-hour maximum, under account `biqc-delta-gpu` and partition `gpuA100x4`. The recorded request was 32 CPUs and 128 GiB of host memory. The stated owner budget was up to five hours on each of two GPUs, or ten GPU-hours total; the launcher cap was 300 minutes. W&B project and namespace were recorded as `echocritic-diagnostic-pilot-20261025` and `ecpilot1025skaveti1`.

At the latest live check, **2026-10-27 18:56:24 UTC** (**2026-10-27 13:56:24 CDT**), job `23173276` was **FAILED**, not pending. Slurm records that it allocated node `gpua059` with two GPUs, 32 CPUs and 128 GiB memory at 06:45:54 CDT, then failed after nine seconds at 06:46:03 CDT. There is no current start estimate because this job is terminal. Its exit code is `2:0`. The campaign root contains the manifest, launch config, and preserved Slurm output/error logs; it contains no per-run outputs.

Slurm records the submission as **2026-10-24 23:20:04 CDT**, equivalent to **2026-10-25 04:20:04 UTC**. At the latest check it had been pending for about **45 hours 7 minutes**. Account and QoS are `biqc-delta-gpu`; partition is `gpuA100x4`. At 03:18 UTC on Oct 26, among 2,030 pending entries in the partition, the pilot was at priority order **1,234**. This historical rank is not a start-time estimate because Slurm can backfill jobs and higher-ranked jobs may be blocked by other constraints. At the Oct 26 afternoon checks, Slurm's projected start times (16:12–16:49 CDT) were still in the future relative to those checks; previous text claiming otherwise was the time-zone error described above. The latest estimate is stated explicitly in the status paragraph above.



### Failed allocation (2026-10-27)

Live `sacct` reports job `23173276` FAILED, `ExitCode=2:0`, on `gpua059`; it held the requested two GPUs, 32 CPUs and 128 GiB for nine seconds. The preserved `slurm-23173276.err` says: `BLOCKED: allocation must expose exactly two unique GPU tokens; no guessed global ordinals`, followed by Slurm's task exit code 2. The controller's GPU-binding preflight therefore stopped the job before starting any of the ten executions. The Slurm queue no longer contains this job, and there is no revised start estimate. The output directory has only the original manifest and launch config plus empty stdout and the 144-byte stderr; no run directories, checkpoints, or W&B results were created. No retry or replacement was submitted, consistent with the no-retries instruction.


### GPU binding failure analysis and resubmission estimate

The frozen PR head is `57d2beb78be7cf0e1145f4a9479082a07924cd16`. Its wrapper starts one `srun` task with `--gpus-per-task=2 --gpu-bind=none`. The controller then reads only `CUDA_VISIBLE_DEVICES`, splits it on commas, and requires exactly two distinct numeric or `GPU-...` tokens before querying hardware or creating `pilot_started.json`. The recorded error is exactly this guard's message, so the failure occurred at that guard; no ten-run process was started. Slurm did allocate two GPUs to the job. The violated assumption is that two allocated GPUs necessarily appear as two acceptable tokens in that step variable. The job did not log the variable itself, so the exact observed value is unknown; it may have been absent, one token, or a different format. It is not justified to blame a specific Delta setting or a particular token value.

Previous Delta work in this repository recorded a one-GPU step with `SLURM_JOB_GPUS=2`, `SLURM_STEP_GPUS=2`, and `CUDA_VISIBLE_DEVICES=0`, showing that Slurm's physical ID and CUDA's task-local ID can differ. The next launcher revision must capture the raw `CUDA_VISIBLE_DEVICES`, `SLURM_JOB_GPUS`, `SLURM_STEP_GPUS`, `SLURM_GPUS_ON_NODE`, NVIDIA UUID inventory and device cgroup before evaluating the guard, then reconcile allocation ownership and exactly two distinct UUIDs. It must pass a single verified UUID to each child, require each child/JAX process to see its assigned one GPU, and abort with retained evidence if identity cannot be proven. This is a runtime binding correction, not a change to the scientific method. Focused tests should cover an absent/single-token visibility value, physical versus task-local ordinals, duplicates, and an actual Delta execution path; CPU mocks alone cannot establish actual binding. No corrected code or new GPU verification was completed as of this report.

The prior job waited **55 hours 25 minutes 50 seconds** from submission at 2026-10-24 23:20:04 CDT to start at 2026-10-27 06:45:54 CDT. That is one historical wait, not a current queue estimate. A new job has a new queue position; the old position cannot be recovered. The Delta SSH multiplexed connection answered its liveness check but remote command sessions timed out during the latest attempted scheduler query, so a current `squeue --start` estimate was unavailable. A planning allowance of roughly **2–3 days in queue, possibly shorter or longer**, is based only on the prior 55-hour wait. Slurm can estimate a start only after the corrected job is submitted and pending.

### Read-only resource-placement diagnosis (2026-10-25 19:13 UTC)

The submitted job asks for one node with **2 A100 GPUs, 32 CPUs and 128 GiB memory**. I parsed live `scontrol show node -o` records for all 99 nodes in `gpuA100x4` (`gpua002` through `gpua100`), using each node's configured four GPUs and current `AllocTRES`, CPU allocation and memory allocation. The snapshot showed:

| Nodes | GPUs apparently free | CPUs apparently free | Memory apparently free | Scheduler state | Fits this job now? |
|---|---:|---:|---:|---|---|
| 94 nodes | 0 | varies | varies | available-state | No: no free GPU |
| 2 nodes | 1 | varies | varies | available-state | No: only one free GPU |
| `gpua056` | 2 | 32 | 11,870 MiB | `MIXED+PLANNED` | No: less than 128 GiB free |
| `gpua082` | 2 | 32 | 73,310 MiB | `MIXED+PLANNED` | No: less than 128 GiB free |
| `gpua059` | 3 | 48 | 196,190 MiB | `MIXED+DRAIN+REBOOT_REQUESTED` | No: node is drained |

Thus, **none of the 99 nodes appeared to satisfy all three resource requirements at the 19:13 UTC check**. The one node with enough apparent free GPUs, CPUs and memory (`gpua059`) was drained and had `Reason=Kill task failed (JobId=23173970 StepId=0)`, with reboot requested. The two non-drained nodes with two apparently free GPUs did not have the requested memory headroom. This is the concrete resource-placement explanation supported by that snapshot.

A refreshed inventory at **2026-10-26 01:16 UTC** showed 93 nodes with zero free GPUs, five with one free GPU, and only `gpua059` with at least two free GPUs; `gpua059` was drained. At **03:19 UTC**, 98 nodes had zero free GPUs and one had only one free GPU. Thus, in that snapshot **no node had two free A100s**, before considering CPU or memory. A node, `gpua072`, briefly appeared at 03:18 UTC to have two free GPUs, 44 CPUs and ample memory, but was marked `MIXED+PLANNED`; the subsequent node query showed all four GPUs allocated and three running jobs on it. That transient reading was not a usable free placement.

At **19:10 UTC** (14:10 CDT), Slurm projected `gpua079` with a local start window around 16:12–16:49 CDT, later than the check. That node then had all four A100 GPUs allocated across four running jobs, 52 of 64 CPUs allocated, and about 24 GiB free. At **19:34 UTC** (14:34 CDT), the projected node had moved to `gpua082` with a 15:22 CDT estimate, about 48 minutes later than that check; it then had all four GPUs allocated across three running jobs. These estimates were future backfill projections at the time, not guaranteed reservations. The current check at 20:26 CDT now projects 22:52 CDT on Oct 27 (03:52 UTC Oct 28), but remains tentative because the job is pending for `Priority` and has no node allocated.

The latest job record reports a recent `LastSchedEval=2026-10-26T20:26:38` CDT and a pending reason of `Priority`; the earlier `LastSchedEval` values were also CDT and should not be compared as if they were UTC. Resource placement is tight, as shown by the snapshots above, and job priority is currently the scheduler's stated reason. The account/QoS are attached and the five-hour wall cap is below the partition's two-day maximum. No setting was changed, and the job was not canceled or resubmitted.

The durable campaign root was inspected directly. At the snapshot, it contained exactly `pilot_manifest.json` (53,956 bytes) and `unqualified_launch_config.json` (2,127 bytes), using 64 KiB total on disk. There were no `runs/C01G0`…`runs/C05G1` directories, Slurm stdout/stderr files, checkpoints, telemetry, probe artifacts or completion receipts. Therefore there are no per-run records to analyze and no evidence that any child reached the all-ten barrier. The launch JSON files and scheduler records were retrieved and verified for this report; the run data do not exist at the inspected root yet.

The unqualified launch config records the owner-authorized scope to run this exact ten-execution pilot without prior five-process-per-GPU hardware qualification, with a 10 physical GPU-hour cap and 300-minute walltime. It explicitly leaves A100 CUDA execution, simultaneous five/five fit and full-endpoint throughput unverified. `packing_qualified` is false and `throughput_VRAM_storage_actual_measurements` is null. These are meaningful limitations on any eventual interpretation of runtime and resource behavior.

Two earlier qualification jobs, `23172617` and `23172708`, were canceled before starting at the user's direction. Their recorded accounting was `CANCELLED by 97950`, zero elapsed time and no allocated TRES. They were qualification attempts, not any of the ten pilot executions, and are not counted as pilot runs.

## What the runs were meant to preserve

The scientific question and original experiment settings were to remain unchanged. The adapter was intended to add observation and artifact retention around the existing probe rather than alter its generation or fitting method. In particular, the planned method retained:

- Original probe inputs, target generation, target scaling, and each critic's own-mean offset. The literal targets were to be saved; no shared-offset sensitivity variant was authorized.
- The original continuous 2,000-step fitting procedure, optimizer, minibatch schedule, and training conditions. There was to be no refit or segmentation of the fit.
- The source-defined training/probe/evaluation random-number streams, with provenance captured alongside retained states.
- Original metrics, endpoint schedule, statistical thresholds, and interpretation rules. No threshold or result direction was to be changed to make the pilot look successful.
- Failure and partial output preservation. A failed run was not to be deleted, replaced, or automatically retried.

The source manifest described the nominal training endpoint as 500,000 interaction steps per execution, with a 1,000,000-step configured environment budget, two updates per interaction step, and the original fork/extension behavior. The pilot's purpose was to retain additional evidence at prescribed probe checks while leaving the underlying training and fitting protocol intact.

## Planned retained evidence

The adapter's capture design included the following evidence per run:

1. **Probe inputs and targets:** replay sample indices, raw observations/actions and normalized observations, generated base/teacher values, each critic's own offset, literal targets, shapes, dtypes, non-finite counts and SHA-256 hashes. The manifest's selected retained checks were 0, 5, 35 and 50.
2. **Errors and learning curves:** per-round fresh/trained errors and curves, fit minibatch losses and scheduled full-pool Q/MSE snapshots at the defined fit checkpoints (0, 100, 500, 1,000 and 2,000). Minibatch fit losses were to remain distinct from full-pool evaluation metrics.
3. **Model and training state:** initial critic trees and tree structure, teacher/base state, checkpoint file hashes and sizes, interaction/update counters, replay-buffer indices/size, observation-normalizer state, environment restore state, and relevant NumPy/Python/agent/probe/evaluation RNG provenance.
4. **Runtime/resource evidence:** per-process CPU and RSS, GPU UUID and memory/utilization, compile and steady-state timing, probe/checkpoint durations and bytes, shared filesystem I/O/free space, process/barrier/exec timing and failure events.
5. **Completion and integrity:** per-run completion records bound to run identity, plan/configuration and source, with file-level hashes. Partial data and failure records were to remain available if a child stopped early.

These are the intended capture fields from the adapter/controller design. Their presence and integrity in actual campaign output are **not yet verified**. A planned schema or a successful CPU mock does not establish that Delta produced those artifacts.

## Allocation and qualification caveat

The five-process-per-GPU fit was not qualified on actual Delta hardware before this run. The user explicitly authorized proceeding with this exact ten-run layout without that qualification, capped at five hours per GPU. Therefore, the successful queue submission must not be described as proof that five jobs fit safely or complete within the cap. Actual GPU binding, VRAM, host RAM, CPU contention, filesystem throughput, compilation behavior and endpoint throughput require the run's allocation and telemetry evidence.

This report does not substitute a different layout or claim the originally proposed 48-hour allocation. The submitted cap was 300 minutes. The last known accounting observation was pending with zero usage, so no GPU-hours or actual runtime can be reported here.

## Implementation and verification recorded before submission

The PR work recorded in the task handoff added the observation adapter and focused adapter coverage, corrected numeric int/float configuration hashing (the resolved configuration uses numeric `0.0` where a pinned JSON snapshot stores `0`), and added an explicit unqualified owner-authorized controller path. The corresponding reported PR commits were:

- `32bec42e9a97b1eb26110083cd527e6e1d5b195a` — adapter and focused tests.
- `501bc6fc25f1be612acb23598f92c653da41b2e0` — numeric config-hash handling and focused all-five config test.
- `57d2beb78be7cf0e1145f4a9479082a07924cd16` — explicit owner-authorized unqualified launch path.

The actual frozen execution checkout was verified on Delta: `git rev-parse HEAD` returned **`57d2beb78be7cf0e1145f4a9479082a07924cd16`**, matching the `expected_commit` in the unqualified launch config; `git status --short --untracked-files=all` was empty. The source-bound training commit recorded in the manifest was `f597ee9898e4d401912092c336c3e5330cf2fda1`. The manifest's campaign identity was `discrepancy-10-simultaneous-v2`.

The actual Delta SHA-256 values were:

| Artifact | SHA-256 |
|---|---|
| `pilot_manifest.json` | `cdebd3fe41bbf30288e860f14dde7e9e178c2015fd6bbdd1f9f4b95ff66adfef` |
| `unqualified_launch_config.json` | `7decb0e08e5721e7743578e02e604cd6ebde4cfec62803885b9a903ecbd9cea4` |
| `main/audits/probe_discrepancy/adapter.py` | `0e2c430c3f908ced638c01a6e76a80ab25eb09dcbc3991c074fc6463d263f6dc` |
| `main/audits/probe_discrepancy/launch.py` | `c054ff3bb8e207ed8012f154ab58c4a2f6914de1cf5390f113cc879d4b10ec9d` |
| `main/scripts/sbatch_echocritic_diagnostic_pilot.sh` | `fe87abba1a7371138b290d7b91eb0a3094408a330a6bc16f0b78c128c0575eac` |

The scientific and resolved-configuration hashes in the actual manifest are listed for every execution below. Each GPU pair has identical hashes, as expected for the same condition/seed replicated on the two allocation slots.

| Pilot ID | Scientific config SHA-256 | Resolved config SHA-256 |
|---|---|---|
| C01G0, C01G1 | `7bf157ae74e5a34b649b9d94c2ef5c241a9c00aed640ecab657ab95ccd5f3f0e` | `88e5597dc2fd37d71dd8dd03db1bb9de76c93a6cbc9b22d2b34383c6c1d1ebb9` |
| C02G0, C02G1 | `fb09116e0dda791be2de3891b127920da939fdf0deb5d4183061d4d685b7b5df` | `c90101895721875a01c7d784bd214ba7807b6636c5caebe91ff8db2aff053e54` |
| C03G0, C03G1 | `13b411be41f12f3be423e4e3cd26cd3fbc44d0555b0cbf561da2ef4b3d299b24` | `8ed9919c3e6b7e90e54bf8e65cb7b6b443c9bbf5a0e76d46063588b3638a5c3b` |
| C04G0, C04G1 | `d4198b5b3e2513eb66bc8b36fd324a011ad166480a4288cf40c3ff3f52f15486` | `367f8c9ad3cc1a363396ec0596ed3e5bcd1bdd2c6b22ed5e6c9243bca0d7afbb` |
| C05G0, C05G1 | `65c05cf4b2512b8aced9a8b005ae8fa49f14cd85388414e9c14dd970c7b5aada` | `b7ef6084fadb2c80e36cd573eb517d9a916530ac9232d9baed584c11f7c67aab` |

The pre-submission handoff reports that 72 CPU tests passed before adapter completion, and that focused checks subsequently passed for adapter behavior, original target generation and continuous 2,000-step fitting on a small `TinyCritic`, selected-check target/full-pool-MSE retention, and all five pinned scientific configurations. It also records successful Python compilation and `git diff --check`. These are CPU/source checks only; they do not verify GPU execution, simultaneous five/five packing, full training, or scientific results. This report has not rerun those checks.

## Results

**There are no experimental result values because the job was still pending and no run outputs existed at the live check.** This is stronger than “results not yet fetched”: the Slurm accounting shows zero elapsed/allocated resources, and the campaign root has no run directories or scheduler output logs. The only valid run count at that instant is **0 of 10 started**; **0 of 10 completed**; **0 GPU-hours used**. These counts describe the checked snapshot only. The job may change state after that time, so they are not a claim about its present-day state.

No adaptation errors, per-round curves, trained/fresh critic comparison, endpoints, checkpoints, exit statuses for child processes, W&B run summaries, GPU UUIDs, VRAM/RAM peaks, storage rates, compilation timings or final accounting exist in the inspected output. A Slurm submission ID is evidence of queue acceptance, not evidence of a successful pilot. Do not calculate condition means, compare DMC against MyoSuite, claim replication or infer a cause from this snapshot.

The scientifically relevant outcome—whether the measured DMC/Myo discrepancy was reproduced and what the retained diagnostics say about it—remains undetermined because none of the executions had started. There is no causal explanation to assess yet.

## What is needed to complete this report

The authenticated SSH connection is available and was used for the read-only status, accounting, manifest, source, scheduler and filesystem checks recorded above. The latest follow-up at 01:26:41 UTC on Oct 27 still found no allocation and no run artifacts. To complete the scientific report, continue monitoring the same job without resubmitting or creating replacements. After it starts or terminates, query final `sacct` state and inspect batch stdout/stderr; enumerate all ten run directories; validate completion/failure receipts and file hashes; summarize errors/curves and endpoints without treating GPU replicas as independent seeds; and report allocation, resource and timing telemetry. Preserve any partial output. If the job remains pending, report that fact and do not invent or substitute results.
