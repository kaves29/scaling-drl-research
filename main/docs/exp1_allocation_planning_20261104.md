# Four-GPU Exp1 allocation planning — 4 November 2026

This is a planning assessment, not launch approval. All requested layouts use
exactly four GPUs. No GPU tests, training, Slurm queries or submissions were run.

## Revised reference after checking raw clocks

The recheck reads **all88 completed raw histories**, not just the derived runtime
column. All reach1,000,000 raw steps without a logged step or runtime reversal.
Median first-to-last timestamp spans are5.3138 /13.2996 /18.5900 h, agreeing with
the earlier runtime medians to seconds. Largest gaps between any adjacent log
records are344 /751 /939 seconds by architecture. This rules out a multi-hour
unlogged pause as the explanation in these retained histories; it does not prove
continuous GPU utilization between records. Planned fork save/restore appears in
some logs, so these are not claimed to be uninterrupted single-trainer objects.

Concrete example: large pen-twirl run `14fmtkz6`, raw_api/14fmtkz6/history.jsonl,
first record21October22:53:00.797618UTC and final record22October17:21:41.301629UTC:
18h28m40.504s elapsed, raw step0→1,000,000. Its largest adjacent gap is697.56s.
The original output.log records a20.5s fork save at interaction30,000, followed
by a1.4s restore into the control continuation. That is genuine resumed state,
not evidence that eighteen hours was queue time or just a metadata-unit error.

The important correction is to interpretation: **18.6 h is shared-resource
process elapsed time, not18.6 exclusive GPU-hours per run**. All96 campaign
metadata records identify gpuc05, an eight-A100 host; at most44 retained logged
run intervals overlap. This demonstrates overlapping run intervals, not exact
per-GPU assignment or simultaneous GPU compute. The historical scheduler
documentation describes7/5/4 homogeneous packing and bounded mixtures; it is a
design record, not a recovered per-child packing receipt. Do not divide elapsed
time by7/5/4, or multiply it by a new packing ratio, as a verified speed estimate.

For a more cautious **conditional reference**, the generator now uses each
architecture's largest observed elapsed time plus a stated20% planning allowance,
rounded up: ceil(5.526×1.2)=7h, ceil(13.615×1.2)=17h,
ceil(19.236×1.2)=24h per run. The20% allowance is an analyst assumption, not a
measurement, owner-frozen resource limit or statistical confidence bound. It
does **not** quantitatively account for different packing throughput or new tasks.

|Critic|Proposed processes/GPU|Assumed per-run duration|Reference allocations on four GPUs|Total wall hours|Allocated GPU-hour ceiling|
|---|---:|---:|---|---:|---:|
|D2W512,78 runs|10|7h|one14h allocation (78 runs)|14|56|
|D4W1024,78 runs|6|17h|two34h allocations (48+30 runs)|68|272|
|D4W1536,86 runs|3|24h|four48h allocations (24+24+24+14 runs)|192|768|

Total reference ceiling: **1,096 allocated GPU-hours**, conditional on the
assumptions and running these allocations for their full requested times. These
are conservative relative to retained elapsed measurements, **not guaranteed
completion estimates for the proposed packing**. Actual consumption is lower
if allocations exit early and higher if additional allocations are needed.
These are three architecture-specific work lists, requiring seven reference
allocations under a48-hour maximum; they are not three single-node submissions.
The48h limit is retained for planning, not verified live scheduler eligibility.
Dynamic replacement may improve utilization; this reference deliberately counts
fixed waves and includes short hopper/swimmer tasks at full reference duration,
since their50-check probe budget is not halved. No claim that total experiment
cost is exactly1,096 GPU-hours is justified.

`--architecture-dir` additionally writes `allocation_reference.json`, containing
the per-allocation cell identities, four-GPU layout, assumptions and false
approval/execution/packing/throughput flags. The three command lists and scientific
argv are unchanged. It neither sets a Slurm time limit nor submits/resumes jobs.
Exact raw-clock rows, input checksums, tested source and regenerated lists are
outside Git under `/workspace/artifacts/exp1-runtime-recheck-20261104/`.

The sections below preserve the original requested limits and earlier
median-based comparison, rather than silently relabeling them as new approvals.

## Initial source boundary

The approved remote integration ref was verified as
844e3c0cc24998b25c3d1e667bcbfe056b1b336c. This planning-only successor extends
PR25's 6adbe42f6d4bc3c25d41380ec21336d609f3d788; its runtime/configuration remains
PR24's d885854f8fdb3c2888fd87219d79e2733d574107 candidate, not promoted production.

## Population and requested limits

Historical completion exclusions and unresolved seed admission are documented in
[the remaining-run plan](exp1_remaining_runs_20261104.md). Do not credit or resume
the eight partials. They remain fresh reruns. Eleven environments are retained.

|Critic|Remaining|Processes/GPU|Four-GPU slots|Waves if fixed waves|Requested walltime|Maximum allocated GPU-hours|
|---|---:|---:|---:|---:|---:|---:|
|D2W512|78|10|40|2 (40+38)|10 h|40|
|D4W1024|78|6|24|4 (24+24+24+6)|17–18 h|68–72|
|D4W1536|86|3|12|8 (7×12+2)|48 h|192|

Total requested maximum: **300–304 allocated GPU-hours**. This is not a runtime
prediction or evidence of completion within budget. Filling a slot as soon as a
run finishes can reduce wave barriers; these command lists implement neither
fixed waves nor a scheduler.

For fixed equal-duration waves, the respective per-run limits would be 5 h,
4.25–4.5 h, and 6 h. Even with perfect dynamic slot filling and no overhead, the
allowed mean durations are only 5.128 h, 5.231–5.538 h, and 6.698 h. Tail imbalance,
compilation, checkpoint I/O and failure recovery consume additional time.

## Measured runtime evidence

Source: the archived comprehensive package's `derived/run_inventory.csv` and
`derived/probe_trajectories.csv`; select cohort `exp96`, state `finished`, then
maximum retained `_runtime` per run. Exclude all eight partials. The statistics
below use all 88 completed runs, not probe rounds as independent runs.

|Critic|Completed runs|Minimum|Median|Maximum|
|---|---:|---:|---:|---:|
|D2W512|32|2.335 h|5.315 h|5.526 h|
|D4W1024|32|10.631 h|13.301 h|13.615 h|
|D4W1536|24|18.197 h|18.591 h|19.236 h|

These are W&B elapsed process-clock values at the last retained check, not
isolated compute benchmarks, Slurm allocation durations or physical GPU-hours.
Historical process packing/pauses are not established by these tables. The four
sampled environments are dog-run, humanoid-run, myo-key-turn and myo-pen-twirl;
seven official environments have no comparable full regular-probe runtime here.

As an explicitly **conditional reference scenario**, assign every remaining run
its architecture's historical median and assume packing does not change its
speed. Fixed waves would need approximately **10.63 / 53.21 / 148.73 hours**.
Ideal slot-work lower bounds in that same scenario are **10.36 / 43.23 / 133.23
hours**, before tail and overhead. Thus 10 h is tight even in this optimistic
small-critic comparison; 18 h medium and 48 h large require approximately
2.4× and 2.8× better aggregate throughput than that reference scenario.
These are not measured runtimes of the proposed layouts and not new approved
resource limits. Increasing concurrency usually increases process elapsed time;
it cannot be assumed to multiply completed-work throughput linearly.

The successful six-run alternative allocation 23387585 lasted 5:44:53 on two
A100-SXM4-40GB GPUs, 32 allocated CPUs and 128 GiB allocated host RAM. It used a
mixed small+medium+large process set per GPU. It is **not** a six-hour benchmark
for official regular-probe runs: that alternative used 440,000 probe optimizer
updates per job versus 2,020,000 in official regular evaluation. Official
checkpoint frequency and optional rank/SVD diagnostics also differ. Do not
multiply runtime by the update ratio as if compilation/SAC/I/O scaled identically.

## Memory evidence and its limits

The six-run archive retains `*/wandb/system_metrics.csv` and
`dog-small/common/slurm_accounting.txt`, plus `common/campaign/allocation.json`.
Across all six exports, sampled device-wide `gpu.0.memoryAllocatedBytes` and
`gpu.1.memoryAllocatedBytes` peak at **8.414 and 8.416 GiB**. The identical values
across child exports are shared device measurements; **do not sum them or call
8.416 GiB an individual process measurement**. `gpu.*.memory` is memory-controller
utilization, not allocated-memory percentage. Sampling does not capture every
transient peak or establish per-process demand.

Sampled process RSS maxima (GiB): dog small3.865 / medium6.460 / large8.847;
pen-twirl small3.323 / medium5.399 / large7.717. Multiplying the larger observed
RSS by the proposed node-wide process counts gives **154.6 / 155.0 / 106.2 GiB**
respectively, before launcher, OS, observer and environment differences. These
are extrapolations from alternative runs, not proven peak RAM requirements.
The Slurm Python-step MaxRSS was 42,100,116 KiB; it is not child VRAM evidence.
Host `availableMB` is node-wide and does not establish job cgroup entitlement.

On nominal 40 GB devices, dividing capacity by concurrency gives only about
4 / 6.67 / 13.33 GB per process for small/medium/large **before shared overhead
and headroom**. The archive does not provide attributable per-child peak VRAM
for these exact homogeneous layouts. The mixed three-process result is
encouraging but cannot certify 10 small, 6 medium or 3 large per GPU. Large×3
also differs from one large plus two smaller processes. No OOM-safety claim is
supported for any requested layout.

A full four-GPU allocation does not pool their VRAM. Actual four-GPU node CPUs,
host-RAM entitlement, partition walltime allowance, child-to-UUID binding,
CPU contention with 40/24/12 processes, and I/O at these concurrency levels must
come from the Delta allocation/launcher record. They are not available to this
Cloud session. Do not infer them from the earlier two-GPU allocation. This
assessment requests no extra GPUs and does not require another experiment to
produce the three lists; it leaves the unmeasured packing claims unresolved.

## Separate command lists and use

Run from `main/` with a fresh, absolute durable campaign root:

```bash
python scripts/generate_remaining_exp1_manifest.py \
  --root /absolute/fresh_campaign_root \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --out /absolute/remaining.json \
  --architecture-dir /absolute/new_architecture_lists
```

Produces D2W512.txt (78), D4W1024.txt (78), D4W1536.txt (86). Each line retains the
original argv/configuration exactly. Lists are disjoint and cover all242 missing
cells. No training/probe setting, GPU binding or concurrency is changed by the
split. Directory and files must not exist. IDs1–10 remain the existing preview;
seed9's development exclusion still needs resolution before official launch.
The program refuses seed lists inconsistent with the official source population.

These are **three program lists, not executable Slurm launchers**. Feed each into
the reviewed launch/preflight path after the source/seed/storage decisions are
resolved. Preserve per-cell output directories and partial checkpoints. Do not
execute all lines in parallel, restart partials as new runs, or use the lists to
bypass admission/budget gates. If a requested single allocation cannot finish,
the closest structural option is the same three size-specific lists consumed
across multiple bounded allocations with reviewed checkpoint/resume handling,
still four GPUs at a time. That does not require changing the scientific method;
continuation/resume and total budget must be explicit rather than silently
shortening runs. No such launch/resume orchestration is added here.

## Validation and evidence

Six focused CPU tests cover the existing selection/rejection cases plus the real
CLI split: exact78/78/86, unique242 union, identical shell-parsed commands, and
no overwrite on a second invocation. The actual configuration-generating CLI is
also run on a fresh artifact path. Only planning code/docs/tests changed; no
production runtime/configuration changed. One existing GLFW DISPLAY warning is
recorded. Source/test SHA and command receipts, input hashes, all88 runtime rows,
summaries and generated lists are retained outside Git at
`/workspace/artifacts/exp1-allocation-planning-20261104/`.

**Assessment:** separate size-specific lists are ready for review. Requested
packing is not established; requested completion times are not supported by the
retained timing evidence. Do not label these proposed allocations qualified.
