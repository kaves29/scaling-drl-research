# Metric-transfer investigation: Delta job 22756705

## Conclusion and scope

The complete downloaded evidence confirms a stall inside **native CUDA host-copy
submission for already-ready metric arrays**. Python tree traversal, metric
replay, and outstanding SAC updates are not the observed blocking operations.
Exact upstream source also exposes a concrete conditional self-deadlock: a full
worker queue can execute a transfer task inline while its caller holds the same
buffer-event mutex that the task tries to acquire.

**The existing Python stacks cannot prove that this job took that native lock
cycle.** They do not identify the native wait, the leaf index, or whether any
copies completed between samples. General CUDA/runtime contention remains an
alternative. The corrected accumulator drains one existing metric group before
submitting another, removing the unbounded prefetch pattern without changing
values or training. An A100 validation is still required.

Work continues from `bf20692e16700a87cd8375b2e3a851089ba83074` on
`codex/metric-transfer-fix`. That previous commit removed a redundant host-tree
traversal; it did not address the first transfer. No GPU access, SSH, Delta
modification, Slurm submission, push, merge, or other research-branch change was
performed during this follow-up. Original artifacts remain unchanged.

## Complete artifact inspection

All four files under `/Users/shouryakaveti/Downloads/a100-job-22756705/` were
read in full. Every JSONL record was parsed and checked for monotonic ordering,
matched stage nesting, errors, and terminal state. The stack file contains four
complete watchdog snapshots. Profile and backend contents were inspected in full.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| trace.jsonl | 2,195,691 | `6ebdeb1c9647b8afc2451920d228966743b66dff4a3c59ef5931a8819f6cb0fd` |
| trace.stacks.log | 11,807 | `7b3f645e8b77544ced0880a2183d40b6c86345638968894ab65b3c379376830e` |
| profile.log | 326 | `73b52186d94b83c91153e8aedc75212003a1f742838d244b69ccf7a28f140588` |
| backend.json | 329 | `fef906c25ed93acf38daba92c18b98632520009038883e30629c7841deda8ad8` |

Backend: A100-SXM4-40GB, Python 3.12.13, JAX/JAXLIB 0.4.34, `cuda,cpu`, source
`f6b7f73b018eac365819ce64c90d22942b211f42`. Profile reports 76.12M parameters
and contains no additional error. Exit 124 and the 300-second timeout were
reported by the user; these four files do not include Slurm accounting/status.

The trace has **10,935 records**: 5,438 begins, 5,435 ends, 59 progress records,
one wrapper start, one trace start, and one training begin. All recorded ends
match their begins; no error is recorded. The final line is newline-complete.
Only `entrypoint`, `diagnostic_metric_flush`, and `metric_device_get` remain open.
There is no terminal summary or train exit.

Times below are seconds from `wrapper_start`:

| Time | Evidence |
| ---: | --- |
| 25.201 | Trace initialized with `synchronize=true`; entrypoint starts |
| 43.863 | Trainer initialization completed |
| 52.931 | Initial evaluation/start completed |
| 59.842 | Interaction 2000 flush completed with zero pending groups |
| 91.363 | First structural metrics completed |
| 98.177 | Interaction 4000 flush completed with zero pending groups |
| 98.321 | Second structural metrics completed |
| 150.859 | Last progress: interaction 5900, update 1802 |
| 153.333 | Interaction 6000 `update_many` call 1001 begins |
| 153.353 | Its post-update synchronization and update call complete |
| 153.511 | Flush begins: 1001 groups, 19019 leaves, 152152 bytes |
| 153.666 | `metric_device_get` begins; no later trace record |

The last post-update synchronization took 8.262 ms. `RuntimeTrace.ready` waits
for the returned metric tree and the actor, critic, target critic, temperature,
and RNG state. Thus this is not simply a logging call waiting for a backlog of
unsynchronized updates. There are 2002 completed update rows at this boundary.
No nonempty previous bulk metric flush exists to establish a failure threshold.

The first watchdog snapshot is in an earlier structural-metric compilation;
the second is in an earlier post-update readiness wait. The final **two**, one
watchdog interval apart, are both:

```text
array.py:612 copy_to_host_async
api.py:2481 device_get
runtime_trace.py:249 device_get
exp12/diagnostics.py:78 flush
exp12/trainer.py:184 train
```

At JAX 0.4.34 line 612, Python has called the native single-device host-copy
method. `device_get` is still in its copy-submission loop; its subsequent
`tree_map` materialization has not been reached. These are Python snapshots,
not native CUDA stacks. They cannot distinguish one blocked request from a
sequence of very slow native requests.

## Native source investigation and CPU/GPU difference

[JAX 0.4.34's XLA pin](https://github.com/jax-ml/jax/blob/jax-v0.4.34/third_party/xla/workspace.bzl)
is `cd6e808c59f53b40a99df1f1b860db9a3e598bff`; that tree pins Eigen
`33d0937c6bdf5ec999939fb17f2a553183d14a74`. Both reviewers inspected these
specific versions, rather than assuming current-main behavior. These are the
upstream version pins, not independently recovered build IDs from Delta's binary.

[PyHostValue](https://github.com/openxla/xla/blob/cd6e808c59f53b40a99df1f1b860db9a3e598bff/xla/python/py_array.cc#L1459)
returns immediately from async-copy submission for ordinary zero-copyable CPU
buffers, and NumPy conversion acquires their host memory. CUDA buffers instead
allocate a destination and invoke IFRT/PJRT host-copy machinery. Consequently,
a fast CPU run does **not** exercise CUDA staging, stream events, host callbacks,
or the GPU transfer-task queue, and cannot disprove a GPU-specific failure.

The exact upstream lock cycle is:

1. [BufferSequencingEvent scheduling](https://github.com/openxla/xla/blob/cd6e808c59f53b40a99df1f1b860db9a3e598bff/xla/pjrt/tracked_device_buffer.cc#L140)
   holds its mutex while scheduling the task for an already-defined buffer.
2. [TSL's scheduler](https://github.com/openxla/xla/blob/cd6e808c59f53b40a99df1f1b860db9a3e598bff/third_party/tsl/tsl/platform/threadpool.cc#L128)
   delegates to Eigen. [Pinned Eigen](https://github.com/eigen-mirror/eigen/blob/33d0937c6bdf5ec999939fb17f2a553183d14a74/Eigen/src/ThreadPool/NonBlockingThreadPool.h#L100)
   has queues of 1024 tasks each and executes a task synchronously on the caller
   when pushing onto the selected queue fails.
3. The [ToLiteral task](https://github.com/openxla/xla/blob/cd6e808c59f53b40a99df1f1b860db9a3e598bff/xla/pjrt/pjrt_stream_executor_client.cc#L1749)
   accesses the same buffer's definition event, whose
   [GetDefinedStatus](https://github.com/openxla/xla/blob/cd6e808c59f53b40a99df1f1b860db9a3e598bff/xla/pjrt/tracked_device_buffer.h#L140)
   acquires that mutex again. Inline execution therefore self-deadlocks.

This establishes a conditional implementation defect in the pinned upstream
path. A burst of 19019 requests is a credible trigger. It does not establish
that a queue filled in job 22756705: no native backtrace, queue occupancy,
thread-pool size, per-leaf progress, or GPU/node telemetry was captured.
Queue capacity is per worker, not one global 1024-copy limit. No pinned-memory
exhaustion or driver deadlock is claimed as confirmed.

| Explanation | Result |
| --- | --- |
| Python traversal or logger replay | Excluded as the observed blocking site by both stalled snapshots |
| Pending SAC computation | The final update's output/state readiness completed before transfer |
| Compilation or trace-file I/O | Earlier costs completed; stalled samples are inside native copy submission |
| Raw payload bandwidth | 152 KB alone is insufficient to characterize 19019 native request sequences |
| Queue overflow and inline mutex reentry | Concrete upstream failure mechanism; actual occurrence needs native evidence |
| Other CUDA/runtime wait or severe per-copy overhead | Remains possible without native stacks and copy progress |

## Minimal engineering correction

`PendingUpdateMetrics._materialize` now calls `jax.device_get` separately for
each existing group and finishes that group's host conversion before submitting
the next. Both base and diagnostic flushes use it. For this job that changes the
prefetch bound from 19019 arrays to **19 arrays**, while still copying every
value. All groups are materialized before diagnostics or row replay begins, so
a later transfer failure does not partially update either collector.

There is no new compiled program, packing, dtype promotion, reduction, RNG use,
per-update synchronization, early logging, or changed metric. The original row
loop, Python float accumulation order, cosine filtering, diagnostic definitions,
precision policy, and logging cadence remain unchanged. Scientific configs,
trainer/SAC/network/optimizer/probe/evaluation code are untouched. Tracing now
naturally records one transfer event per group; aggregate flush metadata remains
19019 leaves / 152152 bytes. This changes diagnostic event volume, not research
metric cadence.

Files changed relative to `bf20692`:

- `experiments/angle_1.py`: shared group-draining materializer.
- `experiments/exp12/diagnostics.py`: use the shared materializer.
- `tests/test_metric_transfer.py`: copy-queue bound, late-failure atomicity,
  exact float bits/mixed dtypes, and full-payload row/meter/diagnostic parity.
- `tests/test_runtime_observability.py`: expect one completed transfer per group
  before replay, with no added readiness calls.
- `scripts/diagnose_metric_transfer.py`: isolated targeted transfer diagnostic.
- This report: complete evidence, source reasoning, validation, and next test.

## Validation and independent review

All **41 relevant methods passed in 56.977 seconds**: metric transfer,
observability, actor diagnostics/precision, cosine cadence, Exp12 runtime, and
SAC checkpointing. The exact 1001-group fixture compares all 2002 replayed rows,
every meter field, diagnostic values/std, and cosine sample selection against
the original bulk implementation. Float32 signed zero, infinities, a NaN payload,
a subnormal, int32, and host float64 precision are preserved by materialization.

A finite-capacity test backend demonstrates that bulk prefetch can exceed its
capacity while the new path peaks at 19 pending requests. This is a regression
of the bound, **not a reproduction of CUDA or the native self-deadlock**. A failure
on the second group leaves pending rows, diagnostics and meters unconsumed;
retry collects and replays each row once.

The real SAC/DMControl engineering fixture reached interaction 6001 / 2004
updates / four metric rows in plain, traced, and synchronized modes. Compared
with both the original bulk fixture and the previous-commit fixture, all six
complete-state comparisons returned `[]`. Every one of the 107 agent-state
leaves had identical dtype, shape, and bytes, and metric rows serialized
identically. Within the new version, both traced modes also matched plain state.
D1W8 networks/capacity 7000 are isolated test fixtures; production config was
not edited. Durations are not a claimed CUDA speedup.

The targeted diagnostic transferred and bit-verified all 19019 leaves on CPU
under bulk, groups, and serial strategies in separate processes. Each transfer
phase took approximately 0.05 seconds. Forced timeouts, unavailable-debugger
reporting, native-collector invocation using a fake debugger, and refusal to
reuse an existing evidence directory were verified. No real debugger attachment
or GPU execution occurred locally. Existing Linux-launcher checks were not
repeated: the prior run documented macOS's missing GNU `realpath -m`; those
unchanged launcher files remain outside this change.

Independent review of the complete evidence, pinned native source, correction,
diagnostic and report found no actionable correctness defects. The reviewer
independently passed all 18 focused methods in 3.131 seconds and confirmed the
limits of the native-deadlock attribution. Generated logs/checkpoints, source
extracts with provenance, and comparisons remain outside Git under
`/private/tmp/metric-transfer-*`.

## Smallest next A100 experiment — prepared, not executed

After explicit approval for GPU access and a fresh Delta output directory,
run the following inside an approved A100 allocation with the pinned environment:

```bash
python scripts/diagnose_metric_transfer.py \
  --platform cuda --groups 1001 --strategies bulk groups \
  --timeout 30 --native-stacks --out-dir NEW_ABSOLUTE_OUTPUT_DIRECTORY
```

This launches each strategy in a fresh process, creates the identical number
and size of metric arrays, explicitly waits for readiness, then changes only
prefetch depth. It defaults to CPU unless `--platform cuda` is supplied. It
records backend metadata, copy/materialization phase and sampled leaf progress,
five-second Python stacks, exact output-bit verification, and process status.
A timed-out worker optionally gets a five-second `gdb` native-backtrace attempt
before termination. Missing debugger, missing symbols, or ptrace restrictions
can prevent native identification; failure is reported without changing
permissions/settings or requesting privilege escalation. No Slurm command is
embedded in the diagnostic.

Interpretation:

- **Bulk stalls, groups succeeds:** isolates concurrency in native copy
  submission. A native stack showing inline scheduling under
  `ExecuteOrAddToFutureTasks` followed by the same event's `GetDefinedStatus`
  mutex acquisition confirms the predicted lock cycle.
- **Both stall:** run the optional `serial` strategy to test one-request depth;
  inspect the native wait for general driver/runtime/transfer failure.
- **Bulk progresses slowly:** sampled leaf progress and transfer/materialization
  timing distinguish per-request overhead from a complete stop.
- **Both succeed:** the synthetic case does not reproduce the actual context.
  The next necessary experiment is the original dog-run D4W1536, seed 990 warm-up
  only through interaction 6000, with bulk/group transfer selection and native
  stacks at the first populated flush. Retain the original precision, batch,
  UTD, sampling onset and cadence; do not substitute a shortened scientific run
  as evidence of campaign qualification.

A successful targeted test must then be followed by one approved original-workload
validation of the committed correction, checking completed transfer/replay,
unchanged counters/metric keys/cadence, and the existing profile validator.
No claim is made that the full diagnostic now fits 300 seconds. The remaining
uncertainty is native CUDA behavior and the unobserved native wait, which cannot
be resolved by rerunning equivalent zero-copy CPU tests.
