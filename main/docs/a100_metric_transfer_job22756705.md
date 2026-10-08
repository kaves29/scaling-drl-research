# Metric-transfer investigation: Delta job 22756705

Work is isolated on `codex/metric-transfer-fix`, based on `f6b7f73`.
The A100 root cause is **not established**. The verified change removes a
redundant traversal after the first transfer; it cannot fix a stall inside
that first transfer. No GPU was accessed, no job was submitted, and no remote
file was changed. No push, merge, or research-branch modification was performed.

## Evidence access and limits

The supplied diagnostic directory is
`/work/hdd/biqc/skaveti1/exp12_diagnostic_outputs/f6b7f73_20261008T184434Z_692401`.
Read-only SSH reached Delta's authentication banner but failed with
`Permission denied (gssapi-with-mic,password)`. Local `klist` reports no ticket.
Authentication was requested from the user; credentials were not requested in
chat. Consequently, **none of the four required remote artifacts has been read**:
`trace.jsonl`, `trace.stacks.log`, `profile.log`, or `backend.json`.

The following is user-supplied evidence, not an independently verified trace:
interaction 6000 completed SAC updates; `diagnostic_metric_flush` began with
1,001 pending groups; `metric_device_get` began with 19,019 leaves / 152,152
bytes; neither operation ended before the 300-second timeout; exit status 124.
Do not substitute job 22740708's earlier artifacts or report for this job.

## Source findings

JAX/JAXLIB 0.4.34 is installed in the existing local Python 3.12.13 environment.
Inspection of that installed JAX source confirms that `jax.device_get(tree)`:

1. Flattens the tree and calls `copy_to_host_async()` separately on each leaf.
2. Maps a host conversion over the tree; array conversion can wait for readiness.

Passing the entire tree in one call does not pack its buffers into one transfer.
The shape agrees with 1,001 groups × 19 metrics × two float32 values = 152,152
bytes. The payload is independent of the critic's parameter count.

Before this change, `DiagnosticPendingUpdateMetrics.flush` transferred the
whole tree, collected diagnostics, replaced pending entries with host copies,
then called the parent's flush. The parent called `device_get` again on NumPy
arrays. That second traversal is redundant; it is not a second GPU transfer.
It occurs after the operation reported stalled in this job.

The checked-in diagnostic launcher enables synchronization. If this job used
that unchanged invocation, the trace's `update_many` wrapper waits for both
returned metrics and agent state after each call. Completed updates would then
argue against an accumulated training backlog, but the actual trace and stacks
must be inspected before drawing that conclusion about this job.

## CPU reproduction

`scripts/reproduce_metric_transfer_cpu.py` forces CPU before importing JAX and
separately measures readiness, flattening, copy requests, materialization, a
redundant host-tree traversal, and fresh `device_get`. It uses distinct arrays
returned by a compiled synthetic metric function. There is no training or GPU
performance claim. One run of the exact 1,001-group payload measured:

| Phase | Seconds |
| --- | ---: |
| Readiness before measuring transfers | 0.00614 |
| Flattening | 0.00104 |
| Per-leaf copy requests | 0.04297 |
| Host materialization | 0.01026 |
| Redundant host-tree traversal | 0.00736 |
| Fresh complete `device_get` | 0.09500 |

Counts 1, 1,001, 2,000, and 4,096 all completed. These timings do not bound CUDA
latency. Python traversal alone does not reproduce the stall on CPU. Excessive
CUDA per-buffer overhead, waits in transfer/materialization, native runtime
failure, and node contention remain unconfirmed possibilities.

The existing real SAC/DMControl fixture was also run before and after the
change, in plain, traced, and synchronized-trace modes. Each of the six runs
reached interaction 6001, performed 2004 updates, and saved four metric rows.
All three old-versus-new complete-state comparisons returned `[]`, including
agent/optimizer state, replay, normalizer, environment state, global RNG,
counters, diagnostics, and metrics. The existing comparator ignores only the
WandB run ID. Within each version, traced-versus-plain comparisons also returned
`[]`. Tiny D1W8 networks and replay capacity 7000 are engineering fixtures;
production configurations were not edited. Concurrent local tests and cache
warming prevent using run durations as speedup estimates.

## Verified change

- `experiments/angle_1.py`: extract the existing row loop into `_replay`.
- `experiments/exp12/diagnostics.py`: replay the already-materialized host tree
  directly, avoiding replacement of pending entries and the second `device_get`.
- `experiments/exp12/runtime_trace.py`: observe `_replay` so transfer and host
  aggregation remain separately visible without adding waits.
- `tests/test_metric_transfer.py`: compare all 2,002 rows and every meter field
  with the literal old implementation, including diagnostics, cosine filtering,
  mixed group lengths, host float64 values, cancellation order, transfer errors,
  and empty/repeated flushes.
- `tests/test_runtime_observability.py`: require one transfer and ordered,
  completed transfer/replay stage events.
- `scripts/reproduce_metric_transfer_cpu.py`: retain the CPU phase reproduction.

The nested replay loop and its Python float accumulation order are unchanged.
No reduction, dtype conversion, packing, changed cadence, or additional barrier
was introduced. SAC, networks, optimizers, trainer, probes, methodology,
configuration, precision policy, and evaluation code remain unchanged.
For base `PendingUpdateMetrics`, `metric_replay` now times host replay alone;
its previous trace span included the transfer. Account for this when comparing
trace timings between revisions.

Independent review found no actionable correctness defects and independently
passed the 15 focused metric/observability methods. It confirmed that this
cleanup does not resolve a stall in the first transfer.

## Validation and local environment errors

The initial 15 focused tests passed. The final relevant suite passed all
**38 methods in 54.786 seconds**, covering metric transfer, observability,
actor diagnostics and precision, cosine cadence, Exp12 runtime, and SAC
checkpointing. The CPU reproduction script, `git diff --check`, and Black
checks for the two new Python files also passed.

An expanded attempt including the unchanged Linux Slurm launcher suite ran
67 methods but failed with 15 failure reports and 10 error reports. The launcher
requires GNU `realpath -m`; macOS's installed `realpath` rejects `-m`, causing
preflight failure and missing output artifacts in those fixtures. The launcher
and its tests have no diff from the base. This is an unresolved local platform
limitation, not a passing launcher qualification. No launcher change was made.
An initial import also aborted in GLFW; rerunning with `MUJOCO_GL=disable`
resolved it. A new dependency-install attempt encountered sandbox DNS failure;
the existing pinned environment was then found and used without modification.

CPU commands, from `main/` with the existing environment:

```bash
export JAX_PLATFORMS=cpu MUJOCO_GL=disable EXP12_JAX_CACHE_DIR=off
export PYTHONPATH=/private/tmp/exp12-validation-deps:.
CPU_PYTHON=/Users/shouryakaveti/VS_Projects/sparse-ppo-drl-research/.venv/bin/python
"$CPU_PYTHON" scripts/reproduce_metric_transfer_cpu.py
"$CPU_PYTHON" -m unittest -v tests.test_metric_transfer \
  tests.test_runtime_observability tests.test_exp12_diagnostics \
  tests.test_angle1_real_entry_point.ActorGradCosineCadenceTest \
  tests.test_exp12_runtime tests.test_sac_agent_checkpoint
"$CPU_PYTHON" scripts/reproduce_runtime_logging_cpu.py --out-dir NEW_ABSOLUTE_DIRECTORY
```

Generated evidence is local under `/private/tmp/metric-transfer-*`: CPU phase
JSONL, before/after fixture checkpoints and traces, complete-state comparison,
and test logs. These are not remote research outputs or committed binaries.

## Remaining investigation and A100 validation

First restore read-only SSH authentication and read all four complete artifacts,
recording sizes and hashes. Match repeated stacks to the complete trace timeline
and confirm backend, synchronization mode, timeout provenance, and errors.
A stack in `copy_to_host_async` favors transfer submission/runtime; a stack in
array conversion can represent waiting or materialization; neither proves a
native deadlock. A Python traversal stack requires inspection for repeated work
and host contention. Native transfer/synchronization failures may require a
CUDA timeline and node telemetry.

Only after explicit approval for GPU access and new Delta work, measure the
unchanged transfer path and the relevant remedy on the same A100 model and
pinned runtime. Separate already-ready leaf transfer cost from readiness cost.
If evidence supports packing, compare exact byte-preserving, bounded packing
against the original path without changing metric reductions or SAC outputs;
qualify CPU state and metric parity before proposing an A100 trial.

Then repeat the original workload through its first populated logging boundary,
retaining complete trace, stacks, backend, profile, and exit status in fresh
output locations. Require completed transfer/replay events, unchanged counters,
metric keys/cadence, numerical precision and training state, and the existing
output validator. Passing this cleanup alone would not establish that the
original transfer bottleneck is fixed, that the entire profile fits 300 seconds,
or that the scientific campaign is qualified.
