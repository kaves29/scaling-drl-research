# A100 runtime investigation: job 22740708

## Scope and conclusion

Base: **312bb01a7a6aa2e3a34a74dd557f05543081d0f5**. Work is isolated in
`codex/a100-runtime-debug`, created directly from that commit. This is an unmerged,
local diagnostic improvement, not a demonstrated correction of the GPU stall.
No Delta access, submission, push, deletion, scientific change, or incorporation
of `a3f44aa56455e3aae4b36459a78c97237bad62a8` occurred.

**CONFIRMED:** the 300-second internal timeout expired during a long unobserved
interval in initial training, after steady progress through interaction 5900.
The recorded compilation and metrics costs do not account for the entire timeout.
**UNRESOLVED:** the exact blocking operation. The leading source-supported
hypothesis is the first populated metric flush at interaction 6000, particularly
its many-small-array host transfer. Neither entry to 6000 nor entry to that flush
was recorded. CPU reproductions complete it successfully. Do not claim the A100
runtime issue is fixed or the full diagnostic fits five minutes.

## Original evidence, inspected in full

Original ZIP (retained, not executed):
`/workspace/attachments/1f88cc40-9a08-4c07-b7d0-7dd63ce078a4/delta_job_22740708.zip`.
Size 3,087,822 bytes; SHA-256
`06a0f6d70c9e6fe8e65de943b0137bc1228b3b6be29ae67b0d3bf836cfa69823`.
All **589 members** were read and passed ZIP CRC validation. Extraction is outside
the repository, at `/workspace/scaling-drl-runtime-evidence/job22740708/`.

Inside its `312bb01_20261008T012054Z_1627981/` directory:

| File | Bytes | Evidence |
| --- | ---: | --- |
| `trace.jsonl` | 383,283 | All 1,833 JSON records valid; newline-complete final record |
| `profile.log` | 326 | Backend metadata and 76.12M parameter count; no error/warning |
| `backend.json` | 329 | A100-SXM4-40GB, CUDA and CPU, Python 3.12.13, JAX/JAXLIB 0.4.34 |
| `exit_status.txt` | 4 | `124` |
| `commit.txt` | 41 | Exact base revision |

Sibling `312bb01_20261008T012054Z_1627981.slurm-22740708.out` and `.err`
are present, both zero bytes, not corrupt. No `profile.json`, `validation.json`,
Python stack capture, GPU timeline, GPU utilization, or process memory telemetry
was produced. There are 288 executable-cache files and 288 access-time files;
cached executables total 2,837,879 bytes. No inference is made from unavailable
per-architecture measurements.

The trace SHA-256 is
`4164d56828babae934494fc697745cf15f1aa56771bb16c8b5515630111bd3cf`.
Reanalysis, hash manifest, CPU traces, checkpoints, and test logs remain outside
Git under `/workspace/scratch/runtime-debug/`. The analysis script is committed;
generated evidence and cache binaries are not.

## Three attempts and timeout provenance

| Job | Revision | Outcome and status |
| --- | --- | --- |
| 22731248 | 781dc6c | Reported 4-second failure: detached checkout rejected by named-branch guard. Previously reproduced and corrected in e01f3b6. **CONFIRMED engineering history**, not remeasured GPU evidence. |
| 22738928 | e01f3b6 | Reported 45-second failure: `JAX_PLATFORMS=cuda` excluded explicit CPU parameter initialization. Previously reproduced and corrected by `cuda,cpu` in 312bb01. **CONFIRMED engineering history**. |
| 22740708 | 312bb01 | Original artifacts inspected here: backend/trainer/training succeeded; 124 after reported 308 seconds; **CONFIRMED timeout**, blocking stage unresolved. |

`scripts/sbatch_exp12_runtime_diagnostic.sh:69` encloses backend preflight and
profiling in `timeout -k 15 300 bash`; the Slurm allocation limit is 420 seconds.
ZIP file timestamps place `commit.txt` at 20:25:18, `trace.jsonl` at 20:27:24,
and `exit_status.txt` at 20:30:18. They corroborate 300 seconds and approximately
174 seconds after the last trace write. ZIP times have two-second resolution and
unspecified export timezone; use relative differences, not a claimed exact signal
timestamp. The extra reported eight seconds include outer startup/teardown but
cannot be allocated precisely. This was not the seven-minute Slurm limit.

## Full trace reconstruction

All 886 begin events have correctly nested matching ends: **no unfinished emitted
begin**, no recorded exception. There are 59 progress events, one `trace_start`,
one `train_begin`, and no `train_exit` or terminal summary. The final event is
interaction 5900 / environment step 11800 / update 1802, five initial evaluation
episodes, zero post-fork episodes, `last_check=null`.

Trace wall times are UTC: first `2026-10-08T01:25:42.073354Z`, last
`2026-10-08T01:27:24.109302Z`. The monotonic span is 102.035947 seconds. The earlier
Python/dependency import/backend startup interval was outside the old trace.

| Seconds after trace start | Stage | What completed |
| ---: | --- | --- |
| 0.000 | backend/cache setup | Device lookup 0.000016s; cache configuration 0.000636s |
| 0.305–13.315 | trainer initialization | 13.010s; actual 76.12M architecture initialized |
| 13.315–22.267 | start / initial evaluation | Five episodes, 2500 environment interactions; 8.949s evaluation |
| 22.267 | first training call | Target last interaction 6001 |
| 29.178–46.710 | structural metrics at 2000 | 17.532s, dominated by cold work; completed |
| 53.576–53.719 | structural metrics at 4000 | 0.143s; completed |
| 57.242–66.900 | first UTD group at 5000 | 9.659s; first SAC scan compilation |
| 66.910–73.436; 73.442–80.140 | next two groups | 6.526s and 6.698s; two further scan compilations |
| 82.533–102.036 | progress 5100–5900 | 800 interactions / 1600 updates in 19.503s, 41.019 interactions/s |
| After 102.036 | unobserved | No more events; no evidence that interaction 6000 completed or began |

Cumulative, overlapping stage measurements at EOF:

| Stage | Calls | Seconds |
| --- | ---: | ---: |
| backend compilation | 288 | 25.019 |
| cache read | 288, zero hits | 0.114 |
| cache write | 288 | 16.407 |
| structural metrics | 2 | 17.675 |
| update-many | 901 | 40.366 |
| training environment step | 5900 | 14.721 |
| replay sampling | 1804 | 0.417 |
| synchronization | 40,807 | 11.359 |

These durations are nested, not additive exclusive costs. Between 5100 and 5900
there are 8008 synchronization calls totaling 7.051 seconds inside 19.503 wall
seconds. Wait time includes required GPU execution; it is not removable observer
overhead. Compilation counts are 138 by interaction 100, 276 by 2000, 286 by
5000, and 288 by 5100; unchanged through EOF. Three distinct SAC scan cache keys
are present. The 288 calls represent many primitive/modules/shape variants, not
288 compilations of the full SAC scan. The last cache write completes before
80 seconds; no cache-corruption warning is present in this job.

## Exact control flow near 6000

Unmodified source references are relative to `main/`, at the base revision:

1. `scripts/profile_exp12.py:92`: compose base Exp12, dog-run, D4W1536, seed 990;
   seed NumPy/Python; construct trainer; start and evaluate five episodes.
2. `profile_exp12.py:103`: warm-up target = buffer minimum 5000 + 1001 = 6001.
   No production probe callback is passed to `train`. No training/fork checkpoint
   is scheduled during this phase.
3. `experiments/exp12/trainer.py:144`: assign interaction counter, sample action
   (including observation-normalizer behavior), environment step, replay add.
4. `trainer.py:171`: two sampled batches, `update_many`, retain metric result,
   increase update counter by two. At 6000 this would yield 2002 updates.
5. `trainer.py:182`: every 2000 interactions, **pending.flush first**, then sampled
   structural metrics and actor diagnostic window metrics.
6. `experiments/exp12/diagnostics.py:77`: device_get the entire pending info tree,
   collect gradient norms, replace pending entries with host copies, call shared
   `experiments/angle_1.py:104` flush. That second device_get traverses a host tree;
   it is not a second model-sized GPU transfer. Replay rows in original order into
   `scale_rl/common/logger.py:83` average meters. No filesystem write here.
7. `trainer.py:194`: snapshot averages/media, append metrics row, call logger,
   reset meters. `profile_exp12.py:64` stubs WandB init/log, so online WandB traffic
   is not involved. No ordinary checkpoint write occurs at 6000.
8. Only afterward, `trainer.py:202`, tracing's after-step callback can emit progress
   6000. Evaluation cadence is 50000, not 6000.

Previous flushes at 2000/4000 contain no SAC update metrics. At 6000 the first
populated flush would contain 1001 groups × two rows. Old tracing suppresses
individual action, environment, replay, and update begin/end events after three
calls; its pre/post waits and metric flush are also unlogged. A stop anywhere in
5901–6000 therefore fits the evidence. Structural metrics would emit a coarse
begin before their pre-wait: its absence strongly favors a stage before that call
(or a blocked attempt to write that begin). It does not prove flush entry.

After warm-up, the intended 60-step timing interval, probe initialization, warm
and timed five-round probes, 475000-transition synthetic fork save/restore, and
later evaluation timings were **not reached in the recorded execution**. Initial
evaluation did complete; later profiling stages did not produce measurements.

## Ranked mechanisms

| Rank | Classification | Mechanism and evidence |
| --- | --- | --- |
| 1 | **POSSIBLE; leading source-supported hypothesis** | First populated 6000 flush: many separate small device arrays, bulk host transfer and host replay untraced. Missing structural-metrics begin is consistent. CPU completes it; CUDA transfer/driver contention remains untested. |
| 2 | **POSSIBLE** | An action, environment, replay-add, update, or pre/post synchronization in 5901–6000. Fine stages are suppressed. Regular preceding throughput lowers the evidence for persistent slowness but cannot exclude a one-off stall. |
| 3 | **POSSIBLE** | Trace write/flush or host scheduling/resource contention. The last progress event itself has a flushed newline. A subsequent write could block; local microbenchmark cannot rule out Delta filesystem behavior. |
| 4 | **CONFIRMED cost; UNRESOLVED contribution** | Existing tracing barriers serialize execution. 40807 waits are real; they may perturb scheduling. No matched CUDA observer-free measurement or GPU timeline exists. |
| 5 | **STRONGLY SUPPORTED exclusion for silent tail** | Already-recorded compilation/cache I/O cannot alone explain the unobserved tail: all complete before it, no pending recorded compiler stage or warning. A compiler call whose begin write blocks is still observationally possible. |
| 6 | **STRONGLY SUPPORTED exclusion** | Third structural-metric computation: no begin recorded. Earlier 2000 cold metrics are expensive but complete; 4000 is fast. |
| 7 | **RULED OUT by scheduled source path for 6000** | Evaluation, probe fitting, fork serialization, remote WandB upload, checkpoint save, and legacy experiment training dispatch are not scheduled at this boundary in this profile invocation. |

**No confirmed GPU deadlock, out-of-memory condition, cache-corruption cause,
repeated work bug, or oversized metric payload.** Source and CPU inspection show
19 length-two scalar arrays per group, independent of critic parameter count.
Scalar metrics require norms of model arrays on device; their scalar output does
not mean the underlying update computation is cheap.

The original Block B report (job 22706349, source b4a90cb) contains truncated-cache
warnings and a six-hour timeout, but no corresponding complete boundary trace or
stack. It uses shared logging infrastructure, so a common issue is possible.
Different source, workloads, concurrency, and cache conditions prevent attribution
of both failures to a common cause. See the existing separate legacy audit for
import/call-site evidence. Angle1's metric accumulator is a live shared dependency;
it is not obsolete code merely because of its name. No evidence supports legacy
methodology overlap as the cause of job 22740708.

## CPU reproduction and independent verification

Local Python **3.12.14**, JAX/JAXLIB **0.4.34**, `TFRT_CPU_0`; no GPU available.
`scripts/reproduce_runtime_logging_cpu.py` uses the actual DMControl dog-run,
Exp12 trainer, SAC, logger, pending accumulator and complete-state saver. Fixture
changes are explicit: D1W8 networks, capacity 7000, raw-step budget 14000. Sampling
starts at 5000; batch 256, UTD=2, logging 2000, evaluation 50000, five initial
episodes, seed 990 retained. No configuration file is edited. No probes substituted
or disabled in production; the profile's first phase already has no probe callback.

| Fixture | To interaction 6001 | Updates | Metrics snapshots | Complete-state comparison |
| --- | ---: | ---: | ---: | --- |
| No tracing | 38.338s | 2004 | 4 | Reference |
| Tracing without extra readiness waits | 36.344s | 2004 | 4 | No differences |
| Tracing with existing readiness waits | 42.484s | 2004 | 4 | No differences |

At 6000: **19019 array leaves, 152152 bytes**, 1001 groups/2002 rows. Flush took
0.385s without extra waits, 0.345s with them; first device_get 0.147/0.138s,
host-tree device_get 0.0165/0.0130s, row replay 0.0379/0.0324s. The entire current
state comparison covers optimizer/parameters, buffers, normalizer, diagnostics,
environment, global RNG, counters and metrics. Initial evaluation explains the
fourth metrics snapshot in addition to three logging windows.

Runs share a process and compilation warms in order; these elapsed times are
observations, **not a causal estimate of tracing overhead or CUDA performance**.
One separate writer microbenchmark emitted/flushed 10000 records, 1105664 bytes,
in 0.0354s (~3.54 microseconds/record) on the local filesystem. Two fixture traces
were 514889 and 1179547 bytes. This does not bound Delta filesystem latency.

Regressions additionally exercise actual metric flush/replay for 0, 1, 1001 and
4096 groups, original row order, failed host transfer, slow/failed trace writes,
pre/post readiness exceptions, startup import failure, boundary selection,
malformed/unmatched trace records, output reuse, and periodic real Python stack
capture during a deliberately blocked entry point. No numerical threshold is
selected for scientific validation.

## Implemented engineering changes

**CONFIRMED defect:** the old wrapper invoked pre-call readiness outside its
exception/finally handling. A CPU reproduction from the literal 312bb01 source
produces `begin(operation)` then `summary(RuntimeError)` without an operation end.
The correction in `experiments/exp12/runtime_trace.py:124` covers pre-wait failures,
preserves the exception and reports phase; timing includes an additional wall-span
field while successful original stage timing still excludes the pre-wait.
This is a trace error-reporting fix, not an established timeout fix.

Diagnostic-only additions:

- `runtime_trace.py:79`: flushed synchronization begin/end/error/phase around
  **existing waits**, without adding wait calls.
- `runtime_trace.py:105`: optional detailed action/environment/replay/update events
  during the last progress interval before a logging boundary; only while training.
- `runtime_trace.py:117`: inspect shapes/bytes without materializing arrays.
- `runtime_trace.py:225`: flush, host transfer, host replay, diagnostic-window,
  logger-log and reset observers. New logging observers add no synchronization;
  original transfers, order and quantities remain intact.
- `scripts/trace_exp12_runtime.py:40`: exclusively create journal before dependency
  import; flushed startup/install/entry-point stages; retain final summary convention.
- `trace_exp12_runtime.py:67`: opt-in standard-library repeating Python stack dump,
  separate exclusively created `.stacks.log`, cancel before closing it. No SIGTERM
  handler replacement or early exit. Launcher enables 60-second dumps in all modes.
- `scripts/sbatch_exp12_runtime_diagnostic.sh:112`: observer flags only; same backend
  policy, profile/identity workloads, success validator and 300-second timeout.
- Small full-trace analyzer, explicit CPU-only reproduction script and regressions.

Adversarial review caught a local failure in an initial candidate: very rapid
0.1-second stack dumps armed during dependency import exited with SIGSEGV while
importing Pydantic. Exact interpreter/extension cause was not established. That
candidate was discarded: the committed watchdog arms **after imports**; startup
is still covered by flushed stage records. The repeated-stack regression then
passed. Python stacks cannot prove a GPU-side stall and are not native CUDA stacks.
If the output filesystem blocks, both journal and stack-file writes may block;
external filesystem/node telemetry would then be necessary. No watchdog is armed
during dependency imports; an incomplete startup stage localizes that interval.
Review also made errors from deliberately sampled-out fine calls explicit via
`begin_recorded=false`, so the analyzer does not misclassify their error end as
a corrupted nesting sequence. The regression exercises an actual fourth-call
exception after the first three calls have been logged.

## Validation and preservation review

Integration run: **126 methods in 999.426s, 124 passed, two HumanoidBench dependency
skips, zero failures/errors, exit 0**. Modules: launcher, observability, runtime,
diagnostics, SAC checkpoint, foundations, pipeline, fork, orchestration, manifest
and registry. This includes complete-state/optimizer parity, initialization,
restore across processes, kill/resume before/at/after forks, logger/diagnostic
quantities, and the 195 distinct parent grid and its approved budgets.

Expanded final launcher/observability run: **40 passed in 7.733s**, including four
new observer cases beyond those loaded by the integration run. After the final
phase-label and sampled-error metadata adjustments, **all 11 observer tests passed
again in 3.745s**.
Across runs, **130 distinct methods: 128 passed and two skipped**. Skips are the
HumanoidBench pipeline and Reach evaluation seeding tests because that dependency
is unavailable. No skips resulted from new instrumentation or chosen tolerances.
The initial narrower 38-method runtime/launcher run also passed.

These are relevant targeted runs, not a new full Exp12 discovery run or a rerun
of the 72 scientific break checks; prior full-suite totals are not reasserted.
Shell syntax, Python AST parsing, full original-trace JSON/nesting/monotonic-order
checks, and `git diff --check` passed. The analyzer explicitly rejects malformed
JSON rather than silently accepting an incomplete record.

Reproduction commands from `main/`, with the existing CPU environment:

```bash
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu EXP12_JAX_CACHE_DIR=off
export MUJOCO_GL=disable WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/workspace/scratch/mpl_methodology
export XDG_CACHE_HOME=/workspace/scratch/xdg_methodology
CPU_PYTHON=/workspace/scaling-drl-research/.venv/bin/python
"$CPU_PYTHON" -B -m unittest -v tests.test_runtime_diagnostic_launcher \
  tests.test_runtime_observability tests.test_exp12_runtime tests.test_exp12_diagnostics \
  tests.test_sac_agent_checkpoint tests.test_exp12_foundations tests.test_exp12_pipeline \
  tests.test_exp12_fork tests.test_exp12_orchestration tests.test_exp12_manifest \
  tests.test_experiment_registry
"$CPU_PYTHON" -B scripts/reproduce_runtime_logging_cpu.py \
  --out-dir /workspace/scratch/runtime-debug/NEW_CPU_EVIDENCE_DIRECTORY
"$CPU_PYTHON" -B scripts/analyze_runtime_trace.py \
  /workspace/scaling-drl-runtime-evidence/job22740708/312bb01_20261008T012054Z_1627981/trace.jsonl
```

On a future final-tree integration rerun, four added observer methods increase
the selected count from 126 to 130. Outputs must remain outside the checkout.

Reviewed diff against the literal base commit, including every untracked addition,
not just a passing test count. Only opt-in tracer, trace wrapper, launcher observer
flags, relevant tests, isolated CPU scripts and this report differ. Scientific
sources/configuration/authority documents, SAC initialization/optimizer/network
files, trainer, diagnostics, probe/trigger/bootstrap/calibration, fork/checkpoint,
logging implementation, evaluation and precision policy are byte-identical.
All 261 tracked files outside the four intended existing-file edits were compared
byte-for-byte against the base commit and were identical. The
methodology-reconstruction branch and all pre-existing checkout contents
remain untouched. No numerical tolerances or experimental thresholds changed.

The corrected trace can affect observed diagnostic runtime through record-writing
overhead; it does not establish GPU bit-exact equivalence. Do not compare its new
wall timings as an observer-free speedup over 312bb01. CPU complete-state parity is
evidence of local semantic preservation, not GPU qualification. The instrumented
revision remains separate and unmerged.

## Minimum remaining evidence and decisions

1. **Engineering blocker:** one authorized A100 diagnostic of the same workload,
   model, seed and 300-second limit, with this diagnostic revision. Preserve its
   entire trace, stacks, backend metadata, cache warning log, status and outputs.
   A last `metric_device_get` begin identifies host transfer entry; `metric_replay`
   identifies host aggregation; an outstanding synchronization identifies the
   Python wait; a stack in `emit` identifies a journal operation. None alone proves
   a native driver deadlock. Keep node/resource context for contention analysis.
2. If it reaches later probes/fork work but still expires, determine completed stage
   costs before any proposal to extend the limit or reduce diagnostic scope. **Such
   changes need explicit approval** under the task, and are not selected here.
3. If a transfer stall is reproduced, inspect CUDA/native timeline and driver/GPU
   resource state before changing metric storage or transfers. An optimization
   would require exact row/order/state equivalence and measured benefit; none is
   implemented speculatively here.
4. Independent scientific gates (fresh-null 13/100; hopper range failure) and the
   separate approved twin-loss reconstruction remain separate decisions/evidence.
   No gate relaxation, precision change, probe budget or statistical amendment is
   proposed to resolve a runtime failure. Campaign readiness is **not established**.

Another GPU diagnostic is necessary to localize the original stall; CPU alone
cannot decide it. Expected allocation: one A100, four CPU cores, 32GB RAM, seven
minutes maximum = **0.117 allocated GPU-hours**, not a monetary estimate. Internal
work remains limited to five minutes; success is not promised. Queue time and
rate/account charging are unknown. Full 195-run cost cannot be estimated from
these incomplete measurements.

## Safe future detached deployment (not executed)

The local commit is not pushed. A small Git bundle outside the checkout contains
the new branch and requires the already-published base. The full commit and bundle
SHA-256 are reported to the lead separately. Transfer the bundle manually only
after reviewing this result; do not copy caches, old output directories or ZIP
executables into the source checkout.

On Delta, after the lead approves the diagnostic, set `DIAG_REV` to the **full
commit reported with this work**, and `DIAG_BUNDLE` to the transferred bundle path.
These commands use a fresh directory and do not alter the frozen checkout:

```bash
set -euo pipefail
DIAG_REV=FULL_COMMIT_FROM_DELIVERABLE
DIAG_BUNDLE=/work/hdd/biqc/skaveti1/a100-runtime-debug.bundle
DIAG_CHECKOUT=/work/hdd/biqc/skaveti1/exp12_runtime_debug_${DIAG_REV}
test ! -e "$DIAG_CHECKOUT"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$DIAG_CHECKOUT"
git -C "$DIAG_CHECKOUT" bundle verify "$DIAG_BUNDLE"
git -C "$DIAG_CHECKOUT" fetch "$DIAG_BUNDLE" refs/heads/codex/a100-runtime-debug
test "$(git -C "$DIAG_CHECKOUT" rev-parse FETCH_HEAD)" = "$DIAG_REV"
git -C "$DIAG_CHECKOUT" checkout --detach "$DIAG_REV"
test "$(git -C "$DIAG_CHECKOUT" rev-parse HEAD)" = "$DIAG_REV"
test -z "$(git -C "$DIAG_CHECKOUT" status --porcelain --untracked-files=all)"
test -f "$DIAG_CHECKOUT/main/scripts/sbatch_exp12_runtime_diagnostic.sh"
```

Manual future submission, using the same previously reviewed CUDA environment:

```bash
conda activate scaling-drl-py31213
cd "$DIAG_CHECKOUT/main"
DIAG_PARENT=/work/hdd/biqc/skaveti1/exp12_diagnostic_outputs
test -d "$DIAG_PARENT"
export OUT="$DIAG_PARENT/${DIAG_REV}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
test ! -e "$OUT"
export EXPECTED_COMMIT="$DIAG_REV"
export EXPECTED_GPU_MODEL=NVIDIA_A100-SXM4-40GB
export DIAGNOSTIC_MODE=profile DIAGNOSTIC_ARCH=D4W1536
sbatch --export=ALL --output="$OUT.slurm-%j.out" --error="$OUT.slurm-%j.err" \
  scripts/sbatch_exp12_runtime_diagnostic.sh
```

The launcher creates `OUT` itself and validates commit, cleanliness, files and
backends. It never submits another job. No commands in this section were executed
against Delta. A future pass still requires exit zero, terminal error-free trace
summary, profile output, original expected call/counter counts and finite positive
timings under the existing validator. It is not scientific calibration approval.
