# Experiment 3 development-pilot infrastructure

Task branch: `codex/exp3-pilot-implementation`, based on approved integration
`844e3c0cc24998b25c3d1e667bcbfe056b1b336c`. This is an opt-in development
tool, not a confirmatory experiment or a GPU/source-run authorization. Exp1/Exp2
production files, configs, budgets, tolerances and methodology are unchanged.
The two user-reported queued Delta jobs are outside this task and untouched.

## Interfaces and artifacts

| Path | Responsibility |
|---|---|
| experiments/exp3/artifacts.py | Trusted-local complete-state adapter; strict config/protocol fingerprints, source/identity/counter/optimizer/replay/normalization checks; injected single/twin reconstruction via existing code. |
| experiments/exp3/streams.py | Explicit arrival order, raw transitions, source RMS and update counts, chunk checksums, atomic publication, prefix/seal handling, timing/memory/storage instrumentation. |
| experiments/exp3/guidance.py | Shared actions/noise and alpha within each actor; per-state Q/action gradients, critic and SAC parameter gradients; matched AdamW updates, policy changes. |
| experiments/exp3/passive.py | Independent passive SAC learners, shared replay indices, existing diagnostics/precision, checkpoint/restart and evaluation RNG isolation. |
| experiments/exp3/target_policy.py | Frozen policies, common frozen target-network evaluator, fixed restored alpha, shared base noise, existing critic optimizer and single/twin loss. |
| experiments/exp3/runner.py | Explicit protocol execution, input fingerprints, streaming evidence, outcomes, sensitivity and restart. No scheduler or automatic source generation. |
| experiments/exp3/recording.py | Process-local instrumentation around the existing Exp1 control or Exp2 arm training loop; disabled unless explicitly invoked. |
| configs/exp3/ | Versioned non-runnable owner-choice templates; no invented scientific numbers. |
| scripts/exp3_pilots.py | audit / manifest / run commands. |
| scripts/exp3_record_source.py | Explicit future source-run recording, only with separate authorization. |
| scripts/exp3_gpu_validation.py | Tiny engineering test harness on an already allocated A100; never submits jobs. |

Artifact specifications use absolute `state` paths to actual committed
`step_<...>` directories, with the actual producing `run_metadata.json` path.
Do not point at a moving LATEST pointer. Pickles/Orbax inputs must come from
trusted sources; this is compatibility checking, not an untrusted-file sandbox.
Model-only checkpoints are rejected; no missing Adam moments, normalization,
RNG, replay or transition history is synthesized. Full source state and all
auxiliary panels/stream manifests are fingerprinted in each run receipt.
Only the fork replay is loaded for Pilots1/3; other replay archives are header-
validated without loading redundant million-transition arrays into RAM.

## Scientific choices remain unset

The two selected DMC IDs are dog-run and humanoid-walk (`dmc_hard`); seeds1–5
come directly from `generate_manifest.EXP12_SEEDS`. MyoSuite has FOUR approved
registered tasks, not two: myo-key-turn → myoHandKeyTurnFixed-v0,
myo-pen-twirl → myoHandPenTwirlFixed-v0, myo-pose-hard → myoHandPoseRandom-v0,
myo-reach → myoHandReachFixed-v0. The owner must select two. Manifest generation
refuses an unset, duplicated or unknown pair. Once selected, it materializes
exactly20 unique environment/seed cells with per-pilot artifact dependencies.
CPU tests' example pair is a test fixture, not a scientific selection.

The owner must also approve architecture and checkpoint selection; common panel
size/provenance and chunk size; diagnostic RNG seed; alpha and normalization
assignment; intervention space, conditions and zero-signal handling; passive
arrival/update budget, normalization, evaluations, checkpoint cadence and
injection m/seed; frozen-target budget/batch size/alpha/critic/evaluator choice.
Templates have `approved:false`; Pilot3 additionally has `enabled:false`.
Filling `approved:true` represents an external owner decision, not an agent's
scientific qualification. No final Exp3 methodology is inferred from this pilot.

Pilot1 `common_fork` uses a declared common-normalization intervention.
`artifact_specific` preserves each actor and critic's saved preprocessing, while
raw states/actions remain matched. `alpha_source=fork` holds fork alpha across
actors; `actor_specific` uses each actor's checkpoint alpha for BOTH critics.
Critic aggregation is singleQ or min(Q1,Q2), as ordinary SAC. Per-state Jacobians
are chunked; the single optimizer comparison aggregates over the complete panel,
starts every condition from the same actor/Adam state, and retains the entropy
gradient. Chunk size is part of the reproducible draw protocol, not an implicit
performance knob to change between comparisons. Optional `source_panels` maps
labels to `{path,size}` and is independently fingerprinted; the common panel is
always retained.

Two implemented intervention spaces require an explicit choice. `action` replaces
dQ/da before the actor Jacobian. `parameter_per_state` replaces the per-state
critic-derived actor gradient before panel averaging. For either space, direction
uses gI×||gU||/||gI||; magnitude uses gU×||gI||/||gU||; full uses gI; sanity uses
gU. Entropy is unchanged. No epsilon, clipping or implicit zero threshold is added.
At an exactly zero norm, `error` refuses the undefined substitution;
`keep_baseline` retains gU. These are alternative diagnostic protocols, not adopted
scientific choices. Undefined cosine is stored with a `direction_defined` mask.
Model/log-density/gradient nonfinites fail. Actual parameter updates and per-state
policy means/SDs/actions are retained, with approved full-FP32 policy-function
diagnostics. Fresh common-seed outcome evaluations are descriptive; one-step
gradient differences or those outcomes alone do not establish harmful guidance.
Longer intervention consequences would require a separately approved duration.

Pilot2 StageA reads only the actual U stream. StageB explicitly adds the actual
I stream and four passive learners; active U/U and I/I are references only. Both
learners start from one immutable pre-injection fork; only I's critic/target is
injected using existing machinery. Actor/Adam/alpha/RNG/normalization/replay must
match initially. The existing256-pair `fork_panel` is required, checked against
restored normalization and used for the approved immediate Check1 comparison.
Initial injection failures stop the diagnostic; no tolerance is selected.
Recorded update counts must match the explicitly approved passive update timing.
Sampling uses a dedicated RandomState restored from the fork; each learner's
own replay must yield identical batches at the same indices. No training env
or sample_actions call exists in passive delivery. Normalization must explicitly
use recorded source statistics or the frozen fork statistics. The latter is a
diagnostic alternative, not ordinary normalization silently disabled.
Evaluation-only environments use the same protocol/seed and restore all learner
and global RNGs. Ordinary SAC actor/alpha/critic/Polyak ordering and existing
actor diagnostic precision/cadence are reused. No active-policy RNG consumption
is injected into passive training. Historical active sampling sequences need
not equal passive sequences; matching is between passive learners.

Pilot3 clones the selected online critic AND Adam state into both learners. The
selected target-network evaluator remains frozen; alpha is the actual scalar
from the explicitly selected checkpoint. Shared Gaussian noise is transformed
by each frozen actor. The actual tanh preimage is retained for log density;
rounded saturated actions are never inverse-tanh reconstructed. Terminal masks,
gamma^n and twin shared-min targets follow SAC. Only critics are updated, using
the existing Trainer/AdamW; no actor, alpha or frozen evaluator update occurs.
`common_fork` versus `artifact_specific` preprocessing is explicit here too.
Alternative target evaluators reuse identical batches/base noise but fit separate
diagnostic clones. Final fixed-panel Q/dQ/actor-guidance measurements and every
target/batch-index/difference/loss are retained. This frozen-target experiment
intentionally does not perform online target-network Polyak changes.

## Reuse, missing data and incremental cost

See [exp3_implementation_plan.md](exp3_implementation_plan.md) for the dependency
matrix. No actual per-environment/seed Exp2 pilot artifacts have been supplied
to this workspace; scratch files are engineering fixtures only. Delta paths
are not inspected. Existing checkpoints/evaluation panels/returns can be reused
if compatibility passes. Missing intermediate checkpoints and chronological
trajectories cannot be reconstructed from a circular replay snapshot.

If an ordered U source is missing, StageA requires one separately authorized U
continuation per eligible cell from the actual fork. StageB needs an additional I
source where missing. These are upper requirements of20 U and20 I continuations,
not20 automatically eligible forks or permission to launch them. Cells without
an actual eligible fork remain blocked. Retained verified streams reduce that
count. Source recording changes I/O cost but performs no extra training steps,
updates, evaluations, checkpoint calls or RNG draws. Completed CPU exact-state
parity is engineering evidence, not a CUDA guarantee. Recorder restart requires
an unsealed checksummed prefix whose cursor EXACTLY matches the restored source
checkpoint. It flushes arrivals at existing save boundaries. A prefix ahead of
or behind the source state is refused; records are not deleted or invented.

Per cell, cost is: compatible artifact reads + matched-panel gradient diagnostics
and outcome episodes + two passive learners' prescribed updates for StageA
(four for B) + two critic fits per prescribed target update/evaluator. Missing
source continuation/evaluation costs are additional. Do NOT assume passive cost
0.4–0.7 of an active arm. Measure environment stepping, transition delivery,
SAC updates, evaluation, backend compilation, memory and storage. First-call
times include tracing/compilation; separately measured backend_compile time
excludes tracing/lowering. Nested timings must not be added. CPU memory is peak
RSS; device memory_stats is null on CPU. GPU time/memory and simulator-speed
estimates remain UNVERIFIED until actual representative A100 measurements.
Stream memory is bounded by chunk size; per-state parameter gradients require
roughly chunk_size×actor_parameter_count×dtype_bytes per retained signal family.
Full runs retain explicit-cadence checkpoints and evidence on disk; provision
storage from measured bytes per checkpoint/arrival, not an invented fixed bound.

## Exact interfaces for future authorized execution

Run from `main/` in a clean checkout of the delivered full task-branch SHA.
Use the existing pinned dependency environment; no dependency upgrades are needed.
Prepare owner-approved JSON/YAML copies outside Git; all placeholder choices must
be resolved first. Output directories must be new absolute paths outside Git.

```bash
python scripts/exp3_pilots.py audit \
  --state /absolute/actual/step_directory \
  --metadata /absolute/producing/run_metadata.json

python scripts/exp3_pilots.py manifest \
  --config /absolute/owner-approved/exp3-pilots.yaml \
  --artifact-root /absolute/exp3-inputs --out /absolute/new/manifest.json

python scripts/exp3_pilots.py run \
  --spec /absolute/owner-approved/pilot1.json --out /absolute/new/exp3/pilot1
python scripts/exp3_pilots.py run \
  --spec /absolute/owner-approved/pilot2.json --out /absolute/new/exp3/pilot2
python scripts/exp3_pilots.py run \
  --spec /absolute/owner-approved/pilot3.json --out /absolute/new/exp3/pilot3

# Same source/spec/input fingerprints; only interrupted Pilots2/3 resume.
python scripts/exp3_pilots.py run \
  --spec /absolute/owner-approved/pilot2.json --out /absolute/existing/exp3/pilot2 --resume
```

Only after separate authorization for a NEW source continuation:

```bash
python scripts/exp3_record_source.py --experiment exp1 \
  --args /absolute/ordinary-exp1-arguments.json \
  --stream /absolute/new/stream_u --chunk-size OWNER_APPROVED_INTEGER \
  --authorized-source-run --benchmark
python scripts/exp3_record_source.py --experiment exp2_arm \
  --args /absolute/ordinary-exp2-arm-arguments.json \
  --stream /absolute/new/stream_i --chunk-size OWNER_APPROVED_INTEGER \
  --authorized-source-run --benchmark
```

These reuse the existing entry-point arguments and schedules. Do not use queued
job directories, completed-run directories or existing streams as fresh output.
`--resume-stream` applies only to an explicitly authorized interrupted recording
with matching source checkpoint/prefix; it does not reconstruct historical data.

## Minimal A100 qualification plan, not a submission

First authorize a separate tiny engineering validation on one A100. The existing
four-CPU/32GB/seven-minute/300-second diagnostic envelope is a proposed bound for
this NEW test, not inherited permission from gpu_resume_95. No Slurm script or
submission is generated by this task. Within an owner-provided allocation:

```bash
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=disable WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION XLA_FLAGS
unset NVIDIA_TF32_OVERRIDE XLA_PYTHON_CLIENT_MEM_FRACTION
timeout -k 15 300 python -u scripts/exp3_gpu_validation.py \
  --expected-commit FULL_DELIVERED_TASK_SHA --out /absolute/new/exp3_gpu_evidence \
  > /absolute/new/external-exp3-gpu.log 2>&1
```

Preserve shell exit status, log, backend.json, tests.log and validation.json, plus
Slurm stdout/stderr/allocation/exit records and archive checksum. Harness PASS
requires every selected single/twin gradient/target/stream assertion to execute
and pass without skips, actual A100 default backend and available CPU backend,
exact source and clean tree. Timeout/missing outputs/nonfinite/mismatch is
INCOMPLETE/failed qualification; never increase budgets or relax assertions
automatically. Unit-test oracle tolerances are not Exp3 scientific thresholds.
Completion within300seconds and full-width memory/runtime are unmeasured.

Next, once checkpoint/m/panel/budget choices and actual compatible data are
approved, validate one real-width representative cell: input restoration, Check1,
one prespecified tiny diagnostic workload for each pilot, interrupted restart,
recording on/off complete-state equivalence and device/precision evidence.
Validate MyoSuite source/evaluation restoration independently, and twins where
future diagnostics use them. Additional simulator/source runs require separate
authorization. Neither the tiny harness nor CPU tests qualifies full-length pilots,
causal conclusions, final Exp3 or Exp1/Exp2 scientific gates.
