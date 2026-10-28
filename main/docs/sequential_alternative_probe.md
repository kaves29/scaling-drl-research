# Opt-in sequential random-target diagnostic

Implementation branch `codex/sequential-alternative-probe`, based on pilot
commit `57d2beb78be7cf0e1145f4a9479082a07924cd16`. This is a **new diagnostic**
inspired by Lyle et al.2023, not a sequential method validated by that paper.
The approved probe, training functions, manifests and active Kaggle code are
unchanged. Nothing is registered or enabled in the main experiment.

## Contract and outputs

K=1..8 denotes task phases seen in sequence, never simultaneous heads. A starts
from the current SAC critic; B from the exact saved pre-SAC fresh initialization;
C restarts from that fresh initialization separately for each current target K.
A/B carry parameters between phases. Every arm/phase resets AdamW state and uses
2,000 continuous updates with the current configured LR, decay and batch size.
Transformation objects may be cached; optimizer state is never cached/carried.
The same phase-specific indices and literal target arrays enter all three arms.

A matched environment/seed block has **one** immutable, deterministic eight-
teacher bank and fixed initial raw/preprocessed replay pool/RMS. A designated
producer samples it at warmup using a dedicated stream; other learner sizes
must load it, not independently regenerate pools. This guarantees identical inputs even if independently collected same-seed
Myo rollouts differ; it does not establish a historical mismatch or cause. The producer convention is explicit and needs pilot approval.
Teacher architecture is caller-specified and frozen independently of learner
size; the proposed fixed teacher is D2W512. Teacher keys include environment,
seed and target index, not learner architecture/checkpoint. Independent run
seeds get independent teacher streams. Teacher parameters returned from the same
compiled target-generation call are persisted, not reinitialized for capture.

Each architecture/run separately freezes one offset equal to its saved fresh
critic's initial fixed-pool mean. Its eight literal float32 targets are stored
once and shared bitwise across arms/checkpoints. The common teacher/base/input
bank is shared across learner sizes; **literal targets across sizes may differ
by their prescribed architecture-specific frozen offsets**. No current-critic
or checkpoint-time centering occurs. No new learner heads/task IDs exist.

The shared fresh-derived offset deliberately leaves any current SAC output-mean
difference in A's initial residual. Thus the alternative tests adaptation to an
identical literal target, including mean adjustment; this differs from the
approved per-learner-offset probe. A large positive contrast is not automatically
evidence of reduced plasticity. The initial full-pool losses and prediction
vectors must be interpreted alongside subsequent improvement. No offset is
changed to obtain a preferred result.

Source-defined score is `P=b−MSE_final`, preserving `b=Var(sine base)` and source
float32 score/difference arithmetic. `targets.npz` separately reports empirical
literal-target variance; offset-addition roundoff can make it differ slightly
from source b. Neither value is silently substituted for the other. There are
no ratios, clipping, new statistical thresholds or target-resampling rules.

Reported per K: source b, raw final MSE and score in every arm; `C_minus_A`
(continual-versus-fresh-current-task) and `B_minus_A` (matched sequence). Positive
means A fits worse than its named reference. B/C are expected to be invariant across SAC checkpoints in deterministic
execution of this fixed package; the CPU example verifies that identity.
They are recomputed rather than silently reusing another checkpoint's fits.
No IQM or significance over K is
invented. A-versus-C includes the difference in sequence exposure as designed;
A-versus-B matches exposure counts/order but not the resulting path-dependent
learner state. Neither is automatically a pure causal measure of plasticity. `trajectory.json` preserves all eight phases and seed/architecture.

Each phase stores per-update **pre-update minibatch losses**, separately labeled
full-pool predictions/losses at0/100/500/1000/2000, exact targets and minibatch
indices. A/B model states support continuation; C does not need continuation.
Inputs, raw actions, sampled replay indices, normalizer arrays/metadata, teacher
parameters/key bank, target hashes/shapes/dtypes and fresh state are persisted.
Full-pool final source MSE uses JAX; host arithmetic on stored predictions is
also retained for inspection and may differ by reduction roundoff.

## Dedicated alternative-only runner

Use `diagnostics.sequential_runner.run(args, alternative)` for this dedicated
opt-in pilot. With no alternative or `enabled=false`, it delegates the original
args object directly to `experiments.exp1.run`: the original `RunProbes` path is
unchanged. With the alternative enabled, it constructs the real `Exp12Trainer`
directly and **never constructs or executes `RunProbes`**, regardless of the
unchanged `cfg.probe.enabled` value. Ordinary SAC updates, replay sampling,
normalization, actor/environment evaluation and training logging still use the
existing trainer. There is no global method patch or approved entrypoint edit.
The previous `Hooks` context is retained only for API compatibility; it runs
both diagnostics and is **not** the dedicated pilot integration.

The runner requires an explicitly reviewed request JSON, not the unapproved
pilot manifest. From `main/`, the entry point is:

```bash
python -m diagnostics.sequential_runner --request /absolute/reviewed-request.json
```

This documents invocation; no campaign, GPU execution or launch is authorized
by this implementation. The JSON has `args` and `alternative` objects:

- `args`: existing `config_path`, `config_name`, `overrides`, absolute
  `checkpoint_dir` and positive `checkpoint_interval`. The alternative uses
  recovery/evaluation/interval checkpoints; default-entrypoint extras such as
  `checkpoint_start_frac` do not control these recovery saves.
- `alternative`: `enabled=true`; absolute `bank`, `reference`, `output` paths;
  `environment`, `seed`, actual `architecture` label, frozen `source` identity;
  `bank_producer` boolean; explicit `teacher_architecture` label; and a `teacher`
  object containing `block_type`, `num_blocks`, `hidden_dim`, `dtype=float32`.
  There is no default teacher, environment, seed allocation or pilot cadence.

The configuration must explicitly disable `fork.enabled`: this alternative has
no approved-probe trigger, injected arm or automatic natural-control extension.
It rejects twin/sparse/mixed-precision critics, multiple training environments,
and update ratios below one, rather than changing them. Configured probe
steps must remain 2,000. Normal SAC evaluation cadence drives the later
alternative checkpoints; the initial observation occurs at replay warmup,
immediately before the first replay sampling/SAC update. The saved critic is an
exact functional copy of the current parameters, not a separately seeded model.

One checkpoint is saved at startup and at the last complete step before warmup
capture, in addition to configured periodic/evaluation checkpoints. There is no
per-step checkpointing throughout warmup. A capture interruption restores that
complete boundary and deterministically replays the last warmup transition.
Later evaluation checkpoints are saved **before** the alternative runs. Restore
rechecks the pending checkpoint before advancing training. Completed phase
triplets are reused without refits; an unfinished triplet can be recomputed.
The runner emits a separate `ALTERNATIVE_DONE.json`, not the approved experiment
ledger or `DONE` marker. W&B failure status is retained on exceptions.

The non-producer fails closed if the designated shared bank is absent. Do not
substitute a separately collected pool. The runner verifies source/configuration,
module hashes, backend/precision, reference and checkpoint identities on resume.
`Observer` remains a lower-level API; `evaluate(enabled=False)` returns before
file/model/RNG work. Only private functional model copies are fitted.

## Backend and precision contract

Inputs, teacher/learner parameter leaves, model computation, target arrays and
model outputs must be finite float32. Unsupported float16, bfloat16, float64 or
integer inputs/parameters fail before silent JAX narrowing or fitting. The
network must produce one scalar per pair; mutable extra model collections,
twin/injected/sparse architectures are outside this runner's supported contract.
RMS statistics retain their source precision and normalization formula; only
the resulting input pool is float32, as in the approved source.

The identity records Python, JAX/JAXLIB, Flax, Optax, NumPy and Orbax versions;
backend and platform version, device kind, x64, matmul precision, PRNG
implementation, dtype promotion, and XLA/TF32 override flags. x64 must be off;
bfloat16 matmul precision is rejected. CPU and GPU are recognized backends,
but **only CPU is verified**. A CPU bank cannot silently become a GPU bank;
backend/precision changes require a separately reviewed package and qualification.
Dtype validation can synchronize/copy large parameter trees to the host; these
optional checks and all fitting/storage costs require resource qualification.

Bank/reference creation refuse existing paths. Evaluation resume binds the
checkpoint and starting-state hashes, bank/reference hashes, settings, module
implementation bytes and Python-package versions/precision. A committed K
contains all three arms and A/B continuation states plus checksum receipt.
Files/directory are fsynced before atomic directory publication. Completed K
phases are verified and loaded without refitting. Corruption or identity drift
fails closed rather than silently overwriting evidence. Failure JSON identifies
the uncommitted K; that whole K may be recomputed on resume, never a completed
K. This is phase-boundary resume, not mid-optimizer-step resume. Unsupported
filesystem atomicity/fsync is a qualification blocker, not proof of durability.

## CPU demonstration and verification

Run from `main/` in the repository's existing pinned CPU environment:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_sequential_probe.py \
  tests/test_sequential_runner.py \
  tests/test_sequential_mutations.py \
  tests/test_method_c_ten_reference.py \
  tests/test_probe_discrepancy_adapter.py \
  tests/test_forensic_probe_audit.py

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' python \
  scripts/sequential_probe_cpu_example.py --out "$NEW_EXTERNAL_OUTPUT"
```

These are reproducibility commands, not requests for the owner to execute them.
The demo forces CPU and uses synthetic inputs/tiny real SAC critics. Checkpoint1
is a synthetic parameter perturbation, not SAC training or historical evidence.
It writes all K/arm arrays, receipts and reloadable state for example inspection.

Coverage includes deterministic banks, teacher state reconstruction, target
shape/dtype/diversity, same bank across learner widths, bitwise B=C at K1 and
A=B through all K at fresh initialization; distinct-current and cross-checkpoint
target/index reuse; carry/reset semantics and direct original-primitive checks;
source score signs/baseline; no input/RMS/parameter/global-RNG mutation; corrupt/
partial resume; disabled hooks and original probe outputs; actual tiny residual
SAC networks through8 phases and milestone-prefix equivalence. No GPU, full-
size training or scientific qualification is claimed. No threshold is loosened.
Frozen implementation `545008f4b447bebd6c9b1b9291548b7cfdb751fd` passes
**141 CPU tests,0 failures/skips,231 warnings**:50 alternative-core cases,13
runner lifecycle/rejection cases,6 alternative break/restore cases and72 existing
regressions. The complete existing74-case mutation suite passes twice with
every mutant detected and every restoration passing. All65 simulator restoration
cells pass (13 tasks ×five seeds), verifying exact next transition and action
RNG. Black, Pyflakes, compileall and diff checks pass.

The real-critic example completes two synthetic checkpoints; independent reload
verifies16 phase receipts and48 arm archives, initial A=B, K1 B=C, bitwise B/C
invariance across those checkpoints, target/index hashes and parameter carry.
No additional fit runs during independent verification. The maximum observed
JAX-versus-host final-MSE reduction difference is1.7763568394002505e-15; it is
reported, not used to invent or relax a tolerance.

Warnings are Matplotlib/Pyparsing deprecations and Orbax sharding restoration
notices; stage logs also retain cache/fontconfig diagnostics. The independent
read-only reload emitted one GLFW missing-DISPLAY warning; it did no rendering
or model execution. The receipt preserves exact commands, source/base hashes,
module counts and artifact checksums. Other repository modules were not all
rerun: these are the applicable selected regressions plus the complete mutation
suite, not a claim of entire-repository or full-budget qualification.
Tiny CPU SAC updates in lifecycle tests are engineering validation, not a
campaign, learning result, full-size simulator qualification or GPU test.

## Draft pilot: counts, cost and remaining decisions

See `configs/alternative_diagnostics/sequential_pilot.DRAFT.json` (unapproved,
non-executable). Proposed full matrix: **two preselected tasks × three learner
sizes × five matched independent seed labels =30 training parents**:

|Task proposal|Learners|Seeds|Parents|
|---|---|---|---:|
|dog-run (DMC, eligibility pending)|D2W512, D4W1024, D4W1536|1–5 in each cell|15|
|myo-pen-twirl (Myo, approval pending)|D2W512, D4W1024, D4W1536|1–5 in each cell|15|

These are candidate tasks, not a claim about the current main grid or which DMC
task the owner removed. That removal/eligibility is unresolved from the reviewed
records; owner must confirm selection. The fixed teacher proposal, initial-pool
producer, matched bank/normalizer convention, checkpoint schedule, precision and
resource caps also require review. No approved main manifest is changed.

A **five-total-job development smoke** is separately listed: seed0, dog at all
three sizes, pen at small/large. It is deliberately incomplete, not a balanced
scaling comparison, and has no five-independent-seed inference. Seed0 and task
selection are proposals needing approval, not existing experiment authority.

Capture initial warmup at5,000 interactions and evaluate at50k,100k,...,500k,
matching the nominal SAC evaluation cadence:11 observations/parent. Each costs
3 arms ×8 tasks ×2,000 =48,000 extra probe updates;528,000/parent and15,840,000
for30 parents, **in addition to SAC training only**. The dedicated run does not
execute the approved independent probe. Automatic fork/control extensions are
unsupported; any extended horizon requires an explicit owner decision. Five-job smoke at the same full cadence would add2,640,000 updates;
a shorter engineering schedule needs its own explicit plan/approval.

GPU-hours/walltime cannot be derived from these counts without profiles. Retaining
every phase's A/B model states dominates disk: a75.7M-parameter float32 learner
needs roughly4.5GiB of phase states per observation, about50GiB per large parent
at11 observations; ten large parents alone approach500GiB. The full draft's
rough800–2,000GiB planning range includes smaller models, banks, curves and
metadata but is unmeasured. This is not a13.7GiB Kaggle storage plan. Input
sharing and JIT-transform caching reduce duplication, not the fitting budget.
Expensive evaluation is opt-in and must be separately resource-qualified.

With five seeds/cell, results are exploratory. K/tasks/checkpoints are repeated
measurements; paired seed blocks share banks across architectures. Do not count
30 parents as30 independent target banks or five seeds as confirmatory power.
The pilot can describe sequence-history contrasts and task-order variability;
it cannot prove a general DMC/Myo mechanism or validate an intervention merely
because it yields expected plasticity loss. No Delta/Kaggle job was queued,
launched, cancelled, resumed or changed as part of this implementation.

## Source base and unresolved integration

PR17 remains stacked on PR16 `57d2beb78be7cf0e1145f4a9479082a07924cd16`.
The live shared `main` observed during review was
`087feab341b9be1f377697668f2d1c750ec0467f`, also the merge base. The
AGENTS instruction identifies `main` as the shared target after owner review;
it does not approve adopting every intervening draft dependency. No restack,
force push or merge was performed. See [dependency bindings](../diagnostics/sequential_base_dependencies.json) for
file hashes, importing paths and the commits that last changed them.

The numerical core imports `_mean_prediction`, `_chunked` and
`critic_optimizer` from `experiments/exp12/probe.py` (last changed by
`5fe2ff65ae96cf900b23bf9810f79464730def9d`). The runner uses `Exp12Trainer`
(last changed by `1d849c3b52f1b6bdc2514f8972a2388021b929c7`) and config
composition in `exp1.py` (`20d8c6b47cf9ec5e62f1d35190ea48b5fe115a5b`), plus
their instrumentation/diagnostic imports. The critic calls layers changed by
`f08c5b189e2b53d36b22d06cf165b8ede1f77e15`. These runtime files are inherited
from earlier engineering PRs, not new edits here. The listed helpers also exist
on main; existence is not evidence that all current behavior is equivalent.

No alternative module imports PR16's `audits/probe_discrepancy` adapter,
manifest, launcher or Slurm scripts. PR16-specific pilot changes are therefore
not a direct runtime dependency of the alternative. Removing the inherited
stack still requires an approved integration target and independently validated
dependency versions. That integration remains blocked rather than guessing a
base or silently bringing unrelated draft work into main.

## Review verification summary

The actual tiny lifecycle uses hopper-hop, one residual block/width4, warmup2,
8 interaction steps,14 SAC updates and ordinary evaluations at0/4/8. Alternative
observations occur at2 before any SAC update, then4 and8. These are explicit CPU
fixture settings, not proposed campaign choices. A disabled-probe baseline and
the alternative finish with bit-exact complete state: parameters, target/actor/
temperature and optimizer state, replay and replay metadata, normalizer, counters,
NumPy/Python/JAX RNGs and simulator state. The existing comparison excludes W&B
run ID, which is an operational identity, not learning state.

Deliberate interruptions at the initial observation and a later evaluation
restore from training boundaries1 and4 respectively. Both preserve committed
phase1 bytes/mtime and finish with exactly72 successful fits plus one failed
attempt; committed phases are not fitted again. Default runner construction
still reaches RunProbes; enabled alternative construction forbids it even when
the inherited `probe.enabled` configuration is true.

Alternative arrays and trajectories are written to the configured local output
paths. This runner does not automatically publish them to W&B; ordinary SAC W&B
logging remains unchanged. Artifact archival/transport and full-scale resources
require review before a campaign. The selected regression suite also runs the
PR16 diagnostic-adapter tests as adjacent regressions; those files are validation
dependencies, not imports of this alternative's runtime.
