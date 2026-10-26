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
must load it, not independently regenerate pools. This resolves the possible
Myo same-seed trajectory mismatch without pretending independently collected
pools are equal. The producer convention is explicit and needs pilot approval.
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

## API and optional integration

`diagnostics.sequential_probe` is standalone. `evaluate(enabled=False)` returns
before file/model/RNG work. `diagnostics.sequential_hooks.Hooks` is an explicitly
installed **instance-local** context: disabled does not inspect/patch the trainer;
enabled preserves original fresh-capture/evaluation return values and invokes
this alternative afterward. Original probes still run normally. It does not
patch classes globally or mutate live SAC/target parameters, optimizers, replay,
training RNG, counters or RMS. Only private functional model copies are fitted.
Only ordinary single dense Flax SAC critics are supported by the live observer;
twin/sparse/injected paths and other mutable-network-state families are not
qualified. No existing entrypoint imports these modules automatically.

Explicit dedicated-run integration (not a launch command):

```python
from diagnostics.sequential_hooks import Hooks, Observer

observer = Observer(
    enabled=True, bank=shared_environment_seed_bank,
    reference=architecture_run_reference, output=checkpoint_output_root,
    teacher=fixed_teacher_definition, teacher_architecture=fixed_teacher_label,
    architecture=learner_architecture_label, environment=cfg.env_name,
    seed=int(cfg.seed), source=clean_source_sha,
    bank_producer=is_designated_initial_pool_producer,
)
# Construct the existing trainer and RunProbes by the existing runner first.
# Enter before original warmup/fresh capture and original trainer.train call.
with Hooks(trainer, probes, observer):
    # Existing runner drives original callbacks and configured evaluations.
    existing_run_loop()
```

This pseudocode explains the lifecycle; no executable RL launcher is added.
A non-producer fails closed if the bank is absent. Do not wait inside a scheduled
job or create a different bank to bypass it. On a restored SAC run, restore the
original training state and exact fresh initialization first, then call
`observer.restore(trainer, saved_fresh)` before installing hooks. Calling
`initialize` after SAC updates is refused. Caller integration must supply the
actual saved fresh tree, not a newly seeded approximation. Production trainer/
simulator restore qualification remains separate from these CPU hook tests.

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
Final result:49 passed,0 failed,0 skipped;14 pre-existing Matplotlib/Pyparsing
deprecation warnings. This includes32 new alternative tests and17 existing
regressions. Exact commands/hashes are in the validation receipt and external logs.

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
for30 parents, **in addition to** existing approved probes/training. Any natural
control extension adds observations at the original cadence, not a truncated
horizon. Five-job smoke at the same full cadence would add2,640,000 updates;
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
