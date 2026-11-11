# Revised online Experiment 2 implementation

Implementation base: `1b5db60dfa91dc15a0172113118a919afbf36d5e` on
`codex/exp2-online-prototype-20261107`. The binding requirements are
`online_exp2_pilot_revision_requirements.md` and
`online_exp2_handoff_20261111.md`. Resolve final implementation/package commits
from Git; this file cannot contain its own commit identity.

## Implemented path

The two pilot YAML profiles now contain the approved population eligibility,
one-million actual raw-step reference budget, action repeat two, fifteen raw
checkpoints, full assay settings, and four 10,000-raw-step conditions. The
population scheduler selects the matching DMC or MyoSuite environment profile
for each of twenty independent environment/seed/regime identities. The run
validator rejects unsupported population members, old branch budgets, altered
raw schedules, incompatible environment types, and altered full assay sizes.
The development prototype remains separately available.

`Exp12Trainer` persists an actual executed-raw-step counter, including early
termination within action repeat. Only the revised pilot uses raw-budget loop
termination. RepeatAction supports a per-action allowance to prevent crossing
a reference checkpoint or the branch's final budget. The next-step Gymnasium
autoreset path is explicitly reset after the terminal transition is recorded,
so automatic-reset calls are not mistaken for physical environment steps.
Older same-step final-observation metadata is also supported.

At each checkpoint, the reference is saved before diagnostics. The runner then
runs small/frozen, large/frozen, small/learning, and large/learning sequentially,
followed by the established plasticity assay. Every branch begins with the
same reference actor and actor optimizer. Frozen conditions retain shadow
critic/target/optimizer and reference temperature and observation statistics.
Learning conditions copy the chosen shadow critic and target (including critic
optimizer), reference actor/temperature/optimizers and SAC RNG, and a separate
reference replay. Their source agent `update_many` executes the original SAC
update order, temperature learning, target update and training normalizer.
The local fractional update carry begins at the reference carry.

A branch starts a new environment episode. Its copied n-step queue is cleared
because unfinished reference transitions cannot be joined to a different
branch episode. All already committed reference replay entries remain. The
copy allocates enough capacity for the reference data plus the maximum possible
branch additions, preserving ring behavior without copying unused capacity.
The reference replay is never appended to by a branch.

Each active condition has one atomic restart file containing model/optimizer,
replay, RNG, normalizer, environment, episode, diagnostic and update-schedule
state. A checkpoint progress file records completed conditions. A completed
branch is compacted to its episode/diagnostic rows, counters, model hashes,
replay hash/count and provenance; it no longer needs replay/model tensors for
restart. This avoids retaining sixty large replay/optimizer copies. Reference
restart state retains the latest complete training checkpoint. Partial current
branch work after the last periodic save is recomputed on resume.

Reference/shadow model and optimizer hashes, replay data/count/index/n-step
queue, raw counter, normalizer, RNG and both reference environments are checked
for isolation. Resume requires the same configuration bytes, source commit,
source file hash and run identity. Numeric scalar serialization is canonicalized
for isolation hashes so Python and restored NumPy/JAX scalars compare by value.
No identity check is bypassed.

## Metrics, analysis and W&B

Branch measurements are ordinary completed training-episode returns. No branch
baseline evaluation or separate 5k/10k evaluation is run. The 5k graph marker does not shorten an action repeat or add an interaction;
when an interaction crosses it, the observation is labeled with its actual raw
counter. Partial episode return
is recorded explicitly at markers and never relabeled as a completed return.
Analysis preserves environment, independent seed/regime, condition and
checkpoint; it exports the training curves and raw-step markers, with no old
paired-evaluation estimand or invented independent episode replicas.

Learning conditions call the source actor/critic diagnostic functions and
ActorDiagnostics. Frozen actor updates retain the source update_actor formula,
with the established churn, KL direction, saturation, gradient norm/spread,
source gradient-cosine function and actor/critic feature/weight/rank/dormancy
functions. Diagnostic sampling is isolated from the frozen training sampler.
Condition records retain architecture and reference regime plus source/config
identity and run ID. Reference shadow update evidence streams to disk rather
than accumulating a million Python dictionaries in saved trainer metadata.

Each checkpoint also writes an ordinary training-return PNG with dashed raw-step
markers at 5,000 and 10,000; the same image is published to W&B.

The valid W&B project is `Experiment-2-Pilot-Testing`. The notebook authenticates
from Kaggle Secrets/environment without printing credentials, resolves the
account's default entity and requires access before starting. Reference and
each checkpoint/condition use distinct raw-step axes, avoiding W&B's global-step
rejection when branch and reference counters differ. Real online delivery still
requires verification on Kaggle; disabled-mode CPU tests do not establish it.

## Kaggle packaging and concurrency

New notebook: `main/notebooks/exp2_revised_complete/experiment-2-revised-complete.ipynb`.
It preserves the old notebook and the owner's dirty `Experiment 2 Testing.ipynb`.
Default mode runs full-sized reference regimes, all four full-10k branches,
full plasticity and an interrupted learning-branch restore. A solo large-reference path first establishes measured VRAM use. When twice
that peak fits, two independent qualification processes stress the mixed
dog-run reference regimes on that physical GPU while nvidia-smi samples
aggregate VRAM use; otherwise the other regime runs sequentially. Child process logs go to files, keeping notebook outputs
small. No GPU model/count is assumed. A failed process fails qualification;
it does not become a readiness claim.

Qualification shortens reference training to 10k raw steps and one checkpoint;
this is an explicit engineering fixture, not a full-million-step result. The
pilot mode schedules the actual twenty approved runs with one visible GPU per
process. The ceiling is two independent runs per GPU. Two-job admission needs
an exact-source, exact-device receipt covering full branches, assay, restart,
concurrent success and measured peak VRAM that fits currently free memory.
Absent/different/incomplete/oversized evidence falls back to one. The actual
second-job launch also requires the exact environment/regime pair recorded
for that device; a mixed dog-run receipt never admits large+large or MyoSuite. Scheduler
failures retain logs/checkpoints and are resumable by the same command.

Kaggle runtime setup reuses the prior working dependency setup and its explicit
Python 3.13 substitutions. It is not the pinned CPU software environment.
Notebook `SOURCE_SHA` must be pinned to the reviewed implementation commit
before upload. No Kaggle notebook was pushed or launched by this implementation
task. Preserve `/kaggle/working/exp2-revised` between sessions; a Kaggle session
limit does not change scientific budgets or imply all twenty runs fit.

## Validation and limitations

Final stable-tree CPU validation is recorded below after execution. CPU tests
cover real SAC model updates, four isolated conditions, early termination,
markers, exact raw caps, per-condition durable state, learning-branch
interruption followed by equality against an uninterrupted control, real DMC
integration/completed restart, configuration and concurrency admission. Legacy
online-prototype and diagnostic primitive regressions are included. The local
interpreter is `/workspace/scratch/scaling-drl-venv/bin/python`; unittest is used
because pytest is absent. Test logs stay outside Git.

Completed reference runs still retain their latest full source checkpoint.
Campaign storage therefore grows with the twenty outputs; preserve/export
outputs between sessions and check disk capacity rather than assuming a full
campaign fits Kaggle local storage. Qualification does not establish late-run
full-replay host-memory/storage capacity.

GPU execution, actual full-size memory peaks/concurrent qualification, live
W&B delivery and full-budget learning remain unrun here. The notebook/qualifier
make those checks executable; they do not claim the checks have passed. A
given GPU's two-job qualification must complete before requesting concurrency
two on it. MyoSuite and all-environment full-budget runs remain execution
coverage to establish on Kaggle. Nothing was launched on Delta and no Exp1 job
was changed.

### Validation receipts and independent review

- Main CPU suite: **29 passed in 208.046 s**, log
  `/tmp/exp2-revised-complete-validation.log`. Modules: `test_online_exp2`,
  `test_online_exp2_diagnostic_primitives`, `test_online_exp2_driver`,
  `test_revised_exp2_runner`, `test_revised_exp2_scheduler`.
- Shared trainer/diagnostic/checkpoint regression coverage: **21 distinct
  tests passed**. The first combined run passed all 15 non-W&B tests and three
  W&B tests; three W&B fixtures bypassed construction and lacked `cfg`.
  After setting the fixture's explicit legacy configuration, all **6 W&B
  transport tests passed in 0.620 s**, including the real offline SDK. Logs:
  `/tmp/exp2-shared-regression.log`, `/tmp/exp2-wandb-regression.log`.
- Actual DMC active learning-branch interruption/control comparison passed
  before review in 107.940 s (`/tmp/exp2-real-active-restart.log`).
- Independent review subsequently found two gaps, both fixed: ExactRestore
  now records/replays each action's repeat allowance and retains old-state
  compatibility; second-job admission now checks the device's measured exact
  environment/regime pair, including refills and skipped completed jobs.
  The source hash includes `envs.py`, the repeat wrapper, logger, diagnostics,
  source SAC and runner, so stale restore semantics cannot resume silently.
- Final scoped review-fix suite: **5 passed**, including a real DMC **odd raw
  checkpoint at step 3**, an interrupted learning branch and uninterrupted
  control with identical final model/replay hashes and episodes, plus four
  scheduler/admission tests. Log: `/tmp/exp2-review-fixes.log`.
- A separate real DMC shortened-repeat/old-format restore regression passes;
  final log `/tmp/exp2-env-repeat-restore-final.log`. This proves that the
  physical next transition matches after a `[2, 1]` raw-repeat history, and
  that historical action-only checkpoint histories retain normal repeats.
- All **20 production configurations** compose and validate, and the existing
  matched-population/checkpoint-schedule function passes. Notebook code cells
  parse, outputs are empty, metadata identifies the correct notebook, Black
  checks pass and `git diff --check` passes. No test skip is counted as a pass.

The 29-test main and shared receipts predate the two scoped review fixes; the
five-test scoped suite and direct real-simulator compatibility regression
verify those final changes. Logs preserve earlier failures rather than
presenting them as successful runs. Source/file and log checksums are retained
outside Git under `/workspace/scratch/exp2-revised-validation-20261111/`.

Two-job evidence is **not universal**. The supplied qualifier measures the
mixed small/large **dog-run** reference workload. The scheduler only allows
that measured pair to share its qualified physical GPU; two large-reference
runs, two small-reference runs, and other environments run singly until their
exact pairs are measured. The notebook measures a solo large-reference path
before admitting its two-job stress, uses fresh qualification-attempt paths
so completed-run reloads cannot masquerade as memory stress, and records
actual workload identities in the receipt. Failure of a concurrent stress
fails that attempt rather than emitting a successful receipt.
