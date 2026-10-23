# EchoCritic: self-contained Codex Cloud research and engineering handoff

Snapshot: 2026-10-23 19:47:48 UTC. Documentation source: local files, committed diagnostic source,
archived analysis exports and dated Slurm receipts. Read all twelve sections before
spending GPU time. This document is also stored at
`/Users/shouryakaveti/VS_Projects/agent_context/cloud.md` on the owner's Mac.
The synchronized Git copy is `main/docs/agent_context/cloud.md`.

## Immediate priorities and an important new defect

1. Obtain authorized Delta access from the execution environment you actually use.
2. Reconcile job **23120792** and its artifacts before any new GPU submission.
3. **Correct the diagnostic runner's optimizer-config selection before trusting or
   extending its results.** This defect was discovered while assembling this
   handoff, after earlier progress reports, and supersedes their claim that the
   runner uses the campaign optimizer settings.
4. Use the corrected reconstruction to distinguish measurement/task effects from
   actual adaptability changes. Preserve original data and the four-hour ceiling.

### Confirmed diagnostic-only config-selection defect (new, uncorrected)

`main/scripts/exp96_fixed_task_diagnostics.py`, `Investigation.__init__`, explicitly
loads `main/configs/agent/sac.yaml` and passes it to `critic_optimizer`. That generic
file specifies **critic LR=3e-4, weight_decay=0.0**. The campaign's
`main/configs/base_exp12.yaml` selects **agent:sac_simba**, whose
`main/configs/agent/sac_simba.yaml` specifies **critic LR=1e-4,
weight_decay=1e-2**. The frozen `RunProbes.__init__` constructs its optimizer from
`trainer.cfg.agent`, not the generic SAC file. All 96 canonical configurations in
the archived W&B inventory confirm `(critic LR, critic decay, actor LR, actor
decay)=(0.0001,0.01,0.0001,0.01)`.

Evidence was inspected without running new fits: the two YAML files,
`main/experiments/exp12/run_probes.py`, the runner constructor, and a canonical-ID
join of the archived inventory. Both queued `bcf6cf3` and latest diagnostic
`5974706` inherit this selection defect. The tiny fitting tests themselves use
`AdamW(1e-4, weight_decay=.01)` and therefore did **not** catch the runner's wrong
config selection. CPU persistence/segmentation tests remain useful, but do not
validate the reconstruction configuration.

This is an implementation defect in the **new diagnostic**, not a demonstrated
cause of the original Exp96 anomalies. No source fix or new GPU retry was made
during handoff preparation. No queued job was modified or cancelled. If23120792
has executed, preserve its evidence and inspect its config hash/runtime/receipts;
do not label a run with the wrong optimizer a valid historical reconstruction.
The existing mismatch guard may have stopped it, but that outcome is unverified.
Use each selected original checkpoint's `run_metadata.json` →
`resolved_config.agent` to derive and bind the correct optimizer, validate
agreement across selected parents, and add a targeted runner-config regression.
This restores frozen settings; it is not hyperparameter tuning. Simply changing
scores, tolerances, or the original campaign to make reconstruction pass is not a
correction.

## 1. Project overview and scientific authority

EchoCritic studies how scaling a SimBa SAC critic, with the actor kept fixed,
affects critic adaptability and downstream actor learning. Distinguish four
claims: architecture-dependent finite-budget fitting; training-induced
adaptability loss on a controlled task; association with policy performance; and
causal propagation from critic pathology to actor learning. Only the first and
parts of the observational association are established. Do not assume the
hypothesis must survive the investigation.

Frozen official Exp96 source:
`f597ee9898e4d401912092c336c3e5330cf2fda1`.
Repository: https://github.com/kaves29/scaling-drl-research .
Diagnostic branch: `codex/exp96-fixed-task-diagnostics-20261022`.
Latest tested diagnostic implementation before this documentation:
`5974706b3cabcaa2140f575f2e0712bd40ae236f`.
Queued job source: `bcf6cf3c90726c77f998e5795fedf0d62c5f11e2`.
The handoff's enclosing documentation commit is obtained with
`git log -1 --format=%H -- main/docs/agent_context/cloud.md`; a file cannot contain
its own enclosing commit hash without a self-reference problem.

### Frozen experiment and hyperparameters

The actual source paths below are relative to repository root. Application code
lives under nested `main/`; run Python commands from that directory.

| Item | Frozen setting / evidence |
|---|---|
| Population | 4 environments ×3 critics ×8 seeds1–8 =96 unique parents |
| Environments | `dog-run`, `humanoid-run` (`dmc_hard`); `myo-key-turn`, `myo-pen-twirl` (`myosuite_simba`) |
| Myo IDs | `myoHandKeyTurnFixed-v0`, `myoHandPenTwirlFixed-v0` |
| Critic definitions | D2W512=(2 residual blocks,512 width), D4W1024=(4,1024), D4W1536=(4,1536) |
| Actor | 1 residual block, width 128; fixed across critic conditions |
| Horizon | 500,000 agent interactions, action_repeat2, raw env-step budget1,000,000 |
| SAC | AdamW actor and critic LR 1e-4, decay 1e-2; temperature LR 1e-4, decay 0 |
| Entropy | Initial temperature.01; target entropy coefficient−.5×action dimension |
| Targets | Polyak tau.005; n_step1; single critic for these `episodic:false` envs |
| Discount | Formula `max(min((effective_episode_len/5-1)/(effective_episode_len/5),.995),.95)`; effective length=max_episode_steps/action_repeat; DMC .99, Myo .95 |
| Replay | Uniform NumPy; capacity 1,000,000; warmup 5,000; sample batch 256 |
| Updates | 2 per interaction after warmup; preserve exact frozen update/RNG order |
| Normalization | Running observation normalization enabled via sac_simba; internal residual LayerNorm |
| Precision | FP32 parameters/arrays; `tensorfloat32` GPU matmul policy; mixed_precision=false |
| Probe schedule | Baseline after 5,000 warmup interactions, before first critic update; checks 1–50 at 10,000-interaction intervals |
| Probe target count | Ten per scheduled check; check 0 is fresh-only |
| Fitting | 2,000 AdamW updates with fresh optimizer state per fit;256 minibatch;25,600 pool entries; eval chunks2,560 |
| Targets | A freshly initialized same-architecture teacher; `z=sin(100000*f_omega(x))`; each learner fits own-mean offset plus z |
| Operational onset | Paired per-target score gaps; fractional IQM10;10,000 paired bootstrap resamples;95% interval; lower bound>0; two consecutive checks; eligibility≤95% budget |
| Interpretation | Operational detector; no validated trajectory-wide5% false-positive guarantee |

Authority/implementation references: `AGENTS.md`, `main/.claude/methodology-exp1-exp2.md`,
`main/docs/exp12_decisions.md`, `main/configs/base_exp12.yaml`,
`main/configs/base_exp12_ten.yaml`, `main/configs/agent/sac_simba.yaml`,
`main/configs/buffer/numpy_uniform.yaml`, `main/configs/env/dmc_hard.yaml`,
`main/configs/env/myosuite_simba.yaml`, `main/scripts/exp96_campaign.py` and
`main/generate_manifest.py`.

### Working authority and safety

- Keep the frozen commit, official W&B runs, queues, checkpoints, replay and
  analysis outputs intact. Work on this diagnostic branch in isolated checkouts.
- The owner authorizes **at most4 cumulative physical A100 GPU-hours for this
  investigation**, including allocation idle/compile/failure time. This is not4
  hours per job or a new budget for each agent. Reconcile all attempts before
  further submission. No full SAC training is authorized by this budget.
- Keep LR/optimizer/decay, architecture, SAC/reward/entropy settings fixed. Do not
  implement historical reports' proposed LR/decay sweeps. Extending fitting4k/8k
  is an authorized diagnostic, not a silent change to the official2k detector.
- Implementation defects can be corrected with evidence and targeted tests.
  **Notify the owner and obtain their decision before adopting a scientific
  metric/target/normalization/interpretation change.** Diagnostic ablations may
  test the already-authorized alternatives; they are not production adoption.
- Preserve first job 23120792; the current instruction is not to modify/cancel it.
  Do not duplicate it. Confirm actual consumption before any subsequent job.
- Use small CPU contracts for changed code. Reuse completed evidence; do not
  rerun broad suites, benchmark comparisons or literature reviews by default.
- Commit/push scoped work without force-push/main merge. Owner reviews scientific
  choices. Other Cloud work may exist; inspect live refs before integrating.
- Never invent PASS/approval evidence or replace the official IQM definition.

## 2. Repository architecture and data flow

### Source map

| Path | Responsibility |
|---|---|
| `AGENTS.md` | Source/ownership/validation instructions; read first |
| `main/docs/AGENT_HANDOFF.md`, `main/docs/EXPERIMENT_STATUS.md` | Historical source ledger; dated entries are not live approval/status |
| `main/run.py` | Experiment dispatch |
| `main/experiments/exp1.py` | Hydra composition, metadata and parent lifecycle |
| `main/scripts/exp96_campaign.py` |96-row plan/queue, source/config binding, startup and completion validation |
| `main/scripts/exp96_multigpu.py` | Historical production multi-GPU orchestration; do not run for this investigation |
| `main/experiments/exp12/trainer.py` | SAC collection/update/evaluation and state restoration |
| `main/scale_rl/agents/sac/sac_network.py` |SACEncoder, actor, critic definitions |
| `main/scale_rl/networks/layers.py`, `main/scale_rl/networks/utils.py` | Residual blocks/initialization |
| `main/scale_rl/agents/sac/sac_update.py` | Actor/critic/temperature/target updates and diagnostics |
| `main/scale_rl/agents/wrappers/normalization.py` | Observation normalization and RMS checkpoint handling |
| `main/experiments/exp12/state.py` | Atomic complete training state/replay layout |
| `main/experiments/exp12/run_probes.py` | Fresh capture, probe scheduling, archives, records |
| `main/experiments/exp12/probe.py` | Original numerical target, fit, RNG and IQM implementation |
| `main/experiments/exp12/trigger.py` | Paired bootstrap/onset logic |
| `main/experiments/exp12/ledger.py`, `main/experiments/exp12/fork.py` | Scientific records and natural fork lifecycle |
| `main/experiments/exp12/diagnostic_artifacts.py` | Read-only partial critic/replay/RMS access |
| `main/experiments/exp12/fixed_task.py` | Continuous fitting and additional measurements |
| `main/experiments/exp12/probe_tasks.py` | Immutable numeric tasks and strict pairing |
| `main/scripts/exp96_fixed_task_diagnostics.py` | Retrospective reconstruction and controlled screen; optimizer-selection defect above |
| `main/scripts/sbatch_exp96_fixed_task_diagnostics.sh` | Single-A100 source-bound execution wrapper |
| `main/scripts/check_exp12_validation_checkout.sh` | Checkout/source checks invoked by wrapper |
| `main/tests/test_fixed_task.py` |7 original diagnostic CPU contracts |
| `main/tests/test_probe_tasks.py` |4 immutable-task/pairing contracts |
| `main/tests/test_fixed_task_bank_integration.py` |2 persistence/runner integration contracts |
| `main/docs/exp96_fixed_task_diagnostics.md` | Prospective predictions, scope and dated receipts |
| `main/docs/exp12_cpu_validation_setup.md`, `main/requirements.txt` | Environment/validation references; broad suite is not the next task |

### Data flow

1. Read canonical frozen `queue/queue.json`; it carries parent row identity,
   checkpoint_dir and source binding. Do not claim or alter production queue items.
2. Read fresh critic and selected latest/early-fork critic leaves, RMS and replay
   from durable checkpoints. Extract only needed arrays; bind metadata/leaf hashes.
3. For historical reconstruction regenerate keyed pool positions/teacher/minibatch
   draws under the original config, checking against original archived curves.
   For new controls freeze explicit inputs/targets/indices and label them retrospective.
4. Fit functional parameter copies with freshly initialized AdamW state. Carry
   optimizer state continuously for2k/4k/8k comparisons. Do not mutate SAC state.
5. Save numeric tasks, full minibatch loss curves, full-pool metrics, parameter
   provenance, runtime and synchronized timings under a new diagnostic directory.
6. Analyze paired target outcomes and seed replication separately; produce CSVs,
   plots and a claim/evidence table. A lower loss is not a validity certificate.

### Commit history

| Full SHA | Purpose / validation |
|---|---|
| `f297dd49088a3ef7184fe13c884fa3f7ce8d58b0` | Initial diagnostic modules/runner/wrapper/tests;7 CPU tests3.426s |
| `4f10ca9ef4e4d151f388d41fad775fa71b6bd38e` | Prospective controls and instrumentation |
| `2c9f280128cee37b726d0bee86df348a1a18d3ec` | Direct-script import/source binding, compiled continuation;7 tests3.128s |
| `bcf6cf3c90726c77f998e5795fedf0d62c5f11e2` | Short allocation cap; queued23120792 uses this |
| `5d86e23a3ad5c009f31bd67149e15c11e0ccd16b` | Immutable task banks,4 tests subsequently passed |
| `5974706b3cabcaa2140f575f2e0712bd40ae236f` | Task-bank runner integration;6-test suite1.878s |

At snapshot, committed diagnostic changes are9 added files,1,473 insertions
relative to f597. No original SAC/probe source file was edited. One extra local
untracked file, `main/experiments/exp12/autonomous_control.py`, is a draft only.
Section9 and AppendixA preserve it for Cloud without pretending it is tested or
part of 5974706. Do not run a dirty tree: the source gate intentionally rejects it.

## 3. Chronological investigation history

### Retrospective science,22October (completed before new GPU diagnostics)

- Read the original fitting/scheduling/checkpoint source and located retained
  probes, latest states, natural forks, normalizers and nonwrapped replay prefixes.
- Analyzed all 96 parents over available checks; created per-target/per-seed data,
  cell summaries and plots of normalized error, scores and time-zero contrasts.
- Independently joined canonical durable parent IDs and archived offline W&B IDs
  to the W&B API inventory:96 unique parents of 103 records;7 unrelated records
  excluded by identity, not performance. Retrieved23,800 history rows and4,834
  checks;0 missing checks or score mismatches in the recorded audit.
- Re-read and hash-verified4,834 probe NPZs; examined95,720 curves/score records.
- Quantified pretraining architecture amplification, incomplete fitting convergence,
  replay indexed support versus geometry, moving normalization, initial fitting
  distance and actual LayerNorm/head output parameterization.
- Evaluated return/critic/actor associations; did not infer causality. Completed a
  focused14-paper primary-source review. See section4 for actual findings.
- No new neural fit/GPU job/SAC run was part of these retrospective reports.

### Diagnostic engineering,22October UTC

- 21:57: verified SSH. Obtained/pinned official SimBa and related code.
- 22:09:49: initial diagnostics commit f297dd4. Created fresh Delta evidence root,
  detached checkout; small write/fsync/read/remove test in a new preflight file.
- 22:10:48 receipt:7 real-JAX CPU contracts passed 3.426s.
- 22:16–22:18: predictions/source-binding/AOT continuation additions;7 contracts
  passed 3.128s. Direct-script help/import and real artifact setup succeeded:
  observations(25600,93), actions(25600,39).32 selected critic component schemas
  were checked; this was not32 real GPU fits.
- 22:20:07: submitted normal one-A100/two-hour job 23120674. Receipt construction
  failed afterward due to a duplicate `utc` keyword. Reconciled exact job identity
  via Slurm; no blind retry/duplicate submission.
- 22:26:02: canceled23120674 only while pending; verified elapsed0/noAllocTRES.
- 22:26:07: submitted one-A100/one-hour interactive23120792, sourcebcf6cf3.
- 22:32:38: task-bank commit 5d86e23; tests were initially unrun, later executed.

### 23October UTC

- 00:37:49:4 task-bank tests passed.00:40:09:6-test bank/integration suite passed;
  commit 5974706 pushed to the diagnostic branch.
- Compared scheduling options using read-only previews. Moved pending23120792
  to x8-interactive without resubmission when preview looked earlier. Actual
  estimate contradicted preview; restored x4 eligibility alongside x8 in the same
  job. No concurrent replacement job was created. Further shorter normal/preempt
  previews did not establish a better start; job retained unchanged afterward.
- Local macOS UID501 lookup/DirectoryServices errors prevented ordinary ssh;
  agent socket was unavailable. The existing owner-authenticated SSH master was
  temporarily reused via documented OpenSSH muxv4. No keys/config/credentials
  were changed. A temporary adapter needed to drain output until control EOF;
  that transport bug was corrected. None of this is a scientific probe fix.
- Started an autonomous-continuation helper (AppendixA). Did not implement its
  scheduler controller/worker/dependencies, test it, deploy it, or submit any
  dependent job before the owner requested an audit/handoff.
- At audit around19:05UTC, the master was unavailable and ordinary ssh still
  failed locally. **Current job state and consumption are not verified.**
- Handoff source inspection uncovered the diagnostic optimizer-selection defect
  stated at the top. Verified all 96 archived canonical configs use 1e-4/.01.
  Preserved queued source and original artifacts. This is the newest technical
  finding and the first diagnostic implementation correction to make next.

### Relevant upstream comparisons (reuse; do not repeat)

- Official https://github.com/SonyResearch/simba at
  `7d0358b1f1fa3404fc5ea0ffedf9dc1424b49a45`: residual layers,
  initialization utilities and **generic sac.yaml** were byte-identical.
  Matching generic sac.yaml did not show it was the campaign-selected config;
  this distinction was missed previously and matters for the new defect.
- Encoder input projection, residual ordering and final LayerNorm match inspected
  upstream code. Added encode helpers, metric collection, target-map API spelling,
  normalization checkpoint support and twin-vmap compatibility were distinguished
  from active single-critic math. No original-critic architecture bug demonstrated.
- RunningMeanStd default float64→32 is not an active dtype change: both wrappers
  explicitly pass observation-space dtype. That specific false lead was ruled out.
- SimBaV2 https://github.com/dojeon-ai/SimbaV2 at
  `86899c277cdc697b2b02d827243de1ea93f20a1d` was pinned and inspected at high level;
  its changed methodology is not a drop-in correction.
- BRO https://github.com/naumix/BiggerRegularizedOptimistic at
  `5870a13d4bf4f66eeddfc291a59df6d2009de600` was pinned; detailed audit unfinished.

## 4. Dataset and scientific findings

### Population/provenance

W&B entity/project: `s_kaveti23-seckinger/echocritic_official_exp1`.
https://wandb.ai/s_kaveti23-seckinger/echocritic_official_exp1 .
Canonical inclusion is exact durable parent identity + archived offline W&B ID +
matching env/architecture/seed config. Do not deduplicate by display name or score.
W&B source binding comes from archived queue/metadata, not a SHA in run names.

Recorded counts:96 parents,103 W&B inventory entries,7 excluded;23,800 history
rows;4,834 hash-matched probe artifacts;95,720 target-fit curves. Archived score
CSV comparison max absolute error0; cloud/archive tolerance1e-12 gave0 mismatches.
All 96 have check 35.88 have check 50;8 large seeds7–8 are interrupted.60 parents
were accepted,28 finished training but were validation-blocked,8 interrupted.
Preserve and include the 28; W&B88finished/8crashed is not scientific acceptance.

Check0 was logged after 5k warmup and before critic update. W&B's raw merged step
may show 8000; use `probe/interaction_step`, not a guessed axis. Check0 archives
contain fresh fits only: there is no second independently measured current0 fit.
The source captures the live initial parameter tree; runtime identity testing is
still required. Latest large endpoints may be checks 39–50: do not label them all 500k.

Example canonical W&B IDs (read-only):
- small pen seed 5: `63b389qb`
- small pen seed 7: `515e9nqz`
- large humanoid seed 1: `jiotx3f8`
- large key-turn seed 1: `onrxf1xo`
Use `<project URL>/runs/<id>` or `wandb.Api().run('<entity>/<project>/<id>')` after
authorized authentication. Keep API tokens in the platform secret store, never Git.

### Definitions and aggregation

For each target r: `P=b-MSE_final`, `b=Var(z)`; normalized error `N=MSE_final/b`.
Official loss is **IQM10 of paired FP32 score differences**, not the difference
of two separately aggregated IQMs. For 10 sorted values x, fractional IQM is
`.1*(x[2]+x[7])+.2*sum(x[3:7])`. Across-seed summaries use paired seed-level data;
ten targets are not ten independent SAC seeds. Historical plot bands are seed
IQRs, not a calibrated confidence interval for an intrinsic-capacity effect.

### Fresh deterioration (observational, all 96)

At matched check 35, mean across seed-level IQM10 normalized errors:

| Environment | Small baseline→35 | Medium baseline→35 | Large baseline→35 |
|---|---|---|---|
| dog-run |.00579→.02210|.00974→.04093|.01138→.07451|
| humanoid-run |.00833→.07457|.01235→.12594|.01682→.32652|
| key-turn |.00611→.13098|.00963→.15170|.01217→.29859|
| pen-twirl |.00614→.18565|.01393→.32969|.01642→.51364|

Large humanoid ΔN35=.30970±.02861 seed SD; large pen=.49722±.18758; n8 each.
All 96 worsen at their latest available point, none strictly monotonically.
Variance remains around.5. Fresh initial minibatch errors do not grow similarly:
median check 35/baseline first-loss ratios=.974/.982/.958 small/medium/large.
See local `matched_check35.csv`, `per_seed_check.csv`, `per_target.csv`,
`report_matched35_table.csv` and `plots/fresh_and_trained_error.png`.

Fixed weights still see different `Q_theta0(N_t(s),a)`. Replay pool, RMS, teachers
and minibatches change. This proves moving conditions, not their causal shares.
A deteriorating fresh score suppresses a positive contemporaneous fresh−trained
score gap at fixed trained score; it does not by itself create a larger positive
loss. Time-zero versus contemporaneous reference changes sign in 28/96 latest
contrasts (25/88 final observed endpoints), demonstrating descriptive sensitivity,
not validity of time-zero subtraction. Source: prior report§§3,6,8 and data above.

### Architecture-dependent early amplification (observational)

At baseline, first 100 mean minibatch loss / first minibatch loss is>1 in all 32
medium and 32 large parents, none of 32 small. Cell medians: small .40–.51,
medium 1.27–1.51, large 2.30–3.18. This occurs before SAC updates; it is not evidence
of training-induced parameter damage. At check 35, late minibatch slopes are
negative for every seed in 11/12 cells (medium dog 4/8). These are pre-update
minibatch curves, not intermediate full-pool MSE or a convergence proof.
Source: `review_curve_seed_summaries.csv`, `curve_convergence.csv`,
`report_curve_table.csv`; previous report§4.

Hypotheses: architecture-dependent function-space steps/conditioning, residual
and normalization dynamics, teacher/task variation and finite horizon. Their
quantitative causal contributions are unmeasured. Joint depth+width change
small→medium does not identify each effect. Medium→large still changes teacher
and replay in original runs. No LR/decay sweep is authorized.

### Extreme successful trained pen-twirl fits (observational + algebra)

| Small pen seed, check 50 | Initial trained / fresh minibatch MSE | Final trained / fresh full-pool MSE (target means) |
|---|---|---|
|5|167177.98 / .85185|14.97435 / .14552|
|7|110654.97 /1.36736|14.59003 / .06648|

Reported successful returns≈1831.6(seed 5),2378.3(seed 7). Their trained scores
can be−14.38/−8.69 despite success. Do not mix target mean and target IQM MSE.
For own-offset task `q-[E(q)+z]`, a pure constant translation cancels exactly.
Initial full-pool identity is
`Var(q)+Var(z)-2Cov(q,z)+(E(z))^2`.
Bounded |z|≤1 and archived first minibatches imply sampled
`RMS(q-a)>=sqrt(MSE)-1`:≥407.87 and331.65. This demonstrates large input-dependent
starting discrepancy, not a measured full-pool Q variance or inability to adapt.
First-minibatch versus final-full-pool reduction ratios compare different samples;
do not call them exact percentages of full-pool learning.

Initial/final-error temporal correlations in pen have median≈−.005/−.195/.001 by
small/medium/large, so starting discrepancy alone does not explain residual error.
Small pen seed 7 head bias≈1.393, final-LayerNorm/head affine component≈976.973,
scaled-head norm≈64.726, target mean≈2239.104. A giant head-bias-only mechanism is
contradicted; learned scale is distributed through features/normalization.
Direct FP32 offset-addition rounding bound≈1.22e-4 (squared1.49e-8) cannot explain
MSE9–15; it does not rule out iterative/teacher/TF32 effects.
Sources: `review_initial_distance.csv`, `initial_final_error_associations.csv`,
`checkpoint_output_parameterization.csv`, `target_addition_rounding_bounds.csv`;
prior report§7. No own/shared/incremental controlled fit has been reviewed.

### Replay and normalization (measured inputs, not controlled fitting)

Keyed reconstructed pools have≈4970 unique indices at baseline versus≈24700 at 35.
Measured geometry uses512 distinct indices early/late with the same late RMS.
All 192 sampled full state-action sets have512 distinct rows; repeated Myo
observations can have distinct actions. Later covariance trace is smaller91/96;
median late/early trace:dog.640, humanoid.950, key.848, pen.172. Indexed support is not
geometric spread or teacher complexity. A universal broader-later explanation is
contradicted by these metrics; interpolation-support effects remain unresolved.
Sources: `actual_pool_index_support.csv`, `replay_diversity.csv`, prior report§5.

On identical early key-turn large seed 1 raw observations, actual20k versus500k
normalizers change normalized inputs by RMS.7181; variance ratios.1429–4.7988.
One chronological RMS reconstruction matches archived20k mean/var/count in shape,
dtype and bytes (max error0). Reconstructed5k RMS is retrospective, not a recovered
measurement. Only latest states and 32 scaled-Myo forks remain, not 51 trained
parameter/RMS snapshots per parent. Sources:`normalizer_reconstruction_verified.txt`,
`fixed_input_normalization_diagnostic.txt`, `saved_state_inventory.json`.

### Target construction and validity

Original `_base_targets` uses a new random same-architecture teacher, sine scale 1e5.
A 1e-5 change in teacher output changes sine phase by 1 radian; this is a sensitivity
calculation, not measured phase noise. Similar variance does not imply equal
frequency/geometry/learnability. Historical NPZs do not contain teacher weights,
actual pools/target tensors/minibatches/fitted params. Exact historical target
identity cannot be asserted from agreement in final MSE.

Original paired fits share the base z, pool and minibatches but can have
**different target tensors because offsets are learner-specific**. Same-task
zero/shared-offset controls may instead impose a huge mean-removal burden. They
are not automatically better metrics. Incremental targets `q_start+z` equalize
initial perturbation burden but change the scientific question. Keep named
alternatives, report initial/final full-pool residual decomposition, and ask the
owner before scientific adoption. No controlled causal explanation or validated
scientific fix has been established.

### Critic-to-actor interpretation

Gap/return associations differ by environment/architecture and change after
first differencing. Successful pen policies coexist with poor fitting; trained
large humanoid can fit much better than its fresh reference while returning≈170.
Q/TD-error/gradient/actor churn/KL diagnostics exist, but training gradients are
not probe-fitting gradients. This is association, not a causal propagation test.
Actor remains fixed architecturally. Do not claim critic scaling intrinsically
reduces adaptability from 2k loss alone. Sources:`check35_across_seed_associations.csv`,
`within_seed_associations.csv`, `report_actor_associations.csv`, `metric_coverage.csv`.

### Literature and implementation anchors to reuse

Lyle et al., Understanding Plasticity in Neural Networks(2023),§3.1 andAppA.2:
https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf . It motivates replay/random
teacher/own-mean/2k fitting, not an assumption that this fresh-reference score is
longitudinally stable. Lyle et al., Disentangling the Causes of Plasticity Loss:
https://proceedings.mlr.press/v274/lyle25a.html discusses representation/conditioning
and learned output scale; not proof of this experiment's cause. Official SimBa:
https://arxiv.org/abs/2410.09754 . Prior `literature_review.md` identifies precise
sections for14 primary sources through22October2026. It is a focused, not
exhaustive, review. Avoid reopening it until a measured mechanism requires it.

## 5. Diagnostic implementation interfaces and readiness

### `main/experiments/exp12/diagnostic_artifacts.py`

- `state_dir(checkpoint, kind)`: `kind='early'` selects `fork/state/LATEST`, otherwise
  `state/LATEST`. Fresh parameters are stored separately. LATEST refers to a
  durable snapshot, not an environment reset. Do not use this loader to write state.
- `replay_array(state,name)`: parse uncompressed NPZ ZIP/NPY headers and return
  mode-r NumPy memmap. Compressed replay is rejected; it is not silently copied.
  Typical names `observations`, `actions`; original save trims to filled count.
- `normalizer(state)`: read trusted owner-generated `obs_rms.pkl` only. Pickle is
  not a safe format for arbitrary external uploads.
- `initial_normalizer(observations)`: retrospective RMS update overobs[1:5000]
  with vector shape(1, d), not genuine archived check 0 RMS. Label accordingly.
- `load_critic(checkpoint,kind)`: fresh prefix`params`; early/latest prefix
  `critic.params` in Orbax OCDBT/Zarr. Read only required leaves via TensorStore;
  require finite float32; return nested tree and path/metadata/leaf SHA256 provenance.
- `atomic_json(path,data)`: temp write/fsync/rename, disallow NaN. Original
  complete training-state save logic remains in `state.py` and must not be invoked
  on originals: `commit_state_dir` can prune prior state directories.

Tiny Orbax/tree-hash and readonly replay contracts passed; real selected schema
and setup checks passed. This is not every checkpoint's behavioral restoration.

### `main/experiments/exp12/fixed_task.py`

- `FitState(params,optimizer)` is the continuous scan carry.
- `extended_indices(key,steps,batch_size,pool_size)` preserves frozen first 1k key
  plus second1k fold_in(key,1), then adds1k blocks fold2...7. Minimum2k and multiple1k.
  It is not equal to asking the old function for a single differently-shaped7k draw.
- `fit_segment(network_def,tx,state,obs,act,targets,indices)` uses original scalar
  MSE, `value_and_grad`, AdamW update, and `lax.scan`. No optimizer reset between segments.
- `predictions(...,chunk)` uses original chunked evaluator.
- `pool_metrics(...)` returns MSE, mean_error, centered_error_variance, q_mean, q_variance.
  Identity MSE=variance(residual)+mean(residual)^2 has a CPU test.
- `initial_gradient(...)` measures one minibatch loss/global L2 gradient norm,
  does not update parameters. **First-update function-space changes are not yet
  implemented as a completed diagnostic.**
- `tree_sha256(tree)` hashes path, shape, dtype, bytes.

Seven CPU tests validate small-network operation/continuation. Actual A100
segmented/full equivalence remains unreviewed. Within-allocation Adam continuity
works; cross-allocation optimizer-state checkpoint/resume is **not implemented**.
Saved curves do not constitute a resumable optimizer checkpoint.

### `main/experiments/exp12/probe_tasks.py`

`FrozenTask(observations,actions,targets,minibatch_indices,provenance)` requires
finite float32 input/target tensors, int32 in-range2D indices. Arrays are copied
onto immutable bytes-backed storage; provenance is canonical JSON. `task_id`
hashes dtype/shape/bytes and provenance. `save(new_directory)` refuses overwrite,
fsyncs numeric NPZ/manifest; `load(directory)` verifies hashes and identity.
Same tasks with different source/recipe provenance can have different IDs;
compare declared metadata as well as numeric content when interpreting reuse.

`fixed_task_paired_summary(fresh_mse, current_mse, variance, fresh_task_ids,
current_task_ids, fresh_steps, current_steps)` requires10 distinct matching task IDs,
finite values, matched horizons, and computes IQM of paired FP32 scores. A3-target
screen is intentionally rejected as IQM10. Four API contracts and two integration
contracts passed, including bitwise same CPU curve and final carry hash after reload.

### `main/scripts/exp96_fixed_task_diagnostics.py`

`Investigation(args)` expects root(new evidence), campaign(frozen root), commit(exact
checkout), cap_seconds. It reads `campaign/queue/queue.json`, requires frozen source
and96 canonical parent keys, caches host trees/compiled fitting executables, and
currently constructs `self.tx` from the **wrong generic config** (fix first).

- `model(env,arch,seed,kind)` reads/checksums the specified actual critic and records
  synchronized host→device time.
- `pool(env,seed,check=50,period='late',norm='latest',support=None)` uses the selected
  **large-parent** replay for controlled architecture comparisons. Early prefix5k,
  late500k, length25600. Explicit5000 support is sampled without replacement then
  resampled; it is not perfectly nested in the larger-support pool. Use only
  parents with required replay length; check actual buffer_meta rather than padding.
- `original_reproduction(env,arch,seed,round_index=0)` reconstructs original check 50
  keyed pool/base/indices for small pen seed 5 and large humanoid1, compares archived
  finalMSE/offset/variance and records full-curve difference. Saves numeric task,
  curves and segmented equivalence. Repeat of identical original fresh fit must
  have identical loss array. Current archive guards: variance error≤.002,
  offset error≤.05, final error≤max(.02,10%|reference|); segmented final error
  ≤max(1e-4,.5%|original|). These are engineering review guards, not proof of exact
  historical target identity; full-curve acceptance still requires justified review.
- `fit(name,env,arch,seed,kind,obs,act,z,idx,offset_mode,steps)` supports own mean,
  zero, and incremental q_start+z targets in code. Records full-pool metrics at 0
  and each1k, initial gradient, complete pre-update minibatch curves, optimizer count.
  Shared-offset named sensitivity is not a complete ready-made experiment yet.
- `bank_task(...)` at 5974706 writes and reloads `FrozenTask`, records task_id.
  Learner name is excluded so paired numeric tasks retain identity.
- `run()` is a fixed screen, **not a fully adaptive controller**: original small pen,
  original large hum if time; then3 target rounds over 2 Myo envs×seeds1/2; conditions
  late/latest/full, late/latest/5k-support, early/latest/5k, early/early/5k;
  input-only normalizer control; fresh/early/latest common zero target; latest-own;
  small/medium fresh on the same large-teacher task. Selected seed 1 fits8k, others2k.
  It skips conditions when cap estimates are insufficient. It does not implement
  every requested pen seed 5/7 offset or first-function-step control.

Entry arguments:
`--root PATH --campaign PATH --commit FULL_SHA --cap-seconds SECONDS`.
It requires Slurm, source/clean-tree checks, GPU backend, exactlyone visibleA100,
and TF32 policy. `sync()` blocks JAX arrays before timing. Original compile and
execution times are recorded separately; partial/first-use segment timings must
not be represented as pure steady-state throughput.

Saved outputs: `runtime.json`, `results.json`(rows, timings, inputs, memory stats),
`status.json`, `telemetry.jsonl`, `pool_*.npz`, `original_<run_key>_r0/task.npz`,
`fresh_original_fit.npz`, `latest_original_fit.npz`, `segmented_equivalence.json`,
and per-fit `task.npz`, `curve_through_*.npz`, `progress.json`, plus at 5974706
`numeric_task/arrays.npz` and `manifest.json`.

`status=COMPLETED_BUDGETED_STAGE` means the bounded program returned, not that all
conditions finished or scientific qualification passed. `FAILED` preserves an
exception; process kill may leave no final status but prior durable results.
Do not call missing/partial conditions PASS. Instrumentation includes own-process
NVML memory and allocator stats; shared utilization is not exclusive attribution.

### Autonomous continuation readiness

Only a local untracked186-line draft exists (AppendixA). It has helper functions
for terminal accounting, budget arithmetic, first-result checks and CSV/report/ZIP
writing. It is **not** a scheduler workflow. No controller entry point, dependency
submissions, atomic spending reservation, deduplication/resume, worker, wrapper,
tests or deployment exists. Its prospective gate would also need updating for the
optimizer mismatch. Do not infer automation from its module docstring.

## 6. Delta environment, access and operational commands

### Verified paths and runtime (historical unless live rechecked)

- Login: `skaveti1@login.delta.ncsa.illinois.edu`; account`biqc-delta-gpu`.
- Durable original root:
  `/projects/biqc/skaveti1/exp96/campaign_owner_f597ee9_20261019T230748Z`.
  Its queue is `campaign/queue/queue.json`; checkpoint hierarchy
  `campaign/checkpoints/exp1/<architecture>/<environment>/seed_<n>/`.
- Original source checkout:
  `/projects/biqc/skaveti1/exp96/checkouts/campaign_owner_f597ee9_20261019T230748Z`.
- New investigation root:
  `/projects/biqc/skaveti1/exp96/rootcause_20261022T220000Z`.
- Detached sources beneath it: `checkout`(2c9f280),
  `checkout_debug_bcf6cf3`(bcf6cf3), `checkout_taskbank_5d86e23`(5d86e23),
  `checkout_bank_5974706`(5974706). Do not change an active checkout.
- Python: `/u/skaveti1/.conda/envs/scaling-drl-py31213/bin/python`.
  CPU receipts used Python 3.12.13, JAX/JAXLIB 0.4.34, Flax 0.8.4, Optax 0.2.3,
  NumPy 1.26.4, Orbax 0.5.3; consult actual runtime for CUDA plugins/driver/GPU kind.
- Wrapper initializes `/sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh`, activates
  `scaling-drl-py31213`. Keep shared environment unchanged.
- Source`main/requirements.txt` pins the Python stack but includes Mac-only jax-metal
  via platform marker and does not itself prove the CUDA plugin/driver setup.

Paths `/work/hdd/...` in older messages refer to earlier storage investigations.
The current diagnostic evidence is on `/projects/...`. Do not silently move evidence
to temporary/local storage. Earlier quota timeouts did not prove HDD full. A dated
/projects preflight reported347G free and642459 free inodes; recheck capacity
boundedly only if new output size requires it. Tiny writes alone do not prove quota.

### Jobs and status timeline

| Job | Purpose | Source | Latest verified information |
|---|---|---|---|
|23120674|Initial normal diagnostic,1 GPU/16 CPU/60 GiB/2h|2c9f280|Cancelled while pending22Oct22:26:02UTC; elapsed0/noAllocTRES|
|23120792|Current single-A100 diagnostic,16 CPU/60 GiB/1h|bcf6cf3|Last successful session inspection around23Oct03:18UTC:pending (Priority), elapsed0, noAllocTRES; current status unknown|
|22952754|Original engineering smoke|be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574|Historical completed21:31, exit0; not a new investigation run|
|23023494|Historical official campaign allocation|f597 lineage|Owner requested cancellation; reconcile original records if needed, do not revive it|

The saved03:04:41UTC status predicted23Oct18:20UTC for23120792. That is an expired
prediction, not evidence of start/completion. During handoff, live access failed
because the previous local mux socket was absent. A normal ssh attempt in the
previous audit failed before connection due local UID501 lookup. Do not claim
current pending/zero consumption from this history.

23120792 is eligible for
`gpuA100x4-interactive,gpuA100x8-interactive`, still requests **one** GPU/one node,
not8 GPUs. Slurm stores its original submission line with x4 even after eligibility
updates. Inspect current `Partition`, `ReqTRES`, `AllocTRES`, limits, state and
`Requeue` rather than inferring from name or old SubmitLine.

Evidence paths for23120792:
```
/projects/biqc/skaveti1/exp96/rootcause_20261022T220000Z/gpu_debug1/
/projects/biqc/skaveti1/exp96/rootcause_20261022T220000Z/slurm/debug_23120792.out
/projects/biqc/skaveti1/exp96/rootcause_20261022T220000Z/slurm/debug_23120792.err
```
The queued script uses only Delta paths, Slurm environment and shared conda.
**It runs without MacBook/Codex/SSH session after scheduling.** No continuation
or external results collector was submitted. Latest5974706 is not queued.

### Access from the owner's Mac: exact process used

Use this only when the agent's command execution is on the same Mac and can
access the owner's Unix control socket. The owner performs password/Duo prompts
in **their Terminal**, never in chat or an agent-stored script.

First owner/agent check:
```bash
ssh -O check -o ControlPath="$HOME/.ssh/codex-delta-retry/%C" \
  skaveti1@login.delta.ncsa.illinois.edu
ssh -o BatchMode=yes -o ControlPath="$HOME/.ssh/codex-delta-retry/%C" \
  -o ConnectTimeout=12 skaveti1@login.delta.ncsa.illinois.edu \
  'hostname; id -un; date -u'
```
The alive check proves a master process, not a functioning remote session; require
the second command to succeed before claiming access.

If a shared connection must be established, the owner runs:
```bash
mkdir -p -m 700 "$HOME/.ssh/codex-delta-retry"
ssh -MNf -o ControlMaster=yes -o ControlPersist=8h \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ControlPath="$HOME/.ssh/codex-delta-retry/%C" \
  skaveti1@login.delta.ncsa.illinois.edu
ssh -o BatchMode=yes -o ControlPath="$HOME/.ssh/codex-delta-retry/%C" \
  -o ConnectTimeout=12 skaveti1@login.delta.ncsa.illinois.edu \
  'hostname; id -un; date -u'
```
Enter password/Duo only in Terminal.8h is an idle-persistence setting, not a promise
that server/network policy keeps access alive8h. If a stale control socket prevents
binding, use a fresh owner-only directory/path and report that path to the agent;
do not delete keys or kill another active master. Existing connection reuse needs
no new password, Duo code or agent key lookup.

The old Mac process sometimes failed `getpwuid(501)` before DNS/network/auth,
with DirectoryServices eServerError. Key/config metadata were readable; ssh-agent
socket refused connection. This was not proof of expired Delta auth. A temporary
owner-local `/private/tmp/exp96_mux_transport.py` used standard OpenSSH muxv4
passenger sessions to reuse a still-authenticated master; it was not part of Git
and is not a Cloud access mechanism. It must wait for EOF after exit status to
avoid dropping output. Do not commit or copy an actual control socket, agent socket,
keychain material or helper hardcoded personal socket basename. Do not assume
restarting Codex fixes OS lookup; diagnose where failure occurs first.

### Access from Codex Cloud: what owner authentication can and cannot do

**A Mac-local Unix socket and ssh-agent do not travel with a Git clone.** The
owner saying “authenticated” on their Mac does not authenticate an unrelated
Cloud VM. Establish an approved reachable execution/relay environment first.
Do not ask for a password/Duo code/private key in chat, install a backdoor, change
Delta's MFA policy, expose the control socket publicly, or disable host-key checks.

Use one of these explicit, owner-approved routes:

1. **Run Delta operations in the owner's existing Mac/local Codex executor.**
   Cloud develops and pushes the source; that executor uses the Terminal process
   above, stages an immutable checkout on Delta, and runs the specified bounded
   diagnostics. Return receipts/artifacts through approved project storage or a
   sanitized handoff. This is the already-used route and needs no Cloud SSH secrets.
2. **Use an approved persistent gateway/remote executor accessible to Cloud.**
   The owner establishes SSH+MFA in an interactive shell on that same executor,
   leaving an owner-protected multiplex master there. Cloud must actually be able
   to execute on that gateway under the authorized account; setting ControlPath in
   a different VM is insufficient. Use its local `$HOME/.ssh/...` path, verify
   `hostname;id -un;date -u`, then run operations. Obtain the endpoint/access method
   from the owner; none was provisioned/verified in this investigation.
3. **Direct Cloud SSH, only if Delta and the Cloud execution policy support the
   owner's authorized authentication method.** The owner performs any required
   interactive authentication in that environment's secure Terminal; an agent can
   then reuse that environment's master. Confirm outbound DNS/TCP22, supported
   MFA/interactive capability and policy first. This route was not tested. A
   noninteractive Cloud environment without a gateway or permitted interactive
   authentication is a genuine access blocker, not a reason to transfer passwords.

Start in a new environment with these bounded non-secret checks, then choose the
applicable route:
```bash
hostname
id -un
ssh -V
ssh -o BatchMode=yes -o ConnectTimeout=12 \
  skaveti1@login.delta.ncsa.illinois.edu 'hostname; id -un; date -u'
```
If no authenticated route exists, continue the config-selection correction and
CPU regression locally/Cloud, push the successor and provide exact Delta commands
to the owner's executor. No safe relay/connection secret is embedded here. Do not
claim remote access, running jobs or current spending based on cached reports.

### Read-only Delta check commands after access works

Run on Delta (UTC matters; Delta defaults Central):
```bash
TZ=UTC squeue --start -j 23120792
TZ=UTC scontrol show job 23120792
TZ=UTC sacct -D -j 23120674,23120792 \
  --format=JobID,JobName,State,ExitCode,Start,End,ElapsedRaw,TimelimitRaw,AllocTRES,Partition -P
TZ=UTC squeue -u skaveti1
```
Do not count `.batch`, `.extern` and `.0` steps as additional GPU allocations.
A job absent from squeue may be complete: inspect accounting, not queue position.

Read job evidence without requiring the Mac:
```bash
R=/projects/biqc/skaveti1/exp96/rootcause_20261022T220000Z
for f in "$R/gpu_debug1/status.json" "$R/gpu_debug1/runtime.json"; do
  if test -f "$f"; then cat "$f"; fi
done
if test -f "$R/slurm/debug_23120792.out"; then tail -n 100 "$R/slurm/debug_23120792.out"; fi
if test -f "$R/slurm/debug_23120792.err"; then tail -n 100 "$R/slurm/debug_23120792.err"; fi
```
Confirm recorded optimizer file hash/values. If wrong-config fits exist, preserve
and label them; do not certify them as reconstruction or use them to infer the
original scientific mechanism. Inspect actual stored numerical tasks before
reusing anything. Copy only selected artifacts for analysis, not whole replay by
default. Use `ssh ... 'tar -C VERIFIED_ROOT -czf - SELECTED_RELATIVE_PATHS' > new.tgz`
with actual verified roots/paths, or approved scp/SFTP; do not overwrite official data.

### GPU-hour and SU accounting

Budget file: `.../rootcause_20261022T220000Z/budget.json`, plus submission/partition
receipts. Last verified actual usage was0; **current usage is unknown**. The only
possibly executing new allocation is23120792 at 1 GPU×1h requested. Reconcile it,
including failed/timeout/idle/compile time. Compute physical GPU seconds as the
sum of terminal allocation elapsed seconds×actual GPU count. Inspect duplicate
accounting rows/requeues rather than silently dropping them.

Historical live weights were MAX_TRES-style:
- x4normal:CPU62.5, mem16/GiB, GPU1000 ⇒16 CPU/60 GiB/1 GPU billing1000 ⇒1weightedSU/h.
- x4interactive:CPU125, mem32/GiB, GPU2000 ⇒billing2000 ⇒2weightedSU/h.
- x8normal:CPU93.75, mem6/GiB, GPU1500.
- x8interactive:CPU187.5, mem12/GiB, GPU3000 ⇒16 CPU/60 GiB/1 GPU billing3000 ⇒3weightedSU/h.
Re-read applicable weights and actual AllocTRES when needed; pending ReqTRES may
retain an earlier billing value after a partition update. Weighted usage estimate
is `ElapsedRaw/3600 * AllocTRES.billing/1000`; this is not physical GPU-hours or a
posted ACCESS transaction. Historical account `accounts` output:204 rounded hours
balance,1203 deposited, not an exact fresh balance. Original smoke22952754 used64 GiB
so its billing1024 differed from a60 GiB request; it is outside this investigation.

Before a new allocation reserve its worst-case count×walltime plus all existing
actual/reserved usage≤14400 GPU-seconds. OneGPU is sufficient. Keep a teardown
margin; no automatic retries or overlapping jobs. Do not treat a field in JSON as
enforcement: implement locked reservation, submission intent/reconciliation,
non-requeueing hard limits, dependencies and post-run accounting. Unsettled
accounting, missing artifacts or ambiguous submission means stop admission.

No new GPU job is submitted by this document. If a later reviewed successor is
submitted, create a new detached clean checkout and fresh evidence root, use the
pinned environment/source guards, one GPU and a cap consistent with Slurm. Example
argument shape (NOT an instruction to submit before fixing the config and reconciling):
```bash
# From NEW_CLEAN_CHECKOUT/main, only after valid admission:
# EXPECTED_COMMIT=FULL_REVIEWED_SHA
# DIAGNOSTIC_ROOT=NEW_DURABLE_OUTPUT
# FROZEN_CAMPAIGN=/projects/biqc/skaveti1/exp96/campaign_owner_f597ee9_20261019T230748Z
# DIAGNOSTIC_CAP_SECONDS=3000
# Export those four variables, then use sbatch --parsable --no-requeue with
# --account=biqc-delta-gpu --nodes=1 --ntasks=1 --gres=gpu:1
# --cpus-per-task=16 --mem=60G --time=01:00:00 and verified eligible partition.
# Script: scripts/sbatch_exp96_fixed_task_diagnostics.sh
```

## 7. Completed tests and how to reproduce targeted validation

There are **13 distinct test methods**, not17: the later6-test suite reran4 task
contracts and added2 integrations. No test below is a scientific seed replication.

| Test method | Fixture/assertion | Result/limit |
|---|---|---|
|`test_frozen_indices_prefix`|Exact first 2k draws match original two-key construction|CPU passed; not teacher identity|
|`test_aot_signature_and_original_repeat`|AOT original fit equals normal JIT invocation|CPU arrays equal|
|`test_original_segmented_and_identity`|2×1k continuous scan versus original2k; curve/final allclose rtol2e-5/atol2e-6; same-repeat hash; unchanged starting tree; Adam count 2000; compiled continuation|CPU passed on Tiny network|
|`test_pool_error_decomposition`|MSE=mean_error²+centered_variance|CPU passed|
|`test_own_offset_translation_cancels`|q and q+100 with corresponding own means have same residual|CPU passed; not network training translation invariance|
|`test_continuation_keeps_adam_count`|8×1k endsAdam count 8000|CPU passed; no cross-job state checkpoint|
|`test_partial_checkpoint_and_readonly_replay`|Tiny Orbax restored tree hash; NPZ memmap values exact/read-only|CPU passed; not full simulator restore|
|`test_roundtrip_immutable_and_no_overwrite`|Task ID survives storage; arrays cannot become writable; refuses overwrite|CPU passed|
|`test_corruption_is_rejected`|Changed provenance invalidates manifest|CPU passed|
|`test_paired_iqm_is_not_difference_of_iqms`|Paired arithmetic differs from separately aggregated marginal IQMs|CPU passed|
|`test_mismatched_tasks_horizons_or_target_count_are_rejected`|Wrong task order, horizons,3-targetscreen rejected|CPU passed|
|`test_runner_bank_preserves_fitting_and_pair_identity`|Runner saves/reloads task; same ID across learner directories; identical CPU curve and final parameter+optimizer hash|CPU passed|
|`test_numeric_payload_corruption_is_rejected`|Changed stored target invalidates checksum|CPU passed|

Fixture:32 samples,3obs/2actions, Flax Tiny Dense8+tanh+Dense1, AdamW1e-4/.01,
fixedkeys and8-example minibatches. Receipts:
- f297dd4:7 tests3.426s,22Oct22:10:48UTC.
- 2c9f280:7 tests3.128s,22Oct22:18:25UTC.
- 5d86e23:4 tests.014s,23Oct00:37:49UTC.
- 5974706:6 tests1.878s,23Oct00:40:09UTC.

Additional engineering: direct-script import/help, actual runner pool/one fresh
checkpoint setup and 32 selected schema checks; bash-n and git diff--check.
The runtime pool test did not inspect/catch optimizer config values. This is why
a **new targeted config regression is essential**, not a repeat broad suite.

Run only relevant tests for changed code. In a fresh Cloud environment, reproducing
these13 once to establish environment compatibility is reasonable; if receipts
already apply and code/environment unchanged, reuse them.
```bash
cd /absolute/path/to/scaling-drl-research/main
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=false
export PYTHONDONTWRITEBYTECODE=1 EXP12_JAX_CACHE_DIR=off WANDB_MODE=disabled
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable
/absolute/path/to/pinned/python -B -m unittest \
  tests.test_fixed_task tests.test_probe_tasks tests.test_fixed_task_bank_integration -v
```
Use a new outside-Git output directory and record source/runtime/stdout/stderr/exit.
The hardware helper's rocm-smi warning while CPU is explicitly selected is not a
GPU test. Do not count zero discovered tests as coverage. No A100 gradient/function
step, memory peak, convergence replication or scientific null calibration is proved.

## 8. Artifact inventory and accessibility

### GitHub-accessible

The diagnostic code/tests/docs and this handoff are in the branch. Original source
is reachable by immutable SHA. AppendixA preserves the uncommitted automation draft
as text. Prior reports/raw data, active SSH helpers/sockets and GPU evidence are
**not** in Git. Do not assume absolute Mac paths exist in Cloud.

### Mac-only retrospective evidence

1. `/private/tmp/exp96_fresh_reference_20261022/`: first full drift/timezero report,
   source_provenance.json, raw_csv_validation.json, saved_state_inventory.json,
   per_target.csv, per_seed_check.csv, per_seed_latest.csv, cell_check_statistics.csv,
   cell_latest_statistics.csv, analysis.json;12cell plotsets; fresh_normalized_error.png,
   fresh_normalized_error_by_seed.png, fresh_mse_and_variance.png, score_changes.png,
   normalized_error_changes.png; scripts analyze.py, plot.py, report.py;
   `exp96_fresh_reference_analysis.zip`≈32.75MB. This directory is temporary: ask
   owner to retain/export it through approved storage if needed.
2. `/private/tmp/exp96_deep_mechanisms_20261022/`: full second analysis. Key inventory:
   - `report.md` (49388 bytes)
   - `README.md` (1517 bytes)
   - `checks.json` (951 bytes)
   - `raw_csv_validation.json` (133 bytes)
   - `source_provenance.json` (612 bytes)
   - `canonical_parents.csv` (36200 bytes)
   - `excluded_wandb_records.csv` (1035 bytes)
   - `wandb_inventory.json` (877955 bytes)
   - `wandb_archive_reconciliation.csv` (8344 bytes)
   - `canonical_config_verification.csv` (7027 bytes)
   - `artifact_manifest.csv` (1308034 bytes)
   - `per_target.csv` (22183200 bytes)
   - `per_seed_check.csv` (4729520 bytes)
   - `per_seed_latest.csv` (36484 bytes)
   - `matched_check35.csv` (107753 bytes)
   - `cell_check_statistics.csv` (5621443 bytes)
   - `aggregate_check_statistics.csv` (5543945 bytes)
   - `curve_convergence.csv` (28290677 bytes)
   - `review_curve_seed_summaries.csv` (72183 bytes)
   - `review_initial_distance.csv` (101700 bytes)
   - `replay_diversity.csv` (104979 bytes)
   - `actual_pool_index_support.csv` (119752 bytes)
   - `checkpoint_output_parameterization.csv` (16242 bytes)
   - `initial_final_error_associations.csv` (13273 bytes)
   - `target_addition_rounding_bounds.csv` (238 bytes)
   - `normalizer_reconstruction_verified.txt` (1067 bytes)
   - `fixed_input_normalization_diagnostic.txt` (621 bytes)
   - `saved_state_inventory.json` (44275 bytes)
   - `wandb_training_history.csv` (12361321 bytes)
   - `metric_coverage.csv` (1650 bytes)
   - `within_seed_associations.csv` (80569 bytes)
   - `check35_across_seed_associations.csv` (8227 bytes)
   - `report_actor_associations.csv` (736 bytes)
   - `evidence_table.csv` (3136 bytes)
   - `hypotheses.csv` (4027 bytes)
   - `proposed_controls.csv` (4597 bytes)
   - `source_excerpts.md` (12541 bytes)
   - `literature_review.md` (9265 bytes)
   - `data_dictionary.json` (203707 bytes)
   - `SHA256SUMS.txt` (9798 bytes)
   - `ChatGPT_review_package.zip` (2270080 bytes)
   - `Exp96_full_analysis.zip` (82921980 bytes)
3. `/private/tmp/echocritic_rootcause_20261022/`: new engineering evidence.
   `WORKLOG.md`, stage_cpu.log, final_cpu.log, taskbank_cpu.log, bank_integration_cpu.log;
   submission.log, debug_submission.log; source staging/submission helper scripts and
   Git bundles; simba_file_comparison.json and7saved diffs; references/{simba, simba-v2, bro};
   continuation_20261023.md, acceleration_blocker_20261023.md,
   ssh_diagnosis_20261023.md, execution_blocker_20261023T0307.log.
   These are not a final scientific correction package. Do not commit private
   authentication artifacts or assume /private/tmp is durable storage.

The two earlier archives contain retrospective findings only:
`ChatGPT_review_package.zip`≈2.27MB and`Exp96_full_analysis.zip`≈82.92MB in the
second directory. They are not proof of new GPU controls or a fix.

### Delta-only inputs and outputs

Original checkpoint example:
```
/projects/biqc/skaveti1/exp96/campaign_owner_f597ee9_20261019T230748Z/campaign/checkpoints/exp1/D4W1536/myo-key-turn/seed_1/
```
Subpaths: `fresh_critic/`, `probes/check_00.npz`…`check_50.npz`,
`probes/probe_checks.csv`, `state/LATEST`, `fork/state/LATEST` when retained,
`run_metadata.json`, natural-fork metadata and scientific ledger references.
A snapshot holds `agent_ckpt/`(Orbax),`obs_rms.pkl`,`buffer.npz`,`buffer_meta.pkl`,
`meta.pkl` and simulator/RNG state via original state/trainer implementation.
Latest known large key-turn seed 1 fork was20k; latest example
`state/step_000500000_95dd2783/agent_ckpt` and fork
`fork/state/step_000020000_015805f9/agent_ckpt`. Resolve LATEST, do not hardcode a
snapshot when current provenance must be checked. Most35k/check 35 trained states
are not retained; do not invent them. Do not execute pruning save routines on originals.

Diagnostic root contains `cpu/`receipts/logs, Slurm logs, source bundles/checkouts,
`submission_intent.json`,`submission_reconciled.json`,
`pending_cancellation_23120674.json`,`submission_debug_raw.json`,
`submission_debug.json`,`budget.json`, partition-change/union receipts and dated
status/test-only preview receipts. Existing GPU output is `gpu_debug1` if created.
CPU receipts include `final_receipt.json`, `real_artifact_schema.json`,
`taskbank_5d86e23_receipt.json`, `bank_integration_receipt.json`.
These were read previously; current availability must be reverified after access.

Historical smoke evidence, separate from new budget:
`/projects/biqc/skaveti1/exp96/smoke_be1ffae_20261017T141608Z/`.
Historical launchprep audit:
`/projects/biqc/skaveti1/exp96/launchprep_be1ffae_20261018T153904Z/qualification_audit_20261018.md`.
Do not use its older campaign source/config snapshots as current diagnostic truth.

### Recover evidence without Git bloat

Read canonical configs from checkpoint metadata on Delta; if inaccessible, the
owner can export the selected96-row config/identity tables and compact historical
review ZIP via an approved file channel. Full replay access requires Delta or
an explicitly staged dataset, not W&B alone. The new task bank is designed so a
small selected numeric pool can be exported instead of a whole replay buffer.
Check original artifact SHA256 against `artifact_manifest.csv` before analysis.
No Mac-only raw dataset is silently made Cloud-accessible by this handoff.

## 9. Outstanding engineering tasks, in dependency order

### T1. Reconcile existing execution and preserve evidence — essential

Use job 23120792, its submission receipts, `budget.json`, and the read-only commands
in section 6. Record actual Slurm state, allocated GPU seconds, source/configuration,
logs, and every complete or partial result. If access fails, report the precise
access layer and proceed with T2 CPU work. Do not assume the job is still pending.
Acceptance: an accounting-backed inventory, or an explicit unresolved access limit.
This inspection takes minutes once access works and requires no new GPU allocation.

### T2. Correct diagnostic optimizer provenance — first code task

Reuse `RunProbes.__init__`, `critic_optimizer`, and each selected checkpoint's
`run_metadata.json`. Modify `main/scripts/exp96_fixed_task_diagnostics.py`, optionally
adding a helper in `main/experiments/exp12/diagnostic_artifacts.py`. Add a focused
regression in `main/tests/test_fixed_task.py` or a new dedicated test module.

1. Load the original parent's `resolved_config.agent` rather than generic sac.yaml.
2. Validate source/configuration bindings and agreement among selected parents.
3. Bind the numerical optimizer settings and metadata hash to each diagnostic result.
4. Reuse an equivalent optimizer transformation to avoid unnecessary recompilation.
5. Reject missing or conflicting configuration; never silently fall back to defaults.

Acceptance: a constructor-level fixture selects LR=1e-4 and decay=.01 despite a
conflicting generic 3e-4/0 configuration; missing/mismatched metadata is rejected;
selected actual archived configurations agree. Restore the frozen AdamW defaults,
without tuning. Run the new focused test module and retain stdout/exit status.
Existing Tiny equivalence tests need repeating only if their numerical code changes.
Planning estimate: .5–2 engineering hours, CPU only. Verify actual metadata on Delta
when available. Commit independently and preserve all wrong-config outputs.

### T3. Complete narrow source-bound GPU reproduction — essential

After T1/T2, use a new clean detached successor checkout and fresh durable output.
Reuse archived small pen-twirl seed 5 and large humanoid seed 1, their keyed target
and index construction, correct optimizer and precision. Review full curves as well
as final errors; preserve uncertainty about unavailable historical target tensors.
Do not weaken tolerances to make reconstruction pass. Measure synchronized runtime
before choosing subsequent scope. A .14–.30 A100-hour estimate for selected checks
is a planning proxy, not a measured benchmark. The existing request reserves 1 hour;
reconcile its actual use before admitting another allocation.

### T4. Add missing high-information measurements and conditions — essential

Reuse `fixed_task.py`, `FrozenTask`, and `Investigation.fit` after T2. Add a
single-update measurement of prediction-change RMS/mean and full-pool loss on the
same inputs before/after the update. Comparing unrelated minibatches does not
measure a function-space step. Initial gradients and raw fitting curves already exist.

Add explicit own-offset, shared-offset, and incremental-target contrasts for
successful small pen-twirl seeds 5/7, using the same underlying base target and
minibatch stream. Save the starting prediction tensor for incremental targets and
label that altered scientific question. Use a Tiny CPU one-step reference for new
code. Acceptance: starting parameters unchanged, exact task identities, correct
predicted changes, and no replacement of the historical metric. GPU cost follows
measured T3 throughput; no optimizer or learning-rate sweep is authorized.

### T5. Bounded autonomous continuation — only when needed for unattended stages

Treat Appendix A as an untested draft, not a deployed workflow. Add a CPU controller,
worker entry point under `main/scripts/`, and focused scheduler/budget tests.
Use `afterany` for collecting the first job's outcome; admit a GPU successor only
after artifact/configuration validation. Dependencies alone cannot certify evidence.

Required controls: locked cumulative GPU-second ledger; reserve worst-case allocation
before submission; no requeue; durable submission intent and raw receipt; reconcile
ambiguous submissions before retrying; prevent overlap; skip completed conditions
by starting-parameter/task/optimizer/precision/horizon identity; preserve intermediate
results; generate a CPU report on failures. Missing accounting or reconstruction
mismatch blocks admission. Improve reviewed full-curve acceptance rather than
relying only on final MSE/variance/offset. Test with fabricated Slurm/NPZ fixtures
before any real submission. Automation may execute authorized controls, not approve
new scientific metric definitions. A prospective 1-hour child plus the original
1-hour request would reserve at most 2 hours before other usage; no such child is
currently submitted. Planning estimate: 1–3 engineering hours, CPU validation only.

### T6. Cross-allocation fitting recovery — conditional

Current segments retain Adam state within one process, not across jobs. If an 8k
fit cannot finish within the remaining cap, explicitly skip it or implement a new
atomic diagnostic checkpoint containing parameters, Adam state/count, task/index
prefix, source/configuration/precision IDs and horizon. Test restored next-step
identity on Tiny, then on a necessary real case. Restarting from 2k with a fresh
optimizer is not continuous 8k fitting. Never invoke pruning saves on official data.
This is unnecessary when the whole fit safely fits one allocation.

### T7. Analysis, replication, correction and performance

Select the strongest two or three mechanisms after T3/T4; avoid a full matrix.
Emit per-target/per-seed CSVs, complete curves, initial/final full-pool changes,
uncertainty across actual seeds, and evidence classifications. Replicate principal
contrasts on at least one additional seed and, where appropriate, another environment.
Implement minimal demonstrated bug fixes or owner-approved named measurement changes
in separate commits. Preserve the historical detector and original SAC settings.

Existing scans, read-only mmap, partial loading and compiled-executor reuse are not
measured speedups. Time compilation, synchronized execution, transfer, serialization
and memory separately. Adopt further optimizations only after numerical and timing
comparisons. Do not launch SAC training just to benchmark it; use existing timing
logs until separately authorized.

## 10. Adaptive scientific execution sequence

Before each condition, save its suspected mechanism, controlled intervention,
predictions under competing explanations, meaningful effect criterion and next action.
Adapt after inspecting numerical evidence, rather than executing every planned fit.

| Stage | Existing implementation | Missing work / advancement condition |
|---|---|---|
| 1. Reconcile 23120792 | Commands, paths and receipts | Live access/accounting; wrong-config results cannot qualify reconstruction |
| 2. Original reconstruction | `original_reproduction`, original `_base_targets`/`_fit` | T2 optimizer correction; review actual curves and runtime; localize mismatches before scientific interpretation |
| 3. Architecture contrast | Small/medium/large fresh critics on a common large-parent pool/teacher task | First-update function-space metrics; identical numerical labels, not merely equal RNG seeds |
| 4. Fresh/early/latest | Common zero-offset tasks and continuous fitting segments | Review starting-function burden; exact bank identities; reuse completed conditions |
| 5. Pen target mismatch | Small-pen reconstruction; own/zero/incremental modes | Explicit shared-offset and seed-7 controls, residual decomposition and prediction changes |
| 6. Replay/normalization | Four pool conditions and input-only normalizer crossover | Indexed supports are not fully nested; add stricter crossovers only if prior results justify them |
| 7. Extended fitting | Continuous 2k/4k/8k segments | Runtime/budget admission; retain original 2k metric; add cross-job state only if necessary |
| 8. Replication | Planned two Myo seeds and three target rounds | Actual independent outcomes; three targets do not validate IQM10 |
| 9. Correction | Diagnostic config fix is justified; no validated scientific fix | Controlled predicted effect, replication, regression tests and owner approval for scientific redefinition |

Predictions and interpretation boundaries:

- Identical executable, numerical task and starting state should repeat. Segmented
  fitting should agree with the original loop under justified recorded tolerances.
  Failure is reconstruction evidence, not evidence of scientific plasticity loss.
- Large-network amplification on the same numerical task would exclude teacher or
  replay differences as its sole explanation. Gradient norm alone is not a
  function-space learning rate; measure actual prediction changes.
- Good incremental-function fitting despite huge own-offset MSE would falsify a
  broad claim of inability to learn new perturbations. It would not automatically
  validate a new detector or eliminate specialization/conditioning effects.
- Catch-up by 8k would show that a 2k gap does not establish inability to fit.
  Persistence at 8k would still not establish a global capacity limit.
- Restricted support under the same replay/teacher/normalizer tests conditional
  interpolation burden. Equal support across early/late distributions is a separate
  contrast; do not equate unique indices with geometric diversity.
- Fixed raw inputs, alternate normalizers and fixed labels isolate learner-coordinate
  effects. Regenerating labels adds the teacher path. Do not force additive causal shares.
- Frequency/precision ablations are justified only when earlier results point there.
  Preserve teacher weights and record actual outputs, targets and variances.

A meaningful effect must be comparable to the anomaly, exceed reconstruction noise,
and replicate in direction. Do not invent a 5% scientific PASS threshold. A focused
screen and limited replication may fit 2–4 A100-hours; not every condition will.
The older proxy of three starting states × two tasks × 8k alone was 1.68–3.60 hours,
so choose scope using actual synchronized T3 timings.

## 11. Codex Cloud onboarding

### A. Check out and read the branch

In a new authorized workspace:
```bash
git clone --branch codex/exp96-fixed-task-diagnostics-20261022 \
  https://github.com/kaves29/scaling-drl-research.git scaling-drl-research
cd scaling-drl-research
git status --short --untracked-files=all
git rev-parse HEAD
git log -1 --format='%H %s' -- main/docs/agent_context/cloud.md
```
Read this document, repository `AGENTS.md`, the original probe/RunProbes, T2 config
paths, diagnostic runner/tests, and scientific-authority files in section 2.
Older `AGENT_HANDOFF`/`EXPERIMENT_STATUS` entries concern dated campaigns; live
metadata outranks narrative snapshots. Stay on the diagnostic branch, inspect for
other agents' changes, and coordinate a single writer. Preserve untracked work.

### B. Establish the environment and run only relevant validation

Prefer an existing reviewed Python 3.12 environment. On a fresh Cloud Linux image:
```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r main/requirements.txt
.venv/bin/python -m pip check
```
This recipe is not proof that installation succeeds on every image. Record versions
and dependency conflicts; do not silently upgrade JAX/Flax or mutate shared Delta
Conda environments. CPU wheels do not qualify A100 behavior. Consult
`main/docs/exp12_cpu_validation_setup.md` for historical setup guidance, without
rerunning unrelated environment suites or HumanoidBench setup.

Implement T2 and its focused regression first. Reuse the 13 completed contracts
unless numerical code changes or a new environment warrants one compatibility run.
Use section 7 CPU flags. Capture actual discovery count, stdout, stderr and exit code
outside Git; zero discovered tests is not a pass. No GPU is needed for T2.

### C. Coordinate Delta work and owner authentication

Follow section 6. First reconcile job/accounting, then inspect selected source-bound
artifacts. Owner Mac authentication alone does not grant Cloud access. Without an
approved reachable route, finish T2 and CPU validation, push the successor, and give
the owner/local executor exact SHA, reconciliation commands, artifact paths and a
bounded job command. Never invent remote status or transfer authentication secrets.

Stage reviewed published source in a new detached clean Delta checkout; preserve
active bcf source and all old evidence. A changed SHA alone is not a reason to repeat
the original smoke. Target the affected reconstruction and measurements. Before a
new GPU submission, verify actual usage plus all pending worst-case reservations is
at most four physical A100-hours and check for duplicates. Keep failed/partial results.

### D. Commit and report

Keep implementation corrections, scientific changes, performance changes and docs in
separate commits. Use normal pushes to this diagnostic branch; do not rewrite history.
Check diffs and relevant tests. Record source/config/task/checkpoint identities and
accounting. Do not commit replay/checkpoints, API credentials, private keys or sockets.

Deliver to the scientific analyst: original/new curves; initial/final MSE and normalized
MSE; mean/centered residuals; gradient/function changes; per-target pairings and seed
IDs; precision/runtime/teacher/normalizer bindings; replication; missing conditions;
source commits; claim classifications; and compute ledger. The owner decides scientific
adoption. Software success or lower losses are not scientific approval.

## 12. Current state and immediate action

| Area | Verified state / limit | Next action |
|---|---|---|
| Original experiment | Frozen f597 source/results preserved | Read-only reuse |
| Retrospective evidence | 96-parent descriptive analysis reconciled | Export selected Mac-only evidence if needed |
| Diagnostic source | 5974706; CPU contracts valid in their scope | Correct optimizer provenance before trusting reconstruction |
| New confirmed defect | Generic 3e-4/0 diagnostic optimizer versus original 1e-4/.01 | Metadata-derived optimizer and focused regression |
| GPU job | 23120792, source bcf; historical pending state only | Obtain live state, logs and accounting |
| Budget | Four hours cumulative; last verified zero is not current usage | Reconcile every allocated second, including failures/idle/compilation |
| Automation | Draft only; no dependent jobs deployed | Build/test a minimal bounded continuation only when needed |
| Root cause | No controlled quantitative cause established | Correct reconstruction, then decisive contrasts |
| Scientific fix | None validated | Owner review for methodological changes; controlled validation/replication |
| Performance | No measured new speedup | Synchronized timings, not asynchronous dispatch |
| Cloud evidence access | Source/handoff in Git; raw evidence on Mac/Delta | Authorized route or selected sanitized exports |

**Immediate CPU task:** implement T2's metadata-bound optimizer selection and targeted
regression on this branch. **Immediate remote task:** reconcile job 23120792 and its
accounting. Preserve wrong-config results; they cannot certify original reconstruction.
A scientific decision is unnecessary to restore original optimizer settings. Later
changes to the scientific measurement remain owner decisions.

Current job status/end/actual hours, whether any fits ran, exact historical target
identity, causal contributions, new GPU performance/memory, a generalized correction,
and a working Cloud-to-Delta route remain unverified. Do not present those as completed.

## Appendix A. Preserved uncommitted automation draft (not executable qualification)

The following is the exact local draft at handoff. It is included here so Cloud
can recover the work without depending on a Mac-only untracked file. It is **not**
installed as a controller, CPU-tested, or deployed. It submits no job and does not
constitute budget enforcement. Its nominal module purpose is not evidence it works.
Ifreused, firstaddressT2andT5. Sourcepathatcapture:
`main/experiments/exp12/autonomous_control.py` (untracked locally).
Snapshot SHA256:`a990f20f6bed00c322de03299723798777a2fe6d4c32ebfc994ffb947ddd7ef4`.

```python
"""Fail-closed admission and reporting for one diagnostic continuation."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import os
import zipfile

import numpy as np

FROZEN = "f597ee9898e4d401912092c336c3e5330cf2fda1"
FIRST_SOURCE = "bcf6cf3c90726c77f998e5795fedf0d62c5f11e2"
FIRST_JOB = "23120792"
CEILING_SECONDS = 14400
CHILD_SECONDS = 3600
CORE_FILES = (
    "experiments/exp12/probe.py",
    "experiments/exp12/fixed_task.py",
    "experiments/exp12/diagnostic_artifacts.py",
    "experiments/exp12/precision.py",
    "scale_rl/agents/sac/sac_network.py",
    "scale_rl/networks/layers.py",
    "scale_rl/networks/utils.py",
    "configs/agent/sac.yaml",
)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def command(args, cwd=None):
    result = subprocess.run(
        args, cwd=cwd, env={**os.environ, "TZ": "UTC"},
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr[:1500]}")
    return result.stdout


def accounting(job_ids):
    raw = command([
        "sacct", "-D", "-nP", "-j", ",".join(job_ids),
        "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,Partition",
    ])
    result = {}
    for line in raw.splitlines():
        fields = line.split("|")
        if len(fields) < 6 or fields[0] not in job_ids:
            continue
        job, state, exit_code, seconds, tres, partition = fields[:6]
        if job in result:
            raise ValueError("duplicate/requeued accounting requires human review")
        resources = dict(x.split("=", 1) for x in tres.split(",") if "=" in x)
        elapsed = int(seconds)
        gpu = int(resources.get("gres/gpu", "0"))
        terminal = state.split()[0] in {
            "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
        }
        if not terminal or (elapsed and gpu != 1):
            raise ValueError("unsettled or unexpected GPU accounting")
        result[job] = {
            "state": state, "exit_code": exit_code, "elapsed_seconds": elapsed,
            "gpu_count": gpu, "physical_gpu_seconds": elapsed * gpu,
            "alloc_tres": tres, "partition": partition,
            "resource_weighted_SU_estimate": elapsed / 3600 * float(resources.get("billing", 0)) / 1000,
        }
    if set(result) != set(job_ids):
        raise ValueError("accounting missing; no GPU admission")
    return result


def budget_admission(usage, reservations):
    if set(usage) != {"23120674", FIRST_JOB}:
        raise ValueError("unexpected investigation job set")
    if any(str(x["job_id"]) not in usage for x in reservations):
        raise ValueError("another reservation already exists")
    parent = usage[FIRST_JOB]
    if parent["state"] != "COMPLETED" or parent["exit_code"] != "0:0" or parent["gpu_count"] != 1:
        raise ValueError("first GPU stage did not complete cleanly")
    seconds = sum(x["physical_gpu_seconds"] for x in usage.values())
    if seconds + CHILD_SECONDS > CEILING_SECONDS:
        raise ValueError("insufficient physical A100 budget")
    return seconds


def validate_first(root, first_main, current_main):
    root = Path(root)
    status, runtime, results = [read_json(root / n) for n in ["status.json", "runtime.json", "results.json"]]
    if status["status"] != "COMPLETED_BUDGETED_STAGE":
        raise ValueError("first stage failed or unfinished")
    if runtime["job_id"] != FIRST_JOB or runtime["expected_commit"] != FIRST_SOURCE:
        raise ValueError("first job/source binding mismatch")
    env = runtime["runtime"]
    if env["platform"] != "gpu" or "A100" not in env["device_kind"] or env["matmul_precision"] != "tensorfloat32":
        raise ValueError("wrong hardware/precision")
    if results["source"] != FIRST_SOURCE or results["frozen_source"] != FROZEN:
        raise ValueError("result source binding mismatch")
    hashes = {}
    for name in CORE_FILES:
        old = (Path(first_main) / name).read_bytes()
        new = (Path(current_main) / name).read_bytes()
        if old != new:
            raise ValueError("numerical core changed: " + name)
        hashes[name] = hashlib.sha256(old).hexdigest()
    reproduced = [x["original_reproduction"] for x in results["rows"] if "original_reproduction" in x]
    required = {("myo-pen-twirl", "D2W512", 5), ("humanoid-run", "D4W1536", 1)}
    summaries = []
    for env_name, arch, seed in sorted(required):
        directory = root / f"original_exp1_{arch}_{env_name}_seed{seed}_r0"
        with np.load(directory / "task.npz", allow_pickle=False) as task:
            if task["obs"].shape[0] != 25600 or task["indices"].shape != (8000, 256):
                raise ValueError("incomplete reconstruction task")
            if any(not np.isfinite(task[n]).all() for n in ("obs", "act", "base")):
                raise ValueError("nonfinite original numerical task")
        for kind in ("fresh", "latest"):
            rows = [x for x in reproduced if (x["environment"], x["architecture"], x["seed"], x["kind"], x["round"]) == (env_name, arch, seed, kind, 0)]
            if len(rows) != 1:
                raise ValueError("required original fit absent/duplicated")
            row = rows[0]
            if not all(np.isfinite(row[k]) for k in ("final_mse", "reference_mse", "offset_error", "variance_error", "curve_max_abs_error")):
                raise ValueError("nonfinite original fit receipt")
            if row["variance_error"] > .002 or row["offset_error"] > .05 or abs(row["final_mse"] - row["reference_mse"]) > max(.02, .1 * abs(row["reference_mse"])):
                raise ValueError("historical reconstruction mismatch")
            with np.load(directory / f"{kind}_original_fit.npz", allow_pickle=False) as data:
                losses = data["losses"]
                if losses.shape != (2000,) or not np.isfinite(losses).all():
                    raise ValueError("incomplete original curve")
                summaries.append({**row, "initial_minibatch_mse": float(losses[0]), "early_peak_mse": float(losses[:100].max())})
        equivalent = read_json(directory / "segmented_equivalence.json")
        fresh = next(x for x in summaries if (x["environment"], x["kind"]) == (env_name, "fresh"))
        if not np.isfinite(equivalent["final_abs_error"]) or equivalent["final_abs_error"] > max(1e-4, .005 * abs(fresh["final_mse"])):
            raise ValueError("GPU segmented equivalence failed")
    # The frozen runner reaches COMPLETED only after its exact repeat checks.
    return {"original_fits": summaries, "numerical_core_sha256": hashes, "runtime": runtime}


def write_report(auto_root, first_root, usage=None, error=None):
    root = Path(auto_root)
    rows = []
    for stage, location in [("first", Path(first_root)), ("continuation", root / "gpu")]:
        p = location / "results.json"
        if not p.exists():
            continue
        data = read_json(p)
        for row in data["rows"]:
            if "measurements" not in row:
                continue
            for point in row["measurements"]:
                rows.append({"stage": stage, "name": row["name"], "environment": row["environment"], "architecture": row["architecture"], "seed": row["seed"], "kind": row["kind"], "task_id": row.get("task_id", "legacy-unbound"), **point})
    columns = sorted({k for r in rows for k in r})
    with (root / "fitting_measurements.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Bounded EchoCritic continuation", "", "This is diagnostic evidence, not an approved detector change or a causal actor claim.", ""]
    if error:
        lines.extend(["BLOCKED: " + error, "No further GPU job is admitted by this report.", ""])
    if usage is not None:
        atomic_json(root / "accounting.json", usage)
        seconds = sum(x["physical_gpu_seconds"] for x in usage.values())
        lines.extend([f"Allocated physical A100 hours: {seconds / 3600:.8f} / 4.", "Weighted charges in accounting.json are Slurm resource-time estimates, not posted ACCESS transactions.", ""])
    for row in rows:
        if row["step"] not in (0, 2000, 4000, 8000):
            continue
        lines.append(f"- {row['stage']} {row['name']}, update {row['step']}: MSE={row['mse']:.7g}, mean residual={row['mean_error']:.7g}, centered residual variance={row['centered_error_variance']:.7g}.")
    lines.extend(["", "Interpretation limits: zero-offset fixed tasks are supplementary contrasts; different starting functions remain a confound. Three-target screens are not IQM10 qualification. Missing/partial conditions are not passes. No scientific correction is applied automatically. Owner review is required before a scientific metric change."])
    (root / "findings.md").write_text("\n".join(lines) + "\n")
    with zipfile.ZipFile(root / "review_package.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(root.glob("*.json")):
            archive.write(p, p.name)
        for name in ("findings.md", "fitting_measurements.csv"):
            archive.write(root / name, name)
```

## Appendix B. Verification index

All paths below were checked to exist in the source worktree at handoff. Use this
list to detect missing/new layouts before editing. Placeholder paths in commands
are explicitly examples; absolute Delta paths are historically verified locations,
not asserted accessible from Cloud.

- `AGENTS.md`
- `main/run.py`
- `main/generate_manifest.py`
- `main/requirements.txt`
- `main/.claude/methodology-exp1-exp2.md`
- `main/docs/exp12_decisions.md`
- `main/docs/AGENT_HANDOFF.md`
- `main/docs/EXPERIMENT_STATUS.md`
- `main/docs/exp12_cpu_validation_setup.md`
- `main/docs/exp96_fixed_task_diagnostics.md`
- `main/configs/base_exp12.yaml`
- `main/configs/base_exp12_ten.yaml`
- `main/configs/agent/sac.yaml`
- `main/configs/agent/sac_simba.yaml`
- `main/configs/buffer/numpy_uniform.yaml`
- `main/configs/env/dmc_hard.yaml`
- `main/configs/env/myosuite_simba.yaml`
- `main/experiments/exp1.py`
- `main/experiments/exp12/trainer.py`
- `main/experiments/exp12/state.py`
- `main/experiments/exp12/probe.py`
- `main/experiments/exp12/run_probes.py`
- `main/experiments/exp12/trigger.py`
- `main/experiments/exp12/ledger.py`
- `main/experiments/exp12/fork.py`
- `main/experiments/exp12/precision.py`
- `main/experiments/exp12/diagnostic_artifacts.py`
- `main/experiments/exp12/fixed_task.py`
- `main/experiments/exp12/probe_tasks.py`
- `main/scale_rl/agents/sac/sac_network.py`
- `main/scale_rl/agents/sac/sac_update.py`
- `main/scale_rl/agents/wrappers/normalization.py`
- `main/scale_rl/networks/layers.py`
- `main/scale_rl/networks/utils.py`
- `main/scripts/exp96_campaign.py`
- `main/scripts/exp96_multigpu.py`
- `main/scripts/exp96_fixed_task_diagnostics.py`
- `main/scripts/sbatch_exp96_fixed_task_diagnostics.sh`
- `main/scripts/check_exp12_validation_checkout.sh`
- `main/tests/test_fixed_task.py`
- `main/tests/test_probe_tasks.py`
- `main/tests/test_fixed_task_bank_integration.py`

Documentation excludes private keys, tokens, passwords, actual SSH socket basenames,
agent environment values and personal context unrelated to research. Only public cluster
endpoints, project paths, run identities, commands, scientific metadata and code are included.
