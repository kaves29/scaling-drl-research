# Pilot root-cause audit and engineering corrections — 23 October 2026

The source audit did **not** find the later retrospective optimizer-selection
bug in the specified pilot source. Two independently reproduced engineering
defects are corrected here: terminal success can be attributed to a reset
instead of the completed episode, and the retrospective fixed-task summary
accepts invalid fitting horizons. Neither establishes an explanation for the
pilot's loss spikes, large trained errors, or return drop. Those observations
must be preserved, not repaired by changing the scientific protocol.

## Provenance and coverage

- Dedicated branch: `codex/pilot-root-cause-corrections-20261023`, based on
  `55166a643c979240510f3bdb76484c9777d72fee` (PR #9, itself based on PR #8).
  PR #8 remains open at `868cab4346f2f3124d1a459e34a2d66fed28c794`.
- Implementation: `89cb004` fixes and CPU attribution harness; `51f33a3` adds
  the configured-decay known answer and explicit final errors. Resolve full SHAs
  and later documentation-only commits from Git.
- Specified pilot source: `f8619dd5f494399d9febc79fc9092010234c714d`.
  Frozen Exp96 source: `f597ee9898e4d401912092c336c3e5330cf2fda1`.
- `/workspace/echo_pilot_local_partial.zip` is **absent**. A filename search
  under `/workspace` and `/tmp` found no recovered pilot archive. Seven raw
  runtime metadata files, W&B history and per-episode return records were not
  available. This audit cannot certify the actual settings or reproduce the
  seven-run population statistics. User-reported anomalies remain observations
  supplied by the owner, not newly remeasured results.
- The later seven-parent scheduler names training source `087feab...`, not
  `f8619dd...`, and does not exist at f861. Between these revisions the relevant
  changes include project name, disabled forking and actor diagnostic behavior.
  Do not silently treat scheduler placement as proof of actual pilot provenance.
  The harness resolves its seven listed architecture/environment placements
  against f861 as **reconstructed configurations**, not runtime evidence.
- Read-only exported pilot source and evidence reside outside Git under
  `/workspace/artifacts/pilot-root-cause/`. The harness verifies source bytes
  against immutable Git objects before any fit. It also verifies that the
  imported network, optimizer constructor and normalization implementations
  match those pilot bytes. Resized constructor calls are observed, not inferred
  from YAML alone. No historical Git object or archived measurement is modified.

## Source trace and effective settings

These links refer to the specified immutable pilot source unless indicated.
Resolved configurations for all seven reconstructed placements are retained
in `final-controls/receipt.json`, including every configuration field.

| Stage | Source and effective behavior |
| --- | --- |
| Configuration | [exp1.compose_config](https://github.com/kaves29/scaling-drl-research/blob/f8619dd5f494399d9febc79fc9092010234c714d/main/experiments/exp1.py#L52) composes base_exp12 and overrides; generate_manifest.exp12_overrides supplies architecture, environment group, seed and locations. No optimizer override in this grid recipe. Actual launch overrides require runtime metadata. |
| SAC constructors | [sac_simba.yaml](https://github.com/kaves29/scaling-drl-research/blob/f8619dd5f494399d9febc79fc9092010234c714d/main/configs/agent/sac_simba.yaml): actor residual D1W128, critics D2W512/D4W1024/D4W1536, actor/critic AdamW lr=1e-4, decay=.01. [SACAgent/_init_sac_networks](https://github.com/kaves29/scaling-drl-research/blob/f8619dd5f494399d9febc79fc9092010234c714d/main/scale_rl/agents/sac/sac_agent.py#L97) passes both explicit fields to Optax. Temperature lr=1e-4, decay=0, initial .01, entropy target −action_dim/2. |
| Actual constructor check | Resized pilot-compatible actor/critic instantiated on CPU: captured AdamW calls are actor (1e-4,.01), critic (1e-4,.01), temperature (1e-4,0). This does not instantiate seven production networks or inspect their historical moments. |
| Optimizer state | Optax .2.3 defaults beta1=.9, beta2=.999, eps=1e-8, eps_root=0; constant learning rate, unmasked decay including biases/LayerNorm. Trainer.apply_gradient carries moments and count and applies updates with current parameters. SAC scan carries updated actor, critic and target. |
| SAC targets | [update_critic](https://github.com/kaves29/scaling-drl-research/blob/f8619dd5f494399d9febc79fc9092010234c714d/main/scale_rl/agents/sac/sac_update.py#L70): reward + gamma^n (1−terminated)(targetQ−temperature×log_prob); n=1, target_tau=.005; time-limit truncation bootstraps. Single critic except episodic HumanoidBench twins, which use min targetQ. No target or Q normalization/clipping was found in this path. |
| Replay/training | Uniform replay max1M, min5000, minibatch256; UTD2; one train/eval environment, action_repeat2. Raw replay observations are normalized by the current running observation statistics before SAC updates and before probe sampling; statistics update only during training action selection. |
| Environment-specific values | DMC dog/humanoid gamma=.99,1M raw steps/500k interactions; Myo gamma=.95,1M/500k, max_episode_steps100 (50 interactions under action repeat), reward scale1; H1 gamma=.99,2M/1M, max_episode_steps1000, twin critics. Myo's registered key-turn time limit can be200, but the explicit SimBa wrapper limit100 is intentional. Do not substitute myosuite_hard's1000 limit. |
| Fresh/current | [RunProbes.capture_fresh/current_critic](https://github.com/kaves29/scaling-drl-research/blob/f8619dd5f494399d9febc79fc9092010234c714d/main/experiments/exp12/run_probes.py#L64): fresh is the actual initial critic captured before the first SAC update; current is the online critic at the check. Fresh parameters remain fixed across checks; replay, normalizer and target draws change. Both get freshly initialized probe AdamW state; neither continues SAC's optimizer moments. |
| Probe task | [probe_round](https://github.com/kaves29/scaling-drl-research/blob/f8619dd5f494399d9febc79fc9092010234c714d/main/experiments/exp12/probe.py#L112): same normalized replay pool25600, same independently initialized same-architecture teacher sin(1e5 f), same minibatch indices256. Own-mean offsets differ by learner. Thus common base and inputs do **not** mean identical numerical targets. |
| Fitting/metrics | [_fit](https://github.com/kaves29/scaling-drl-research/blob/f8619dd5f494399d9febc79fc9092010234c714d/main/experiments/exp12/probe.py#L74): continuous2000 updates, loss[t] before update t+1 on minibatch[t]; final_loss is full-pool MSE after2000. Score=Var(base)−finalMSE; paired L=score_fresh−score_current. Five rounds use historical scipy25% trim_mean (one value per tail), not ten-target weighted IQM. |
| RNG/precision | Separate PROB fold-in stream for seed/check/round; separate NumPy replay sampling generator; first1000 minibatches from index key, next1000 from fold_in(key,1). FP32 mixed_precision=false; GPU matmul precision tensorfloat32, CPU unaffected; Check1 uses highest. sin(1e5 f) is highly sensitive to teacher prediction rounding, so matching source/seed and target variance alone cannot establish identical historical targets. |
| Evaluation | [evaluate_episodes](https://github.com/kaves29/scaling-drl-research/blob/f8619dd5f494399d9febc79fc9092010234c714d/main/experiments/exp12/trainer.py#L43): separate seeded eval environment, fresh reset each episode, training=false so RMS frozen, policy temperature0 (tanh mean actions). Sum actual rewards through termination/truncation; average episode returns. Five episodes every50k interactions, final evaluation50 episodes. Auto-reset consumes an extra reset before the next explicit reset; this is existing protocol, not evidence of a return-sum bug. |
| Logging/checkpoints | Trainer logs every2000 interactions at env_step=interaction_step×action_repeat (one env); initial evaluation at0. Local CSV preserves explicit steps and per-episode returns. Pilot rounds5 do not enter later RunProbes rounds10 W&B blocks. A later sidecar publishes validated completed-parent plasticity against raw env steps; its baseline-relative degradation percentage is not paired L. Checkpoint restore carries parameters, moments/count, RNG, RMS, replay, environments and loop/log state. No evidence here associates a late drop with a restart. |

The pilot and frozen Exp96 training/optimizer/network implementations agree in
the inspected paths. Exp96 changes include ten rounds,50 checks, weighted IQM10
and direct probe W&B logging. PR #8 binds its retrospective optimizer to archived
metadata and adds fixed-task alternatives; its zero/shared targets are not the
pilot's own-offset definition. The later generic3e-4/0 regression does not appear
in the pilot constructor. See the [separate optimizer audit](exp96_optimizer_validity_audit.md).

## Confirmed defects, fixes and scope

| Issue | Severity | Root cause and correction | Historical scope |
| --- | --- | --- | --- |
| Terminal success selected from reset info | Logging/reporting defect | evaluate_episodes prioritized top-level success even when SyncVectorEnv had auto-reset and placed the completed episode in final_info. Now terminal final_info has priority; reset info cannot count as success for the finished episode. Regression reproduces both false-negative and false-positive cases. | Present in pilot source. Affects success only when reset info contains success; occurrence in actual seven runs is unverified. Return accumulation, actions, resets and lengths are unchanged. It cannot explain an eval/avg_return drop. |
| Invalid completed horizons accepted | Significant experimental confound in retrospective summary validation | fixed_task_paired_summary only compared equality, accepting0, negative, fractional or boolean horizons with finite supplied losses. Now requires positive integer horizons before pairing. Four invalid-horizon cases failed before correction and pass after. | New retrospective API, absent from original pilot. Does not invalidate any recorded valid2000-step pilot result. Caller still must establish completeness/provenance; a positive number alone is not a success receipt. |
| First-update parameter movement absent | Reporting addition, not an established numerical defect | fixed_task.first_update now records global parameter L2 and per-element RMS delta as well as prediction RMS, using the same copied update. Constant-critic known answer checks both against .4. | New diagnostics only; no optimizer/update or fitting definition changed. |

Pre-correction receipt: `before-fixes.log` records7 tests, five assertion failures
(four horizon subcases plus success) and one missing-instrumentation error.
The expected missing field was a feature gap, not counted as a historical bug.
Original passing tests and historical artifacts were not changed to obtain PASS.

## Classification of unresolved interpretations

| Issue | Classification | Disposition |
| --- | --- | --- |
| Fresh early spikes from architecture/first-step amplification | Unconfirmed scientific hypothesis | No production-scale reproduction in this audit; preserve curves. |
| Own-offset targets and changing replay/normalizer across checks | Significant experimental confound for stronger equal-task/intrinsic-capacity interpretations | Correctly implements the specified measurement; do not silently change it. |
| Large-offset decay sensitivity and FP32 cancellation | Numerical robustness concern | Reproduced in synthetic fixtures, causal contribution to pilot unknown. |
| Late policy-return drop | Unconfirmed scientific hypothesis / missing measurement verification | Source has no demonstrated return bug; raw episode/history comparison remains necessary. |
| A100 segmented/continuous discrepancy | Numerical robustness concern and failed engineering qualification | Still unresolved; fail-closed guard retained. |
| f861 versus later scheduler's087 source identity | Significant experimental confound if conflated | Reconcile actual metadata before attributing seven runs to either source. |

No newly confirmed critical scientific correctness defect in the original pilot
fit/update path is established by the available evidence.

## Anomaly A: early fresh loss spikes

There is no verified optimizer mismatch or state-reset defect in the pilot path.
Own-offset targets, LayerNorm/residual networks and AdamW first steps can create
architecture-dependent function-space movement; this remains a hypothesis for
the production spikes. The first logged loss is before any update, but losses
at adjacent indices use different minibatches. A deterministic zero-learning
fixture produces [1,100] simply by changing samples, while fixed full-pool MSE
stays50.5. An increasing training curve alone cannot distinguish sampling
variation from deterioration of the same fixed loss.

New tests verify common inputs/base/minibatch order, distinct own offsets,
pre-update initial loss, gradient/update arithmetic, seed reproducibility via
existing contracts and unchanged source parameters. Existing first-update
regressions compare against explicit JAX gradients/Optax updates. No gradient
clipping, normalization, optimizer change or precision workaround was added.

## Anomaly B: trained key-turn errors and adaptation scores

With z=sin(1e5 f_teacher), own-offset target y=mean(q)+z, initial residual is
(q−mean(q))−z. Up to arithmetic rounding, initial pool MSE equals
Var(q)+Var(z)−2Cov(q,z)+mean(z)^2. A large Q **mean alone** does not establish
large initial error; Q spread and alignment matter. A trained critic's large
initial error is not equivalent to loss of learnability.

Own offsets implement the stated methodology. Fresh/current share inputs and
base function within a round but have distinct absolute targets and starting
functions; equal optimizer and update counts do not equalize initial difficulty.
Final score is conditional finite-budget fitting performance. Check0 fresh score
uses a different replay/normalizer/teacher draw than later paired checks; the
sidecar percentage relative to that score is not a controlled longitudinal
measurement of intrinsic capacity. Changing to common targets, centering outputs,
removing bias decay or normalizing scores requires a separately approved
scientific comparison, not an engineering patch.

A technically correct mechanism was reproduced: AdamW decay makes subsequent
fitting non-invariant to output translation even when initial residuals nearly
match. The new constant-critic known answer starts with zero loss and zero
first gradient; approved decay alone moves a large bias away from its own
fixed target. Six residual fixtures compare bias0 with an artificial +10000
output bias. D2W16 initial MSEs .861406/.861230 become final .096905/36.629318.
This is an artificial stress control, **not** a trained key-turn reproduction.
The other final MSE pairs are D4W24 .000108/5.188416 and D4W32
3.95e-9/1.007216; all are one seed on resized networks. Raw curves and an
explicitly labeled synthetic-control PNG/PDF accompany the receipts.
It does not identify decay as the causal explanation for the pilot. Both
conditions retain1e-4/.01 and identical streams; no settings are tuned for a
better curve. FP32 cancellation adds small initial differences at large offsets.

## Anomaly C: late D4W1536 key-turn return drop

Source inspection found no verified action-selection, return-summing or logging
error that explains the reported drop. Return and success are separate metrics;
the corrected terminal-success bug does not change returns. Evaluation freezes
RMS and policy parameters; temperature0 actions are deterministic for a given
observation, though SAC's key still advances as in the original protocol.
Truncations, reward rescaling (eval scale1) and episode lengths are explicit.
The larger final episode count changes estimator variance and should be shown
alongside individual episodes rather than interpreted as a training event.

Without the seven runs' raw episode CSVs, W&B history, resolved metadata and
checkpoint/restart sequence, this audit cannot distinguish a genuine recorded
return decline from a plotting/export mistake, and cannot attribute causality.
Minimum read-only verification: join rows by explicit env steps and run identity;
recompute means from episode returns; compare local CSV to unsmoothed W&B rows;
check final episode count, state restores, source and RMS provenance. No new
policy rollout is needed for this first check. Historical returns remain
untouched as recorded measurements, without a new claim of causal validity.

## Segmentation attribution

`audit_pilot_root_cause.py` executes the actual byte-verified pilot _fit against
D2W16/D4W24/D4W32 residual FP32 fixtures, each with bias0/+10000, pool128,
obs6/action3, minibatch16, chunk32, fixed seed1/check0/round0 and2000 updates.
For each case it compares (1) original own-offset monolithic fit, (2)2000-step
materialized-target scan, (3)1000+1000 steps with full FitState serialized and
restored at the boundary. Adam counts are0→1000→1000→2000; restored params,
moments and count hash identically; all six final states and loss curves match
bitwise across the materialized paths, and all six frozen-versus-segmented
curves/final errors are zero. A deliberately reset optimizer state is detected
by the regression. Numeric arrays and halfway state bytes are retained.

This rules out those errors in these CPU executions only. The failed A100
job23120792's .00405173 final discrepancy versus guard .000751385 remains a
failed qualification. No serialized boundary was used in that original failure;
therefore disk restore is not evidenced as its cause. Both old paths used the
same wrong optimizer. Fixed versus internally constructed targets, compiler
fusion, TF32/GPU reduction arithmetic and amplified trajectory divergence remain
unresolved candidates. The existing threshold is unchanged. The runner retains
its fail-closed segmentation guard; CPU PASS must not be relabeled A100 PASS.

## Validation and evidence

- Focused suite:49 tests pass, zero skips, covering seven initial new contracts,
  PR #8 fitting/plan/provenance/execution/export and existing probe/pilot checks.
- Additional configured-decay known-answer test passes (one test); it preserves
  the approved behavior instead of asserting that translations must fit equally.
- Five additional distinct restoration/parity/probe-isolation tests pass. The
  first restoration invocation ran12 tests with11 passes and one setup error
  because MUJOCO_GL=disable violates HumanoidBench's rendering requirement.
  That same per-suite restoration test passes under EGL (146.453 seconds),
  including H1 and both pre-update/mid-episode restore splits, without source
  changes. The failed setup receipt is preserved. Total distinct passing unit
  tests across these runs:55; duplicates are listed/deduplicated in validation.json.
- All74 mutation checks detect the deliberate break and pass after restoration;
  zero PROBLEM results, exit0. Black, Pyflakes and git diff --check pass.
- Forged pilot source is rejected before creating an output directory. The exact
  immutable f861 evaluator independently reproduces both terminal-success errors;
  the corrected evaluator preserves returns/lengths in the same fixtures.
- Final source-bound CPU harness: `final-controls/receipt.json`; source hashes,
  seven reconstructed resolved configs, resized constructor calls, six complete
  attribution cases, full raw curves and halfway msgpack states. Earlier harness
  attempts and the configuration-serialization API error are retained separately;
  they are not counted as successful fits or additional independent replications.
- CPU stack: Python3.12.14, JAX/JAXLIB0.4.34, Flax0.8.4, Optax0.2.3,
  NumPy1.26.4; FP32/x64 disabled, CPU backend. This does not qualify production
  architectures, GPU precision or the actual pilot's task geometry.

Reproduce from repository main/ using the pinned CPU environment and a fresh
output path (the source export must match immutable f861 bytes):

```bash
python scripts/audit_pilot_root_cause.py --repository /absolute/repo \
  --pilot-main /absolute/f861-export/main --out /absolute/new-output
python -m unittest tests.test_pilot_root_cause tests.test_fixed_task \
  tests.test_exp96_diagnostic_engineering tests.test_exp96_diagnostic_execution
```

## Minimum GPU qualification proposal — not authorized or submitted

First obtain read-only pilot metadata, exact archived arrays and episode CSVs;
resolve the f861-versus087 source discrepancy and fresh accounting. These are
prerequisites, not GPU work. Use an isolated source-bound pilot comparator;
PR #8's Exp96 runner intentionally rejects other-source parent metadata. Never
relax that binding or change historical metadata to admit pilot data.

One A100/16CPU/60GiB allocation,15-minute maximum, eight-minute workload cap
plus bounded shutdown, no retry/extension. Request ceiling900 physical seconds;
large-critic timing is unbenchmarked, so this is a reservation ceiling rather
than a demonstrated runtime. If safe completion cannot fit, preserve partials
and stop. Last verified investigation usage181 seconds is stale: after a full
900-second request total would be1081/14400 seconds, leaving13319 seconds,
subject to fresh accounting showing no other consumption.

Smallest useful sequence: corrected-source reproduction of the previously
failed D2W512 fresh pen seed5 task; then one source-verified D4W1536 key-turn
fresh case and its trained case at the reported anomalous check. Each case
shares exact numeric tensors **within** its monolithic/segmented comparisons,
uses its archived optimizer and own-offset definition,2000 steps, no SAC update.
Compare original monolithic, materialized monolithic and1000+1000 continuation;
repeat the large fresh monolithic once (ten2000-update fits total). Do not call
fresh/trained own-offset targets identical. Select actual checkpoint IDs/check
from verified pilot records, rather than inventing unavailable paths.

Acceptance: source/config/tensor identity verified; complete finite2000 losses;
Adam count and serialized state preserved; unchanged starting weights; exact
repeat contract retained; continuous-versus-segmented final error within the
existing max(1e-4,.005×abs(referenceMSE)) guard. Retain full curve error and first
update parameter/prediction deltas, including failures. Any mismatch halts later
cases. Passing this is engineering qualification only, not reproduction of
seven-run prevalence or proof of plasticity loss. Broader seeds, optimizer
sensitivity, common-target alternatives or policy rollouts need separate owner
approval. No allocation or scientific-methodology change is authorized here.
