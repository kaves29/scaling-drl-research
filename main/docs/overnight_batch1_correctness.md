# Overnight Batch1 — implementation correctness and reproducibility

This is an engineering audit of the current review stack. It does not analyze
or explain the pilot results. It adds no numerical stabilizer, changes no
approved optimizer value or diagnostic definition, and submits no GPU job.

## Source and scope

- Branch: `codex/overnight-batch1-correctness-20261023`.
- Exact base: `c05d77cc9899e50e857c170faf9937e5635c3663` (completed PR #10).
  This preserves the PR #10→#9→#8 stack rather than replacing PR #8's branch.
  PR #8's reviewed source is `868cab4346f2f3124d1a459e34a2d66fed28c794`.
  The default main branch is not claimed to contain these review branches.
- Core fixes/tests: `97fbb20f4c6637840798f5f5d2ce3e3604d4498f`.
  The subsequent `705490fbaa9225259889f7efb1ab7dab56497bc4` correction
  strengthens the retrospective diagnostic contract. Recovery found this exact
  local commit, its branch, report draft and test receipts accessible; the
  remote branch was absent. Recovery correction `e26e37f3cb353e212be02c151fd2d84aded63f53`
  completes static sparse-mask validation and repairs stale infrastructure
  test fixtures. Resolve final documentation/publication revisions from Git.
- Historical source inputs: pilot `f8619dd5f494399d9febc79fc9092010234c714d`,
  Exp96 `f597ee9898e4d401912092c336c3e5330cf2fda1`.
- Authority: chronological amendments in `.claude/methodology-exp1-exp2.md`
  and approved answers in `docs/exp12_decisions.md`, especially G1,
  time-limit amendment(s), twin amendment(z), and continuous2000-step fitting.
  The [entropy reconciliation](entropy_temperature_reconciliation.md) retains
  the approved signed target; this task does not select another interpretation.
- Coverage concerns the configured dense FP32, n_step=1 SAC paths, single/twin
  critics and current engineering tools. Sparse variants additionally receive
  deterministic static-mask restoration and next-update CPU tests. Neither
  sparse nor mixed-precision variants receive production/GPU qualification. Prioritized replay is not enabled by the factory and is not
  certified as an alternative algorithm.

## Verified defects and corrections

| Defect and precise source | Reproduction and correction | Could historical numerical results be affected? |
| --- | --- | --- |
| Replay reset retains old write pointer — `scale_rl/buffers/numpy_buffer.py:NpyUniformBuffer.reset` | After3 writes, reset sets length0 but leaves pointer3; the next item is written outside the filled prefix from which sampling reads. Reset now resets pointer0. A sentinel fixture proves the next sampled observations are the new item, not unfilled storage. | Yes if a populated buffer was reset and reused. Normal fresh construction starts at0 and is unchanged; no occurrence in archived runs is asserted. |
| Multi-step truncation has an incorrect bootstrap exponent — same file's n-step accumulation, with `sac_update.update_critic` | At n=3,gamma=.5, a first-transition truncation retains a one-reward prefix and terminal next observation but the target uses gamma^3 instead of gamma^1. Executing immutable pre-fix code gives target2.25 versus6 in a known-answer fixture. The buffer now explicitly rejects n_step≠1; no new multi-step scientific definition or checkpoint format was introduced. | All supplied study configurations have n_step=1, so their numerical behavior is unchanged. Any independently overridden multi-step run needs review. Supporting correct variable-horizon returns would require a separate implementation/protocol decision. |
| Invalid replay dimensions/warmup accepted — `NpyUniformBuffer.__init__` | Zero capacity, zero batch sizes, warmup0, warmup larger than capacity, invalid n_step types could be constructed or fail late. Explicit positive-integer/capacity checks now reject them before allocation. | Valid configured runs unchanged. This is input validation, not evidence that archived runs used invalid settings. |
| Invalid replay metadata accepted or rejected after mutation — `experiments/exp12/state.py:load_buffer` | Partially filled replay accepted an unrelated pointer; malformed/oversized pending queues could be truncated or raise after replay arrays were written. Validate pointer=count when not full, queue length, six transition fields and array shapes/types before copying arrays. | Valid saved states are unchanged. Corrupt/inconsistent metadata is now rejected; no historical state was edited or declared corrupt without evidence. This is not a claim of transactional rollback after arbitrary I/O corruption. |
| Explicit entropy target silently overwritten — `scale_rl/agents/sac/sac_agent.py:SACAgent.__init__` | An explicit target−3 with2 actions became−1 from the coefficient. Derive from the coefficient only when target is None. The ordinary None/−.5 configuration still gives−action_dim/2. | Could affect configurations that explicitly supplied a different target; metadata alone previously overstated that field's effect. Default pilot/Exp96 settings use None and are numerically unchanged. This does not authorize a new entropy setting for the study. |
| Invalid SAC scalar settings reach initialization — `SACConfig.__post_init__`, new `configuration.validate_config` | NaN/Inf, negative optimizer settings, invalid gamma/tau, nonpositive initial temperature and invalid sparsity/n_step are rejected before network construction. Zero learning rates remain usable in controlled fixtures. | Valid settings unchanged. No sign, schedule, clipping or optimizer transformation changed. |
| Standalone checkpoint can restore moments under another optimizer — `SACAgent.save_checkpoint/load_checkpoint`, new `configuration.py` | Orbax restores dynamic Trainer state but obtains static tx from the destination instance. A checkpoint could silently resume under3e-4 instead of1e-4, zero instead of nonzero decay, or another discount/entropy target. New saves add `sac_config.json`; compare update configuration, dimensions and AdamW descriptor/Optax version before restoring. Corrupt/new incompatible bindings fail closed. | New mismatched restores are prevented. Existing Exp1 entry points already validate run_metadata before restore; this is not proof any historical resume was mismatched. Legacy checkpoints have no sidecar and remain readable under existing external provenance gates; absence does not establish verified optimizer provenance. |
| Sparse masks can differ despite identical restored dynamic state — `Trainer.network_mask` static field and `configuration.check_checkpoint_config` | Immutable/recovered code reproduced equal restored parameters/moments/counters but different constructor-derived masks and a divergent next update. Retain the initialization-seed comparison if either actor or critic is sparse; only dense checkpoints waive it. Tests reject changed actor/critic mask seeds, require identical same-seed masks and next-update state, and preserve dense cross-seed restoration. | Could affect standalone sparse restores under another seed. Current study configurations are dense and unchanged. Legacy sidecar-free restores still require original configuration/source/seed evidence; this guard does not fabricate missing provenance. |
| PR #8 accepted unsupported reconstruction settings — `experiments/exp12/diagnostic_artifacts.py:optimizer_binding` | A matching row/metadata hash still admitted mixed_precision=true, nonzero sparsity or missing precision fields, while its runner constructs a dense FP32 critic. Require explicit mixed_precision=false and critic_sparsity=0. Four negative cases fail before correction and pass afterward. Synthetic metadata fixtures now include the actual required fields. | Canonical dense FP32 Exp96 settings remain accepted. Unsupported or incomplete metadata is rejected; no archived result is recomputed or changed. |

Regression receipts before correction contain23 assertion failures plus one
error across13 tests/subcases, four failures in the additional diagnostic
contract test, and two sparse-seed rejection failures in four restore tests. Missing validation is distinguished from a demonstrated error
in valid production data. The passing legacy read and valid-roundtrip controls
were retained; existing failures or numerical thresholds were not disabled.

Dense checkpoint binding deliberately does not require the destination
constructor's seed: the full saved RNG and parameters replace that initialization.
Sparse masks are static Flax fields excluded from the restored Trainer pytree;
their constructor seed must match. The guard runs before any Orbax restore.
This preserves masks by validated reconstruction, rather than changing the
historical checkpoint tensor layout. The saved
seed field describes construction configuration, not independent certification
of run identity. `run_metadata.json` remains the source for experiment/seed/env
identity and original source. The sidecar binds the base SAC configuration;
Exp12 metadata still reconstructs injection structures before restoration.
It is not a substitute for a complete source/runtime manifest or cryptographic
binding to every tensor file. Legacy archives are not rewritten to add it.
A standalone SACAgent checkpoint is model/optimizer/RNG component state, not
the complete training loop: Exp12 separately persists pending metric windows,
replay, RMS, environment state and counters. Legacy Angle1's reset-based resume
is not certified as bit-exact full trajectory continuation; its entry-point
smoke tests do not establish that stronger property. Full continuation evidence
here refers to the current Exp12 engine.

## Components checked without a new arithmetic defect

| Component | Intended behavior checked | Supporting evidence |
| --- | --- | --- |
| Bellman target | r+gamma^n(1−terminated)(Q_target−alpha logpi); truncation bootstraps from the terminal observation, not auto-reset observation. | New `test_batch1_sac_equations.SacEquationTest.test_bellman_targets_and_critic_gradients_terminal_truncated_and_continuing`: independent NumPy target, loss and analytic weight/bias gradients, both single/twin paths, gamma0/.95/1. Existing `DiscountAndHorizonTest` exercises actual time limits and terminal capture. |
| Actor objective/gradient | Mean(alpha logpi−Q), with min(Q1,Q2) for twins. Differentiate through sampled actions and tanh; critic parameters remain fixed in the actor update. | New actor known answer uses independent tanh density/analytic action derivatives, not the production loss helper. State hashes prove original actor/critic/target/temperature unchanged. Existing twin actor-min and independent transformed-density tests cover actual modules. |
| Gradient isolation | Actor update optimizes only actor params; critic regression optimizes only critic params, treating sampled target values as constants for that gradient. Temperature differentiates only log alpha. | New analytic-gradient tests and before/after whole-state hashes; existing twin test isolates each Q's parameter slice. No stop-gradient was inserted into the action path. |
| Temperature | J=alpha(H−h*), dJ/dlogalpha=alpha(H−h*), H=−mean(logpi); approved default h*=−.5 action_dim. | New below/at/above-target known answers and existing independent entropy/density reference. No alternative sign chosen. |
| Update order | Actor first; temperature from that actor entropy; critic uses the updated actor/temperature and old target; Polyak uses new critic. | Source trace `sac_agent._sac_update`, existing scalar-versus-scanned update/diagnostic-isolation tests, new actual single/twin scan continuation, and unchanged-source compatibility comparisons. This is the repository/SimBa order, not an assumed critic-first variant. |
| Target networks/parameter sharing | Initial online/target values equal; targets have no tx/moments; functional Polyak theta_target=tau theta_online+(1−tau)theta_target. Updating online state leaves old targets unchanged. | New exact initialization/Polyak tests at tau0/.005/1, both single and twin; existing twin checkpoint/Polyak tests. Equal immutable JAX values are not evidence of mutable parameter aliasing. |
| Twin semantics | Minimum target for both Qs, sum of the two MSE terms, minimum Q in actor loss; independent parameter slices. | New critic equations and existing `TwinCriticPathTest`, twin gradient/diagnostic tests and checkpoints. No divide-by-two convention was substituted. |
| Replay/RNG | Uniform replacement sampling from filled storage, copies on add/sample; NumPy stream continues after restore, including wraparound writes. | New sentinel/copy/wraparound/RNG tests; real entry-point replay-stream tests, full Exp12 restoration and crash tests. |
| Discount/action repeat | Undiscounted reward sum within RepeatAction, stop on done; gamma once per interaction for n=1. Myo gamma.95/limit100; DMC/H1 gamma.99/limit1000; repeat2, per approved heuristic. | Existing suite/config known answers,65 task/seed simulator restoration cells, independent195-parent methodology/configuration reference. Logged env_step is the protocol's interaction×repeat clock, not an additional simulator-step counter. |
| Evaluation | Temperature0 gives tanh Gaussian mean; repeated identical inputs give identical actions. Training uses stochastic sampling; restoring its key reproduces samples exactly. RMS is frozen in eval; parameters do not change. | New evaluation/key test plus existing normalization, real trainer, per-episode evaluation and restoration tests. Evaluation still advances the SAC RNG as before; changing eval schedule is not assumed trajectory-invariant. |
| Optimizer/state/counters | Fresh AdamW moments/count0, no schedule; next update carries moments and increments count/Trainer step. SAC checkpoints preserve RNG, optimizer and counters. Probes intentionally initialize fresh optimizer state each fitting task. | New Orbax roundtrip followed by exact next update, actual single/twin SAC scan6 versus3+3, existing Trainer/probe/8000-step continuation tests, and preserved full trainer counters/metrics/environment-state tests. |
| Run/source configuration | Resolved config/protocol hashes and source/identity validation before Exp1 resume; new standalone binding; diagnostic metadata has no generic fallback. | Existing run_metadata/configuration/manifest tests, PR #8's96-parent synthetic binding tests, new incompatible-checkpoint and dense-FP32 rejection tests. Reconstructed configs are not96 independently reread historical runtime files. |
| Failure/completeness | Partial and failed diagnostic results retain distinct non-success statuses; guard failures halt; source/run/budget checks are fail closed. | Existing PR #8 execution/export/budget tests and untouched segmentation guard. No incomplete result is promoted to a successful experiment. |

## Optimizer provenance and default compatibility

`audit_batch1_compatibility.py` loads immutable constructor source for each
historical revision, composes that revision's configuration files, captures
actual resized constructor calls and retains source hashes/config snapshots.
Shared update/network/normalizer dependencies are required to be byte-identical
across those revisions and the current tree before compatibility is asserted.
These are reconstructed source configurations, not recovered per-run metadata.

| Source/path | Actor | Critic | Temperature | State/schedule |
| --- | --- | --- | --- | --- |
| Pilot f861, base_exp12→sac_simba | AdamW1e-4/.01 | AdamW1e-4/.01 | AdamW1e-4/0 | Constant; SAC moments carried |
| Exp96 f597, base_exp12_ten→sac_simba | Same | Same | Same | Same |
| PR #8 training constructors868cab4 | Same | Same | Same | Same |
| PR #8/current retrospective fitting | Not optimized | AdamW from validated parent fields; canonical1e-4/.01 | Not optimized | Fresh fit moments; carried continuously within the fit |
| Generic configs/agent/sac.yaml | Separate configuration | Critic3e-4/0 | Separate configuration | Not the intended diagnostic fallback |

For Optax0.2.3, AdamW is scale_by_adam(beta1=.9,beta2=.999,eps=1e-8,
eps_root=0)→add_decayed_weights→scale_by_learning_rate, unmasked, constant
schedule, no added clipping/normalization or Nesterov. Decay includes biases
and LayerNorm parameters under the established implementation. PR #8's original
metadata binding correction is retained. Its source/identity/config/launch
checks and runtime pins remain active; the new check adds supported precision
and sparsity requirements. The [earlier optimizer audit](exp96_optimizer_validity_audit.md)
contains the exact regression-introduction lineage; it is not reinterpreted here.

A direct old/new comparison executes the current default dense FP32 agent and
the completed PR #10 constructor with resized networks, for single and twin
critics. Initial state, four real SAC updates starting at external counter11,
all returned metrics (including diagnostic NaNs), RNG and dynamic Trainer states
match bitwise. This is a bounded compatibility proof, not production training,
full-size GPU equivalence or a claim that unsupported configurations are equal.

## Reproducibility and segmentation disposition

New tests compare six actual SAC updates with3+3 updates, serialize/restore the
entire intermediate dynamic state, preserve external diagnostic-counter phase,
and require bitwise final parameters, moments, RNG, counters and metrics for
single and twin critics. Separate real Orbax tests restore a populated agent
and compare the next update exactly. Replay tests include wraparound and the
next sampling stream; simulator and full trainer tests include environment RNG,
RMS, Python/NumPy/JAX states, pending logging windows and loop counters.

The existing probe tests retain the frozen first2000 minibatch draws, fresh
probe moments, continuous2000 fitting,1000+1000 equivalence, explicit8000-step
carry and unchanged starting parameters. The earlier six residual CPU cases at
PR #10 remain historical CPU evidence; no pilot-mechanism exploration is added.

A100 job23120792 remains a failed equivalence check: final difference
.0040517300 exceeded .0007513852. No actual saved A100 tensor/state reproduction
or new GPU run was available here. A successful small CPU state test cannot
resolve compiler/device-precision differences in that execution. There was no
disk-restoration boundary in the original failure; source inspection does not
establish serialization as its cause. Both old paths used the same later-wrong
optimizer, so an internal optimizer mismatch is not established either.
The accepted guard remains max(1e-4,.005×abs(referenceMSE)); no tolerance changes
or bypasses were made. Any proposed GPU reproduction must retain the failure.

## Validation receipts

FINAL_VALIDATION_RESULTS

Generated evidence lives outside Git in `/workspace/artifacts/overnight-batch1/`:
full isolated per-module receipts/log hashes, final pytest output, mutation
checks, simulator/reference checks, before/after regressions, compatibility
receipts and the delivery checksum manifest. Zero-test unittest modules are not
counted as pytest coverage. Test fixture networks and SGD derivative controls
are explicitly separate from production AdamW and full-size architectures.

The first broad invocation was intentionally stopped before completing the new
diagnostic provenance fix; it is retained as preliminary, not final coverage.
The first pytest invocation found pytest absent. Pinned pytest8.3.4,
pluggy1.5.0 and iniconfig2.0.0 were added to the test overlay without replacing
the pinned scientific stack; the subsequent receipts record actual execution.
A missing archived-pilot test input remains a skip, never an archive-hash PASS.

## Remaining qualification and historical impact

- No archived outputs or frozen commits were modified. No result is recomputed
  to improve its appearance; no pilot-return/loss explanation is proposed.
- The default n=1 dense FP32 update arithmetic is unchanged in the direct CPU
  compatibility check. Fixes change reset reuse, explicit non-default target
  handling and admission of invalid/mismatched states/configurations. They do
  not establish that any historical run encountered those cases.
- Legacy checkpoint tx cannot be inferred from Adam moments alone. Use original
  run_metadata/source/runtime provenance for such restores; the absence of a
  new sidecar is not proof of compatibility. Standalone checkpoint I/O is not
  claimed to roll back every in-memory component after arbitrary corruption.
- Nontrivial multi-step replay is now rejected, not scientifically implemented.
  Production-scale sparse/mixed-precision SAC variants and cross-device restore
  remain outside the new qualification; only the bounded sparse CPU restore
  contract has been added. Supported retrospective reconstruction is
  explicitly restricted to dense FP32 residual single critics.
- Minimum GPU follow-up, requiring separate owner authorization and fresh
  accounting: reuse exact saved failing-task tensors/starting weights and
  source-bound optimizer; run original monolithic, materialized-target
  monolithic, segmented continuation and an identical repeat on one A100 with
  recorded TF32/runtime pins. Retain initial target hashes, first-divergence
  index and complete parameter/moment/count snapshots at the split. Compare
  in-memory and disk-restored continuation only on identical numeric tasks.
  Preserve the existing guard and exact-repeat requirements; halt on failure.
  Repeating the old3e-4/0 execution, if later authorized, needs an isolated,
  explicitly labeled legacy-condition comparator; do not falsify parent metadata
  or disable PR #8's binding. Qualifying corrected1e-4/.01 fitting is a separate
  condition, not a retroactive PASS for the failed legacy execution.
  Full-size single/twin checkpoint/device behavior is a separate qualification,
  not inferred from these fixtures. No GPU allocation is authorized by this report.
