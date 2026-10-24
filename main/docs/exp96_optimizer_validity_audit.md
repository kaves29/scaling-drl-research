# Exp96 optimizer consistency and historical validity audit

Audit of frozen experiment `f597ee9898e4d401912092c336c3e5330cf2fda1`, diagnostic handoff `6af1bb463eba6d8856606bc040d003d1cb4df2be`, and PR #8 implementation `868cab4346f2f3124d1a459e34a2d66fed28c794`. Audit work is on `codex/exp96-optimizer-validity-audit-20261023`. No frozen training/probe implementation, optimizer setting, threshold or historical artifact was modified. No GPU job was launched.

## Principal result and evidence limits

**The late retrospective runner's wrong optimizer must not be conflated with the original 96-agent training or its in-training probe archive.** The frozen SAC constructor and original RunProbes both receive the resolved `sac_simba` agent configuration. All 96 canonical source configurations independently resolve to AdamW critic LR `1e-4`, decay `.01`; no environment/architecture/seed optimizer overrides occur in that queue. The late diagnostic runner selected `3e-4`/`0` from generic `sac.yaml`. It did not rewrite the original agents or archived probe curves.

The initial retrospective runner was actually correct-valued: `f297dd4` hardcoded `1e-4`/`.01`. Commit **`4f10ca9ef4e4d151f388d41fad775fa71b6bd38e` introduced the regression** by replacing those literals with a generic YAML lookup. Later `2c9f280`, executed `bcf6cf3`, and `5974706` inherited it. PR #8 fixes the selection/binding, not a demonstrated corruption of original SAC training.

Evidence levels must remain distinct:

| Evidence | What this audit verified | Limit |
|---|---|---|
| Immutable frozen executable and canonical queue | Read exact Git objects; byte-verified 17 source/config inputs; composed all 96 canonical configurations; exercised the real frozen agent constructor in a small CPU fixture | Source/configuration proof, not a read of 96 running processes |
| Selected original parent metadata | The local execution agent read pen-twirl D2W512 seed 5 metadata. Its source is frozen f597, identity matches, and recorded config hash `f29e24a325823316478db1459cfd2684a345293fb5dd105ad597a89a00d20262` exactly matches this audit's independently reconstructed canonical row | One selected original parent, not an independent population-wide checkpoint-state audit |
| Archived 96-row configuration reconciliation | Handoff reports all 96 archived configurations agree on `(critic LR, critic decay, actor LR, actor decay)=(1e-4,.01,1e-4,.01)` | The Mac-only `canonical_config_verification.csv` and all 96 raw metadata files were not imported into this Cloud task; that prior result is reported evidence, not newly recomputed here |
| Executed diagnostic job | Read the local worker's actual captured `results.json`, runtime, status and Slurm accounting. The recorded optimizer YAML SHA256 exactly matches the `bcf6cf3` Git blob | Runtime evidence for job 23120792; not for every hypothetical/manual invocation |
| Historical 64/64 and 96/96 counts | Traced documented provenance to original archived probes and verified their generating optimizer path | The 4,834 NPZs and Mac analysis CSV/scripts were not independently reprocessed here. Exact count/statistic replication remains a read-only evidence task, needing no GPU |

The supported original-training conclusion is therefore **consistent intended and recorded `1e-4`/`.01`, with no evidence of the late mismatch contaminating the original population**. Do not upgrade this into a claim that this Cloud session independently inspected every original optimizer state. Full 96-file runtime verification is still a stated coverage gap.

## Phase 1: executable optimizer trace

Every link below points to an immutable source revision. Line numbers refer to that revision, not a moving branch.

| Source/function | Exact reference | Trace |
|---|---|---|
| Canonical 96-row launcher, `rows` | [f597 exp96_campaign.py:52](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scripts/exp96_campaign.py#L52), lines 57–85 | Four environments × three architectures × seeds 1–8; explicit Hydra overrides and row hashes |
| Config selection | [base_exp12_ten.yaml:3](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/configs/base_exp12_ten.yaml#L3); [base_exp12.yaml:44](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/configs/base_exp12.yaml#L44) | Ten-target config inherits base_exp12, which selects `agent:sac_simba` |
| Actual parameter values | [sac_simba.yaml:15](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/configs/agent/sac_simba.yaml#L15), lines 15–22 | Actor and critic LR `1e-4`, decay `.01`; metadata/census support these values |
| Composition and metadata | [exp1.py:52](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp1.py#L52), lines 52–83 and 94–115 | Resolve all Hydra overrides; save resolved config, hash, source and launch provenance before constructing trainer |
| Agent creation | [trainer.py:73](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp12/trainer.py#L73); [agents/__init__.py:18](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/agents/__init__.py#L18) | Pass `cfg.agent` into SACAgent; observation normalization wrapper does not replace optimizer settings |
| Immutable SACConfig and constructor | [sac_agent.py:44](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/agents/sac/sac_agent.py#L44); [sac_agent.py:447](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/agents/sac/sac_agent.py#L447) | Constructor derives temperature target entropy, copies config into a frozen dataclass, initializes networks; it does not rewrite critic LR/decay |
| Critic AdamW creation | [sac_agent.py:165](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/agents/sac/sac_agent.py#L165), lines 165–175 | Explicit `optax.adamw(learning_rate=cfg.critic_learning_rate, weight_decay=cfg.critic_weight_decay)` |
| State creation/update | [networks/trainer.py:65](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/networks/trainer.py#L65); [trainer.py:88](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/networks/trainer.py#L88) | Initialize state once; `tx.update(grads, opt_state, params)`; apply updates and carry returned state/counter |
| Actual SAC update | [sac_update.py:131](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/agents/sac/sac_update.py#L131); [sac_agent.py:395](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/agents/sac/sac_agent.py#L395) | Each critic update calls that Trainer; scan carries the new critic, retaining moments/count rather than resetting each update |
| Checkpoint restore | [sac_agent.py:687](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/scale_rl/agents/sac/sac_agent.py#L687), lines 693–734 | Save critic parameters and optimizer state; restore into constructed Trainer structures. Static `tx` is not an independent serialized hyperparameter certificate |
| Original probe construction/storage | [run_probes.py:27](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp12/run_probes.py#L27), lines 34–38; [run_probes.py:148](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp12/run_probes.py#L148) | Build probe optimizer from `trainer.cfg.agent`; pass it through run_probe and save resulting NPZ curves |
| Probe AdamW and fresh state | [probe.py:212](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp12/probe.py#L212); [probe.py:78](https://github.com/kaves29/scaling-drl-research/blob/f597ee9898e4d401912092c336c3e5330cf2fda1/main/experiments/exp12/probe.py#L78) | Same scalar LR/decay; fresh Adam state per fit, carried continuously for the full 2k scan; it intentionally does not continue the SAC optimizer state |

The source search found no LR schedule, gradient clipping, environment-dependent critic optimizer replacement or per-seed optimizer override on this original path. Sparse and mixed-precision branches exist, but all 96 canonical configurations set critic sparsity zero and mixed precision false. Target critics use `tx=None` and Polyak parameter updates, which is intentional. Injection uses a separate optimizer path for intervention arms; it is not an override of the original 96-parent SAC training settings.

### AdamW semantics verified

Pinned Optax 0.2.3 constructs `scale_by_adam → add_decayed_weights → scale_by_learning_rate`: decay is decoupled from the loss gradient and is scaled by LR. Defaults are beta1 `.9`, beta2 `.999`, epsilon `1e-8`, eps_root `0`, mu_dtype `None`, nesterov false, mask `None`. Scalar LR is constant. With no mask, kernels, biases and LayerNorm parameters all receive decay; no kernel-only exception is implemented. This describes executable semantics, not an independent re-review of every SimBa paper implementation detail.

For zero gradient, one step is `theta_next=(1-lr*decay)*theta`. Under the campaign settings the factor is `.999999`, not `.99`. Over 2,000 zero-gradient steps it is approximately `.998002`; actual learning is nonlinear and must not be inferred from that illustration. Frozen Trainer checks verified this update on both kernels and nonzero biases and Adam counts 0→1→2. Further nonzero-gradient checks matched the frozen training wrapper against the independent Optax update for all 36 small control fits.

No population-wide claim is made about the contents/counters of every saved optimizer state; raw checkpoint-state verification remains distinct from correct executable state handling. Stored moments/count alone do not identify LR/decay, which live in the static transformation/configuration provenance.

## Phase 2: optimizer by pipeline and executed scope

| Pipeline/revision | Critic fitting optimizer | Evidence and scope |
|---|---|---|
| Original SAC training, frozen f597 | AdamW `1e-4` / `.01` | Executable trace + all 96 canonical configs + selected actual metadata match; reported archived 96-row reconciliation agrees |
| Original in-training fresh and current probes, f597 | Fresh-state AdamW `1e-4` / `.01` | `RunProbes` uses trainer config for both; original archive source for the historical analysis |
| Read-only retrospective statistics over original NPZs/CSV | No new optimizer | Handoff documents archive analysis and no new neural fits; actual Mac analysis scripts/CSV not freshly re-run in Cloud |
| First diagnostic draft f297dd4 | Hardcoded AdamW `1e-4` / `.01` | [initial constructor:71](https://github.com/kaves29/scaling-drl-research/blob/f297dd49088a3ef7184fe13c884fa3f7ce8d58b0/main/scripts/exp96_fixed_task_diagnostics.py#L71); aligned values, not metadata-bound |
| Regression 4f10ca9 and descendants through 6af1bb4 | AdamW `3e-4` / `0` | [introducing diff](https://github.com/kaves29/scaling-drl-research/commit/4f10ca9ef4e4d151f388d41fad775fa71b6bd38e); [executed bcf constructor:79](https://github.com/kaves29/scaling-drl-research/blob/bcf6cf3c90726c77f998e5795fedf0d62c5f11e2/main/scripts/exp96_fixed_task_diagnostics.py#L79) |
| Tiny diagnostic CPU contracts before PR #8 | Explicit AdamW `1e-4` / `.01` | [test_fixed_task.py:32](https://github.com/kaves29/scaling-drl-research/blob/5974706b3cabcaa2140f575f2e0712bd40ae236f/main/tests/test_fixed_task.py#L32); tested kernels, not the wrong Investigation constructor |
| Corrected PR #8 | Validated archived LR/decay; `1e-4` / `.01` for canonical metadata | Metadata-bound constructor regression and CPU tests; no corrected-source GPU execution yet |
| This audit's CPU controls | Four explicitly labelled optimizer conditions | Immutable numeric tasks, same starting parameters, frozen `_fit`, CPU float32; no production-size fitting |

Generic `sac.yaml` also contains different architecture/entropy/normalization defaults, but the late runner's `critic_optimizer` reads **only LR and decay**. Its critic architecture and saved normalization still come from the explicit residual model/checkpoint path. Do not claim that this bug changed the retrospective network to an MLP or changed original SAC entropy/normalization.

### Actual job 23120792

The captured runtime binds source `bcf6cf3c90726c77f998e5795fedf0d62c5f11e2`, JAX/JAXLIB/CUDA plugin `0.4.34`, one A100 and TF32. Its optimizer YAML hash is `49a050ea5b55baad3a2b4865ccdb21c85b6ac21077fcddc4e9a159ec1c188258`; this audit recomputed that hash from the immutable Git blob and matched it. Thus `3e-4`/`0` is supported by executed-artifact evidence, not just a nearby YAML.

Only one reconstruction result exists: **fresh D2W512 pen-twirl seed 5, round 0, check-50-derived task**. It ran a 2k frozen fit, an identical 2k repeat, and two continuous 1k segments before throwing. The repeat loss check passed on the observed code path. It did **not** reach trained/latest fitting, large-humanoid reconstruction, or the later architecture/state/target/normalization matrix. An `(8000,256)` saved index bank is not evidence that an 8k fit ran.

The wrong optimizer would affect both fresh and trained fits if those paths ran, because they share `self.tx`; only the first fresh path is observed executed. The old `fit` path measured initial gradient norm before optimizer updates, but the actual failed reconstruction did not call that instrumentation. First-update function-space measurements were not implemented in the old runner; PR #8 prepared them, but no production GPU values exist yet.

Two different discrepancies must not be merged:

- Against original archive: fresh final MSE `.1502770483` versus `.1423999518`, and maximum curve error `5.1054697037`. This is **not** an optimizer-controlled historical reconstruction, even though final error passed the old loose scalar guard.
- Within the wrong-config executable: segmented/monolithic curve error `.0512650609`, final error `.0040517300`, exceeding its unchanged final guard `.0007513852`. **Both internal paths used the same wrong optimizer.** This is not evidence that one internal path accidentally used `1e-4` and the other `3e-4`. LR-dependent numerical sensitivity, compilation/fusion/target-materialization differences and other numerical causes remain hypotheses.

The original snapshots/arrays were read functionally; the runner wrote separate diagnostic output, not original training states or probe archives.

## Phase 3: controlled CPU numerical results

Executed 36 fits: three small residual profiles × three fixed fixture seeds × four optimizer settings. Each profile/seed is one fixed task. Parameters, input/target tensors, 2,000 minibatch indices, architecture, seeds, dtype and update count are unchanged within each four-condition group. The frozen source's `_fit` executes a materialized own-offset target with an explicit zero extra offset; this removes optimizer-dependent target recomputation. A fixed small teacher generates the base once. These are small constructed tasks, not recovered original Exp96 data or production-size training.

Conditions are campaign `(1e-4,.01)`, legacy diagnostic `(3e-4,0)`, LR-only change `(3e-4,.01)`, and decay-only change `(1e-4,0)`. The extra two controls separate the effects of the two jointly changed hyperparameters; they are CPU demonstrations, not an authorized GPU sweep.

Backend CPU, Python 3.12.14, JAX/JAXLIB 0.4.34, Flax 0.8.4, Optax 0.2.3, NumPy 1.26.4, float32, x64 disabled; identical precision within all comparisons. The 4,529/19,417/34,081-parameter fixtures are vastly smaller than the actual pen-input critics: 4,271,105/33,722,369/75,749,377 parameters at dimensions obs=93/actions=39. Production counts were obtained by abstract shape evaluation, not allocating/fitting those networks.

All initial full-pool MSEs and initial gradient norms agree within each fixed-task group. The first gradient is computed once from the common objective and shared, as required for a pure optimizer control; frozen training-wrapper nonzero-gradient checks independently agree. Decoupled decay does not enter that loss gradient. First-step function-change RMS with the legacy setting is **2.929–2.999×** the campaign setting across the nine tasks. This is an observation on these fixtures, not a scaling law for full critics.

Medians over the three fixture seeds, with `R100=mean(loss[0:100])/loss[0]`:

| CPU profile / parameters | Optimizer LR / decay | First-step prediction RMS | R100 | Final full-pool MSE after 2k |
|---|---|---:|---:|---:|
| D2W16 / 4,529 | `1e-4` / `.01` | .0149733 | .802328 | .137585 |
| D2W16 / 4,529 | `3e-4` / `0` | .0448651 | .582932 | .000310907 |
| D4W24 / 19,417 | `1e-4` / `.01` | .0239668 | .527214 | .00000556253 |
| D4W24 / 19,417 | `3e-4` / `0` | .0702011 | .374167 | .000740713 |
| D4W32 / 34,081 | `1e-4` / `.01` | .0307016 | .724390 | .000000126872 |
| D4W32 / 34,081 | `3e-4` / `0` | .0899677 | .540355 | .000206237 |

Both configurations had one individual R100>1 task among these nine (the same small-profile seed); neither reproduces the production medium/large pattern. All 36 curves and final MSEs were finite. Some higher-LR proxy trajectories reached low minibatch error and later rebounded (visible in the raw curves); finite values are not a convergence certificate. These are minibatch observations, not saved intermediate full-pool measurements or proof of a floating-point failure. A repeated execution produced exactly identical arrays in all 54 numeric archives (tasks, initial parameters and condition outputs); this is a determinism check, **not another set of independent replications**.

The LR-only/decay-only results and all 72,000 raw pre-update losses are exported alongside the complete curves. They show why neither “higher LR always causes early amplification” nor “correcting the optimizer must improve final fit” follows from this discrepancy. The legacy condition improved final error in one proxy family and worsened it in the other two. Long-run decay effects can be nonlinear even when its direct first-step contribution is small.

Only `loss[0]` is optimizer-independent for a fixed start/task. `loss[t]` is measured before update t+1, **after t previous updates**. Calling the whole curve “pre-update losses” does not make its first-100 mean or later trajectory independent of the optimizer.

Reproduce using the authenticated engineering executor, not by making the owner run commands:

```bash
# Export frozen source to a fresh directory outside Git first:
set -euo pipefail
mkdir /ABSOLUTE/FRESH_SNAPSHOT
git archive f597ee9898e4d401912092c336c3e5330cf2fda1 | tar -x -C /ABSOLUTE/FRESH_SNAPSHOT
python -B scripts/audit_exp96_optimizer_consistency.py \
  --repository /ABSOLUTE/REPOSITORY --frozen-main /ABSOLUTE/FRESH_SNAPSHOT/main \
  --out /ABSOLUTE/NEW_CENSUS --census-only
python -B scripts/audit_exp96_optimizer_consistency.py \
  --repository /ABSOLUTE/REPOSITORY --frozen-main /ABSOLUTE/FRESH_SNAPSHOT/main \
  --out /ABSOLUTE/NEW_CPU_CONTROLS
```

The script enforces CPU, frozen dependency pins, source bytes and new output directories. The real frozen constructor is tested with explicitly resized networks to bound memory; its actual three AdamW calls were actor `(1e-4,.01)`, critic `(1e-4,.01)`, temperature `(1e-4,0)`. It does not build 96 full agents or instantiate simulators.

## Phase 4: finding-by-finding validity assessment

The classification concerns **this optimizer mismatch**, and does not substitute for scientific interpretation or independent reanalysis of every raw historical statistic.

| Finding | Classification | Technical reason / remaining limit |
|---|---|---|
| Original 96 agents' intended training optimizer | Unaffected by optimizer mismatch | Original config/constructor path selects `1e-4`/`.01`; canonical census and selected metadata agree. Full 96 raw runtime/state audit remains a coverage gap |
| Reported 64/64 medium/large baseline early amplification | Unaffected by optimizer mismatch, given documented archive lineage | Original archived probe curves were generated by the intended optimizer, not the late runner. Exact 64-count has not been recomputed from the raw Mac archive here |
| Interpretation of those baseline curves as training-induced intrinsic plasticity loss | Potentially confounded / not established | Baseline precedes SAC updates; first-100 dynamics depend on optimizer, initialization, architecture and task. Correct intended optimizer can still produce finite-budget amplification; this bug is not its demonstrated cause |
| Reported 96/96 longitudinal fresh-reference deterioration | Unaffected by optimizer mismatch, given documented archive lineage | Descriptive statistic uses original intended-config probe fits. It is **fit-dependent**, but not contaminated by a different later program. Exact count not freshly recomputed; latest endpoints vary and are not all check 50 |
| Interpreting longitudinal fresh-score deterioration as loss of fresh weights' intrinsic capacity | Potentially confounded | Fresh weights are fixed while replay, RMS, teachers and minibatches change; source supports a changing-task phenomenon, not an isolated causal contribution |
| Pen-twirl initial trained minibatch MSE ~167178 / ~110655 | Unaffected by optimizer mismatch | First loss is computed on the existing starting checkpoint before the fitting optimizer updates it. It still depends on historical SAC training and the chosen inputs/targets, which must remain bound |
| Original pen-twirl final fitting errors and score/return associations | Unaffected by late mismatch, given archive lineage | They depend on original fits, which used intended settings. Association and finite-horizon errors do not prove intrinsic inability or causal actor damage |
| Original archived returns, replay geometry and RMS comparisons | Unaffected by optimizer mismatch | No new diagnostic optimizer update is involved; this does not independently certify every previous statistical calculation |
| Old runner initial MSE, target variance, offsets, sampled input geometry and pre-update gradient norm for a fixed start/task | Unaffected directly | Computed before new optimizer updates. Production first-gradient instrumentation was not reached in job 23120792; hypothetical capability is not measured evidence |
| Job 23120792 new 2k curve/final MSE as a reconstruction of the intended historical optimizer | Invalid for the intended optimizer-controlled comparison; requires reproduction | It used `3e-4`/`0`. Preserve it as explicitly labelled alternative-condition evidence, not a reproduction certificate |
| Job 23120792 segmented-vs-monolithic failure | Requires reproduction under corrected settings; cause unresolved | It is a valid observed internal numerical discrepancy under the old config; both paths shared that config, so it cannot be attributed to an internal LR mismatch |
| Claimed new GPU architecture/state/shared/incremental/normalization contrasts from that job | Not enough evidence to classify | They were not executed before the first failure; no such results exist in the observed job |
| Existing 13 Tiny CPU contracts | Unaffected within their scope | Explicit correct optimizer, but they did not cover the old runner constructor or production A100 behavior |
| PR #8 corrected production behavior | Requires GPU reproduction | CPU tests validate binding/engineering contracts, not corrected full-size TF32 trajectories or scientific conclusions |

**Direct answers:** the documented 64/64 and 96/96 findings were not computed by the known wrong-config GPU runner; source/provenance supports their original intended optimizer. This audit does not freshly certify their numerical counts. Extreme initial pen losses are directly independent of the subsequent fitting optimizer. None of these observations licenses claiming the mismatch explains the original pathology, or that correct settings remove the historical amplification.

## Additional engineering issues and limitations

1. **Configuration-selection regression and test gap:** matching the generic upstream SAC file was not evidence of matching the campaign-selected configuration. Early Tiny tests injected their own correct transform, bypassing Investigation.__init__. PR #8's constructor regression addresses this specific gap.
2. **Historical reproduction guard was insufficient by itself:** variance/offset/final-MSE bounds allowed a run with maximum curve error 5.10547 to proceed. Full-curve error was recorded but not an acceptance operand. Matching moments or a loose final loss cannot establish historical target identity. PR #8 intentionally preserved the old tolerance; report it as an engineering guard, not a scientific validity certificate. A justified full-curve rule needs explicit review; no threshold is silently changed here.
3. **Task IDs are provenance-rich, not optimizer-invariant numeric IDs:** PR #8 `bank_task` includes optimizer hash and diagnostic source in task provenance, and FrozenTask hashes that provenance. Thus byte-identical numerical arrays under old/new optimizers can have different task IDs, and the current identical-ID pairing API is not a drop-in cross-optimizer comparator. This audit uses an optimizer-independent array manifest/task ID and records optimizer as a separate condition. Preserve old IDs; do not relabel a ten-target optimizer contrast as the original fresh-vs-current IQM10 metric.
4. **Stale budget evidence:** remote budget.json still said zero after the failed allocation. The actual 181 allocated seconds, including idle/I/O/compilation, count. Old kernel timings are not the allocation bill. No historical file was overwritten.
5. **Population evidence gap:** raw archived 96-row config inventory and all original checkpoint-state records remain outside this Cloud task. A complete independent runtime/population re-audit needs read-only access/import, not more GPU time. Current tools can read the local worker's captured outputs but cannot dispatch fresh commands to it.
6. **Segmented GPU equivalence unresolved:** no causal diagnosis or corrected-source A100 pass is inferred from small CPU tests. Cross-allocation fit-state resume is also not implemented; curves alone do not contain resumable optimizer state.

## Phase 5: smallest proposed GPU follow-up — not authorized or submitted

No GPU experiment is needed to establish the regression's source or to re-count archived curves once the raw files are accessible. Before allocating GPU time, independently re-check the archived 64/64 statistic and all 96 metadata/config bindings through read-only CPU work where possible. If the owner wants a controlled full-size optimizer-sensitivity test, the minimum useful screen is:

| Item | Proposal |
|---|---|
| Checkpoints | Fresh critics at `/projects/biqc/skaveti1/exp96/campaign_owner_f597ee9_20261019T230748Z/campaign/checkpoints/exp1/{D2W512,D4W1024,D4W1536}/myo-pen-twirl/seed_5/fresh_critic`; corresponding parent metadata, original `probes/check_00.npz` and replay prefix/RMS evidence. Verify existence/leaf hashes before admission |
| Why fresh/baseline | The reported 64/64 phenomenon occurs before SAC training. Trained checkpoints or full SAC reruns are unnecessary for this narrow question |
| Fixed tasks | Materialize the original check-0/round-0 recipe once **per architecture**, using its frozen fresh critic, keyed replay indices, teacher/base, initial-normalizer reconstruction and first 2k minibatches. Save exact obs/actions/targets/indices before either optimizer. Each old/new pair gets identical arrays and starting parameters. These are three fixed paired tasks; not a claim that different-architecture teachers produce one common task |
| Historical identity limit | Initial RMS/teacher/task reconstruction is retrospective; absent original teacher/target tensors prevent an automatic exact historical identity claim. Compare corrected curves to archive, retain all differences and separate reconstructed-task controls from certified archive replication |
| Conditions | AdamW `(1e-4,.01)` versus `(3e-4,0)` only; same starting state, fresh moments per fit, network, target, minibatches, 2k horizon and A100/TF32 float32 policy. Six monolithic fits; one exact repeat of the corrected large fit gives seven fits total |
| Kernel | Use the frozen production monolithic `_fit`, not an unqualified segmented kernel. This avoids making the sensitivity test depend on resolving a separate segmentation failure first. Separately qualify segmentation before relying on extended segmented production fits |
| Measurements | Shared pre-update full/minibatch MSE and gradient norm; first-update function-change RMS/mean and pool MSE; all 2k pre-update losses; original R100 and early peak ratio; final pool MSE; finite-value checks; source/task/parameter/optimizer/precision IDs; compile/execute/allocated seconds and peak memory |
| Engineering acceptance | Exact within-pair numeric/start hashes; fresh Adam state and 2000 updates; complete finite outputs; source/precision binding; repeat corrected-large losses under the existing exact-repeat rule. Preserve failures; do not relax historical or segmentation guards |
| Scientific readout | Does R100>1 (the historical descriptive definition) remain in the corrected medium/large case? Report paired changes versus measured repeat noise. One seed/one target cannot reproduce 64/64 or establish a mechanism. A larger first step or lower final error is not a scientific PASS. Owner determines any replication/inference criterion |
| Time/resources | One A100, 16 CPUs, 60 GiB. Provisional workload estimate 6–15 allocated minutes, **not benchmarked for the larger critics**; reserve at most 15 minutes (.25 physical GPU-hours), with an 8-minute work cap and bounded shutdown. Stop if compilation/timing/memory threatens the cap; no automatic retries or extension |
| Basis for estimate | The old small-critic 2k kernel took .782936 s, compile 1.422341 s, but the whole failed allocation consumed 181 s. Medium/large cost is unmeasured; do not extrapolate the small kernel as a guaranteed runtime or discard setup/idle time |
| Budget | Last independently reconciled investigation usage: 181/14400 GPU-seconds, 14219 seconds remaining. A full new 900-second reservation would bring accounted-plus-reserved time to 1081 seconds and leave 13319 seconds (3h41m59s). Reconcile all new attempts and reservations immediately before submission; this is not fresh live accounting |

A strict common-target comparison across architectures would require choosing a shared teacher/offset objective and would answer a different conditional question from the original architecture-specific detector. That is a scientific design decision for the owner, not an automatic replacement. Likewise, separating LR and decay on full-size GPUs adds conditions and needs separate approval; the small CPU factorial does not authorize it.

Expected outputs are per-condition `task_manifest.json`, immutable numeric arrays, `losses.npz` with exactly 2000 losses and final pool error, a CSV of step/condition metrics, optimizer/counter/state/provenance receipts, and reconciled Slurm accounting. Keep original artifacts and the failed job evidence intact. No broad matrix, new SAC training, GPU scheduling or scientific redefinition was performed during this audit.

PR #8 deliberately rejects optimizer overrides that conflict with original metadata. The proposed old-versus-campaign GPU screen therefore needs an isolated, explicitly labelled sensitivity comparator after scientific approval; do not falsify metadata or disable the production binding to run it. The CPU comparator delivered here is not a GPU execution tool.

## Retained audit evidence

Cloud task evidence is under `/workspace/artifacts/exp96-optimizer-audit/`: `canonical-verified/census.json`, the 96-row CSV, `cpu-controls-verified/comparison.json`, `comparison.csv`, `loss_curves.csv` (72,000 rows), per-case task/parameter/condition NPZs, `cpu_optimizer_controls.png`/`.pdf`, `source-guard.json`, `determinism-check.json`, `audit-evidence.json`, and captured `job23120792-results-from-worker.json`. This directory is task evidence, not a promise of restoration elsewhere. The delivery provides a compact downloadable package; numeric arrays remain outside Git.
