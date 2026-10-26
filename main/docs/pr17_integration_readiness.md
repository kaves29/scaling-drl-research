# PR17 approved-integration readiness audit — 26 October 2026

**Current PR17 must not be retargeted wholesale.** A clean alternative-only candidate
can avoid the later runtime changes; its exact source and CPU results are recorded below.
PR17 itself remains unchanged at `a6958337f411c30f6ebe1a1b3d5c5a0403287780`. No job, merge, force-push or integration
promotion is authorized or performed by this audit.

Dated raw evidence: `/workspace/artifacts/pr17-integration-readiness-20261026/`.
Raw filenames below refer to that directory. The tracked proposal receipt is
`main/diagnostics/sequential_integration_readiness_receipt.json`; the existing
PR17 receipt remains associated with its original source and is not rewritten.

## Approved integration and instruction drift

- Repository-designated shared development source: **`integration/exp12` at `844e3c0cc24998b25c3d1e667bcbfe056b1b336c`**.
  Both live API and Git refs agree. [Its AGENTS.md lines28–29](https://github.com/kaves29/scaling-drl-research/blob/844e3c0cc24998b25c3d1e667bcbfe056b1b336c/AGENTS.md#L28)
  say to start from the latest approved `origin/integration/exp12` head.
- [Approved handoff](https://github.com/kaves29/scaling-drl-research/blob/844e3c0cc24998b25c3d1e667bcbfe056b1b336c/main/docs/AGENT_HANDOFF.md#L6)
  identifies reviewed executable checkpoint `78d04d167a5f65c35f6905cb261876bfe709d85a`.
  Comparing that checkpoint with published `844e3c0cc24998b25c3d1e667bcbfe056b1b336c` changes only the handoff/status
  documents: no executable/configuration change was found. This is engineering
  integration authority, not GPU, scientific-pilot or production-launch approval.
- `main` remains `087feab341b9be1f377697668f2d1c750ec0467f`, a different, later
  five-target scientific baseline. Its AGENTS.md also names integration/exp12.
  A newer branch date does not make it the approved integration target.
- PR17 base: draft PR16's `codex/source-bound-discrepancy-pilot` at
  `57d2beb78be7cf0e1145f4a9479082a07924cd16`.
- The original `main/diagnostics/sequential_base_dependencies.json` has valid
  hashes for all16 source bindings, but compares against **main**, not approved
  integration. It must not be used as evidence of integration readiness. Its
  “unchanged from main” status for state.py also masks a difference from integration.
- PR17 inherited the workflow wording introduced on the consolidation review
  lineage (`c44a7e45cc8c4be85eac78e1e27784ea8dbaa2c7`). That wording is not in the
  approved integration or current main instructions. The earlier main-based
  readiness assessment conflated scientific baseline with integration authority.

## Dependency assessment: file differences are not necessary dependencies

The conservative static closure contains 118 repository modules,
including implicit package initializers, lazy imports and default/fork branches.
The19 imported Python files below differ from approved integration. **None of
these later deltas is required by the standalone alternative's current API.**
Keeping the existing stacked versions, however, keeps their later import dependencies;
changing only PR17's base would not remove them.

All five numerical helpers are AST-identical, including decorators:
`_mean_prediction`, `_chunked`, `critic_optimizer`, `_fit`, `_base_targets`.
The alternative uses the first three directly; the last two supply regression
references. Its own fit loop, stored phase indices, scores and bank implement the
explicitly requested new sequential diagnostic, not later ten-round IQM/trigger logic.

| Differing runtime file | Last change in PR17 ancestry | Does the alternative require the later change? |
|---|---|---|
| [main/analysis/campaign/__init__.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/analysis/campaign/__init__.py) | `54aeca0921e78d60b334016dbb9390819506367e` | No. Package initialization for later campaign instrumentation; no required alternative API. |
| [main/analysis/campaign/ingest.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/analysis/campaign/ingest.py) | `54aeca0921e78d60b334016dbb9390819506367e` | No. Only reached through later provenance instrumentation; alternative uses its own identity/receipts and no campaign ingestion. |
| [main/analysis/campaign/provenance.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/analysis/campaign/provenance.py) | `20d8c6b47cf9ec5e62f1d35190ea48b5fe115a5b` | No. DiagnosticSession is only a later instrumentation dependency; not required for alternative-owned bank/reference/phase provenance. |
| [main/experiments/exp1.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/experiments/exp1.py) | `20d8c6b47cf9ec5e62f1d35190ea48b5fe115a5b` | No. compose_config exists unchanged on approved integration; default delegation should retain approved run. Later decorator, stale-fresh cleanup, round metadata and fork extension are unnecessary for the alternative-only route. |
| [main/experiments/exp12/diagnostics.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/experiments/exp12/diagnostics.py) | `12839f656e53b00a3327384f2cb4fc1be47e73fe` | No. Later recording-on/off compilation policy is not used by alternative code. Approved ActorDiagnostics and DiagnosticPendingUpdateMetrics interfaces suffice; candidate comparisons use the approved behavior. |
| [main/experiments/exp12/fork.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/experiments/exp12/fork.py) | `d65f9bb798f5c5c74b1857a6329669e068ddf517` | No. Later atomic save_npz is fork/default-route infrastructure. Alternative rejects forks and commits its own phase files; approved default route remains untouched. |
| [main/experiments/exp12/instrumentation.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/experiments/exp12/instrumentation.py) | `f8ad82f064547c76e12539804c98be75b32656d8` | No. Later opt-in lifecycle recorder is not invoked by alternative-owned modules. Approved exp1/trainer versions do not import it. |
| [main/experiments/exp12/ledger.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/experiments/exp12/ledger.py) | `5fe2ff65ae96cf900b23bf9810f79464730def9d` | No. Later ten-round ledger/schema behavior is unused: alternative emits its own JSON/NPZ and no original-probe ledger. |
| [main/experiments/exp12/probe.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/experiments/exp12/probe.py) | `5fe2ff65ae96cf900b23bf9810f79464730def9d` | No. Imported _mean_prediction, _chunked and critic_optimizer are AST-identical. _fit and _base_targets used by regressions are also identical. Later IQM and two-part index generation are not called by the alternative. |
| [main/experiments/exp12/run_probes.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/experiments/exp12/run_probes.py) | `5fe2ff65ae96cf900b23bf9810f79464730def9d` | No. Alternative-only runner never constructs it. Later ten-round W&B transport is unnecessary; disabled/default path retains approved RunProbes. |
| [main/experiments/exp12/state.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/experiments/exp12/state.py) | `d65f9bb798f5c5c74b1857a6329669e068ddf517` | No. latest_state_dir/save/restore APIs already exist. Later fsync strengthens host-crash durability, not logical process-interruption state restoration. Do not transfer a stronger power-loss guarantee to the candidate without approving that fix. |
| [main/experiments/exp12/trainer.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/experiments/exp12/trainer.py) | `1d849c3b52f1b6bdc2514f8972a2388021b929c7` | No. All used start/train/before_first_update/save/restore/_sac_agent interfaces exist. Later success attribution and optional instrumentation/critic metrics are not required by the alternative. Clean candidate deliberately retains approved evaluation behavior. |
| [main/experiments/exp12/trigger.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/experiments/exp12/trigger.py) | `5fe2ff65ae96cf900b23bf9810f79464730def9d` | No. Later ten-round fractional-IQM bootstrap is unused by alternative: no trigger or K aggregation; forks are rejected. |
| [main/experiments/exp2_arm.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/experiments/exp2_arm.py) | `20d8c6b47cf9ec5e62f1d35190ea48b5fe115a5b` | No. Only conditional/default experimental paths; alternative does not execute an Exp2 arm. Later instrumentation is unnecessary. |
| [main/scale_rl/common/logger.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/scale_rl/common/logger.py) | `1d849c3b52f1b6bdc2514f8972a2388021b929c7` | No. Existing ordinary logger and get_architecture_id suffice. Later update_critic_diagnostics is called only by later optional observer code, not the alternative. |
| [main/scale_rl/diagnostics/__init__.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/scale_rl/diagnostics/__init__.py) | `f08c5b189e2b53d36b22d06cf165b8ede1f77e15` | No. Initialization for the later optional critic-measurement package; not needed on approved trainer. |
| [main/scale_rl/diagnostics/critic_metrics.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/scale_rl/diagnostics/critic_metrics.py) | `f08c5b189e2b53d36b22d06cf165b8ede1f77e15` | No. Later opt-in reference critic measurements are not used by the alternative fitter; approved trainer has no import. Retaining them would separately require their integration approval. |
| [main/scale_rl/diagnostics/wandb_logging.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/scale_rl/diagnostics/wandb_logging.py) | `1d849c3b52f1b6bdc2514f8972a2388021b929c7` | No. Later critic-measurement transport is unnecessary; alternative arrays are local and ordinary approved SAC logging suffices. |
| [main/scale_rl/networks/layers.py](https://github.com/kaves29/scaling-drl-research/blob/a6958337f411c30f6ebe1a1b3d5c5a0403287780/main/scale_rl/networks/layers.py) | `f08c5b189e2b53d36b22d06cf165b8ede1f77e15` | No. Only added sow calls differ. Alternative requests params/Q, not the critic_diagnostics_v1 mutable collection; no arithmetic or parameter routing requires the added capture sites. |

The five existing bindings unchanged even relative to approved integration are
`envs.py`, `precision.py`, `sac_agent.py`, `sac_update.py` and `sac_network.py`.
This is byte equality, not a claim that arbitrary future configurations are qualified.
Full paths/hashes, import edges and helper source line ranges are in
`dependency_audit.json`; exact imported paths are independently checked in
`import_paths.json` to exclude a fallback into another checkout.

Two configuration bindings also differ:

| Configuration | Assessment |
|---|---|
| `base_exp12.yaml` | Approved integration retains its earlier project name and1000-step default independent probe. Do not edit that file. The explicitly authorized alternative needs `probe.steps=2000` in its dedicated request and never constructs the independent probe. The CPU fixture already explicitly supplies2000. This does not promote later Method C defaults into integration. |
| `base_exp12_ten.yaml` | Absent from approved integration. Only the new lifecycle fixture's config-name choice required it. The candidate changes that one new-test string to `base_exp12`, retaining all explicit numeric overrides. Ten rounds/50 original checks are not used by the alternative, so importing that unintegrated YAML is unnecessary. |

Conditional feature/approval dependencies must remain explicit:

- To retain later provenance lifecycle recording, separately integrate/review
  the PR11 implementation (`20d8c6b47cf9ec5e62f1d35190ea48b5fe115a5b` and
  subsequent `f8ad82f064547c76e12539804c98be75b32656d8`), rather than copying it
  as an undocumented dependency. The alternative already has its own source/
  bank/reference/phase provenance.
- To retain later reference critic metrics and W&B mask transport, separately
  integrate/review PR14's `f08c5b189e2b53d36b22d06cf165b8ede1f77e15` and
  `1d849c3b52f1b6bdc2514f8972a2388021b929c7`. They are not alternative fitter requirements.
- Ordinary evaluation is preserved from approved integration, not from the newer
  stack. Later terminal-success attribution (`89cb004d501a3c1692f21802247091fae259999c`)
  is not silently backported. This can affect the success metric and must be
  disclosed in any source comparison; this audit does not repair the approved source.
- Later state/fork fsync strengthening (`d65f9bb798f5c5c74b1857a6329669e068ddf517`)
  is not silently backported. Logical process-interruption restore passes do not
  establish host-power-loss durability. Alternative-owned phase commits retain
  their own fsync/atomic implementation, but the stronger inherited training-state
  durability claim cannot be transferred to this candidate.
- Later actor recording-mode compilation policy (`12839f656e53b00a3327384f2cb4fc1be47e73fe`)
  is not needed to implement alternative isolation on the approved trainer. No
  equality claim across those two different baseline policies is made.

These are features outside approved integration, not a claim that every one is
scientifically forbidden or that an open PR can never have owner approval. No
record promoting these exact runtime versions into the shared integration was found.

## Minimal safe plan; not an executed PR retarget

A base-only retarget of current PR17 exposes269 changed files,51932 insertions and
427 deletions relative to approved integration. It would include unrelated source,
configuration, analysis and workflow changes. Do not do that.

1. Pin the approved integration and PR17 source SHAs above; preserve PR17's head/history.
2. Start an isolated clean branch at the approved SHA. Copy only the four new
   diagnostics Python files, three new test files and tiny CPU example from PR17.
   No approved production, configuration, methodology or instruction file changes.
3. Port only the new lifecycle fixture's configuration name as described above.
   Reuse all other runtime/test bytes, budgets, thresholds and acceptance assertions.
4. Rebuild active provenance/binding/validation documents against the actual
   integration source. Retain PR17's old141-test receipt as source-specific history;
   do not relabel it as validation of the clean source. Preserve the unapproved
   smoke/grid conditions and flags if carrying the proposal forward.
5. Review the clean source profile and CPU receipt. Optional later features above
   require separate exact-source approval/integration if they are desired.
6. Under the current no-force-push/no-merge constraint, a clean successor branch
   is the reviewable route. GitHub cannot switch an existing PR's head branch;
   retargeting its base alone is unsafe. No replacement PR, head rewrite, base edit,
   merge or force push has been performed. A later owner decision can choose how
   to replace/supersede PR17 after reviewing this proposal.

Prepared candidate branch: `codex/pr17-integration-readiness-20261026`.
Tested implementation: `00ed566e4c7f00c784b61d98afc9768e89906665`.
It adds eight files and modifies **zero** baseline files; its only code difference
from PR17's eight corresponding Python files is the one new-test config-name string.
`clean_candidate.patch` preserves the exact proposed code diff.

## Exact-source CPU verification

Current completed regression result: **115 passed,0 failed,0 skipped;227 warnings**.
That includes all69 alternative cases and46 applicable source-resident regressions.
Module counts: `{"test_exp12_diagnostics": 10, "test_exp12_methodology_corrections": 22, "test_exp12_metric_checkpoint_contract": 3, "test_exp12_probe": 9, "test_exp12_twin_reporting": 2, "test_sequential_mutations": 6, "test_sequential_probe": 50, "test_sequential_runner": 13}`.

Six modules in PR17's former141-case command are absent from approved integration:
`test_method_c_ten_reference`, `test_probe_discrepancy_adapter`,
`test_forensic_probe_audit`, `test_campaign_instrumentation`, `test_exp12_method_c`,
`test_exp12_review_fixes`. Their45 cases were not imported, disabled, or counted as
passing. The source-adapted command instead includes approved probe/actor-diagnostic
regressions. This is115 cases on the candidate, not a repeated141-case claim.

Final complete stages: **74/74 mutation detections/restorations in each of two
full runs;65/65 simulator restoration cells; real-critic example and independent
16-phase/48-arm archive reload; Black, Pyflakes, compileall and diff checks all
pass**. Independent reload performs no new fit. All115 regressions pass with
227 retained dependency warnings. No test threshold/assertion was weakened.

Exact commands, environment flags, working directory, exit statuses, timings,
source/base hashes and log checksums are in `readiness_receipt.json` and
`execution_receipt.json`. No broader/absent module is claimed as tested.
All model execution is tiny CPU engineering validation, not a campaign,
scientific experiment or GPU qualification. The tested commit is the clean
candidate, not a rerun of the old PR17 stack. Any subsequent proposal documentation
commit leaves its executable/test bytes unchanged.

## Five-job engineering smoke: exact proposal, not execution evidence

The unchanged proposal in PR17 is:

| Job | Environment | Critic | Training seed |
|---|---|---|---|
| 1 | dog-run | D2W512 | 0 |
| 2 | dog-run | D4W1024 | 0 |
| 3 | dog-run | D4W1536 | 0 |
| 4 | myo-pen-twirl | D2W512 | 0 |
| 5 | myo-pen-twirl | D4W1536 | 0 |

This covers five of the six cells in a2-task ×3-size rectangle: **Myo pen-twirl
D4W1024 is missing**. All three sizes appear only in dog-run. Each represented
cell has one development seed0, not five independent seeds per cell. There are
two environment/seed blocks with shared banks across sizes. K1–8, three arms and
repeated checkpoints are not additional independent training seeds.

If the unchanged full nominal cadence is approved: initial observation after5000
warmup interactions plus50k,100k,…,500k gives11 observations/job. Each observation
has8 targets ×3 arms ×2000 updates =48000 probe updates;528000/job and2640000
across five jobs, in addition to SAC training. The approved independent probe is
not executed in the alternative-only route. No shorter smoke endpoint, seed,
teacher, eligibility rule or resource cap was adopted by this audit.

Engineering uses: exercise source/config/parameter routing at the included sizes,
shared banks, target/index hashes, three-arm fitting, artifact completeness and
restore behavior. Limits: no balanced scaling estimate, no seed-level uncertainty,
no general DMC–Myo claim, no proof of a plasticity mechanism or favorable outcome,
and no measured full-size GPU memory/runtime/throughput/storage qualification.
The CPU lifecycle fixture uses tiny hopper networks and is not execution of these
five configurations. The30-parent alternative proposal is separately2 tasks ×3
sizes ×5 matched seeds, not five jobs; it is also unapproved.

The smoke remains engineering-only, **approved=false, executable=false,
submitted=false**. Teacher choice, task/seed eligibility, cadence and resources
remain proposals requiring owner decisions. Walltime/GPU-hours are unset; no
capacity guarantee follows from these CPU checks.
