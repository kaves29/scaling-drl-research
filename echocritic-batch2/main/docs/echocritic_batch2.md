# EchoCritic Batch 2 offline infrastructure

Base: PR #10 head `c05d77cc9899e50e857c170faf9937e5635c3663`, stacked on #9 `55166a643c979240510f3bdb76484c9777d72fee`, on #8 `868cab4346f2f3124d1a459e34a2d66fed28c794`. Branch `codex/echocritic-batch2-20261024`. No earlier PR is merged. The original A100 segmentation failure remains unresolved; this CPU work grants no GPU qualification.

## Usage

From `main/`, use the repository's pinned CPU environment (NumPy 1.26.4, JAX/JAXLIB 0.4.34, Flax 0.8.4, Optax 0.2.3). Offline-only analysis additionally uses SciPy, pandas and Matplotlib. No network, GPU or live W&B dependency is used by these commands. Output directories must be new.

```bash
python -m analysis.campaign.pipeline /absolute/path/complete_export.zip --out /absolute/path/new_analysis --bootstrap-reps 10000
python -m analysis.campaign.local /absolute/path/metrics.csv /absolute/path/probe_curves.npz --out /absolute/path/new_legacy_import
python -m scripts.campaign_example --out /absolute/path/new_synthetic_example
python -m unittest tests.test_campaign_offline -v
```

The ZIP contains `<any-prefix>/<record>/metadata.json` and `history.jsonl`. ID is `id` or `run_id`, never display name. Metadata supports `config` (including W&B `{value: ...}` entries), `identity`, `code: {commit, dirty}`, `state`, `campaign: exp96|pilot`, `step_key`, `resumed_from`, `expected_checkpoints`, `expected_metrics`, and optional `provenance`. Config must supply `seed`, `env_name`, architecture (or critic dimensions), and `num_interaction_steps`. The default counter is `interaction_step`; `_step` is never silently substituted. Exporter adapters must explicitly select the correct counter and resolved configuration. Source and campaign declarations are assertions from metadata, not independently certified facts. No membership is inferred from the expected 103/96/7 counts or names. The fixture's counts demonstrate capacity, not canonical grid validation.

Outputs: `validation.json` (SHA256, all retained records, quality flags, unresolved relationships), `statistics.json` (individual observed summaries, eligible group estimates, exclusion reasons), and PDF learning curves. Nonfinite values are flagged and represented as JSON null in the normalized report; original ZIP bytes remain unchanged. Duplicate IDs, duplicate ZIP paths, ambiguous nominal conditions, absent histories, missing seeds, bad step counters, missing declared checkpoints/metrics, source uncertainty and partial/crashed records are reported. Malformed JSON/configuration aborts instead of silently skipping a record. Explicit restart links are retained but never authorize history stitching. Multiple records per condition require human or separately validated reconciliation.

## Statistical contract

Training seed is the independent unit. Exact config except top-level seed, source declaration, campaign, architecture and environment partition groups; nested seeds/storage differences conservatively split groups rather than broadening compatibility. Duplicate nominal conditions are excluded from aggregate estimates regardless of display name. Only finished runs spanning step zero through their exact declared horizon enter default aggregate estimates. Partial/crashed trajectories retain descriptive observed endpoints/AUC and explicit exclusions; they are never assigned zero. Complete-only sample size and exclusion count support an initial missing-data sensitivity audit. Population bounds require a scientifically justified return range and are not invented.

Mean, median, IQM, sample SD and deterministic percentile bootstrap CIs resample one value per seed. IQM uses SciPy trim_mean(.25), including its finite-sample trim semantics. A single seed has no SD/CI. Paired comparisons require explicit seed-keyed inputs and report unmatched seeds. Pairing is a caller-authorized assumption, not proof that equal seed integers share RNG coupling. Longitudinal utilities reject multiple observations for the same seed/step and report each step's n. No evaluation episode, probe round or checkpoint becomes a seed. Intervals are pointwise descriptive 95%, not simultaneous or multiplicity-adjusted confirmatory tests.

Final window: observations at >=90% of declared horizon and <=horizon, with observation count. AUC: observed trapezoidal integral in metric-units × interaction steps; no extrapolation. This integration rule does not reconstruct hidden observations. Final/peak contrasts should be within environment and compatible protocol. No smoothing or silent normalization is applied. Exact-step learning means can change composition; n is annotated. Bootstrap count is explicit and seeded at zero by default.

## Provenance and opt-in instrumentation

Existing `utils/run_metadata.py`, `exp12/diagnostics.py`, `fixed_task.py`, immutable task banks and PR #10 first-update measurements are reused. Training/probe kernels are unchanged. `analysis.campaign.provenance.capture` builds a new optional diagnostic sidecar from the exact config, current Git SHA/dirty status (including untracked files), dependency versions, caller-supplied optimizer settings, architectures, actual parameter counts (`parameter_count`), environment versions, replay/target/task identifiers, seed, precision, checkpoint, interaction/update counters, status/failure. Missing fields are listed and provenance is marked incomplete; fields are not guessed. `write_sidecar` uses exclusive creation to prevent replacing historical evidence. `validate` checks config identity and missing source/optimizer fields. Optimizer values must come from validated parent metadata/current runner, never generic defaults. These APIs are opt-in; existing historical runs do not acquire missing provenance retrospectively. Automatic production trainer lifecycle integration is deliberately not part of this additive implementation and remains future engineering work.

`critic_snapshot` records mean/population variance/min/max/quantiles of caller-supplied Q, original Bellman targets and TD residuals, MSE, and fixed-reference prediction deltas. It never constructs a new target. `fitting_snapshot` distinguishes pre-update full-pool, fixed-batch and varying-minibatch MSE without applying an update. Existing actor loss/gradient/entropy/temperature/KL/churn recording and first-update parameter/prediction displacement are reused rather than reimplemented. Snapshot calls are explicit opt-in and incur host transfer only when invoked. Gradient norms and actor measurements must retain original runner semantics. No snapshot changes training hyperparameters.

`validate_pair` requires matching run/seed/round/checkpoint/task/target/optimizer/horizon identities and completed measurements, verifies P=baseline-final MSE, then computes fresh-current P. Own-offset targets may differ and must not be asserted to be common targets; historical own-offset results remain analyzable descriptively under their original semantics. Missing histories are never a successful measurement.

## Figures and limitations

`pipeline.figures`: individual seed returns plus exact-step means and pointwise bootstrap bands within config/environment/architecture. `diagnostic_plots.plot_series`: critic loss, Q, deficits, initial/final error, fixed/varying fitting transients using explicit units/counter labels. `plot_comparison`: final/peak and architecture comparisons with seed dots and means, enforcing one protocol/environment. `plot_relationship`: descriptive critic-policy temporal or initial-final scatter; callers supply aligned checkpoint rows. All APIs avoid interpolating missing points; no causal or predictive model is implied. Publication layout remains the researcher's review responsibility, particularly for large numbers of traces. Full automatic diagnostic artifact-to-long-form metric mapping is not implemented; local adapters preserve numeric payloads without guessing their meanings.

Synthetic export generation imports the shared deterministic fixture and labels its outputs SYNTHETIC. No real campaign export is committed or claimed processed. Real pilot data are preserved separately and are not used as canonical multi-seed fixtures.

## Validation and next steps

See the delivery/PR for exact final test counts. The initial shared environment had NumPy 2.x and could not satisfy the existing tests' NumPy 1.26.4 contract. A temporary `/tmp/echocritic-batch2-deps` overlay installs NumPy 1.26.4 and SciPy 1.14.1 without mutating shared dependencies. It is an isolated testing workaround, not a dependency-lock amendment. Regression commands set `PYTHONPATH` to this overlay and JAX platforms to CPU. Initial failed attempts are retained outside Git. Existing simulation/GPU qualification remains separate.

Before scientific use: inspect the real export schema, validate resolved optimizer/source evidence, supply campaign membership and expected checkpoint/metric manifests, reconcile restarts, and review exclusions. Broader simulator restoration and full-suite qualification are not established by focused tests. Review this dependent PR after #8–#10; do not merge the stack automatically. Corrected-source A100 equivalence remains a separately authorized future gate.

### Opt-in lifecycle example

`DiagnosticSession` writes exclusive immutable launch/measurement/terminal files, retains exception type/message on failure and propagates the original exception. Exiting without measurements records `no_measurements`, never successful scientific evidence. Callers supply exact already-validated fields; omitted provenance remains explicitly incomplete.

```python
from analysis.campaign.provenance import DiagnosticSession, critic_snapshot

with DiagnosticSession(new_output, repository, resolved_config,
                       optimizer=validated_optimizer, task_id=task_id,
                       seed=training_seed, checkpoint=checkpoint_id) as session:
    session.record(critic_snapshot(q, original_bellman_targets),
                   interaction_step=interaction_step,
                   optimizer_update=optimizer_update,
                   checkpoint=checkpoint_id)
```

Final focused offline validation adds lifecycle failure/no-measurement coverage: 19 tests passed. Earlier combined regression ran 81 tests successfully (17 new plus 64 existing); the expanded 19-test suite is separately rerun. These counts overlap and must not be added as independent coverage. Required mutation results are recorded in the delivery when complete. The unchanged broad simulator restoration suite was not rerun because this change does not touch environments or training state.
