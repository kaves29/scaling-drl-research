# Retained-data probe accounting

These scripts import no model, training, fit or JAX module. They analyze the
supplied immutable JSON/NPZ evidence with NumPy/SciPy/Matplotlib only. The
[report](../../docs/probe_difficulty_accounting.md) defines scope and limitations.

From `main/`, with existing CPU analysis dependencies:

```bash
python -m audits.probe_difficulty.accounting \
  --package "$COMPREHENSIVE_PACKAGE" --metric-package "$METRIC_PACKAGE" \
  --out "$NEW_EXTERNAL_OUTPUT" \
  --search-root /workspace/artifacts --search-root /workspace/attachments \
  --search-root /workspace/library-files
python -m audits.probe_difficulty.figures \
  --analysis "$NEW_EXTERNAL_OUTPUT" --package "$COMPREHENSIVE_PACKAGE" \
  --out "$NEW_EXTERNAL_FIGURES"
python -m pytest -q tests/test_probe_difficulty_accounting.py
```

These are reproducibility instructions, not a request for the owner to run them.
Outputs must be fresh directories outside repository and input roots. Original
source snapshots are verified through immutable Git blobs; fetch full source
history if a new checkout is shallow. No supplied analysis script/pickle runs.

`analysis_plan.json` fixes new analysis definitions and labels this as exploratory,
unblinded work. The source IQM/paired subtraction are unchanged. Seed bootstrap
intervals resample independent training seed labels, never individual rounds.
Domain differences retain task/architecture/seed/check identity; two tasks/domain
are not a population environment experiment. Conditions retain crashed-run
coverage and no imputed checkpoints.

Important outputs:

- `round_components.csv`: observed scores plus b/MSE/ratios where available;
  unknown values blank. Bounds are explicitly labeled, not estimates.
- `probe_points.csv`, `condition_intervals.csv`, `seed_trends.csv`: source-defined
  IQMs, chronological trends, seed-level intervals and component bounds.
- `domain_seed_contrasts.csv`, `domain_intervals.csv`: matched common-check domain
  comparisons. Primary check35; maximum all-parent common check39.
- `candidate_factor_accounting.csv`: observed or missing quantity per parent/
  factor; causal contributions remain null where unidentified.
- `ALGEBRAIC_WITNESSES_NOT_OBSERVED.csv`: counterexamples to identifying variance
  versus fitting-error drift from scores alone. They are not teacher/optimization
  reconstructions, realistic-fit estimates or replacements for historical data.
- `archived_*`: the sole original archive's component, precision and curve checks.
  Changing-minibatch losses never become intermediate full-pool measurements.
- `source_predicted_sampling_NOT_OBSERVED.csv`: source-conditional occupancy/
  expected index diversity; actual physical-state coverage remains unobserved.
- `unresolved_wandb_manifest_references.csv`: listed zero-size manifest references,
  not downloaded or interpreted as actual checkpoint contents.
- `workspace_artifact_inventory.csv`: provenance screening; generated/synthetic
  or unbound workspace arrays are not original campaign evidence.

`all12_condition_trajectories.pdf` contains one nine-panel page per condition;
blank panels explicitly indicate components not retained. Individual parents and
pointwise95% seed-bootstrap bands are shown only where data exist. One-parent
component points have no invented seed CI. `DMC_vs_Myo_common.pdf` compares
shared checks through78%; `original_archived_curves.pdf` separates minibatch
curves and final pool errors. PNG copies are provided.

Reference-arithmetic checks retain the prior32-float64-epsilon bound. Structural
bounds use the sine range; actual-offset precision analysis is a fixed-prediction
roundoff bound, not a bound on changed optimization/GPU/teacher trajectories.
The existing scientific definitions and acceptance criteria are not edited.
