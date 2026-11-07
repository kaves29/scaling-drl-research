# EchoCritic full-record investigation

Start with REPORT.md, DISCOVERY_LEDGER.md and ROADMAP.md. This package analyzes all103 W&B records but does not claim all original disk artifacts are present. Missing full campaign probe curves are an explicit scientific limit. The earlier INPUT_STATUS.md/coverage files document the initial unavailable-data state and are superseded for current coverage by this full API export; they are preserved as history.

## Evidence and outputs

- `raw_api/`:103 immutable run directories, full unsampled `history.jsonl`, metadata/config/summary/attrs, file inventories and downloaded small files. Source project is `s_kaveti23-seckinger/echocritic_official_exp1`, API `https://api.forge.coreweave.com`.
- `raw_api_manifest.json`:SHA256/size for697 raw files. `raw_archived/`:one separately recovered original NPZ, linked to raw W&B rounds by validation. `archived_probe_provenance.json`:source location/hash/scope.
- `source/`:original source snapshots for f597 campaign and f861 pilot, with SHA256 manifest. Main analysis never imports or changes frozen training code.
- `derived/`:run inventory, duplicate display-name/condition tables, seed endpoints, source-matched score verification, paired effects, all collapse cases, curve dynamics, predictive cross-validation/ablations, time-adjusted correlations, onset and sensitivity tables.
- `figures/`:unsmoothed group curves/pointwise seed uncertainty, common-stage policy/probe comparisons, reference drift, scale comparisons, event alignment, original fitting curves and a96-page seed atlas. No figure treats probe repetitions as independent trained seeds.
- `VALIDATION.json`:12 independent numerical/provenance checks, all passed. Source/logger definitions and scientific limitations are in REPORT.md.

## Reproduce offline

Use Python3.12 with packages listed in requirements-analysis.txt. Retrieval used a separate installed Python3.14 W&B environment; it is unnecessary once raw_api exists. Example using the existing Mac isolated analysis overlay:

```bash
cd /Users/shouryakaveti/VS_Projects/scaling_rl_research/echocritic-comprehensive-20261024
export PYTHONPATH=/tmp/echocritic-batch2-deps
export MPLCONFIGDIR=/tmp/echocritic-mpl
analysis_python=/Users/shouryakaveti/VS_Projects/scaling_rl_research/.venv/bin/python
"$analysis_python" analyze_full.py
"$analysis_python" audit_provenance.py
"$analysis_python" deep_checks.py
"$analysis_python" predictive_ablation.py
"$analysis_python" reference_decomposition.py
"$analysis_python" scale_checks.py
"$analysis_python" extra_figures.py
"$analysis_python" validate_results.py
"$analysis_python" write_report.py
```

Run these sequentially; each script overwrites only derived outputs/figures, never raw data. Analyze_full.py resamples 10,000 times with seed20261024; deep_checks.py uses6141 and predictive_ablation.py1428. Save separate output package copies if comparing changed analysis definitions. The available-data coverage_audit.py is historical and points to the old partial export; it is not the full-campaign analysis entry point.

For data retrieval, `fetch_campaign.py` uses authorized W&B credentials without printing them and saves original run files once. It does not log new runs, train, change metadata remotely or start jobs. Its large unsampled page covers the sparse million-step history range; export row counts are independently matched to W&B historyLineCount. A naive1000-step page would cause many empty network requests and is avoided. A rerun skips completed raw files; for a fresh project snapshot use a fresh directory and rebuild the manifest, rather than mixing retrieval dates. Do not commit private exports or credentials to Git.

## Interpretation contracts

Campaign membership follows frozen source and resolved conditions, never display name alone. There are96 unique conditions; the seven pilot records are excluded. Seven pilot/campaign seed1 pairs have identical observed training/evaluation trajectories and are not independent confirmations. Source labels do not establish clean-tree/optimizer-state provenance for every parent.

Full-horizon large-critic summaries have six seeds; common70% contrasts have eight. No missing outcome is set to zero. Score arithmetic uses the ten-round source fractional IQM, whereas seed-level IQM uses the ordinary finite-sample trimmed estimator. Probe rounds are conditional repeats; trajectory observations are repeated within run. Intervals are pointwise/descriptive; all exploratory paired p-values receive a broad Holm correction and are not used as confirmatory findings. Predictive correlations are not causal effects. Terminal evaluations use50episodes versus5at earlier checks. Original success-attribution defects are not retroactively repaired in raw data.

## Scope and resource receipt

No GPU training, new fitting, interventions, source corrections or remote jobs were performed. Computation was local CPU reading numeric arrays, resampling, regression and plotting. One read-only SSH listing attempt failed authentication; no remote data were changed. Raw W&B access succeeded. All original data and preliminary reports remain unchanged. Complete causal explanation is limited by missing measurements; the report names exactly what can be tested next.
