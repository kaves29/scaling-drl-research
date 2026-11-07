#!/bin/sh
set -eu
cd "$(dirname "$0")"
analysis_python=${ANALYSIS_PYTHON:-python3}
for script in analyze_full.py audit_provenance.py deep_checks.py predictive_ablation.py reference_decomposition.py scale_checks.py extra_figures.py validate_results.py write_report.py; do
  "$analysis_python" "$script"
done
