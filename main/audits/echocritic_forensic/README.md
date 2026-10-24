# Independent forensic controls

These modules are audit tools. Production runners do not import them. They
never alter original archives or production code. Read the
[report](../../docs/forensic_probe_audit_20261024.md) and
[data-flow map](../../docs/forensic_probe_dataflow.md) for evidence scope.

After activating the documented CPU environment, change back to this checkout's
`main/` directory (activation changes cwd). Set `JAX_ENABLE_X64=false`.
Outputs must be fresh directories outside both repository and extracted inputs.
`PACKAGE` and `METRICS` below mean the extracted comprehensive/metric ZIP roots;
`REPO` means this checkout, `OUT` an external evidence directory. Originals are
identified by SHA256 in the report. These are reproduction instructions for the
agent/CI, not a request for the owner to run commands.

```bash
python -m audits.echocritic_forensic.evidence \
  --package "$PACKAGE" --metric-package "$METRICS" \
  --repository "$REPO" --out "$OUT/reconstruction"
python -m audits.echocritic_forensic.statistics_checks \
  --package "$PACKAGE" --metric-package "$METRICS" --out "$OUT/statistics"
python -m audits.echocritic_forensic.plots \
  --points "$OUT/reconstruction/independent_probe_points.csv" --out "$OUT/plots"
python -m pytest -q tests/test_forensic_probe_audit.py \
  tests/test_probe_tasks.py tests/test_fixed_task_bank_integration.py
```

Source controls use a separate immutable Git snapshot's `main/` directory:

```bash
python -B audits/echocritic_forensic/controls.py \
  --source-root "$SNAPSHOT/main" --revision "$SHA" --out "$OUT/control"
```

Run separately at full campaign f597ee9, pilot f8619dd and current f8ad82f SHAs
listed in the report. Original source is copied from `git archive`, not checked
out over an active branch; dependencies resolve in the CPU environment. The
controls use tiny synthetic networks/pools and analytically altered weights,
not original parent training or production probe reruns. Default includes a
four-transition warmup fake-env control with zero SAC updates. `--flow-only`
executes just that control. New tests exercise original and proposed module
copies; production files remain unchanged.

Reference-reader checks bound independent float64 reduction differences by
32 epsilons times value scale. This is not a changed experiment criterion.
No tolerance was added to original fitting, GPU equivalence or trigger logic.

`metric_aggregation_excerpt.json` preserves only the seven-line aggregation loop
from the supplied metric analyzer, with full analyzer SHA256. Tests execute
that isolated loop on synthetic data with a stub CI, never the supplied script,
its pickle reader or its plotting/file-writing code. This verifies an effective
sample-count correction leaves all original estimator fields unchanged.

`proposed_patches/` contains **unapplied** review candidates. See its README.
The compact validation receipt is tracked; detailed inputs, logs, numeric CSVs,
NPZ evidence and PDFs are retained outside Git. Large raw histories are not
republished in the repository. Per-point tables contain no interpolation;
figures show all three architectures/four environments on both linear and
symlog axes, with per-seed trained traces and matched observed-seed anchors.
