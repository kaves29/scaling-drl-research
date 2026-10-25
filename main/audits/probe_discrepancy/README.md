# Source-bound paired discrepancy follow-up

The [report](../../docs/source_bound_probe_discrepancy.md) distinguishes original
evidence, source-control tests, blocked recovery and proposed measurements.
`pilot_manifest.DRAFT.json` is **not executable or approved**; it has no launch
command, output root or frozen diagnostic implementation. The ledger is a
source-linked evidence assessment, not scientific authority to change settings.

From `main/`, using existing CPU NumPy/SciPy/Matplotlib/pytest dependencies:

```bash
python -m audits.probe_discrepancy.investigate \
  --package "$COMPREHENSIVE_PACKAGE" --out "$FRESH_EXTERNAL_OUTPUT"
python -m audits.probe_discrepancy.inventory \
  --root "$ACCESSIBLE_EVIDENCE_ROOT" --out "$NEW_EXTERNAL_INVENTORY_JSON"
python -m pytest -q tests/test_probe_discrepancy.py tests/test_probe_difficulty_accounting.py
```

These commands document reproduction; the owner is not asked to execute them.
Full original Git source history is required. Output directories must be fresh
and outside the repository/immutable package. No supplied analysis script or
pickle is executed. Inventory lists archive members without extraction; small
ZIP NPZ bytes are hashed, not deserialized. It does not search remote storage.

`source_checks.py` extracts only selected functions from the pinned original
Git blob, removes decorators/annotations and executes pure routing/arithmetic
with controlled stand-ins. No JAX/model/training module is imported. A fake
teacher and fake fitter do not demonstrate numerical fit/optimizer equivalence.
Six source mutations test the guards. Original campaign optimizer/GPU/compiler
behavior remains a separate qualification requirement.

The generated CSVs include all observed paired checks, seed-level common-check
means/intervals, deterministic outcome-selected retrieval targets and source
guards. Missing remote hashes remain blank; expected paths do not prove file
existence. `paired_all12.pdf/png` shows original per-seed paired-score IQMs.
Unsmoothed original scalar operands and source FP32 subtraction are preserved.

The pilot selects known seeds for diagnostic replication. It does not add14
independent confirmatory seeds, alter the full training/probe recipe or approve
its considerable resource cost. Additional capture/forward endpoints are a
specification only, requiring a separate reviewed adapter and unchanged
on/off/equivalence criteria before any authorized launch.
