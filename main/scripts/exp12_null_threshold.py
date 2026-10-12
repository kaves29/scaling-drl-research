#!/usr/bin/env python3
"""CPU-only computation of the NULL rule's would-be threshold. Never activates it.

Pre-registered NULL rule (docs/exp12_decisions.md, 2026-10-04): if the per-check fire
rate on fresh pairs exceeds 5% at a size, the lead decides whether to adopt a
null-calibrated threshold, "the 95th percentile of L between two independent fresh
critics at that size". This tool:
  1. verifies each supplied null_pairs JSONL against its owner-supplied SHA-256;
  2. re-derives every row with the production `_validate_null_row` (scores, paired
     losses, IQM, seeded bootstrap interval, firing) from the row's own recorded config;
  3. reports per-sample and pooled fire counts and np.percentile(L_IQM, 95), the same
     estimator `probe_fresh_checks.py --mode null` reports;
  4. writes the selected value and the exact config override string, with status
     COMPUTED_NOT_ACTIVATED.
The selection (which samples define the percentile) must be fixed by the owner
before this runs. Activation is a separate, owner-signed queue release.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))

ARCH = "D4W1024"
LIMIT = 0.05


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sample(label, path, expected_sha256):
    from omegaconf import OmegaConf
    from experiments.exp12.probe import probe_config
    from experiments.exp12.trigger import trigger_config
    from scripts.probe_fresh_checks import _validate_null_row

    path = Path(path)
    if _sha(path) != expected_sha256:
        raise ValueError(f"{label}: SHA-256 differs from the supplied value")
    rows, pairs, provenance = [], set(), None
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("arch") != ARCH:
                raise ValueError(f"{label}: row is not {ARCH}")
            if row.get("pair") in pairs:
                raise ValueError(f"{label}: duplicate pair {row.get('pair')}")
            pairs.add(row["pair"])
            if provenance is None:
                provenance = row.get("provenance")
            if row.get("provenance") != provenance:
                raise ValueError(f"{label}: mixed provenance within one sample")
            cfg = OmegaConf.create(provenance["resolved_config"])
            tc = trigger_config(cfg)
            if tc.null_threshold != 0.0:
                raise ValueError(f"{label}: null sample was not recorded at threshold 0")
            _validate_null_row(row, probe_config(cfg), tc)
            rows.append(row)
    if not rows:
        raise ValueError(f"{label}: empty sample")
    if any(not r["valid"] for r in rows):
        raise ValueError(f"{label}: invalid pairs present; the rule's percentile is undefined")
    return {
        "label": label,
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "pairs": len(rows),
        "env": provenance.get("env"),
        "seed": provenance.get("seed"),
        "code": provenance.get("code"),
        "rows": rows,
    }


def summarize(rows):
    losses = np.array([r["loss_iqm"] for r in rows], dtype=np.float64)
    fired = int(sum(bool(r["fired"]) for r in rows))
    return {
        "pairs": len(rows),
        "fired": fired,
        "per_check_fire_rate": fired / len(rows),
        "exceeds_5pct": fired / len(rows) > LIMIT,
        "p95_of_L": float(np.percentile(losses, 95)),
    }


def compute(samples, selection):
    loaded = [load_sample(*s) for s in samples]
    labels = [s["label"] for s in loaded]
    if len(set(labels)) != len(labels):
        raise ValueError("sample labels must be unique")
    if len({s["sha256"] for s in loaded}) != len(loaded):
        raise ValueError("the same file was supplied twice")
    per_sample = {s["label"]: summarize(s["rows"]) for s in loaded}
    pooled_rows = [r for s in loaded for r in s["rows"]]
    pooled = summarize(pooled_rows)
    if selection == "pooled":
        chosen_rows, chosen = pooled_rows, pooled
    elif selection in per_sample:
        chosen_rows = next(s["rows"] for s in loaded if s["label"] == selection)
        chosen = per_sample[selection]
    else:
        raise ValueError("selection must be 'pooled' or one sample label")
    threshold = chosen["p95_of_L"]
    # Descriptive only: the same rows' per-check firing had the threshold been in force.
    would_fire = {
        label: sum(bool(r["valid"]) and r["ci_low"] > threshold for r in s["rows"])
        for label, s in zip(labels, loaded)
    }
    return {
        "schema_version": 1,
        "status": "COMPUTED_NOT_ACTIVATED",
        "rule": "NULL rule, docs/exp12_decisions.md 2026-10-04: 95th percentile of L between "
        "two independent fresh critics at that size; adoption is the lead's decision",
        "architecture": ARCH,
        "samples": [{k: v for k, v in s.items() if k != "rows"} for s in loaded],
        "per_sample": per_sample,
        "pooled": pooled,
        "selection": selection,
        "selected_threshold": threshold,
        "selected_threshold_repr": repr(threshold),
        "override": f"trigger.null_threshold={threshold!r}",
        "applies_to": {"architecture": ARCH, "trigger_compares": "bootstrap lower bound > threshold"},
        "descriptive_null_firings_at_selected_threshold": would_fire,
        "selected_rows": len(chosen_rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        action="append",
        required=True,
        metavar="LABEL=PATH=SHA256",
        help="one null_pairs_D4W1024.jsonl with its expected SHA-256",
    )
    parser.add_argument("--selection", required=True, help="'pooled' or one sample LABEL")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    samples = []
    for spec in args.sample:
        label, path, digest = spec.split("=", 2)
        samples.append((label, path, digest))
    report = compute(samples, args.selection)
    with Path(args.out).open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "selection", "override", "pooled", "per_sample")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
