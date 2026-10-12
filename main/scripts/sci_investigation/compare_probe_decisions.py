"""CPU-only exploratory D1 analysis. Never writes production configuration."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


def percentile95(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("expected nonempty finite one-dimensional IQM values")
    return float(np.percentile(values, 95, method="linear"))


def exact_binomial_interval(fires, total, confidence=0.95):
    if not 0 <= fires <= total or total < 1:
        raise ValueError("invalid binomial count")
    tail = (1 - confidence) / 2
    return [float(beta.ppf(tail, fires, total - fires + 1)) if fires else 0.0,
            float(beta.ppf(1 - tail, fires + 1, total - fires)) if fires < total else 1.0]


def null_decisions(rows, threshold):
    """Each fresh pair is a *single check*, not a production trajectory."""
    if not rows:
        raise ValueError("empty null dataset")
    return [bool(row["valid"] and np.isfinite(row["ci_low"]) and row["ci_low"] > threshold)
            for row in rows]


def replay_checks(rows, threshold, checks=20, eligible_fraction=0.95):
    """Replay the strict lower-bound and first two contiguous fired checks.

    Returns only an observed prefix decision; incomplete trajectories do not prove
    the absence of a later trigger. Forced test-hook checks are refused.
    """
    last = int(np.floor(checks * eligible_fraction + 1e-9))
    prior, streak, seen = None, 0, set()
    for row in sorted(rows, key=lambda r: int(r["check_index"])):
        k = int(row["check_index"])
        if row.get("forced") in (True, "True", "true", "1"):
            raise ValueError("forced test-hook check is not a natural trajectory")
        if k < 1 or k > checks or k in seen:
            raise ValueError("invalid or duplicate check index")
        seen.add(k)
        low = float(row["ci_low"])
        valid = row["valid"] in (True, "True", "true", "1")
        fired = bool(valid and np.isfinite(low) and low > threshold)
        streak = streak + 1 if fired and prior == k - 1 else int(fired)
        prior = k
        if streak == 2:
            return {"check_index": k, "interaction_step": int(row["interaction_step"])} if k <= last else None
    return None


def load_null(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 100 or {row["pair"] for row in rows} != set(range(100)):
        raise ValueError(f"{path}: expected exactly 100 unique pairs")
    if any(row["check_index"] != row["pair"] + 1 for row in rows):
        raise ValueError(f"{path}: pair/check mismatch")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--null-990", type=Path, required=True)
    parser.add_argument("--null-991", type=Path, required=True)
    parser.add_argument("--trajectory-csv", type=Path, help="optional real, unforced probe_checks.csv")
    args = parser.parse_args()
    a, b = load_null(args.null_990), load_null(args.null_991)
    for label, rows in (("990", a), ("991", b)):
        if {r["seed"] for r in rows} != {int(label)}:
            raise ValueError(f"seed mismatch in {label}")
        if any(r["resamples"] != 10000 or r["confidence"] != .95 or r["null_threshold"] != 0
               for r in rows):
            raise ValueError(f"unexpected original trigger settings in {label}")
        if any(null_decisions([r], 0)[0] != r["fired"] for r in rows):
            raise ValueError(f"saved trigger verdict differs in {label}")
    thresholds = {"seed990": percentile95([r["loss_iqm"] for r in a]),
                  "seed991": percentile95([r["loss_iqm"] for r in b]),
                  "pooled": percentile95([r["loss_iqm"] for r in a + b])}
    out = {"thresholds": thresholds, "quantile": "numpy percentile method=linear, p95 IQM(L)", "counts": {}}
    for label, rows in (("990", a), ("991", b), ("pooled", a + b)):
        tests = {"A_zero": 0, "B_pooled_in_sample": thresholds["pooled"]}
        if label == "990":
            tests["B_991_to_990"] = thresholds["seed991"]
        if label == "991":
            tests["B_990_to_991"] = thresholds["seed990"]
        out["counts"][label] = {}
        for name, threshold in tests.items():
            fires = sum(null_decisions(rows, threshold))
            out["counts"][label][name] = {"fires": fires, "checks": len(rows),
                                           "exact_binomial_95": exact_binomial_interval(fires, len(rows))}
    if args.trajectory_csv:
        with args.trajectory_csv.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        out["trajectory_prefix"] = {name: replay_checks(rows, threshold) for name, threshold in
                                    (("A_zero", 0), ("B_pooled_exploratory", thresholds["pooled"]))}
        out["trajectory_prefix"]["observed_checks"] = len(rows)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
