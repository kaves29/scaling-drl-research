#!/usr/bin/env python3
"""Propose the four CUDA diagnostics-isolation bounds from recorded NUMERICS JSONL (CPU).

Logged rule (docs/exp12_decisions.md, Block B decisions 2026-10-05, item 2): each bound
is 10x the largest measured deviation, rounded UP to one significant digit. This tool
only proposes; tests/exp12_helpers.GPU_TOLERANCES stays None until the owner approves
values in a reviewed commit. Records must name their source commit to be usable.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

NAMES = (
    "diagnostics_update/highest_deterministic",
    "diagnostics_update/tf32",
    "diagnostics_training/highest_deterministic",
    "diagnostics_training/tf32",
)


def one_digit_up(value):
    if value <= 0:
        return 0.0
    exponent = math.floor(math.log10(value))
    return math.ceil(value / 10**exponent) * 10**exponent


def propose(files):
    records, sources = [], []
    for path in files:
        path = Path(path)
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    out = {}
    for name in NAMES:
        values = [r["deviation"] for r in records if r.get("name") == name]
        if any(not math.isfinite(v) for v in values):
            out[name] = {"status": "NONFINITE_DEVIATION", "records": len(values)}
        elif not values:
            out[name] = {"status": "MISSING", "records": 0}
        else:
            out[name] = {"status": "PROPOSED", "records": len(values), "max_deviation": max(values),
                         "proposed_bound": one_digit_up(10 * max(values))}
    return {
        "status": "PROPOSED_NOT_APPLIED",
        "rule": "10x the largest measured deviation, one significant digit (rounded up)",
        "sources": sources,
        "bounds": out,
        "complete": all(v["status"] == "PROPOSED" for v in out.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerics", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = propose(args.numerics)
    with Path(args.out).open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["bounds"], indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
