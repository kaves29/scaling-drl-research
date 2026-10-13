#!/usr/bin/env python3
"""Propose the four CUDA diagnostics-isolation bounds from one measurement job's records (CPU).

Logged rule (docs/exp12_decisions.md, Block B decisions 2026-10-05, item 2): each bound
is 10x the largest measured deviation, rounded UP to one significant digit. This tool
only proposes; tests/exp12_helpers.GPU_TOLERANCES stays None until the owner approves
values in a reviewed commit.

Source-bound and fail-closed: the input is one job directory written by
scripts/sbatch_exp12_numeric_bounds.sh. Its submission.json and every record must name the
expected commit; every record must come from the GPU backend on the approved A100, in the
mode of the file it is in; each name must have exactly the expected number of records; and
every deviation must be a finite, nonnegative number. Any violation makes the whole proposal
INVALID with no proposed bound.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

GPU_KIND = "NVIDIA A100-SXM4-40GB"
MODES = {"tf32": "numerics_tf32.jsonl", "highest_deterministic": "numerics_deterministic.jsonl"}
# Records each mode's job writes: the single and the twin critic update test, and one training run.
EXPECTED = {"diagnostics_update": 2, "diagnostics_training": 1}
NAMES = tuple(f"{kind}/{mode}" for kind in EXPECTED for mode in MODES)
SHA = re.compile(r"[0-9a-f]{40}")


def one_digit_up(value):
    if value <= 0:
        return 0.0
    exponent = math.floor(math.log10(value))
    return math.ceil(value / 10**exponent) * 10**exponent


def _valid_deviation(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def propose(root, expected_commit):
    root = Path(root)
    errors, sources, records = [], [], {name: [] for name in NAMES}
    if not SHA.fullmatch(str(expected_commit)):
        errors.append("expected commit must be a full 40-character hash")
    try:
        submission = json.loads((root / "submission.json").read_text())
        if submission.get("commit") != expected_commit:
            errors.append(f"submission commit {submission.get('commit')!r} is not {expected_commit}")
    except (OSError, ValueError) as exc:
        errors.append(f"submission.json unreadable: {exc}")
    for mode, filename in MODES.items():
        path = root / filename
        try:
            text = path.read_text()
        except OSError as exc:
            errors.append(f"{filename} missing: {exc}")
            continue
        sources.append({"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()})
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            where = f"{filename}:{number}"
            try:
                record = json.loads(line)
            except ValueError:
                errors.append(f"{where}: not JSON")
                continue
            name = record.get("name")
            if name not in records:
                errors.append(f"{where}: unexpected record name {name!r}")
                continue
            if record.get("commit") != expected_commit:
                errors.append(f"{where}: record commit {record.get('commit')!r} is not {expected_commit}")
            if record.get("backend") != "gpu":
                errors.append(f"{where}: backend {record.get('backend')!r} is not gpu")
            if record.get("device_kind") != GPU_KIND:
                errors.append(f"{where}: device {record.get('device_kind')!r} is not {GPU_KIND}")
            if record.get("mode") != mode or not name.endswith("/" + mode):
                errors.append(f"{where}: mode {record.get('mode')!r}/{name} does not match {mode}")
            if not _valid_deviation(record.get("deviation")):
                errors.append(f"{where}: deviation {record.get('deviation')!r} is not finite and nonnegative")
            records[name].append(record)
    for name, rows in records.items():
        expected = EXPECTED[name.split("/")[0]]
        if len(rows) != expected:
            errors.append(f"{name}: expected {expected} records, found {len(rows)}")
        if name.startswith("diagnostics_update/") and len(rows) == 2 and sorted(
                bool(r.get("twin")) for r in rows) != [False, True]:
            errors.append(f"{name}: needs one single-critic and one twin-critic record")
    if errors:
        bounds = {name: {"status": "INVALID", "records": len(rows)} for name, rows in records.items()}
    else:
        bounds = {}
        for name, rows in records.items():
            largest = max(r["deviation"] for r in rows)
            bounds[name] = {"status": "PROPOSED", "records": len(rows), "max_deviation": largest,
                            "proposed_bound": one_digit_up(10 * largest)}
    return {
        "status": "INVALID" if errors else "PROPOSED_NOT_APPLIED",
        "source_commit": expected_commit,
        "rule": "10x the largest measured deviation, one significant digit (rounded up)",
        "sources": sources,
        "errors": errors,
        "bounds": bounds,
        "complete": not errors,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="the numerics job directory")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = propose(args.root, args.expected_commit)
    with Path(args.out).open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "errors": report["errors"], "bounds": report["bounds"]},
                     indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
