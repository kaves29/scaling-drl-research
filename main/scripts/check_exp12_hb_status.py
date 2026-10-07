#!/usr/bin/env python3
"""Validate existing Block HB outputs and propagate every required-step failure."""

import argparse
import json
import math
import re
from pathlib import Path


def check(out, identity_archs, speed_archs):
    expected = {"tests", "speed_h1-run-v0"} | {
        f'identity_D{a.split(":")[0]}W{a.split(":")[1]}_{env}'
        for a in identity_archs
        for env in ("dog-run", "myo-key-turn", "h1-run-v0")
    }
    errors = []
    try:
        rows = [
            line.split("\t") for line in (out / "status.tsv").read_text().splitlines()
        ]
        names = [r[0] for r in rows]
        if len(names) != len(set(names)) or set(names) != expected:
            errors.append("missing, unexpected or duplicate required steps")
        for row in rows:
            if len(row) != 3 or row[1] != "0":
                errors.append("required step failed: " + repr(row))
        log = (out / "tests.log").read_text()
        if not re.search(r"^Ran [1-9][0-9]* tests? in ", log, re.M) or not re.search(
            r"^OK$", log, re.M
        ):
            errors.append(
                "tests missing, failed or skipped; unset numerical criteria remain unresolved"
            )
        for name in expected - {"tests", "speed_h1-run-v0"}:
            value = json.loads(
                (
                    out / "identity" / f'{name.removeprefix("identity_")}.json'
                ).read_text()
            )
            if (
                value["pass"] is not True
                or value.get("differences")
                or any(
                    value[k] != value["fork_step"] + 1000
                    for k in ("interaction_step", "identity_interaction_step")
                )
            ):
                errors.append(name + ": identity failed or wrong snapshot steps")
        speed = json.loads((out / "speed_h1-run-v0.json").read_text())
        if len(speed) != len(speed_archs) or {r["arch"] for r in speed} != set(
            speed_archs
        ):
            errors.append("missing or duplicate profiling architectures")
        for row in speed:
            for key in (
                "train_it_per_s_probes_off",
                "probe_check_s",
                "probe_overhead_pct_of_wallclock",
                "peak_device_bytes",
            ):
                if row.get(key) is None or not math.isfinite(row[key]) or row[key] < 0:
                    errors.append(
                        f'{row.get("arch")}: missing/nonfinite profiling quantity {key}'
                    )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(str(exc))
    return {"pass": not errors, "errors": errors, "expected_steps": sorted(expected)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identity-archs", nargs="*", required=True)
    parser.add_argument("--speed-archs", nargs="+", required=True)
    args = parser.parse_args()
    result = check(args.out, args.identity_archs, args.speed_archs)
    (args.out / "qualification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
