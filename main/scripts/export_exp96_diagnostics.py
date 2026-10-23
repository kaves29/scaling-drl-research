#!/usr/bin/env python3
"""Export selected diagnostic evidence and CSVs; never copies replay/checkpoints."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile


def export(root, destination, include_tasks=False):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("export destination must be outside original evidence")
    results_path = root / "results.json"
    if results_path.is_symlink():
        raise ValueError("symlinked evidence is not supported")
    if not any(
        (root / name).is_file()
        for name in ("results.json", "runtime.json", "status.json")
    ):
        raise ValueError("no diagnostic execution evidence found")
    runtime = (
        json.loads((root / "runtime.json").read_text())
        if (root / "runtime.json").is_file()
        else {}
    )
    results = (
        json.loads(results_path.read_text())
        if results_path.exists()
        else {
            "source": runtime.get("expected_commit"),
            "frozen_source": None,
            "rows": [],
        }
    )
    rows = []
    selected = {results_path} if results_path.exists() else set()
    for name in ("runtime.json", "status.json", "reproduction_review_required.json"):
        if (root / name).exists():
            selected.add(root / name)
    for record in results["rows"]:
        if "original_reproduction" in record:
            rows.append(
                {
                    "record_type": "original_reproduction",
                    **record["original_reproduction"],
                }
            )
        elif "name" in record:
            for measurement in record.get("measurements", [{}]):
                base = {
                    k: v for k, v in record.items() if not isinstance(v, (dict, list))
                }
                base.update(
                    {
                        "first_update_" + k: v
                        for k, v in record.get("first_update", {}).items()
                    }
                )
                rows.append({"record_type": "controlled_fit", **base, **measurement})
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("refusing symlinked diagnostic evidence: " + str(path))
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if len(relative.parts) > 3:
            continue
        name = path.name
        if (
            name in ("progress.json", "segmented_equivalence.json")
            or (
                name.endswith(".npz")
                and (
                    name.startswith("curve_through_")
                    or name
                    in (
                        "fresh_original_fit.npz",
                        "latest_original_fit.npz",
                        "segmented_fit.npz",
                    )
                )
            )
            or (
                include_tasks
                and (
                    name in ("task.npz", "arrays.npz", "manifest.json")
                    or name.startswith("pool_")
                    and name.endswith(".npz")
                )
            )
        ):
            selected.add(path)
    # Check CRC and numerical payload before describing selected arrays as intact.
    import numpy as np

    numeric_issues = []
    for path in selected:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                for name in arrays.files:
                    a = arrays[name]
                    if a.dtype.kind in "fc" and not np.isfinite(a).all():
                        numeric_issues.append(
                            {
                                "file": str(path.relative_to(root)),
                                "array": name,
                                "issue": "nonfinite values retained",
                            }
                        )
    destination.mkdir(parents=True, exist_ok=False)
    csv_path = destination / "measurements.csv"
    fields = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": 1,
        "source": results["source"],
        "frozen_source": results["frozen_source"],
        "plan_sha256": results.get("plan_sha256"),
        "status": (
            json.loads((root / "status.json").read_text())
            if (root / "status.json").exists()
            else {"status": "MISSING_STATUS"}
        ),
        "conditions": results.get("conditions", []),
        "numeric_tasks_included": include_tasks,
        "numeric_issues": numeric_issues,
        "interpretation": "Technical evidence only; partial/failed conditions are not passes.",
        "files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(selected)
        },
        "measurements_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    archive = destination / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(selected):
            tar.add(p, arcname="evidence/" + str(p.relative_to(root)), recursive=False)
        for name in ("manifest.json", "measurements.csv"):
            tar.add(destination / name, arcname=name, recursive=False)
    (destination / "SHA256SUMS.txt").write_text(
        "".join(
            hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
            for p in (archive, csv_path, destination / "manifest.json")
        )
    )
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--include-tasks",
        action="store_true",
        help="Include potentially large numeric tasks, never replay",
    )
    args = p.parse_args()
    print(json.dumps(export(args.root, args.out, args.include_tasks), indent=2))


if __name__ == "__main__":
    main()
