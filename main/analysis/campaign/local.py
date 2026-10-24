"""Normalize legacy numeric artifacts without inventing provenance or success."""

import argparse
import hashlib
import json
from pathlib import Path
from .ingest import local_artifacts
from .pipeline import json_safe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    index = []
    for i, name in enumerate(args.inputs):
        source = Path(name)
        data = local_artifacts(source)
        dest = f"artifact_{i:04d}.json"
        (out / dest).write_text(
            json.dumps(json_safe(data), sort_keys=True, allow_nan=False)
        )
        index.append(
            {
                "source": str(source.resolve()),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "normalized": dest,
                "provenance": "incomplete_legacy",
                "measurement_status": "unverified",
            }
        )
    (out / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
