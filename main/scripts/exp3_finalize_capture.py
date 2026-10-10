"""CPU-only recovery of a completed source's unsealed capture; no training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stream", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--expected-commit", required=True)
    args = p.parse_args()
    from utils.run_metadata import code_version
    from experiments.exp3.recording import finalize_capture
    from experiments.exp3.streams import StreamReader

    source = StreamReader(args.stream, require_complete=False).manifest["provenance"][
        "source_code"
    ]
    if (
        code_version() != {"commit": args.expected_commit, "dirty": False}
        or source != code_version()
    ):
        raise ValueError("exact clean capture source required")
    print(json.dumps(finalize_capture(args.stream, args.run), indent=2))


if __name__ == "__main__":
    main()
