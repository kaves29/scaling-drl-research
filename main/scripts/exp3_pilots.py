"""Explicit Exp3 audit/manifest/execute CLI. Does not launch Slurm or source runs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp3.protocol import load_spec, manifest  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--state", required=True)
    audit.add_argument("--metadata", required=True)
    m = commands.add_parser("manifest")
    m.add_argument("--config", required=True)
    m.add_argument("--artifact-root", required=True)
    m.add_argument("--out", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--spec", required=True)
    execute.add_argument("--out", required=True)
    execute.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "manifest":
        value = manifest(args.config, args.artifact_root)
        with open(args.out, "x") as stream:
            json.dump(value, stream, indent=2)
    elif args.command == "audit":
        from experiments.exp3.artifacts import inspect_artifact

        info, meta, obs, act = inspect_artifact(args.state, args.metadata)
        print(
            json.dumps(
                {
                    "provenance": info,
                    "interaction_step": meta["interaction_step"],
                    "update_step": meta["update_step"],
                    "dimensions": [obs, act],
                },
                indent=2,
            )
        )
    else:
        from experiments.exp3.runner import run

        run(load_spec(args.spec), args.out, args.resume)


if __name__ == "__main__":
    main()
