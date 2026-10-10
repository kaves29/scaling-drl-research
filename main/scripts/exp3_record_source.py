"""Owner-authorized future Exp2 source run with opt-in ordered arrival recording."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--args",
        required=True,
        help="JSON of ordinary experiments.exp2_arm.run arguments",
    )
    parser.add_argument("--stream", required=True)
    parser.add_argument("--chunk-size", required=True, type=int)
    parser.add_argument("--experiment", required=True, choices=("exp1", "exp2_arm"))
    parser.add_argument("--authorized-source-run", action="store_true", required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--resume-stream", action="store_true")
    opts = parser.parse_args()
    from importlib import import_module
    from experiments.exp3.recording import record_exp2_scope
    from experiments.exp3.streams import Benchmark

    args = json.loads(Path(opts.args).read_text())
    with record_exp2_scope(
        opts.stream,
        opts.chunk_size,
        Benchmark() if opts.benchmark else None,
        opts.resume_stream,
    ):
        import_module("experiments." + opts.experiment).run(args)


if __name__ == "__main__":
    main()
