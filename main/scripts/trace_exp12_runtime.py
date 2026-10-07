#!/usr/bin/env python3
"""Trace a short existing entry point without overriding its scientific configuration."""

import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.hardware import configure_hardware_env

configure_hardware_env()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--synchronize", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("entrypoint", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.entrypoint
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide -- <existing Python entry point> <arguments>")
    from experiments.exp12.runtime_trace import RuntimeTrace

    previous = sys.argv
    try:
        sys.argv = command
        with RuntimeTrace(args.out, args.synchronize, args.progress_every) as trace:
            trace.install()
            runpy.run_path(command[0], run_name="__main__")
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
