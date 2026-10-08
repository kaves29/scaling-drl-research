#!/usr/bin/env python3
"""Trace a short existing entry point without overriding its scientific configuration."""

import argparse
import faulthandler
import json
import math
import runpy
import sys
import time
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.hardware import configure_hardware_env

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--synchronize", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--boundary-detail", action="store_true",
                        help="trace fine stages in the progress interval before each logging boundary")
    parser.add_argument("--stack-after", type=float,
                        help="periodically dump Python stacks after this many seconds; does not stop execution")
    parser.add_argument("entrypoint", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.entrypoint
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide -- <existing Python entry point> <arguments>")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.stack_after is not None and (not math.isfinite(args.stack_after) or args.stack_after <= 0):
        parser.error("--stack-after must be finite and positive")

    previous = sys.argv
    with ExitStack() as resources:
        stream = resources.enter_context(Path(args.out).open("x"))

        def emit(event, **fields):
            stream.write(json.dumps(dict(event=event, wall_time=time.time(),
                                         monotonic_time=time.perf_counter(), **fields)) + "\n")
            stream.flush()

        emit("wrapper_start")
        stacks = None
        if args.stack_after is not None:
            stacks = resources.enter_context(Path(args.out).with_suffix(".stacks.log").open("x"))
        for stage, action in (("hardware_env", configure_hardware_env),
                              ("runtime_import", lambda: __import__("experiments.exp12.runtime_trace", fromlist=["RuntimeTrace"]))):
            emit("begin", stage=stage)
            start = time.perf_counter()
            error = None
            try:
                result = action()
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                emit("end", stage=stage, seconds=time.perf_counter() - start, error=error)
        RuntimeTrace = result.RuntimeTrace
        # Arm after dependency imports: rapid dumps during extension/import setup
        # can be unsafe on some Python builds. Flushed startup events cover that gap.
        if stacks is not None:
            faulthandler.dump_traceback_later(args.stack_after, repeat=True, file=stacks)
            resources.callback(faulthandler.cancel_dump_traceback_later)
        try:
            sys.argv = command
            with RuntimeTrace(args.out, args.synchronize, args.progress_every,
                              boundary_detail=args.boundary_detail, stream=stream) as trace:
                for stage, action in (("trace_install", trace.install),
                                      ("entrypoint", lambda: runpy.run_path(command[0], run_name="__main__"))):
                    trace.emit("begin", stage=stage)
                    start = time.perf_counter()
                    error = None
                    try:
                        action()
                    except BaseException as exc:
                        error = type(exc).__name__
                        raise
                    finally:
                        trace.emit("end", stage=stage, seconds=time.perf_counter() - start, error=error)
        finally:
            sys.argv = previous


if __name__ == "__main__":
    main()
