#!/usr/bin/env python3
"""Compare ready-array transfer strategies in isolated, time-limited processes."""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def worker(args):
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if args.platform == "cuda" else "cpu"
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np

    root = Path(args.out_dir)
    with (root / "events.jsonl").open("x") as journal, (root / "stacks.log").open(
        "x"
    ) as stacks:

        def emit(event, **fields):
            journal.write(
                json.dumps(
                    dict(
                        event=event, time=time.perf_counter(), pid=os.getpid(), **fields
                    )
                )
                + "\n"
            )
            journal.flush()

        faulthandler.dump_traceback_later(5, repeat=True, file=stacks)
        try:
            emit("begin", phase="backend")
            device = jax.devices()[0]
            assert jax.__version__ == jaxlib.__version__ == "0.4.34"
            assert not jax.config.jax_enable_x64
            if args.platform == "cuda":
                assert device.platform == "gpu"
                assert device.device_kind == "NVIDIA A100-SXM4-40GB"
            else:
                assert device.platform == "cpu"
            emit(
                "end",
                phase="backend",
                device=str(device),
                device_kind=device.device_kind,
                jax=jax.__version__,
                jaxlib=jaxlib.__version__,
                python=sys.version,
            )

            @jax.jit
            def metrics(value):
                return {f"metric_{index:02d}": value + index for index in range(19)}

            emit("begin", phase="create")
            infos = [
                metrics(jnp.asarray([group, group + 1], dtype=jnp.float32))
                for group in range(args.groups)
            ]
            leaves, structure = jax.tree_util.tree_flatten(infos)
            emit(
                "end",
                phase="create",
                groups=len(infos),
                leaves=len(leaves),
                bytes=sum(leaf.nbytes for leaf in leaves),
            )
            emit("begin", phase="readiness")
            jax.block_until_ready(infos)
            emit("end", phase="readiness")

            submitted = materialized = 0
            slowest_submit = slowest_materialize = 0.0

            def copy(leaf):
                nonlocal submitted, slowest_submit
                index = submitted + 1
                sampled = index <= 3 or index % 256 == 0
                if sampled:
                    emit("begin", phase="copy", leaf=index)
                start = time.perf_counter()
                leaf.copy_to_host_async()
                seconds = time.perf_counter() - start
                submitted += 1
                slowest_submit = max(slowest_submit, seconds)
                if sampled:
                    emit("end", phase="copy", leaf=index, seconds=seconds)

            def materialize(leaf):
                nonlocal materialized, slowest_materialize
                index = materialized + 1
                sampled = index <= 3 or index % 256 == 0
                if sampled:
                    emit("begin", phase="materialize", leaf=index)
                start = time.perf_counter()
                result = np.asarray(leaf)
                seconds = time.perf_counter() - start
                materialized += 1
                slowest_materialize = max(slowest_materialize, seconds)
                if sampled:
                    emit("end", phase="materialize", leaf=index, seconds=seconds)
                return result

            emit("begin", phase="transfer", strategy=args.worker)
            start = time.perf_counter()
            host_leaves = []
            size = {"bulk": len(leaves), "groups": 19, "serial": 1}[args.worker]
            for offset in range(0, len(leaves), size):
                batch = leaves[offset : offset + size]
                for leaf in batch:
                    copy(leaf)
                host_leaves.extend(materialize(leaf) for leaf in batch)
            emit(
                "end",
                phase="transfer",
                seconds=time.perf_counter() - start,
                submitted=submitted,
                materialized=materialized,
                slowest_submit_s=slowest_submit,
                slowest_materialize_s=slowest_materialize,
            )
            host = jax.tree_util.tree_unflatten(structure, host_leaves)
            for group, info in enumerate(host):
                for index, value in enumerate(info.values()):
                    expected = np.array([group + index, group + index + 1], np.float32)
                    np.testing.assert_array_equal(
                        value.view(np.uint32), expected.view(np.uint32)
                    )
            emit("verified", groups=len(host), dtype="float32", exact_bits=True)
        except BaseException as exc:
            emit("error", kind=type(exc).__name__, message=str(exc))
            raise
        finally:
            faulthandler.cancel_dump_traceback_later()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--platform", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--groups", type=int, default=1001)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--native-stacks",
        action="store_true",
        help="try gdb on timed-out workers before terminating them",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["bulk", "groups", "serial"],
        default=["bulk", "groups"],
    )
    parser.add_argument(
        "--worker", choices=["bulk", "groups", "serial"], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.groups < 1 or not 0 < args.timeout <= 300:
        parser.error("groups must be positive and timeout must be in (0, 300]")
    if len(set(args.strategies)) != len(args.strategies):
        parser.error("strategies must be distinct")
    if args.worker:
        worker(args)
        return

    root = Path(args.out_dir).resolve()
    root.mkdir()
    outcomes = []
    for strategy in args.strategies:
        output = root / strategy
        output.mkdir()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            strategy,
            "--platform",
            args.platform,
            "--groups",
            str(args.groups),
            "--out-dir",
            str(output),
        ]
        start = time.perf_counter()
        native_status = None
        with (output / "process.log").open("x") as log:
            with subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT
            ) as process:
                try:
                    status = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    status = "timeout"
                    try:
                        if args.native_stacks:
                            debugger = shutil.which("gdb")
                            native_status = "unavailable"
                            if debugger:
                                with (output / "native-stacks.log").open("x") as native:
                                    try:
                                        result = subprocess.run(
                                            [
                                                debugger,
                                                "-batch",
                                                "-ex",
                                                "set pagination off",
                                                "-ex",
                                                "thread apply all bt",
                                                "-p",
                                                str(process.pid),
                                            ],
                                            stdout=native,
                                            stderr=subprocess.STDOUT,
                                            timeout=5,
                                            check=False,
                                        )
                                        native_status = result.returncode
                                    except subprocess.TimeoutExpired:
                                        native_status = "timeout"
                    finally:
                        process.kill()
                        process.wait()
        outcome = dict(
            strategy=strategy,
            status=status,
            wall_seconds=time.perf_counter() - start,
            native_stack_status=native_status,
        )
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)
        (root / "summary.json").write_text(json.dumps(outcomes, indent=2))
    if any(outcome["status"] != 0 for outcome in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
