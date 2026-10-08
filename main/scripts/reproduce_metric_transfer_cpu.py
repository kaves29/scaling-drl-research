#!/usr/bin/env python3
"""Measure metric tree traversal and materialization on CPU, without GPU access."""

import argparse
import json
import os
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


@jax.jit
def metrics(value):
    return {f"metric_{index:02d}": jnp.sin(value + index) for index in range(19)}


def timed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def payload(groups):
    return [
        metrics(jnp.asarray([index, index + 0.5], dtype=jnp.float32))
        for index in range(groups)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 1001, 2000, 4096])
    args = parser.parse_args()
    if any(groups < 1 for groups in args.groups):
        parser.error("group counts must be positive")
    for groups in args.groups:
        infos = payload(groups)
        _, ready = timed(lambda: jax.block_until_ready(infos))
        leaves, flatten = timed(lambda: jax.tree_util.tree_leaves(infos))
        _, copy_requests = timed(lambda: [leaf.copy_to_host_async() for leaf in leaves])
        host, materialize = timed(lambda: jax.tree_util.tree_map(np.asarray, infos))
        _, host_tree = timed(lambda: jax.device_get(host))
        fresh = payload(groups)
        jax.block_until_ready(fresh)
        _, device_get = timed(lambda: jax.device_get(fresh))
        print(
            json.dumps(
                dict(
                    cpu_only=True,
                    jax=jax.__version__,
                    devices=list(map(str, jax.devices())),
                    groups=groups,
                    leaves=len(leaves),
                    bytes=sum(leaf.nbytes for leaf in leaves),
                    readiness_s=ready,
                    flatten_s=flatten,
                    copy_requests_s=copy_requests,
                    materialize_s=materialize,
                    redundant_host_tree_s=host_tree,
                    fresh_device_get_s=device_get,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
