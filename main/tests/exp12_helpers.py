"""Shared fixtures for the Exp 1/2 tests: fake WandB, tiny configs, state comparison."""

import contextlib
import json
import os
import sys
from pathlib import Path
from unittest import mock

import wandb

from experiments.exp12.state import load_agent_tree, state_differences  # noqa: F401  (re-exported for the tests)

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs")


class FakeWandbRun:
    def __init__(self):
        self.name = "fake-run"
        self.id = "fake-run-id"
        self.summary = {}

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def fake_wandb_init(*args, **kwargs):
    run = FakeWandbRun()
    if kwargs.get("id"):
        run.id = kwargs["id"]
    wandb.run = run
    return run


def patch_wandb():
    patchers = [mock.patch("wandb.init", side_effect=fake_wandb_init), mock.patch("wandb.log")]
    for p in patchers:
        p.start()
    return patchers


def tiny_overrides(env_name="hopper-hop", env_group="dmc_medium", seed=1, steps=300, extra=()):
    """steps is in interaction steps (env steps / action_repeat). Probes are shrunk
    (same procedure, fewer steps / smaller pool) so a CPU test finishes quickly.
    Entries in `extra` replace defaults with the same key."""
    base = [
        f"env_name={env_name}",
        f"env={env_group}",
        f"seed={seed}",
        "actor_num_blocks=1", "actor_hidden_dim=8",
        "critic_num_blocks=1", "critic_hidden_dim=8",
        f"env.num_env_steps={2 * steps}",
        "buffer.min_length=10", "buffer.max_length=1000", "buffer.sample_batch_size=8",
        "evaluation_per_interaction_step=100", "logging_per_interaction_step=20",
        "num_eval_episodes=1",
        "probe.steps=20", "probe.pool_size=64", "probe.batch_size=16", "probe.eval_chunk=32",
    ]
    replaced = {e.split("=", 1)[0] for e in extra}
    return [o for o in base if o.split("=", 1)[0] not in replaced] + list(extra)


def compose(overrides, config_name="base_exp12"):
    from experiments.exp1 import compose_config

    return compose_config(CONFIG_PATH, config_name, overrides)


# GPU numerics, as in PyTorch's tf32 "on and off" tests. On the GPU a bit-exact CPU comparison runs in one of two modes:
#  - "highest_deterministic" (the process has DETERMINISTIC_FLAG): both sides under a local 'highest', tight tolerance;
#  - "tf32": the side under test at the run setting (TF32), the reference under 'highest', loose tolerance.
# Each tolerance is set from a measured GPU deviation (rule: 10x the largest measured value, one significant digit,
# decisions log 2026-10-05). None = not measured yet: the test records the deviation and skips.
DETERMINISTIC_FLAG = "--xla_gpu_deterministic_ops=true"
GPU_TOLERANCES = {
    "diagnostics_update/highest_deterministic": None,
    "diagnostics_update/tf32": None,
    "diagnostics_training/highest_deterministic": None,
    "diagnostics_training/tf32": None,
}


def on_gpu():
    import jax

    return jax.default_backend() == "gpu"


def gpu_mode():
    return "highest_deterministic" if DETERMINISTIC_FLAG in os.environ.get("XLA_FLAGS", "") else "tf32"


def precision(name):
    """A local matmul-precision context; None leaves the job's setting alone."""
    import jax

    return contextlib.nullcontext() if name is None else jax.default_matmul_precision(name)


def record_numerics(name, **values):
    """Prints a measured deviation and appends it to $EXP12_NUMERICS_OUT (read by Block B's report)."""
    line = json.dumps({"name": name, **values}, default=float)
    print("NUMERICS " + line, file=sys.stderr, flush=True)
    path = os.environ.get("EXP12_NUMERICS_OUT")
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def max_relative_deviation(xs, ys, allow_nan_indices=()):
    """Float deviations use the existing scale; integer state must match exactly.

    Only explicitly designated diagnostic leaves may carry matching NaN masks.
    Model leaves, infinities, differing shapes, dtypes or leaf counts fail closed.
    """
    import numpy as np

    if len(xs) != len(ys):
        raise ValueError("state leaf counts differ")
    if any(i < 0 or i >= len(xs) for i in allow_nan_indices):
        raise ValueError("invalid diagnostic NaN leaf index")
    worst = 0.0
    for i, (x, y) in enumerate(zip(xs, ys)):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"leaf {i}: shape or dtype differs")
        if x.dtype.kind in "biu":
            if not np.array_equal(x, y):
                worst = float("inf")
            continue
        if x.dtype.kind != "f":
            raise ValueError(f"leaf {i}: unsupported dtype {x.dtype}")
        if i in allow_nan_indices:
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                raise ValueError(f"leaf {i}: diagnostic NaN masks differ")
            keep = ~np.isnan(x)
            x, y = x[keep], y[keep]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"leaf {i}: nonfinite state")
        if not x.size:
            continue
        x, y = x.astype(np.float64), y.astype(np.float64)
        scale = np.abs(y).max()
        worst = max(worst, float(np.abs(x - y).max() / scale) if scale > 0 else float(np.abs(x - y).max()))
    return worst


def check_gpu_tolerance(testcase, name, deviation, **context):
    import math

    testcase.assertTrue(math.isfinite(deviation), "nonfinite deviation or unequal integer state")
    tolerance = GPU_TOLERANCES.get(name)
    record_numerics(name, deviation=deviation, tolerance=tolerance, **context)
    if tolerance is None:
        testcase.skipTest(f"{name}: measured deviation {deviation:.3g}; no tolerance yet (set from a GPU measurement)")
    testcase.assertLessEqual(deviation, tolerance, f"{name}: deviation {deviation:.3g} > tolerance {tolerance:.3g}")
