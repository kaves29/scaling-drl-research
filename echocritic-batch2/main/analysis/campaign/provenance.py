"""Opt-in diagnostic sidecars; preserve original run metadata and algorithm."""

import importlib.metadata
import json
import subprocess
from pathlib import Path
import numpy as np
from .ingest import digest

VERSION = "echocritic-diagnostics/1"
REQUIRED = (
    "optimizer",
    "architectures",
    "parameter_counts",
    "environment",
    "replay",
    "target",
    "task_id",
    "seed",
    "precision",
    "checkpoint",
    "interaction_step",
    "optimizer_update",
    "status",
)


def capture(repository, config, **fields):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], text=True
        ).strip()

    result = {
        "schema": 1,
        "runner_version": VERSION,
        "config": config,
        "config_sha256": digest(config),
        "source": {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        },
        **fields,
    }
    result["missing_fields"] = [
        k for k in REQUIRED if k not in fields or fields[k] is None
    ]
    result["provenance_complete"] = not result["missing_fields"]
    result["packages"] = {}
    for name in (
        "numpy",
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "mujoco",
        "dm-control",
        "myosuite",
    ):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    return result


def write_sidecar(path, receipt):
    """Exclusive creation prevents accidental historical evidence replacement."""
    with Path(path).open("x") as f:
        json.dump(receipt, f, sort_keys=True, indent=2, allow_nan=False)


def parameter_count(tree):
    import jax

    return sum(np.asarray(a).size for a in jax.tree_util.tree_leaves(tree))


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("finite nonempty values required")
    return {
        "n": a.size,
        "mean": float(a.mean()),
        "variance": float(a.var()),
        "min": float(a.min()),
        "max": float(a.max()),
        "quantiles": np.quantile(a, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
    }


def critic_snapshot(q, bellman_targets, previous_q=None):
    """Opt-in fixed-reference function statistics; callers supply original targets."""
    q = np.asarray(q)
    target = np.asarray(bellman_targets)
    if q.shape != target.shape:
        raise ValueError("Q/target shapes differ")
    out = {
        "q": distribution(q),
        "bellman_target": distribution(target),
        "td_error": distribution(q - target),
        "td_mse": float(np.mean((q - target) ** 2)),
    }
    if previous_q is not None:
        p = np.asarray(previous_q)
        if p.shape != q.shape:
            raise ValueError("reference predictions must align")
        out["fixed_reference_prediction_delta"] = distribution(q - p)
        out["fixed_reference_prediction_delta_rms"] = float(
            np.sqrt(np.mean((q - p) ** 2))
        )
    return out


def fitting_snapshot(network, state, obs, act, targets, fixed_indices, varying_indices):
    """No optimizer update. Initial/fixed/varying are explicitly separate."""
    # One unchunked reference avoids imposing a divisibility requirement.
    q = np.asarray(network.apply({"params": state.params}, obs, act)).reshape(-1)
    y = np.asarray(targets)
    return {
        "initial_full_pool_mse": float(np.mean((q - y) ** 2)),
        "fixed_batch_mse": float(np.mean((q[fixed_indices] - y[fixed_indices]) ** 2)),
        "varying_minibatch_mse": float(
            np.mean((q[varying_indices] - y[varying_indices]) ** 2)
        ),
        "measurement_position": "before_optimizer_update",
    }


def validate(receipt):
    issues = []
    for k in REQUIRED:
        if k not in receipt or receipt[k] is None:
            issues.append("missing:" + k)
    if receipt.get("config_sha256") != digest(receipt.get("config", {})):
        issues.append("config_hash_mismatch")
    source = receipt.get("source", {})
    if not source.get("commit") or source.get("dirty") is not False:
        issues.append("unverified_source")
    opt = receipt.get("optimizer", {})
    if not isinstance(opt, dict) or not opt or any(v is None for v in opt.values()):
        issues.append("incomplete_optimizer")
    return issues


class DiagnosticSession:
    """Opt-in runner lifecycle: immutable launch, measurement and terminal receipts."""

    def __init__(self, output, repository, config, **fields):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.receipt = capture(repository, config, **{**fields, "status": "running"})
        self.count = 0
        write_sidecar(self.output / "launch.json", self.receipt)

    def __enter__(self):
        return self

    def record(self, metrics, interaction_step, optimizer_update, checkpoint):
        for counter in (interaction_step, optimizer_update):
            if type(counter) is not int or counter < 0:
                raise ValueError("explicit nonnegative integer counters required")
        event = {
            "runner_version": VERSION,
            "interaction_step": interaction_step,
            "optimizer_update": optimizer_update,
            "checkpoint": checkpoint,
            "metrics": metrics,
            "task_id": self.receipt.get("task_id"),
        }
        write_sidecar(self.output / f"measurement_{self.count:06d}.json", event)
        self.count += 1

    def __exit__(self, exception_type, exception, traceback):
        terminal = {
            **self.receipt,
            "status": "failed" if exception is not None else "complete",
            "measurement_count": self.count,
            "failure": (
                {"type": exception_type.__name__, "message": str(exception)}
                if exception is not None
                else None
            ),
        }
        # No measurements is explicitly distinct from a successful diagnostic.
        if exception is None and not self.count:
            terminal["status"] = "no_measurements"
        write_sidecar(self.output / "terminal.json", terminal)
        return False
