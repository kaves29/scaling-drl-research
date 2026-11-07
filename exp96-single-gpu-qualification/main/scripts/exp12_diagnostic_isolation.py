#!/usr/bin/env python3
"""Source-bound, three-arm diagnostic-isolation measurement (never a production run).

Import the frozen Exp1 entry point from the current working directory.  The only
runtime hook records state after each interaction; it does not alter the step
body, optimizer, RNG, configuration, or checkpoint logic.
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np


STEPS = 400
ARMS = (("off_a", False), ("on", True), ("off_b", False))
EXCLUDED_META = ("wandb_run_id", "metrics_rows", "actor_diagnostics")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _digest(value):
    """Stable typed digest of a live numeric state component."""
    h = hashlib.sha256()

    def add(item):
        if isinstance(item, (np.ndarray, np.generic, jax.Array)):
            array = np.asarray(jax.device_get(item))
            h.update(b"array\0" + array.dtype.str.encode() + b"\0")
            h.update(json.dumps(array.shape).encode() + b"\0")
            h.update(array.tobytes())
        elif isinstance(item, dict):
            h.update(b"dict\0")
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode() + b"\0")
            for child in item:
                add(child)
        else:
            h.update(b"scalar\0" + pickle.dumps(item, protocol=4))

    add(value)
    return h.hexdigest()


def fingerprint(trainer):
    """Named agent leaves and ancillary state at one completed interaction."""
    from experiments.exp12.envs import env_restore_state

    agent = trainer._sac_agent
    tree = {"actor": agent._actor, "critic": agent._critic,
            "target_critic": agent._target_critic, "temperature": agent._temperature,
            "rng": agent._rng, "churn_ref_batch": agent.churn_ref_batch}
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    training = {"/".join(str(part) for part in path): _digest(value) for path, value in leaves}
    buffer = trainer.buffer
    ancillary = {
        "obs_rms": _digest(getattr(trainer.agent, "obs_rms", None).__dict__
                           if hasattr(trainer.agent, "obs_rms") else None),
        "replay/count": _digest(buffer._num_in_buffer),
        "replay/index": _digest(buffer._current_idx),
        "replay/n_step": _digest(list(buffer._n_step_transitions)),
        "observations": _digest(trainer.observations),
        "timestep": _digest(trainer.timestep),
        "numpy_rng": _digest(np.random.get_state()),
        "python_rng": _digest(random.getstate()),
        "train_env": _digest(env_restore_state(trainer.train_env)),
        "eval_env": _digest(env_restore_state(trainer.eval_env)),
        "update_step": _digest(trainer.update_step),
        "update_counter": _digest(trainer.update_counter),
    }
    for key in ("_observations", "_actions", "_rewards", "_terminateds", "_truncateds",
                "_next_observations"):
        ancillary["replay/" + key] = _digest(getattr(buffer, key)[:buffer._num_in_buffer])
    return {"interaction_step": trainer.interaction_step, "update_step": trainer.update_step,
            "training": training, "ancillary": ancillary}


def differences(left, right):
    return {kind: sorted(set(left[kind]) ^ set(right[kind]) |
                         {key for key in left[kind].keys() & right[kind].keys()
                          if left[kind][key] != right[kind][key]})
            for kind in ("training", "ancillary")}


def agent_deviations(left, right):
    """Final agent leaf deviations with the original metric and named contributors."""
    from experiments.exp12.state import load_agent_tree

    a = dict(jax.tree_util.tree_leaves_with_path(load_agent_tree(left)))
    b = dict(jax.tree_util.tree_leaves_with_path(load_agent_tree(right)))
    if a.keys() != b.keys():
        raise ValueError("agent leaf paths differ")
    values = []
    for path in a:
        x, y = np.asarray(a[path]), np.asarray(b[path])
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"agent leaf {path}: shape or dtype differs")
        if x.dtype.kind in "biu":
            deviation = 0.0 if np.array_equal(x, y) else float("inf")
        elif x.dtype.kind == "f":
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"agent leaf {path}: nonfinite value")
            scale = float(np.abs(y.astype(np.float64)).max()) if y.size else 0.0
            delta = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if y.size else 0.0
            deviation = delta / scale if scale else delta
        else:
            raise ValueError(f"agent leaf {path}: unsupported dtype {x.dtype}")
        if deviation:
            values.append({"path": "/".join(str(part) for part in path), "relative_deviation": deviation})
    values.sort(key=lambda row: row["relative_deviation"], reverse=True)
    return {"max_relative_deviation": values[0]["relative_deviation"] if values else 0.0,
            "differing_leaf_count": len(values), "largest_leaves": values[:10]}


def compare(root):
    rows = {}
    for name, _ in ARMS:
        path = root / name / "fingerprints.jsonl"
        rows[name] = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows[name]) != STEPS or [r["interaction_step"] for r in rows[name]] != list(range(1, STEPS + 1)):
            raise ValueError(f"{path}: expected one record for each of {STEPS} interactions")
    pairs = {}
    for label, a, b in (("off_off", "off_a", "off_b"), ("on_off", "on", "off_a")):
        first = None
        for left, right in zip(rows[a], rows[b]):
            diff = differences(left, right)
            if any(diff.values()):
                first = {"interaction_step": left["interaction_step"], "components": diff}
                break
        pairs[label] = {"first_divergence": first,
                        "final_components": differences(rows[a][-1], rows[b][-1])}
    from experiments.exp12.state import latest_state_dir, state_differences

    for label, a, b in (("off_off", "off_a", "off_b"), ("on_off", "on", "off_a")):
        a_state = latest_state_dir(root / a / "state")
        b_state = latest_state_dir(root / b / "state")
        if a_state is None or b_state is None:
            raise ValueError(f"{label}: missing complete final state")
        names = state_differences(a_state, b_state, ignore_meta=EXCLUDED_META)
        pairs[label]["complete_checkpoint_differences"] = {
            "training": [n for n in names if n.startswith("agent:")],
            "ancillary": [n for n in names if not n.startswith("agent:")],
        }
        pairs[label]["final_agent_deviations"] = agent_deviations(a_state, b_state)
    baseline = pairs["off_off"]
    effect = pairs["on_off"]
    if baseline["first_divergence"] is not None or any(baseline["complete_checkpoint_differences"].values()):
        verdict = "BLOCKED_OFF_OFF_BASELINE_DIVERGES"
    elif effect["first_divergence"] is not None or any(effect["complete_checkpoint_differences"].values()):
        verdict = "BLOCKED_DIAGNOSTICS_ASSOCIATED_DIVERGENCE"
    else:
        verdict = "PASS_EXACT_ISOLATION_AT_TESTED_CONFIGURATION"
    return {"verdict": verdict, "pairs": pairs,
            "scope": "400-interaction tiny Exp1, highest precision, one A100; no tolerance inferred"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen-commit", required=True)
    parser.add_argument("--harness-commit", required=True)
    args = parser.parse_args()
    frozen_dir = Path.cwd()
    harness_dir = Path(__file__).resolve().parents[2]
    if git_head(frozen_dir) != args.frozen_commit or git_head(harness_dir) != args.harness_commit:
        raise ValueError("source revision mismatch")
    if jax.default_backend() != "gpu" or "A100" not in jax.devices()[0].device_kind:
        raise ValueError("A100 GPU backend required")
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"frozen_commit": args.frozen_commit, "harness_commit": args.harness_commit,
               "harness_sha256": sha(__file__), "gpu": jax.devices()[0].device_kind,
               "jax_version": jax.__version__, "steps": STEPS, "precision": "highest",
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "xla_flags": os.environ.get("XLA_FLAGS"),
               "arms": [name for name, _ in ARMS], "verdict": "INCOMPLETE"}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    from experiments import exp1
    from experiments.exp12.trainer import Exp12Trainer
    from tests.exp12_helpers import CONFIG_PATH, patch_wandb, tiny_overrides

    original_train = Exp12Trainer.train
    for name, enabled in ARMS:
        arm_dir = args.out / name
        arm_dir.mkdir()
        with (arm_dir / "fingerprints.jsonl").open("w") as stream:
            def observed_train(self, last_step, after_step=None, before_first_update=None):
                def observed_step(trainer):
                    if after_step is not None:
                        after_step(trainer)
                    if jax.config.jax_default_matmul_precision != "highest":
                        raise ValueError("precision changed during the training run")
                    stream.write(json.dumps(fingerprint(trainer), sort_keys=True) + "\n")
                    stream.flush()
                return original_train(self, last_step, after_step=observed_step,
                                      before_first_update=before_first_update)

            Exp12Trainer.train = observed_train
            patchers = patch_wandb()
            overrides = tiny_overrides(steps=STEPS, extra=[
                f"diagnostics.enabled={str(enabled).lower()}",
                f"results_root={arm_dir / 'results'}"])
            receipt[name] = {"overrides": overrides, "status": "RUNNING"}
            (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            try:
                with jax.default_matmul_precision("highest"):
                    if jax.config.jax_default_matmul_precision != "highest":
                        raise ValueError("highest precision context did not activate")
                    exp1.run({"experiment": "exp1", "config_path": CONFIG_PATH, "config_name": "base_exp12",
                              "overrides": overrides,
                              "checkpoint_dir": str(arm_dir), "checkpoint_interval": 10**9,
                              "checkpoint_start_frac": 0.0})
            finally:
                Exp12Trainer.train = original_train
                for patcher in patchers:
                    patcher.stop()
        receipt[name]["status"] = "COMPLETE"
        receipt[name]["fingerprints_sha256"] = sha(arm_dir / "fingerprints.jsonl")
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result = compare(args.out)
    receipt.update(result)
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"verdict": result["verdict"], "first_divergence": {
        key: value["first_divergence"] for key, value in result["pairs"].items()}}, indent=2))
    return 0 if result["verdict"] == "PASS_EXACT_ISOLATION_AT_TESTED_CONFIGURATION" else 3


if __name__ == "__main__":
    sys.exit(main())
