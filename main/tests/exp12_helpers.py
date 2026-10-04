"""Shared fixtures for the Exp 1/2 tests: fake WandB, tiny configs, state comparison."""

import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import orbax.checkpoint
import wandb

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


def _leaves(tree):
    import jax

    return jax.tree_util.tree_leaves_with_path(tree)


def load_agent_tree(state_dir):
    return orbax.checkpoint.PyTreeCheckpointer().restore(str(Path(state_dir) / "agent_ckpt"))


def state_differences(dir_a, dir_b, ignore_meta=("wandb_run_id",)):
    """Names of every component that differs bit-wise between two saved states."""
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    diffs = []
    tree_a, tree_b = load_agent_tree(dir_a), load_agent_tree(dir_b)
    leaves_a, leaves_b = _leaves(tree_a), _leaves(tree_b)
    if len(leaves_a) != len(leaves_b):
        diffs.append("agent:structure")
    for (path, a), (_, b) in zip(leaves_a, leaves_b):
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            diffs.append("agent:" + "/".join(str(p) for p in path))
    with open(dir_a / "obs_rms.pkl", "rb") as f:
        rms_a = pickle.load(f)
    with open(dir_b / "obs_rms.pkl", "rb") as f:
        rms_b = pickle.load(f)
    for k in rms_a:
        if not np.array_equal(rms_a[k], rms_b[k]):
            diffs.append(f"obs_rms:{k}")
    with np.load(dir_a / "buffer.npz") as ba, np.load(dir_b / "buffer.npz") as bb:
        for k in ba.files:
            if not np.array_equal(ba[k], bb[k]):
                diffs.append(f"buffer:{k}")
    with open(dir_a / "meta.pkl", "rb") as f:
        meta_a = pickle.load(f)
    with open(dir_b / "meta.pkl", "rb") as f:
        meta_b = pickle.load(f)
    for k in meta_a:
        if k in ignore_meta:
            continue
        if pickle.dumps(meta_a[k]) != pickle.dumps(meta_b[k]) and not _deep_equal(meta_a[k], meta_b[k]):
            diffs.append(f"meta:{k}")
    return diffs


def _deep_equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b
