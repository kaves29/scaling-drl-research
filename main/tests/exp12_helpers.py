"""Shared fixtures for the Exp 1/2 tests: fake WandB, tiny configs, state comparison."""

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
