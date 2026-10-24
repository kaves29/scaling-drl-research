"""Validate SAC scalars and bind new checkpoints to their update configuration."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import optax

from utils.atomic_io import atomic_write_text

CONFIG_FILE = "sac_config.json"


def validate_config(cfg):
    for name in (
        "actor_learning_rate",
        "critic_learning_rate",
        "temp_learning_rate",
        "actor_weight_decay",
        "critic_weight_decay",
        "temp_weight_decay",
        "temp_initial_value",
        "temp_target_entropy",
        "temp_target_entropy_coef",
        "gamma",
        "target_tau",
        "actor_sparsity",
        "critic_sparsity",
    ):
        value = getattr(cfg, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError("SAC configuration requires finite numeric " + name)
        if name.endswith(("learning_rate", "weight_decay")) and value < 0:
            raise ValueError("SAC configuration requires nonnegative " + name)
    if cfg.temp_initial_value <= 0:
        raise ValueError("SAC configuration requires positive temp_initial_value")
    if not 0 <= cfg.gamma <= 1 or not 0 <= cfg.target_tau <= 1:
        raise ValueError("SAC configuration gamma and target_tau must be in [0,1]")
    if not 0 <= cfg.actor_sparsity < 1 or not 0 <= cfg.critic_sparsity < 1:
        raise ValueError("SAC configuration sparsities must be in [0,1)")
    if type(cfg.n_step) is not int or cfg.n_step < 1:
        raise ValueError("SAC configuration n_step must be a positive integer")


def checkpoint_config(agent):
    config = asdict(agent._cfg)
    seed = config.pop("seed")
    return {
        "schema": 1,
        # Restoration replaces the full saved PRNG; a constructor seed is not
        # an optimizer setting. Preserve its provenance without requiring it.
        "initialization_seed": seed,
        "configuration": config,
        "observation_dim": agent._observation_dim,
        "action_dim": agent._action_dim,
        "optimizer": {
            "algorithm": "optax.adamw",
            "optax_version": optax.__version__,
            "b1": 0.9,
            "b2": 0.999,
            "eps": 1e-8,
            "eps_root": 0.0,
            "schedule": "constant",
            "mask": None,
        },
    }


def save_checkpoint_config(agent, directory):
    atomic_write_text(
        Path(directory) / CONFIG_FILE,
        json.dumps(checkpoint_config(agent), sort_keys=True, allow_nan=False) + "\n",
    )


def check_checkpoint_config(agent, directory):
    path = Path(directory) / CONFIG_FILE
    if not path.exists():
        # Historical payloads have no sidecar; Exp1's run_metadata gate remains
        # the authority for those resumes. Do not fabricate a historical binding.
        return False
    saved, current = json.loads(path.read_text()), checkpoint_config(agent)
    if not isinstance(saved, dict) or set(saved) != set(current):
        raise ValueError("SAC checkpoint configuration must be an object")
    saved.pop("initialization_seed", None)
    current.pop("initialization_seed")
    if saved != current:
        raise ValueError("SAC checkpoint configuration/optimizer mismatch")
    return True
