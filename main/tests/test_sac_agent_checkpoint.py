"""Regression coverage for SACAgent checkpointing's churn_ref_batch handling
(see sac_agent.py's load_checkpoint for the bug: a None-shaped restore
target silently discarded a real saved churn_ref_batch on every resume).
Tests the realistic case through the full SACAgent, and the
before-first-update (None) case directly against orbax - the latter would
hit an unrelated, pre-existing orbax/jax-Metal sharding bug through the full
class (same one in test_angle_2b_smoke's normalizer round-trip test).
"""

import shutil
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import orbax.checkpoint
from omegaconf import OmegaConf

from scale_rl.agents import create_agent

OBS_DIM = 4
ACT_DIM = 2


def _make_cfg():
    return OmegaConf.create(
        {
            "agent_type": "sac",
            "seed": 0,
            "num_train_envs": 1,
            "max_episode_steps": 100,
            "normalize_observation": False,
            "actor_block_type": "residual",
            "actor_num_blocks": 1,
            "actor_hidden_dim": 8,
            "actor_learning_rate": 1e-4,
            "actor_weight_decay": 1e-2,
            "critic_block_type": "residual",
            "critic_num_blocks": 1,
            "critic_hidden_dim": 8,
            "critic_learning_rate": 1e-4,
            "critic_weight_decay": 1e-2,
            # False: see test_angle_2b_smoke.py - the CDQ path can't init here.
            "critic_use_cdq": False,
            "temp_target_entropy": None,
            "temp_target_entropy_coef": -0.5,
            "temp_initial_value": 0.01,
            "temp_learning_rate": 1e-4,
            "temp_weight_decay": 0.0,
            "target_tau": 0.005,
            "gamma": 0.99,
            "n_step": 1,
            "mixed_precision": False,
            "actor_sparsity": 0.0,
            "critic_sparsity": 0.0,
        }
    )


def _make_spaces():
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
    return observation_space, action_space


def _make_random_batch(rng: np.random.Generator, batch_size: int = 8) -> dict:
    return {
        "observation": rng.standard_normal((batch_size, OBS_DIM)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(batch_size, ACT_DIM)).astype(np.float32),
        "reward": rng.standard_normal(batch_size).astype(np.float32),
        "terminated": np.zeros(batch_size, dtype=np.float32),
        "truncated": np.zeros(batch_size, dtype=np.float32),
        "next_observation": rng.standard_normal((batch_size, OBS_DIM)).astype(np.float32),
    }


class TestSACAgentChurnRefBatchCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_populated_churn_ref_batch_survives_a_real_checkpoint_round_trip(self):
        observation_space, action_space = _make_spaces()
        cfg = _make_cfg()
        rng = np.random.default_rng(0)

        agent = create_agent(observation_space=observation_space, action_space=action_space, cfg=cfg)
        self.assertIsNone(agent.churn_ref_batch)
        agent.update(0, _make_random_batch(rng))
        self.assertIsNotNone(agent.churn_ref_batch, "update() should lazily populate churn_ref_batch")

        agent.save_checkpoint(self.tmpdir)

        fresh_agent = create_agent(observation_space=observation_space, action_space=action_space, cfg=cfg)
        self.assertIsNone(fresh_agent.churn_ref_batch)
        fresh_agent.load_checkpoint(self.tmpdir)

        self.assertIsNotNone(
            fresh_agent.churn_ref_batch,
            "load_checkpoint must not silently discard a populated churn_ref_batch",
        )
        np.testing.assert_allclose(
            np.asarray(fresh_agent.churn_ref_batch["observation"]),
            np.asarray(agent.churn_ref_batch["observation"]),
        )

    def test_none_churn_ref_batch_round_trips_as_none_at_the_orbax_level(self):
        """Isolated from SACAgent's TrainState restore (see module docstring)."""
        ckpt_path = self.tmpdir + "/agent_ckpt"
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpointer.save(ckpt_path, {"some_other_field": np.array([1.0, 2.0]), "churn_ref_batch": None}, force=True)

        restored = orbax.checkpoint.PyTreeCheckpointer().restore(ckpt_path)
        self.assertIsNone(restored["churn_ref_batch"])


if __name__ == "__main__":
    unittest.main()
