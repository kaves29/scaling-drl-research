"""End-to-end CPU fixture for online collection, diagnostics and restore."""

import tempfile
import unittest
from unittest.mock import patch
import random

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv

from experiments.exp1 import compose_config
from experiments.online_exp2 import OnlineExp2, aggregate_run, validate
from experiments.exp12.trainer import Exp12Trainer
from experiments.exp12.online_exp2 import paired_shadow_update
import jax


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-10., 10., shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)

    def __init__(self):
        self.step_count = 0
        self.observation_space = gym.spaces.Box(-10., 10., shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.action_space.seed(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        self.step_count += 1
        return np.full(3, self.step_count / 10, np.float32), float(np.sum(action)), False, self.step_count >= 5, {}

    def restore_state(self):
        return {"step_count": self.step_count}

    def restore(self, state):
        self.step_count = state["step_count"]


def tiny_envs(**kwargs):
    return SyncVectorEnv([TinyEnv]), SyncVectorEnv([TinyEnv])


def tiny_eval_env(**kwargs):
    return SyncVectorEnv([TinyEnv])


def cfg():
    return compose_config("/workspace/scaling-drl-research/main/configs", "base_online_exp2", [
        "online_exp2.environments=[dog-run]", "online_exp2.training_seeds=[1]",
        "online_exp2.reference_critic={blocks:1,width:8}",
        "online_exp2.training_interactions=3", "online_exp2.checkpoint_steps=[2]",
        "online_exp2.probe_checkpoints=[2]", "online_exp2.evaluation_episodes=1",
        "online_exp2.evaluation_mode=deterministic", "online_exp2.intervention_updates=1",
        "online_exp2.analysis_aggregation=mean_checkpoint_means",
        "online_exp2.resource_cap_gpu_hours=0",
        "online_exp2.dev_architectures=[[small,1,8],[large,1,8]]",
        "buffer.min_length=1", "buffer.sample_batch_size=4", "buffer.max_length=32",
        "probe.rounds=1", "probe.steps=2", "probe.pool_size=4",
        "probe.batch_size=4", "probe.eval_chunk=4",
        "logging_per_interaction_step=100", "evaluation_per_interaction_step=100",
        "num_eval_episodes=1", "actor_grad_cosine_every=100",
    ])


class OnlineExp2DriverTest(unittest.TestCase):
    def test_equal_checkpoint_analysis(self):
        rows = [{"type": "evaluation", "interaction_step": 1, "advantage_large": 1.},
                {"type": "evaluation", "interaction_step": 1, "advantage_large": 3.},
                {"type": "evaluation", "interaction_step": 2, "advantage_large": 6.}]
        self.assertEqual(aggregate_run(rows, [1, 2])["mean_checkpoint_means"], 4.)
        with self.assertRaises(ValueError):
            aggregate_run(rows, [1, 3])

    def test_missing_science_fails_closed(self):
        empty = compose_config("/workspace/scaling-drl-research/main/configs", "base_online_exp2", [])
        with self.assertRaisesRegex(ValueError, "unresolved fields"):
            validate(empty)

    def test_online_fixture_and_restore(self):
        with tempfile.TemporaryDirectory(dir="/workspace/scratch") as root:
            with patch("experiments.exp12.trainer.create_envs", tiny_envs), patch(
                "experiments.online_exp2.create_eval_env", tiny_eval_env
            ):
                run = OnlineExp2(cfg(), root)
                run.run()
                self.assertEqual(run.trainer.interaction_step, 3)
                self.assertEqual(run.shadows["small"].updates, run.trainer.update_step)
                self.assertEqual(run.shadows["large"].updates, run.trainer.update_step)
                self.assertTrue(any(row["type"] == "evaluation" for row in run.rows))
                self.assertTrue(any(row["type"] == "probe" for row in run.rows))
                trained_probes = [row for row in run.rows if row["type"] == "probe" and row["check"] == 2]
                self.assertEqual({row["architecture"] for row in trained_probes}, {"small", "large"})
                for row in trained_probes:
                    self.assertTrue(np.isfinite(row["plasticity_loss_iqm"]))
                    with np.load(row["artifact"], allow_pickle=False) as arrays:
                        self.assertTrue(any(key.startswith("fresh_") for key in arrays.files))
                        self.assertTrue(any(key.startswith("current_") for key in arrays.files))
                intervention = next(row for row in run.rows if row["type"] == "intervention")
                with np.load(intervention["artifact"], allow_pickle=False) as arrays:
                    for key in ("gradient_small", "gradient_large", "delta_small", "delta_large",
                                "actions_baseline", "actions_small", "actions_large"):
                        self.assertIn(key, arrays.files)
                        self.assertTrue(np.all(np.isfinite(arrays[key])))
                reference_cfg = cfg()
                reference_cfg.agent.critic_num_blocks = 1
                reference_cfg.agent.critic_hidden_dim = 8
                reference_cfg.num_interaction_steps = 3
                np.random.seed(1)
                random.seed(1)
                baseline = Exp12Trainer(reference_cfg, root + "/baseline")
                baseline.start()
                baseline.train(3)
                for name, left, right in (("actor", run.trainer._sac_agent.actor.params, baseline._sac_agent.actor.params),
                                    ("critic", run.trainer._sac_agent.critic.params, baseline._sac_agent.critic.params),
                                    ("rng", run.trainer._sac_agent._rng, baseline._sac_agent._rng)):
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(
                        jax.tree.leaves(left), jax.tree.leaves(right))), name)
                np.testing.assert_array_equal(run.trainer.buffer._observations[:3],
                                              baseline.buffer._observations[:3])
                baseline.close()
                restored = OnlineExp2(cfg(), root)
                restored.restore(next((restored.root / "state").glob("step_*")))
                self.assertEqual(restored.shadows["small"].updates, run.shadows["small"].updates)
                self.assertEqual(len(restored.rows), len(run.rows))
                rng_state = np.random.get_state()
                batch_a = run.trainer.buffer.sample()
                np.random.set_state(rng_state)
                batch_b = restored.trainer.buffer.sample()
                for field in batch_a:
                    np.testing.assert_array_equal(batch_a[field], batch_b[field])
                for current, batch in ((run, batch_a), (restored, batch_b)):
                    batch["observation"] = current.trainer.agent._normalize(batch["observation"])
                    batch["next_observation"] = current.trainer.agent._normalize(batch["next_observation"])
                for name in ("observation", "next_observation"):
                    np.testing.assert_array_equal(batch_a[name], batch_b[name])
                left, _ = paired_shadow_update(run.trainer._sac_agent.actor,
                                                run.trainer._sac_agent.temperature, run.shadows, batch_a,
                                                run.key, .99, 1, .005)
                right, _ = paired_shadow_update(restored.trainer._sac_agent.actor,
                                                 restored.trainer._sac_agent.temperature, restored.shadows,
                                                 batch_b, restored.key, .99, 1, .005)
                for name in ("small", "large"):
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(
                        jax.tree.leaves(left[name].critic.params),
                        jax.tree.leaves(right[name].critic.params))))
                original_rows = len(restored.rows)
                restored.run()
                self.assertEqual(len(restored.rows), original_rows)
