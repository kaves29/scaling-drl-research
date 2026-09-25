"""Regression coverage for throughput Steps 1 & 2 (2026-09-24):
- Step 1: actor_grad_cosine is only computed every actor_grad_cosine_every
  update()s (SACAgent.update(), scale_rl/agents/sac/sac_agent.py).
- Step 2: SACAgent.update() no longer calls float() on every value -
  materialization moves to WandbTrainerLogger.averages()/log_metric()
  (scale_rl/common/logger.py).

No GPU available for this session's work - these tests run the real
(tiny) SACAgent on CPU, which is sufficient to verify both changes'
*correctness* (cadence gating, and that deferring float() doesn't change
the averaged numbers beyond ordinary float32-accumulation precision) even
though it says nothing about the *throughput* these changes are meant to
buy - that still needs a real GPU measurement (see profile_throughput.py).
"""

import unittest

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

from scale_rl.agents import create_agent
from scale_rl.common.logger import AverageMeterDict

OBS_DIM = 4
ACT_DIM = 2


def _make_cfg(actor_grad_cosine_every=30):
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
            "actor_grad_cosine_every": actor_grad_cosine_every,
        }
    )


def _make_spaces():
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
    return observation_space, action_space


def _make_random_batch(rng, batch_size=8):
    return {
        "observation": rng.standard_normal((batch_size, OBS_DIM)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(batch_size, ACT_DIM)).astype(np.float32),
        "reward": rng.standard_normal(batch_size).astype(np.float32),
        "terminated": np.zeros(batch_size, dtype=np.float32),
        "truncated": np.zeros(batch_size, dtype=np.float32),
        "next_observation": rng.standard_normal((batch_size, OBS_DIM)).astype(np.float32),
    }


class TestStep1ActorGradCosineCadence(unittest.TestCase):
    def test_key_present_only_on_cadence_steps(self):
        K = 5
        cfg = _make_cfg(actor_grad_cosine_every=K)
        obs_space, act_space = _make_spaces()
        agent = create_agent(observation_space=obs_space, action_space=act_space, cfg=cfg)
        rng = np.random.default_rng(0)

        present_steps = []
        for step in range(1, 3 * K + 1):
            batch = _make_random_batch(rng)
            update_info = agent.update(step, batch)
            if "train/actor_grad_cosine" in update_info:
                present_steps.append(step)

        expected = [s for s in range(1, 3 * K + 1) if s % K == 0]
        self.assertEqual(present_steps, expected)

    def test_default_cadence_is_30(self):
        cfg = _make_cfg()
        self.assertEqual(cfg.actor_grad_cosine_every, 30)


class TestStep2DeferredMaterializationNumericalEquivalence(unittest.TestCase):
    """Compares the OLD behavior (float() immediately, every update, fed
    into a plain-Python-float running average) against the NEW behavior
    (device arrays accumulated via AverageMeterDict, float() once at the
    end) over real SACAgent.update() calls, using the same batches for
    both. Reports the actual measured max absolute difference rather than
    assuming bit-identical."""

    def test_deferred_vs_eager_materialization(self):
        cfg = _make_cfg(actor_grad_cosine_every=5)
        obs_space, act_space = _make_spaces()
        agent = create_agent(observation_space=obs_space, action_space=act_space, cfg=cfg)
        rng = np.random.default_rng(1)

        eager_meters = AverageMeterDict()  # OLD-style: float() at every single update
        deferred_meters = AverageMeterDict()  # NEW-style: device arrays accumulated

        N = 60
        for step in range(1, N + 1):
            batch = _make_random_batch(rng)
            update_info = agent.update(step, batch)  # returns device arrays (Step 2 applied)
            for k, v in update_info.items():
                eager_meters.update(k, float(v))
                deferred_meters.update(k, v)

        eager_avgs = eager_meters.averages()
        deferred_avgs = {k: float(v) for k, v in deferred_meters.averages().items()}

        self.assertEqual(set(eager_avgs.keys()), set(deferred_avgs.keys()))
        self.assertIn("train/actor_grad_cosine", eager_avgs)  # cadence=5, N=60 -> 12 samples, must be present

        max_abs_diff = 0.0
        max_abs_diff_key = None
        for k in eager_avgs:
            diff = abs(eager_avgs[k] - deferred_avgs[k])
            print(f"[step2-equivalence] {k}: eager={eager_avgs[k]!r} deferred={deferred_avgs[k]!r} diff={diff!r}")
            if diff > max_abs_diff:
                max_abs_diff = diff
                max_abs_diff_key = k
        print(f"[step2-equivalence] max abs diff over {N} updates: {max_abs_diff!r} (key={max_abs_diff_key})")

        # Not asserting exact bit-identity: eager accumulates in Python
        # float (float64), deferred accumulates in JAX's default float32 -
        # a real, expected source of divergence at accumulation scale, not
        # a bug. Asserting it stays small (relative to typical loss/metric
        # magnitudes here) rather than silently passing on any value.
        self.assertLess(max_abs_diff, 1e-3, f"Unexpectedly large divergence at key={max_abs_diff_key}")


if __name__ == "__main__":
    unittest.main()
