"""Covers Angle 2A's refactored null-baseline mechanism (2026-09-08): drawing
from the shared baseline-calibration pool (analysis/baseline_calibration_pool.py)
instead of training fresh null_matchup pairs per seed.

Live-tested with 4 real, tiny pool agents (not the full 10) sharing one real
dm_control environment, confirming: all C(4,2)=6 pairs get evaluated, each
pair's diagonal errors are computed via the identical probes.py machinery the
main comparison uses, and pool_null_values pools both sides of every pair
into one flat list.
"""

import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import ProbeCapture
from experiments.angle_2a.env_state import capture_env_state
from experiments.angle_2a.pool_null_baseline import (
    compute_pool_null_distribution,
    load_or_compute_pool_null_distribution,
    pool_null_values,
)
from experiments.angle_2a.storage import save_frozen_agent_snapshot
from scale_rl.agents import create_agent
from scale_rl.envs import create_envs

ENVIRONMENT = "cheetah-run"
TEST_SEEDS = (101, 102, 103, 104)


def _make_agent_cfg(seed):
    return {
        "agent_type": "sac", "seed": seed, "num_train_envs": 1, "max_episode_steps": 20,
        "normalize_observation": False, "actor_block_type": "residual", "actor_num_blocks": 1,
        "actor_hidden_dim": 8, "actor_learning_rate": 1e-4, "actor_weight_decay": 1e-2,
        "critic_block_type": "residual", "critic_num_blocks": 1, "critic_hidden_dim": 8,
        "critic_learning_rate": 1e-4, "critic_weight_decay": 1e-2, "critic_use_cdq": False,
        "temp_target_entropy": None, "temp_target_entropy_coef": -0.5, "temp_initial_value": 0.01,
        "temp_learning_rate": 1e-4, "temp_weight_decay": 0.0, "target_tau": 0.005, "gamma": 0.99,
        "n_step": 1, "mixed_precision": False, "actor_sparsity": 0.0, "critic_sparsity": 0.0,
    }


def _make_base_cfg():
    return OmegaConf.create(
        {
            "gamma": 0.99,
            "env": {
                "env_type": "dmc", "env_name": ENVIRONMENT, "seed": 0,
                "num_train_envs": 1, "num_eval_envs": 1, "rescale_action": True,
                "no_termination": False, "action_repeat": 1, "reward_scale": 1.0, "max_episode_steps": 20,
            },
        }
    )


class TestPoolNullBaseline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _train_and_persist_pool_agent(self, seed, num_steps=15):
        train_env, eval_env = create_envs(
            env_type="dmc", seed=seed, env_name=ENVIRONMENT, num_train_envs=1,
            num_eval_envs=1, rescale_action=True, no_termination=False,
            action_repeat=1, reward_scale=1.0, max_episode_steps=20,
        )
        try:
            single_env = train_env.envs[0]
            observation_space = train_env.observation_space
            action_space = train_env.action_space
            agent_cfg = _make_agent_cfg(seed)
            agent = create_agent(observation_space, action_space, OmegaConf.create(agent_cfg))
            probe_capture = ProbeCapture(
                capacity=num_steps, observation_shape=observation_space.shape[-1:],
                action_shape=action_space.shape[-1:],
            )
            observations, _ = train_env.reset()
            timestep = None
            for interaction_step in range(1, num_steps + 1):
                env_state = capture_env_state(single_env, "dmc")
                if timestep is not None:
                    actions = agent.sample_actions(interaction_step, prev_timestep=timestep, training=True)
                else:
                    actions = train_env.action_space.sample()
                probe_capture.add(interaction_step - 1, observations[0], actions[0], env_state)
                next_observations, _r, _t, _tr, _i = train_env.step(actions)
                timestep = {"next_observation": next_observations}
                observations = next_observations

            snapshot_paths = save_frozen_agent_snapshot(
                ENVIRONMENT, seed, "baseline_pool", "pool", agent, probe_capture,
                agent_cfg=agent_cfg, root=self.tmpdir,
            )
            probe_capture.save(str(snapshot_paths["checkpoint_dir"]))
        finally:
            train_env.close()
            eval_env.close()

    def test_all_pairs_evaluated_with_real_diagonal_errors(self):
        for seed in TEST_SEEDS:
            self._train_and_persist_pool_agent(seed)

        fake_identities = [
            type("Ident", (), {"seed": s})() for s in TEST_SEEDS
        ]
        with mock.patch(
            "experiments.angle_2a.pool_null_baseline.get_baseline_calibration_pool",
            return_value=fake_identities,
        ):
            results = compute_pool_null_distribution(
                environment=ENVIRONMENT,
                base_cfg=_make_base_cfg(),
                num_probes_per_source=3,
                num_mc_rollouts=2,
                analysis_seed=99,
                angle_2a_root=self.tmpdir,
            )

        num_pairs = len(TEST_SEEDS) * (len(TEST_SEEDS) - 1) // 2
        self.assertEqual(len(results), num_pairs)
        seed_pairs = {(r.seed_a, r.seed_b) for r in results}
        self.assertEqual(len(seed_pairs), num_pairs)  # every pair distinct, no duplicates
        for r in results:
            self.assertTrue(np.isfinite(r.e_a))
            self.assertTrue(np.isfinite(r.e_b))

        values = pool_null_values(results)
        self.assertEqual(len(values), num_pairs * 2)

    def test_load_or_compute_caches_and_reuses(self):
        for seed in TEST_SEEDS:
            self._train_and_persist_pool_agent(seed)

        fake_identities = [type("Ident", (), {"seed": s})() for s in TEST_SEEDS]
        cache_root = tempfile.mkdtemp(dir=self.tmpdir)
        call_count = {"n": 0}
        real_compute = compute_pool_null_distribution

        def counting_compute(*args, **kwargs):
            call_count["n"] += 1
            return real_compute(*args, **kwargs)

        with mock.patch(
            "experiments.angle_2a.pool_null_baseline.get_baseline_calibration_pool",
            return_value=fake_identities,
        ), mock.patch(
            "experiments.angle_2a.pool_null_baseline.compute_pool_null_distribution",
            side_effect=counting_compute,
        ):
            result1 = load_or_compute_pool_null_distribution(
                environment=ENVIRONMENT, base_cfg=_make_base_cfg(), num_probes_per_source=3,
                num_mc_rollouts=2, analysis_seed=99, angle_2a_root=self.tmpdir, cache_root=cache_root,
            )
            result2 = load_or_compute_pool_null_distribution(
                environment=ENVIRONMENT, base_cfg=_make_base_cfg(), num_probes_per_source=3,
                num_mc_rollouts=2, analysis_seed=99, angle_2a_root=self.tmpdir, cache_root=cache_root,
            )

        self.assertEqual(call_count["n"], 1, "second call must hit the cache, not recompute")
        self.assertEqual(result1.null_n, result2.null_n)
        self.assertAlmostEqual(result1.null_mean, result2.null_mean, places=10)


if __name__ == "__main__":
    unittest.main()
