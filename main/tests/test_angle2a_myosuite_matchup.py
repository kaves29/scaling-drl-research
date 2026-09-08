"""Confirms Angle 2A's full pipeline runs end-to-end on a MyoSuite
environment, not just env_state.py's capture/restore mechanism in isolation
(that mechanism is covered directly by tests/test_angle2a_env_state.py and
tests/test_angle2a_env_state_determinism_smoke.py).

This is the "at least one small real Angle 2A matchup runs end-to-end on a
MyoSuite environment" verification for the 2026-09-07 MyoSuite-support task:
a tiny, real run_matchup() call (train D and R fresh, sample probes, run
real MC rollouts, compute diagonal errors) on myo-elbow-pose-random - no
mocks, no myosuite check_single_env_type gating disabled early like every
other MyoSuite core-4 task would use in a real Angle 2A invocation.
"""

import os
import shutil
import tempfile
import unittest

from omegaconf import OmegaConf

from experiments.angle_2a.config import RoleArchitecture
from experiments.angle_2a.matchup import run_matchup

SCALED = RoleArchitecture(role="scaled_a", critic_num_blocks=1, critic_hidden_dim=8)
REFERENCE = RoleArchitecture(role="reference", critic_num_blocks=1, critic_hidden_dim=8)


def _make_myosuite_cfg(seed: int = 303):
    return OmegaConf.create(
        {
            "seed": seed, "gamma": 0.99, "num_interaction_steps": 40,
            "env": {
                "env_type": "myosuite", "env_name": "myo-elbow-pose-random", "seed": seed,
                "num_train_envs": 1, "num_eval_envs": 1, "rescale_action": True,
                "no_termination": False, "action_repeat": 1, "reward_scale": 1.0, "max_episode_steps": 20,
            },
            "buffer": {
                "buffer_class_type": "numpy", "buffer_type": "uniform", "n_step": 1, "gamma": 0.99,
                "max_length": 1000, "min_length": 3, "add_batch_size": 1, "sample_batch_size": 4,
            },
            "updates_per_interaction_step": 1,
            "agent": {
                "agent_type": "sac", "seed": seed, "num_train_envs": 1, "max_episode_steps": 20,
                "normalize_observation": False, "actor_block_type": "residual", "actor_num_blocks": 1,
                "actor_hidden_dim": 8, "actor_learning_rate": 1e-4, "actor_weight_decay": 1e-2,
                "critic_block_type": "residual", "critic_num_blocks": 1, "critic_hidden_dim": 8,
                "critic_learning_rate": 1e-4, "critic_weight_decay": 1e-2, "critic_use_cdq": False,
                "temp_target_entropy": None, "temp_target_entropy_coef": -0.5, "temp_initial_value": 0.01,
                "temp_learning_rate": 1e-4, "temp_weight_decay": 0.0, "target_tau": 0.005, "gamma": 0.99,
                "n_step": 1, "mixed_precision": False, "actor_sparsity": 0.0, "critic_sparsity": 0.0,
            },
        }
    )


class TestAngle2AMyosuiteMatchupEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_real_matchup_trains_evaluates_and_computes_diagonal_errors_on_myosuite(self):
        cfg = _make_myosuite_cfg()

        result = run_matchup(
            matchup_name="myosuite_smoke_matchup",
            scaled_architecture=SCALED,
            scaled_architecture_label="D1W8",
            reference_architecture=REFERENCE,
            reference_architecture_label="R1W8",
            onset_step=20,
            onset_source_run_key="myosuite_smoke_test",
            base_cfg=cfg,
            seed=cfg.seed,
            environment="myo-elbow-pose-random",
            experiment_name="angle_2a_myosuite_smoke_test",
            num_probes_per_source=2,
            num_mc_rollouts=2,
            output_root=self.tmpdir,
            wandb_enabled=False,
        )

        self.assertEqual(len(result.probes), 4)  # 2 D-source + 2 R-source
        for probe in result.probes:
            self.assertIsNotNone(probe.q_d)
            self.assertIsNotNone(probe.q_r)
            self.assertIsNotNone(probe.mc_return)
            self.assertIsNotNone(probe.diagonal_error)
            self.assertGreaterEqual(len(probe.mc_rollout_returns), 2)

        self.assertIn("mean_diagonal_error_D", result.run_metadata)
        self.assertIn("mean_diagonal_error_R", result.run_metadata)
        self.assertTrue(os.path.exists(result.output_paths["probes_csv"]))


if __name__ == "__main__":
    unittest.main()
