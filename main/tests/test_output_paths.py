"""Checkpoint paths remain absolute across current Exp1 and Exp2."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = str(REPO / "configs")


def _make_agent(normalize_observation):
    from scale_rl.agents import create_agent

    cfg = OmegaConf.create({
        "agent_type": "sac", "seed": 0, "num_train_envs": 1, "max_episode_steps": 100,
        "normalize_observation": normalize_observation, "actor_block_type": "residual",
        "actor_num_blocks": 1, "actor_hidden_dim": 8, "actor_learning_rate": 1e-4,
        "actor_weight_decay": 1e-2, "critic_block_type": "residual", "critic_num_blocks": 1,
        "critic_hidden_dim": 8, "critic_learning_rate": 1e-4, "critic_weight_decay": 1e-2,
        "critic_use_cdq": False, "temp_target_entropy": None, "temp_target_entropy_coef": -0.5,
        "temp_initial_value": 0.01, "temp_learning_rate": 1e-4, "temp_weight_decay": 0.0,
        "target_tau": 0.005, "gamma": 0.99, "n_step": 1, "mixed_precision": False,
        "actor_sparsity": 0.0, "critic_sparsity": 0.0,
    })
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return create_agent(observation_space=space, action_space=space, cfg=cfg)


class CheckpointBoundaryRejectsRelativePathsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.addCleanup(os.chdir, original_cwd)

    def test_save_and_load_reject_relative_paths_before_touching_disk(self):
        for normalize in (False, True):
            agent = _make_agent(normalize)
            for method in (agent.save_checkpoint, agent.load_checkpoint):
                with self.subTest(normalize=normalize, method=method.__name__):
                    with self.assertRaisesRegex(ValueError, "checkpoint_dir must be an absolute path"):
                        method("relative/ckpt")
                    self.assertEqual(os.listdir(self.tmpdir), [])

    def test_absolute_paths_still_round_trip(self):
        agent = _make_agent(True)
        ckpt = os.path.join(self.tmpdir, "ckpt")
        agent.save_checkpoint(ckpt)
        _make_agent(True).load_checkpoint(ckpt)



if __name__ == "__main__":
    unittest.main()
