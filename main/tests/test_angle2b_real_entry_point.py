"""Exercises experiments.angle_2_b.run() itself, end to end, on tiny/synthetic
data - not the building-block tests (matchup_2b.py/null_baseline.py/etc. are
all individually tested, but no test previously called angle_2_b.run()
itself).

Same two workarounds as test_angle2a_real_entry_point.py, for the same
reasons (see that file's class docstring for the full orbax finding):
  1. wandb.init() mocked (angle_2_b.py's _log_to_wandb calls it
     unconditionally).
  2. SACAgent.save_checkpoint/load_checkpoint patched to absolute-ify their
     checkpoint_dir argument before delegating to the real method - orbax
     rejects relative paths outright, and Angle 2B's config has NO field to
     override the pool root at all (confirmed by inspection: run_cfg has no
     pool_root, and angle_2_b.py's run_angle_2b_analysis call doesn't pass
     one either, so a real invocation would always fall back to
     analysis.baseline_calibration_pool.POOL_STORAGE_ROOT's relative
     default) - a second, independent gap from the orbax path issue.

Also chdirs into a tmpdir (restored in tearDown), since the pool root has no
override and must be reached via its real relative default.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import wandb
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import ProbeCapture
from experiments.angle_2a.storage import save_frozen_agent_snapshot
from experiments.angle_2b.storage import analysis_dir

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs")
ENVIRONMENT = "cheetah-run"
OBS_DIM = 17
ACT_DIM = 6


class _FakeWandbRun:
    def __init__(self):
        self.name = "fake-run"
        self.id = "fake-run-id"
        self.summary = {}

    def update(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def _fake_wandb_init(*args, **kwargs):
    run = _FakeWandbRun()
    wandb.run = run
    return run


def _make_agent_cfg(seed, critic_num_blocks=1, critic_hidden_dim=8):
    return {
        "agent_type": "sac", "seed": seed, "num_train_envs": 1, "max_episode_steps": 20,
        "normalize_observation": False, "actor_block_type": "residual", "actor_num_blocks": 1,
        "actor_hidden_dim": 8, "actor_learning_rate": 1e-4, "actor_weight_decay": 1e-2,
        "critic_block_type": "residual", "critic_num_blocks": critic_num_blocks, "critic_hidden_dim": critic_hidden_dim,
        "critic_learning_rate": 1e-4, "critic_weight_decay": 1e-2, "critic_use_cdq": False,
        "temp_target_entropy": None, "temp_target_entropy_coef": -0.5, "temp_initial_value": 0.01,
        "temp_learning_rate": 1e-4, "temp_weight_decay": 0.0, "target_tau": 0.005, "gamma": 0.99,
        "n_step": 1, "mixed_precision": False, "actor_sparsity": 0.0, "critic_sparsity": 0.0,
    }


def _make_spaces():
    import gymnasium as gym

    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
    return observation_space, action_space


def _persist(environment, seed, matchup_name, role, root, agent_seed, critic_num_blocks=1, critic_hidden_dim=8, n=20):
    from scale_rl.agents import create_agent

    observation_space, action_space = _make_spaces()
    agent_cfg = _make_agent_cfg(agent_seed, critic_num_blocks, critic_hidden_dim)
    agent = create_agent(observation_space, action_space, OmegaConf.create(agent_cfg))
    rng = np.random.default_rng(agent_seed)
    for step in range(3):
        agent.update(step, {
            "observation": rng.normal(size=(16, OBS_DIM)).astype(np.float32),
            "action": rng.uniform(-1, 1, size=(16, ACT_DIM)).astype(np.float32),
            "reward": rng.normal(size=(16,)).astype(np.float32),
            "terminated": np.zeros((16,), dtype=np.float32),
            "next_observation": rng.normal(size=(16, OBS_DIM)).astype(np.float32),
        })
    probe_capture = ProbeCapture(capacity=n, observation_shape=(OBS_DIM,), action_shape=(ACT_DIM,))
    for i in range(n):
        probe_capture.add(
            i, rng.normal(size=(OBS_DIM,)).astype(np.float32), rng.uniform(-1, 1, size=(ACT_DIM,)).astype(np.float32),
            env_state=None,
        )
    save_frozen_agent_snapshot(environment, seed, matchup_name, role, agent, probe_capture, agent_cfg=agent_cfg, root=root)


class Angle2BRealEntryPointTest(unittest.TestCase):
    def setUp(self):
        GlobalHydra.instance().clear()
        self.tmpdir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.wandb_init_patcher = mock.patch("wandb.init", side_effect=_fake_wandb_init)
        self.wandb_init_patcher.start()

        from scale_rl.agents.sac.sac_agent import SACAgent

        real_save, real_load = SACAgent.save_checkpoint, SACAgent.load_checkpoint

        def _abs_save(agent_self, checkpoint_dir):
            return real_save(agent_self, os.path.abspath(checkpoint_dir))

        def _abs_load(agent_self, checkpoint_dir):
            return real_load(agent_self, os.path.abspath(checkpoint_dir))

        self.checkpoint_patcher = mock.patch.multiple(SACAgent, save_checkpoint=_abs_save, load_checkpoint=_abs_load)
        self.checkpoint_patcher.start()

    def tearDown(self):
        self.checkpoint_patcher.stop()
        self.wandb_init_patcher.stop()
        os.chdir(self.original_cwd)
        GlobalHydra.instance().clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_completes_end_to_end_against_the_real_pool(self):
        from experiments.angle_2_b import run

        angle_2a_root = "results/angle_2a"
        pool_root = "results/baseline_calibration_pool"  # real default, unconfigurable - see class docstring

        _persist(ENVIRONMENT, 1, "matchup_1", "D", angle_2a_root, agent_seed=11, critic_num_blocks=2, critic_hidden_dim=16)
        _persist(ENVIRONMENT, 1, "matchup_1", "R", angle_2a_root, agent_seed=12, critic_num_blocks=1, critic_hidden_dim=8)
        for seed in range(1, 5):
            _persist(ENVIRONMENT, seed, "baseline_pool", "pool", pool_root, agent_seed=100 + seed)

        fake_identities = [type("Ident", (), {"seed": s})() for s in range(1, 5)]
        with mock.patch(
            "experiments.angle_2b.null_baseline.get_baseline_calibration_pool",
            return_value=fake_identities,
        ):
            run({
                "config_path": CONFIG_PATH,
                "config_name": "base_angle2b",
                "overrides": [
                    f"env_name={ENVIRONMENT}", "seed=1",
                    "angle_2_b.matchup_names=[matchup_1]",
                    f"angle_2_b.angle_2a_results_root={angle_2a_root}",
                    "angle_2_b.output_root=results/angle_2b",
                    "angle_2_b.num_states_per_source=5",
                ],
            })

        out_dir = analysis_dir(ENVIRONMENT, 1, "matchup_1", root="results/angle_2b")
        self.assertTrue((out_dir / "run_metadata.json").exists())
        self.assertTrue((out_dir / "null_distribution.csv").exists())
        self.assertTrue((out_dir / "gradients.npz").exists())

        import json
        with open(out_dir / "run_metadata.json") as f:
            metadata = json.load(f)
        self.assertEqual(len(metadata["null_pool_pairs_used"]), 6)  # C(4,2)
        for metric in ("d_dir", "d_mag", "d_grad"):
            self.assertTrue(np.isfinite(metadata["primary"][metric]))


if __name__ == "__main__":
    unittest.main()
