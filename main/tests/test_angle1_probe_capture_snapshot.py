"""Covers the save_probe_capture_snapshot opt-in instrumentation added to
experiments/angle_1.py for the shared baseline-calibration pool (2026-09-08 -
see analysis/baseline_calibration_pool.py).

Does NOT invoke experiments.angle_1.run() directly: that function
unconditionally calls wandb.init() (os.environ["WANDB_MODE"] = "online" is
set right before constructing WandbTrainerLogger), and no existing test in
this repo exercises run() end-to-end for exactly that reason - the
established pattern here is real production runs plus dedicated tests for
the sub-components run() calls (metrics_store, pipeline, onset_detection).

Instead, this test directly exercises the same building blocks angle_1.py's
new code path composes (create_envs, create_agent, ProbeCapture,
capture_env_state, save_frozen_agent_snapshot), in the identical order
angle_1.py now uses them, confirming: per-transition (observation, action,
env_state) capture during training works, and the resulting frozen snapshot
+ probe-capture pickle round-trip correctly through Angle 2B's own loader
and ProbeCapture.load_fresh. The default (flag-absent) path in angle_1.py
itself was verified by direct code inspection instead: the new code is
gated behind `if save_probe_capture_snapshot:` (default False) at every
insertion point, so no existing invocation's behavior changes - see the
End-of-Task Summary for what was live-tested vs. reasoned about.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import ProbeCapture
from experiments.angle_2a.env_state import capture_env_state
from experiments.angle_2a.storage import save_frozen_agent_snapshot
from experiments.angle_2b.checkpoint_io import load_frozen_agent_snapshot
from scale_rl.agents import create_agent
from scale_rl.envs import create_envs

AGENT_CFG = {
    "agent_type": "sac", "seed": 6, "num_train_envs": 1, "max_episode_steps": 20,
    "normalize_observation": False, "actor_block_type": "residual", "actor_num_blocks": 1,
    "actor_hidden_dim": 8, "actor_learning_rate": 1e-4, "actor_weight_decay": 1e-2,
    "critic_block_type": "residual", "critic_num_blocks": 2, "critic_hidden_dim": 512,
    "critic_learning_rate": 1e-4, "critic_weight_decay": 1e-2, "critic_use_cdq": False,
    "temp_target_entropy": None, "temp_target_entropy_coef": -0.5, "temp_initial_value": 0.01,
    "temp_learning_rate": 1e-4, "temp_weight_decay": 0.0, "target_tau": 0.005, "gamma": 0.99,
    "n_step": 1, "mixed_precision": False, "actor_sparsity": 0.0, "critic_sparsity": 0.0,
}


class TestProbeCaptureSnapshotMechanism(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_capture_during_training_then_snapshot_round_trips(self):
        num_steps = 15
        train_env, eval_env = create_envs(
            env_type="dmc", seed=6, env_name="cheetah-run", num_train_envs=1,
            num_eval_envs=1, rescale_action=True, no_termination=False,
            action_repeat=1, reward_scale=1.0, max_episode_steps=20,
        )
        try:
            single_env = train_env.envs[0]
            observation_space = train_env.observation_space
            action_space = train_env.action_space

            agent = create_agent(
                observation_space=observation_space, action_space=action_space,
                cfg=OmegaConf.create(AGENT_CFG),
            )
            probe_capture = ProbeCapture(
                capacity=num_steps,
                observation_shape=observation_space.shape[-1:],
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
                next_observations, rewards, terminateds, truncateds, _infos = train_env.step(actions)
                timestep = {"next_observation": next_observations}
                observations = next_observations

            self.assertEqual(len(probe_capture), num_steps)

            snapshot_paths = save_frozen_agent_snapshot(
                "cheetah-run", 6, "baseline_pool", "pool", agent, probe_capture,
                agent_cfg=AGENT_CFG, root=self.tmpdir,
            )
            probe_capture.save(str(snapshot_paths["checkpoint_dir"]))

            # Angle 2B's own loader must accept role="pool".
            loaded = load_frozen_agent_snapshot("cheetah-run", 6, "baseline_pool", "pool", root=self.tmpdir)
            self.assertEqual(loaded.states.shape[0], num_steps)
            self.assertEqual(loaded.actions.shape[0], num_steps)

            # ProbeCapture.load_fresh must recover the env_state-inclusive data
            # save_frozen_agent_snapshot's own NPZ discards (needed for Angle
            # 2A's refactored null-baseline MC rollouts, not just Angle 2B's
            # gradient-only needs).
            reloaded_capture = ProbeCapture.load_fresh(str(snapshot_paths["checkpoint_dir"]))
            self.assertEqual(len(reloaded_capture), num_steps)
            _idxs, _states, _actions, env_states = reloaded_capture.sample(num_steps, np.random.default_rng(0))
            self.assertTrue(all(es is not None and "physics_state" in es for es in env_states))
        finally:
            train_env.close()
            eval_env.close()


if __name__ == "__main__":
    unittest.main()
