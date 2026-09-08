"""Smoke test for the Angle 2A construct-validity prerequisite check
(experiments/angle_2a/prereq_check.py).

Uses a real, tiny dm_control env ("cheetah-run", not a synthetic Box-space
stand-in like test_angle_2b_smoke.py) because env_state.py's capture/restore
requires real dm_control physics - this is exactly the mechanism this check
exercises (pausing training mid-loop for a "pre" evaluation, then resuming),
so a real physics simulator is the point, not incidental.

Live-tested here: real dm_control training, real SACAgent.save_checkpoint/
load_checkpoint round trips, real MC rollouts with real physics state
restore, real onset-ledger lookup. Not tested here: the full 2-architecture
x 3-seed production configuration (kept to 1-2 architectures x 1 seed to
keep this fast; the per-(architecture,seed) mechanics are what's load-bearing
and identical regardless of how many are run).
"""

import os
import shutil
import tempfile
import unittest

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import (
    ProbeCapture,
    build_agent_and_env,
    derive_rng_seed,
    run_training_loop,
    seed_global_rng_for_agent,
)
from experiments.angle_2a.config import RoleArchitecture, architecture_label
from experiments.angle_2a.prereq_check import run_one_prereq_seed, run_prereq_check
from experiments.angle_2a.prereq_storage import prereq_seed_dir
from scale_rl.agents import create_agent
from utils.onset_ledger import WandbIdentity, log_onset_event

ENVIRONMENT = "cheetah-run"
DEDICATED_SEED = 9101
POST_STEP = 30  # the fake t* this test's onset ledger entry declares
NUM_INTERACTION_STEPS = 40  # burn_in_fraction=0.25 -> pre_step = 10
BURN_IN_FRACTION = 0.25


class _FakeRun:
    class _Summary(dict):
        def update(self, d):
            dict.update(self, d)

    def __init__(self, name, id_):
        self.name = name
        self.id = id_
        self.summary = self._Summary()


def _make_cfg(seed: int = DEDICATED_SEED):
    return OmegaConf.create(
        {
            "seed": seed,
            "gamma": 0.99,
            "num_interaction_steps": NUM_INTERACTION_STEPS,
            "env": {
                "env_type": "dmc",
                "env_name": ENVIRONMENT,
                "seed": seed,
                "num_train_envs": 1,
                "num_eval_envs": 1,
                "rescale_action": True,
                "no_termination": False,
                "action_repeat": 1,
                "reward_scale": 1.0,
                "max_episode_steps": 20,
            },
            "buffer": {
                "buffer_class_type": "numpy",
                "buffer_type": "uniform",
                "n_step": 1,
                "gamma": 0.99,
                "max_length": 1000,
                "min_length": 3,
                "add_batch_size": 1,
                "sample_batch_size": 4,
            },
            "updates_per_interaction_step": 1,
            "agent": {
                "agent_type": "sac",
                "seed": seed,
                "num_train_envs": 1,
                "max_episode_steps": 20,
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
            },
        }
    )


class TestAngle2APrereqCheck(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ledger_root = os.path.join(self.tmpdir, "ledgers")
        self.output_root = os.path.join(self.tmpdir, "prereq_out")
        self.architecture = RoleArchitecture(role="scaled_a", critic_num_blocks=1, critic_hidden_dim=8)
        self.arch_label = architecture_label(self.architecture)
        self._log_onset(DEDICATED_SEED, POST_STEP)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _log_onset(self, seed: int, onset_step: int):
        run_key = f"angle1_{self.arch_label}_{ENVIRONMENT}_seed{seed}"
        row = {
            "run_key": run_key,
            "exact_run_name": run_key,
            "wandb_run_id": f"wid_{run_key}",
            "architecture": self.arch_label,
            "environment": ENVIRONMENT,
            "seed": seed,
            "critic_degradation_onset_step": onset_step,
            "critic_degradation_method": "td_variance_p95_sustained_v1",
            "propagation_onset_step": None,
            "propagation_method": None,
            "propagation_lag": None,
            "status": "success",
            "detection_notes": "test row",
        }
        identity = WandbIdentity(run_obj=_FakeRun(run_key, row["wandb_run_id"]))
        log_onset_event("angle_1", self.arch_label, row, identity=identity, root=self.ledger_root, mirror_to_wandb=False)

    def test_pre_and_post_steps_computed_correctly(self):
        result = run_one_prereq_seed(
            architecture=self.architecture, architecture_label=self.arch_label, base_cfg=_make_cfg(),
            seed=DEDICATED_SEED, environment=ENVIRONMENT, onset_source_experiment="angle_1",
            onset_ledger_root=self.ledger_root, burn_in_fraction=BURN_IN_FRACTION,
            num_probes=3, num_mc_rollouts=2, output_root=self.output_root,
        )
        self.assertEqual(result.pre_step, 10)
        self.assertEqual(result.post_step, POST_STEP)

    def test_checkpoints_are_persisted_and_reloadable(self):
        result = run_one_prereq_seed(
            architecture=self.architecture, architecture_label=self.arch_label, base_cfg=_make_cfg(),
            seed=DEDICATED_SEED, environment=ENVIRONMENT, onset_source_experiment="angle_1",
            onset_ledger_root=self.ledger_root, burn_in_fraction=BURN_IN_FRACTION,
            num_probes=3, num_mc_rollouts=2, output_root=self.output_root,
        )
        out_dir = prereq_seed_dir(ENVIRONMENT, DEDICATED_SEED, self.arch_label, root=self.output_root)
        pre_dir = out_dir / "checkpoints" / "pre"
        post_dir = out_dir / "checkpoints" / "post"
        self.assertTrue((pre_dir / "agent_ckpt").exists())
        self.assertTrue((post_dir / "agent_ckpt").exists())

        observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        cfg = _make_cfg()
        reloaded = create_agent(observation_space=observation_space, action_space=action_space, cfg=cfg.agent)
        reloaded.load_checkpoint(str(post_dir))  # must not raise

    def test_pre_and_post_probes_use_different_derived_seeds_and_differ(self):
        result = run_one_prereq_seed(
            architecture=self.architecture, architecture_label=self.arch_label, base_cfg=_make_cfg(),
            seed=DEDICATED_SEED, environment=ENVIRONMENT, onset_source_experiment="angle_1",
            onset_ledger_root=self.ledger_root, burn_in_fraction=BURN_IN_FRACTION,
            num_probes=3, num_mc_rollouts=2, output_root=self.output_root,
        )
        seed_context = f"prereq:{self.arch_label}:D:seed{DEDICATED_SEED}"
        pre_seed = derive_rng_seed(DEDICATED_SEED, f"{seed_context}:pre_probes")
        post_seed = derive_rng_seed(DEDICATED_SEED, f"{seed_context}:post_probes")
        self.assertNotEqual(pre_seed, post_seed)

        out_dir = prereq_seed_dir(ENVIRONMENT, DEDICATED_SEED, self.arch_label, root=self.output_root)
        with np.load(out_dir / "probes_pre_arrays.npz") as pre_arr, np.load(out_dir / "probes_post_arrays.npz") as post_arr:
            self.assertEqual(len(pre_arr["probe_id"]), 3)
            self.assertEqual(len(post_arr["probe_id"]), 3)
            # different sampling pool sizes (10 vs. 30 available transitions)
            # and different derived seeds - not expected to coincide.
            self.assertFalse(np.array_equal(pre_arr["state"], post_arr["state"]))

    def test_summary_is_descriptive_not_a_pass_fail_gate(self):
        result = run_one_prereq_seed(
            architecture=self.architecture, architecture_label=self.arch_label, base_cfg=_make_cfg(),
            seed=DEDICATED_SEED, environment=ENVIRONMENT, onset_source_experiment="angle_1",
            onset_ledger_root=self.ledger_root, burn_in_fraction=BURN_IN_FRACTION,
            num_probes=3, num_mc_rollouts=2, output_root=self.output_root,
        )
        self.assertTrue(np.isfinite(result.e_d_pre))
        self.assertTrue(np.isfinite(result.e_d_post))
        self.assertEqual(result.direction_consistent_with_decline, result.e_d_post > result.e_d_pre)

        summary = result.output_paths["summary_data"]
        self.assertNotIn("passed", summary)
        self.assertNotIn("pass", summary)
        self.assertNotIn("gate_result", summary)
        self.assertIn("direction_consistent_with_decline", summary)

    def test_training_continues_unperturbed_after_pre_checkpoint_evaluation(self):
        """The strongest correctness check: the pre-checkpoint's MC-rollout
        evaluation must not leak into continued training. Verified by
        comparing against an uninterrupted control run of the exact same
        (architecture, seed): since dm_control dynamics and JAX action
        sampling are both deterministic given the same physics state and rng
        stream, if restore_env_state/agent.load_checkpoint correctly undo the
        evaluation's side effects, the two runs' full transition histories
        must be bit-identical through post_step."""
        cfg_control = _make_cfg()
        train_env, eval_env, single_env, buffer, agent, obs_space, act_space = build_agent_and_env(
            self.architecture, cfg_control
        )
        try:
            probe_capture_control = ProbeCapture(
                capacity=POST_STEP, observation_shape=obs_space.shape[-1:], action_shape=act_space.shape[-1:]
            )
            seed_global_rng_for_agent(DEDICATED_SEED, f"prereq:{self.arch_label}:D:seed{DEDICATED_SEED}")
            for _ in run_training_loop(agent, buffer, train_env, single_env, probe_capture_control, cfg_control, POST_STEP):
                pass
            control_observations = probe_capture_control._observations.copy()
            control_actions = probe_capture_control._actions.copy()
        finally:
            train_env.close()
            eval_env.close()

        # Comparison requires access to the interrupted run's own final
        # probe_capture, which run_one_prereq_seed doesn't expose - re-run
        # the identical interrupted procedure inline here instead, so the
        # raw transition arrays can be compared directly.
        from experiments.angle_2a.env_state import capture_env_state, restore_env_state

        cfg_test = _make_cfg()
        train_env2, eval_env2, single_env2, buffer2, agent2, obs_space2, act_space2 = build_agent_and_env(
            self.architecture, cfg_test
        )
        try:
            probe_capture_test = ProbeCapture(
                capacity=POST_STEP, observation_shape=obs_space2.shape[-1:], action_shape=act_space2.shape[-1:]
            )
            seed_global_rng_for_agent(DEDICATED_SEED, f"prereq:{self.arch_label}:D:seed{DEDICATED_SEED}")
            pre_step = 10
            tmp_ckpt = os.path.join(self.tmpdir, "inline_pre_ckpt")
            for step in run_training_loop(agent2, buffer2, train_env2, single_env2, probe_capture_test, cfg_test, POST_STEP):
                if step == pre_step:
                    agent2.save_checkpoint(tmp_ckpt)
                    captured = capture_env_state(single_env2, cfg_test.env.env_type)
                    # Simulate the perturbation MC-rollout evaluation causes:
                    # consume rng via sample_actions and step the env.
                    for _ in range(5):
                        a = agent2.sample_actions(0, {"next_observation": np.zeros((1,) + obs_space2.shape)}, training=False)
                        single_env2.step(a[0])
                    agent2.load_checkpoint(tmp_ckpt)
                    restore_env_state(single_env2, captured)
            test_observations = probe_capture_test._observations.copy()
            test_actions = probe_capture_test._actions.copy()
        finally:
            train_env2.close()
            eval_env2.close()

        np.testing.assert_array_equal(control_observations, test_observations)
        np.testing.assert_array_equal(control_actions, test_actions)

    def test_run_prereq_check_covers_every_architecture_and_seed(self):
        second_architecture = RoleArchitecture(role="scaled_b", critic_num_blocks=2, critic_hidden_dim=8)
        second_label = architecture_label(second_architecture)
        self.arch_label_2 = second_label
        run_key = f"angle1_{second_label}_{ENVIRONMENT}_seed{DEDICATED_SEED}"
        row = {
            "run_key": run_key, "exact_run_name": run_key, "wandb_run_id": f"wid_{run_key}",
            "architecture": second_label, "environment": ENVIRONMENT, "seed": DEDICATED_SEED,
            "critic_degradation_onset_step": POST_STEP, "critic_degradation_method": "td_variance_p95_sustained_v1",
            "propagation_onset_step": None, "propagation_method": None, "propagation_lag": None,
            "status": "success", "detection_notes": "test row",
        }
        identity = WandbIdentity(run_obj=_FakeRun(run_key, row["wandb_run_id"]))
        log_onset_event("angle_1", second_label, row, identity=identity, root=self.ledger_root, mirror_to_wandb=False)

        results = run_prereq_check(
            scaled_architectures={self.arch_label: self.architecture, second_label: second_architecture},
            base_cfg=_make_cfg(), environment=ENVIRONMENT, dedicated_seeds=[DEDICATED_SEED],
            onset_source_experiment="angle_1", onset_ledger_root=self.ledger_root,
            burn_in_fraction=BURN_IN_FRACTION, num_probes=3, num_mc_rollouts=2, output_root=self.output_root,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual({r.architecture_label for r in results}, {self.arch_label, second_label})


if __name__ == "__main__":
    unittest.main()
