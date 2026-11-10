"""End-to-end CPU fixture for online collection, diagnostics and restore."""

import tempfile
import unittest
from unittest.mock import patch
import random
import time

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
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


class GlobalResetTinyEnv(TinyEnv):
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return np.full(3, np.random.random(), np.float32), {}

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward + np.random.random(), terminated, truncated, info


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
    def test_pilot_evaluation_seed_controls_global_environment_rng(self):
        from experiments.exp12.online_exp2_pilot import _evaluate
        from tests.test_online_exp2 import OnlineExp2Test
        OnlineExp2Test.setUpClass()
        with patch("experiments.exp12.online_exp2_pilot.create_eval_env",
                   lambda **kwargs: SyncVectorEnv([GlobalResetTinyEnv])):
            np.random.seed(4)
            before = np.random.get_state()
            first = _evaluate(cfg(), OnlineExp2Test.actor, None, 1234)
            after = np.random.get_state()
            self.assertEqual(before[2], after[2])
            np.testing.assert_array_equal(before[1], after[1])
            np.random.seed(9)
            second = _evaluate(cfg(), OnlineExp2Test.actor, None, 1234)
            self.assertEqual(first, second)

    def test_pilot_resource_cap_stops_before_work(self):
        pilot = cfg()
        OmegaConf.set_struct(pilot, False)
        pilot.run_role = "pilot"
        pilot.online_exp2.pilot_enabled = True
        pilot.online_exp2.reference_regime = "reference_small"
        pilot.online_exp2.resource_cap_gpu_hours = 1.
        with tempfile.TemporaryDirectory(dir="/workspace/scratch") as root:
            with patch("experiments.online_exp2.validate", return_value=(3, [2], [2])), patch(
                "experiments.exp12.online_exp2_diagnostic.reference_roles",
                return_value=((1, 8), (1, 8), (1, 8))
            ), patch("experiments.exp12.trainer.create_envs", tiny_envs):
                runner = OnlineExp2(pilot, root)
                runner._pilot_started_at = time.monotonic() - 7200
                with self.assertRaisesRegex(RuntimeError, "resource cap"):
                    runner._check_resource_cap()
                runner.trainer.close()

    def test_pilot_profiles_are_separate_and_fail_on_unresolved_science(self):
        for regime in ("reference_small", "reference_large"):
            name = f"online_exp2_pilot_{regime}"
            with self.assertRaises(ValueError):
                compose_config("/workspace/scaling-drl-research/main/configs", name, [])
            profile = compose_config("/workspace/scaling-drl-research/main/configs", name, [],
                                     allow_pilot=True)
            self.assertEqual(profile.online_exp2.reference_regime, regime)
            self.assertEqual(profile.online_exp2.measurement_interactions, [0, 5000, 10000])
            with self.assertRaisesRegex(ValueError, "unresolved fields"):
                validate(profile)

    def test_pilot_requires_explicit_regime_and_fixed_horizon(self):
        pilot = cfg()
        OmegaConf.set_struct(pilot, False)
        pilot.run_role = "pilot"
        pilot.online_exp2.pilot_enabled = True
        pilot.online_exp2.reference_regime = "reference_small"
        pilot.online_exp2.diagnostic_horizon = 10000
        pilot.online_exp2.measurement_interactions = [0, 5000, 10000]
        pilot.online_exp2.resource_cap_gpu_hours = 1
        pilot.online_exp2.dev_architectures = None
        pilot.online_exp2.reference_critic = None
        pilot.online_exp2.intervention_updates = None
        pilot.probe.rounds = 10
        pilot.probe.steps = 2000
        pilot.probe.pool_size = 25600
        pilot.probe.batch_size = 256
        pilot.probe.eval_chunk = 2560
        self.assertEqual(validate(pilot), (3, [2], [2]))
        pilot.online_exp2.reference_regime = "unknown"
        with self.assertRaises(ValueError):
            validate(pilot)

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

    def test_pilot_checkpoint_collects_and_updates_without_reference_feedback(self):
        pilot = cfg()
        OmegaConf.set_struct(pilot, False)
        pilot.run_role = "pilot"
        pilot.online_exp2.pilot_enabled = True
        pilot.online_exp2.reference_regime = "reference_small"
        pilot.online_exp2.diagnostic_horizon = 6
        pilot.online_exp2.resource_cap_gpu_hours = 1
        pilot.online_exp2.measurement_interactions = [0, 2, 6]
        pilot.online_exp2.diagnostic_save_every = 2
        with tempfile.TemporaryDirectory(dir="/workspace/scratch") as root:
            with patch("experiments.online_exp2.validate", return_value=(3, [2], [2])), patch(
                "experiments.exp12.online_exp2_diagnostic.reference_roles",
                return_value=((1, 8), (1, 8), (1, 8))
            ), patch("experiments.exp12.trainer.create_envs", tiny_envs), patch(
                "experiments.exp12.online_exp2_pilot._branch_env", lambda *args: SyncVectorEnv([TinyEnv])
            ), patch("experiments.exp12.online_exp2_pilot.create_eval_env", tiny_eval_env):
                runner = OnlineExp2(pilot, root)
                runner.run()
                evaluations = [r for r in runner.rows if r["type"] == "pilot_evaluation"]
                self.assertEqual({r["diagnostic_interactions"] for r in evaluations}, {2, 6})
                self.assertEqual(len(evaluations), 2)
                record = next(r for r in runner.rows if r["type"] == "pilot_diagnostic")
                updates = [r for r in runner.rows if r["type"] == "update" and r["interaction_step"] == 2]
                self.assertEqual(len(updates), 2)
                self.assertNotEqual(updates[0]["reference_actor_snapshot_sha256"],
                                    updates[1]["reference_actor_snapshot_sha256"])
                import hashlib
                import flax.serialization
                last_update = [r for r in runner.rows if r["type"] == "update" and r["interaction_step"] == 3][-1]
                self.assertEqual(last_update["reference_actor_snapshot_sha256"], hashlib.sha256(
                    flax.serialization.to_bytes(runner.trainer._sac_agent.actor.params)).hexdigest())
                self.assertEqual(record["branch_counters"]["small"]["interactions"], 6)
                self.assertEqual(record["branch_counters"]["large"]["interactions"], 6)
                self.assertEqual(record["branch_counters"]["small"]["optimizer_updates"],
                                 record["branch_counters"]["large"]["optimizer_updates"])
                self.assertEqual(set(runner._pilot_analysis()["horizons"]), {2, 6})
                with open(record["progress_artifact"], "rb") as stream:
                    import pickle
                    progress = pickle.load(stream)
                self.assertEqual(len(progress["branches"]["small"]["branch"]["buffer"]["observations"]), 6)
                self.assertEqual(len(progress["branches"]["large"]["branch"]["buffer"]["observations"]), 6)
                np.testing.assert_array_equal(
                    progress["branches"]["small"]["branch"]["buffer"]["observations"][5],
                    np.zeros(3, np.float32))
                self.assertEqual(progress["branches"]["small"]["environment_seeds"],
                                 progress["branches"]["large"]["environment_seeds"])
                self.assertEqual(len(progress["branches"]["small"]["environment_seeds"]), 2)
                self.assertEqual(len(runner.trainer.buffer._observations[:runner.trainer.interaction_step]), 3)
                baseline_cfg = cfg()
                baseline_cfg.agent.critic_num_blocks = 1
                baseline_cfg.agent.critic_hidden_dim = 8
                baseline_cfg.num_interaction_steps = 3
                np.random.seed(1)
                random.seed(1)
                baseline = Exp12Trainer(baseline_cfg, root + "/baseline")
                baseline.start()
                baseline.train(3)
                for field in ("actor", "critic"):
                    left = getattr(runner.trainer._sac_agent, field).params
                    right = getattr(baseline._sac_agent, field).params
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(
                        jax.tree.leaves(left), jax.tree.leaves(right))), field)
                np.testing.assert_array_equal(runner.trainer.buffer._observations[:3],
                                              baseline.buffer._observations[:3])
                baseline.close()

    def test_pilot_resumes_interrupted_branch(self):
        from experiments.exp12.online_exp2_diagnostic import DiagnosticActorBranch
        pilot = cfg()
        OmegaConf.set_struct(pilot, False)
        pilot.run_role = "pilot"
        pilot.online_exp2.pilot_enabled = True
        pilot.online_exp2.reference_regime = "reference_small"
        pilot.online_exp2.diagnostic_horizon = 6
        pilot.online_exp2.resource_cap_gpu_hours = 1
        pilot.online_exp2.measurement_interactions = [0, 2, 6]
        pilot.online_exp2.diagnostic_save_every = 2
        original_act = DiagnosticActorBranch.act
        interrupted = [False]

        def fail_once(branch, observation, normalize=None):
            if branch.interactions == 5 and not interrupted[0]:
                interrupted[0] = True
                raise RuntimeError("simulated pre-action interruption")
            return original_act(branch, observation, normalize)

        with tempfile.TemporaryDirectory(dir="/workspace/scratch") as root:
            with patch("experiments.online_exp2.validate", return_value=(3, [2], [2])), patch(
                "experiments.exp12.online_exp2_diagnostic.reference_roles",
                return_value=((1, 8), (1, 8), (1, 8))
            ), patch("experiments.exp12.trainer.create_envs", tiny_envs), patch(
                "experiments.exp12.online_exp2_pilot._branch_env", lambda *args: SyncVectorEnv([TinyEnv])
            ), patch("experiments.exp12.online_exp2_pilot.create_eval_env", tiny_eval_env):
                with patch.object(DiagnosticActorBranch, "act", fail_once):
                    with self.assertRaisesRegex(RuntimeError, "interruption"):
                        OnlineExp2(pilot, root).run()
                resumed = OnlineExp2(pilot, root)
                resumed.run()
                self.assertTrue(interrupted[0])
                self.assertEqual(resumed.trainer.interaction_step, 3)
                self.assertEqual(len([r for r in resumed.rows if r["type"] == "pilot_diagnostic"]), 1)
                self.assertEqual(len([r for r in resumed.rows if r["type"] == "pilot_evaluation"]), 2)
                import pickle
                resumed_record = next(r for r in resumed.rows if r["type"] == "pilot_diagnostic")
                with open(resumed_record["progress_artifact"], "rb") as stream:
                    resumed_progress = pickle.load(stream)
                with tempfile.TemporaryDirectory(dir="/workspace/scratch") as control_root:
                    control = OnlineExp2(pilot, control_root)
                    control.run()
                    control_record = next(r for r in control.rows if r["type"] == "pilot_diagnostic")
                    with open(control_record["progress_artifact"], "rb") as stream:
                        control_progress = pickle.load(stream)
                    for name in ("small", "large"):
                        self.assertEqual(resumed_progress["branches"][name]["branch"]["actor"],
                                         control_progress["branches"][name]["branch"]["actor"])
                        self.assertEqual(resumed_progress["branches"][name]["update_records"],
                                         control_progress["branches"][name]["update_records"])

    def test_pilot_real_dmc_short_checkpoint(self):
        pilot = cfg()
        OmegaConf.set_struct(pilot, False)
        pilot.run_role = "pilot"
        pilot.online_exp2.pilot_enabled = True
        pilot.online_exp2.reference_regime = "reference_small"
        pilot.online_exp2.diagnostic_horizon = 6
        pilot.online_exp2.resource_cap_gpu_hours = 1
        pilot.online_exp2.measurement_interactions = [0, 3, 6]
        pilot.online_exp2.diagnostic_save_every = 3
        pilot.env.max_episode_steps = 5
        pilot.env.action_repeat = 1
        pilot.action_repeat = 1
        with tempfile.TemporaryDirectory(dir="/workspace/scratch") as root:
            with patch("experiments.online_exp2.validate", return_value=(3, [2], [2])), patch(
                "experiments.exp12.online_exp2_diagnostic.reference_roles",
                return_value=((1, 8), (1, 8), (1, 8))
            ):
                runner = OnlineExp2(pilot, root)
                runner.run()
                self.assertEqual(len([r for r in runner.rows if r["type"] == "pilot_evaluation"]), 2)
                self.assertEqual(len([r for r in runner.rows if r["type"] == "pilot_diagnostic"]), 1)

    def test_online_fixture_and_restore(self):
        with tempfile.TemporaryDirectory(dir="/workspace/scratch") as root:
            with patch("experiments.exp12.trainer.create_envs", tiny_envs), patch(
                "experiments.online_exp2.create_eval_env", tiny_eval_env
            ):
                original_cfg = cfg()
                original_yaml = OmegaConf.to_yaml(original_cfg, resolve=True)
                run = OnlineExp2(original_cfg, root)
                self.assertEqual(OmegaConf.to_yaml(original_cfg, resolve=True), original_yaml)
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
                restored = OnlineExp2(original_cfg, root)
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
