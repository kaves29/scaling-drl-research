"""Deterministic regressions for the Batch1 replay/config/checkpoint audit."""

import copy
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gymnasium as gym
import numpy as np

from experiments.exp12 import state
from experiments.exp12.fixed_task import tree_sha256
from scale_rl.agents import create_agent
from scale_rl.buffers.numpy_buffer import NpyUniformBuffer
from tests.test_sac_agent_checkpoint import _make_cfg, _make_spaces, _make_random_batch


def buffer(**kwargs):
    options = dict(
        n_step=1,
        gamma=0.95,
        max_length=8,
        min_length=1,
        add_batch_size=1,
        sample_batch_size=4,
    )
    options.update(kwargs)
    space = gym.spaces.Box(-100.0, 100.0, (1, 2), np.float32)
    result = NpyUniformBuffer(space, space, **options)
    result.reset()
    return result


def transition(value):
    return dict(
        observation=np.full((1, 2), value, np.float32),
        action=np.full((1, 2), value / 10, np.float32),
        reward=np.array([value], np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        next_observation=np.full((1, 2), value + 1, np.float32),
    )


def agent(**overrides):
    cfg = _make_cfg()
    for key, value in overrides.items():
        cfg[key] = value
    o, a = _make_spaces()
    return create_agent(o, a, cfg)


class ReplayRegressionTest(unittest.TestCase):
    def test_reset_restarts_write_pointer_and_never_samples_unfilled_data(self):
        replay = buffer()
        for value in (1.0, 2.0, 3.0):
            replay.add(transition(value))
        self.assertEqual(replay._current_idx, 3)
        replay.reset()
        self.assertEqual(replay._current_idx, 0)
        replay._observations[:] = -99
        replay.add(transition(7.0))
        np.testing.assert_array_equal(
            replay.sample()["observation"], np.full((4, 2), 7.0)
        )

    def test_unsupported_multistep_early_truncation_fails_closed(self):
        # Fixed gamma**n cannot represent a truncated prefix's gamma**k (k<n).
        # Do not silently add a new return definition to the approved n_step=1 study.
        with self.assertRaisesRegex(ValueError, "n_step=1"):
            buffer(n_step=3)

    def test_invalid_replay_dimensions_and_warmup_fail_before_allocation(self):
        for override in (
            {"max_length": 0},
            {"min_length": 0},
            {"min_length": 9},
            {"add_batch_size": 0},
            {"sample_batch_size": 0},
            {"n_step": True},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                buffer(**override)

    def test_copying_and_sampling_preserve_replay_and_global_rng_continuation(self):
        replay = buffer(max_length=3)
        for value in range(5):
            t = transition(float(value))
            replay.add(t)
            t["observation"][:] = -99
        self.assertEqual(len(replay), 3)
        self.assertEqual(replay._current_idx, 2)
        rng = np.random.get_state()
        first = replay.sample()
        np.random.set_state(rng)
        second = replay.sample()
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        first["observation"][:] = -88
        self.assertFalse(np.any(replay._observations == -88))
        self.assertFalse(np.any(replay._observations == -99))

    def test_bad_pointer_is_rejected_before_restore_mutates_arrays(self):
        replay = buffer()
        replay.add(transition(1.0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            state.save_buffer(replay, path)
            with (path / "buffer_meta.pkl").open("rb") as stream:
                metadata = pickle.load(stream)
            metadata["current_idx"] = 3  # one filled item cannot have pointer3
            (path / "buffer_meta.pkl").write_bytes(pickle.dumps(metadata))
            target = buffer()
            target._observations[:] = -99
            with self.assertRaises(ValueError):
                state.load_buffer(target, path)
            np.testing.assert_array_equal(target._observations, np.full((8, 2), -99))

    def test_invalid_pending_queue_is_rejected_before_any_array_mutation(self):
        for bad in (
            None,
            [transition(1.0), transition(2.0)],
            [{"reward": np.array([1.0])}],
        ):
            with self.subTest(pending=type(bad)), tempfile.TemporaryDirectory() as tmp:
                replay = buffer()
                replay.add(transition(1.0))
                path = Path(tmp)
                state.save_buffer(replay, path)
                metadata = pickle.loads((path / "buffer_meta.pkl").read_bytes())
                metadata["n_step_transitions"] = bad
                (path / "buffer_meta.pkl").write_bytes(pickle.dumps(metadata))
                target = buffer()
                target._observations[:] = -99
                with self.assertRaises(ValueError):
                    state.load_buffer(target, path)
                np.testing.assert_array_equal(
                    target._observations, np.full((8, 2), -99)
                )

    def test_save_restore_replays_wraparound_writes_and_sampling_exactly(self):
        replay = buffer(max_length=3)
        for value in range(5):
            replay.add(transition(float(value)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            state.save_buffer(replay, path)
            restored = buffer(max_length=3)
            state.load_buffer(restored, path)
            for value in (5.0, 6.0):
                replay.add(transition(value))
                restored.add(transition(value))
            for field in state.BUFFER_ARRAYS:
                np.testing.assert_array_equal(
                    getattr(replay, field), getattr(restored, field)
                )
            rng = np.random.get_state()
            one = replay.sample()
            np.random.set_state(rng)
            two = restored.sample()
            for key in one:
                np.testing.assert_array_equal(one[key], two[key])


class ConfigAndCheckpointTest(unittest.TestCase):
    def test_explicit_entropy_target_is_not_silently_overwritten(self):
        self.assertEqual(agent(temp_target_entropy=-3.0)._cfg.temp_target_entropy, -3.0)
        self.assertEqual(agent()._cfg.temp_target_entropy, -1.0)

    def test_invalid_sac_scalars_fail_before_network_initialization(self):
        for override in (
            {"gamma": float("nan")},
            {"gamma": 1.1},
            {"target_tau": -0.1},
            {"critic_learning_rate": -1.0},
            {"critic_weight_decay": float("inf")},
            {"temp_initial_value": 0.0},
        ):
            with self.subTest(override=override), mock.patch(
                "scale_rl.agents.sac.sac_agent._init_sac_networks"
            ) as init:
                with self.assertRaises(ValueError):
                    agent(**override)
                init.assert_not_called()

    def test_new_checkpoint_rejects_different_optimizer_before_restore(self):
        original = agent()
        with tempfile.TemporaryDirectory() as tmp:
            original.save_checkpoint(tmp)
            for override in (
                {"critic_learning_rate": 3e-4},
                {"actor_weight_decay": 0.0},
                {"gamma": 0.95},
                {"temp_target_entropy": -3.0},
            ):
                with self.subTest(override=override):
                    target = agent(**override)
                    before = tree_sha256((target._rng, target.actor, target.critic))
                    with self.assertRaisesRegex(ValueError, "configuration"):
                        target.load_checkpoint(tmp)
                    self.assertEqual(
                        tree_sha256((target._rng, target.actor, target.critic)), before
                    )

    def test_matching_checkpoint_restores_moments_counts_rng_and_next_update(self):
        original = agent()
        batch = _make_random_batch(np.random.default_rng(71))
        original.update(0, copy.deepcopy(batch), compute_actor_grad_cosine=False)
        with tempfile.TemporaryDirectory() as tmp:
            original.save_checkpoint(tmp)
            restored = agent()
            restored.load_checkpoint(tmp)
            for key in (
                "_rng",
                "_actor",
                "_critic",
                "_target_critic",
                "_temperature",
                "churn_ref_batch",
            ):
                self.assertEqual(
                    tree_sha256(getattr(original, key)),
                    tree_sha256(getattr(restored, key)),
                )
            original.update(1, copy.deepcopy(batch), compute_actor_grad_cosine=False)
            restored.update(1, copy.deepcopy(batch), compute_actor_grad_cosine=False)
            for key in ("_rng", "_actor", "_critic", "_target_critic", "_temperature"):
                self.assertEqual(
                    tree_sha256(getattr(original, key)),
                    tree_sha256(getattr(restored, key)),
                )
            self.assertEqual(int(restored.critic.opt_state[0].count), 2)
            self.assertEqual(int(restored.critic.update_step), 2)

    def test_legacy_checkpoint_is_readable_without_claiming_new_binding(self):
        original = agent()
        with tempfile.TemporaryDirectory() as tmp:
            original.save_checkpoint(tmp)
            marker = Path(tmp) / "sac_config.json"
            if marker.exists():
                marker.unlink()  # legacy payload layout, not an archived file
            restored = agent()
            restored.load_checkpoint(tmp)
            self.assertEqual(tree_sha256(original.critic), tree_sha256(restored.critic))

    def test_corrupted_checkpoint_configuration_is_rejected(self):
        original = agent()
        with tempfile.TemporaryDirectory() as tmp:
            original.save_checkpoint(tmp)
            marker = Path(tmp) / "sac_config.json"
            marker.write_text(json.dumps({"schema": 999}))
            with self.assertRaises(ValueError):
                agent().load_checkpoint(tmp)


if __name__ == "__main__":
    unittest.main()
