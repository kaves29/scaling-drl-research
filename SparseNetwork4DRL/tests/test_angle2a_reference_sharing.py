"""Tests for the shared single-R_2x512-trajectory protocol change:

    exactly ONE reference training trajectory per seed, snapshotted at each
    matchup's own (independent) onset step, rather than one fresh reference
    training run per matchup.

Layered like the rest of the Angle 2A test suite:
  - ProbeCapture.snapshot(): pure numpy, no mocking.
  - train_reference_agent_with_snapshots(): mocks create_envs/create_buffer/
    create_agent (the same seam experiments/angle_2a/agent_runner.py already
    calls through) so this runs fast without JAX/dm_control, while still
    exercising the REAL training-loop generator and REAL deepcopy-based
    snapshotting.
  - run_matchup(reference_handle=...): extends the existing independence
    test's mocking style to check the new parameter's contract.
  - experiments.angle_2_a.run(): full orchestration, using a real fabricated
    onset ledger (as in test_angle2a_onset_lookup.py) with two DIFFERENT
    onset steps, mocking only train_reference_agent_with_snapshots and
    run_matchup to keep it fast.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import omegaconf

from experiments.angle_2a.agent_runner import (
    ProbeCapture,
    ReferenceTrajectory,
    TrainedAgentHandle,
    train_reference_agent_with_snapshots,
)
from experiments.angle_2a.config import RoleArchitecture

OBS_DIM = 3
ACT_DIM = 1


# ---------------------------------------------------------------------------
# ProbeCapture.snapshot()
# ---------------------------------------------------------------------------
class TestProbeCaptureSnapshot(unittest.TestCase):
    def _filled_capture(self, capacity, filled):
        pc = ProbeCapture(capacity=capacity, observation_shape=(2,), action_shape=(1,))
        for i in range(filled):
            pc.add(i, np.array([float(i), 0.0], dtype=np.float32), np.array([float(i)], dtype=np.float32),
                    {"physics_state": np.array([float(i)]), "elapsed_steps": None})
        return pc

    def test_snapshot_contains_only_transitions_up_to_the_requested_count(self):
        pc = self._filled_capture(capacity=100, filled=50)
        snap = pc.snapshot(20)
        self.assertEqual(len(snap), 20)
        for i in range(20):
            self.assertEqual(snap._observations[i][0], float(i))

    def test_snapshot_excludes_transitions_collected_after_the_snapshot_point(self):
        pc = self._filled_capture(capacity=100, filled=50)
        snap = pc.snapshot(20)
        # nothing in the snapshot should ever reference indices >= 20
        idxs, states, actions, env_states = snap.sample(20, np.random.default_rng(0))
        self.assertTrue(all(s[0] < 20 for s in states))

    def test_snapshot_of_full_capture_matches_original(self):
        pc = self._filled_capture(capacity=30, filled=30)
        snap = pc.snapshot(30)
        self.assertEqual(len(snap), len(pc))

    def test_snapshot_beyond_collected_count_raises(self):
        pc = self._filled_capture(capacity=100, filled=10)
        with self.assertRaises(ValueError):
            pc.snapshot(20)

    def test_two_snapshots_at_different_steps_are_independent_and_ordering_agnostic(self):
        # exercises "works when either onset is earlier than the other":
        # taking the smaller snapshot first or the larger one first must not
        # matter - each is just a bounded view.
        pc = self._filled_capture(capacity=100, filled=80)
        snap_late = pc.snapshot(70)
        snap_early = pc.snapshot(10)
        self.assertEqual(len(snap_early), 10)
        self.assertEqual(len(snap_late), 70)
        self.assertLess(len(snap_early), len(snap_late))


# ---------------------------------------------------------------------------
# train_reference_agent_with_snapshots()
# ---------------------------------------------------------------------------
class _FakePhysics:
    def get_state(self):
        return np.zeros(1)

    def set_state(self, s):
        pass

    def forward(self):
        pass


class _FakeSingleEnv:
    def __init__(self):
        self.physics = _FakePhysics()

    @property
    def unwrapped(self):
        return self

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}


class _FakeVectorEnv:
    def __init__(self):
        self.envs = [_FakeSingleEnv()]
        self.closed = False
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(OBS_DIM,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1, ACT_DIM))

    def reset(self):
        return np.zeros((1, OBS_DIM), dtype=np.float32), {}

    def step(self, actions):
        next_obs = np.zeros((1, OBS_DIM), dtype=np.float32)
        rewards = np.zeros((1,), dtype=np.float32)
        terminateds = np.zeros((1,), dtype=bool)
        truncateds = np.zeros((1,), dtype=bool)
        return next_obs, rewards, terminateds, truncateds, {}

    def close(self):
        self.closed = True


class _FakeBuffer:
    def reset(self):
        pass

    def add(self, timestep):
        pass

    def can_sample(self):
        return False  # keep the test fast/deterministic: no agent.update() calls

    def sample(self):
        raise AssertionError("should not be called since can_sample() is False")


class _FakeReferenceAgent:
    """Tracks how many times sample_actions() has been called (a stand-in
    for "how far this agent has been trained") so snapshots taken at
    different points can be shown to be frozen at genuinely different
    states, not all aliasing the final trained agent."""

    def __init__(self):
        self.calls = 0

    def sample_actions(self, interaction_step, prev_timestep, training):
        self.calls += 1
        return np.zeros((1, ACT_DIM), dtype=np.float32)

    def update(self, update_step, batch):
        raise AssertionError("should not be called since buffer.can_sample() is False")


def _fake_base_cfg():
    # Real (Omega)dict-like config: agent_runner does `**base_cfg.env`
    # unpacking and OmegaConf.to_container(base_cfg.buffer, ...) with real
    # OmegaConf calls even though create_envs/create_buffer/create_agent
    # themselves are mocked below, so these need to actually be DictConfigs.
    return omegaconf.OmegaConf.create(
        {
            "seed": 7,
            "updates_per_interaction_step": 1,
            "env": {"env_type": "dmc", "num_train_envs": 1, "max_episode_steps": 1000},
            "buffer": {"max_length": 1_000_000},
            "agent": {},
        }
    )


def _patched_pieces():
    return mock.patch.multiple(
        "experiments.angle_2a.agent_runner",
        create_envs=mock.DEFAULT,
        create_buffer=mock.DEFAULT,
        create_agent=mock.DEFAULT,
        build_role_agent_cfg=mock.DEFAULT,
    )


class TestTrainReferenceAgentWithSnapshots(unittest.TestCase):
    def _run(self, snapshot_steps):
        fake_agent = _FakeReferenceAgent()
        fake_env = _FakeVectorEnv()

        with _patched_pieces() as mocks:
            mocks["create_envs"].return_value = (fake_env, mock.Mock())
            mocks["create_buffer"].return_value = _FakeBuffer()
            mocks["create_agent"].return_value = fake_agent
            mocks["build_role_agent_cfg"].return_value = {}

            reference = RoleArchitecture(role="reference", critic_num_blocks=2, critic_hidden_dim=512)
            trajectory = train_reference_agent_with_snapshots(
                architecture=reference,
                architecture_label="D2W512",
                base_cfg=_fake_base_cfg(),
                snapshot_steps=snapshot_steps,
                seed_context="shared_reference:D2W512",
            )
        return trajectory, fake_agent, fake_env, mocks

    def test_exactly_one_reference_agent_is_constructed(self):
        _trajectory, _agent, _env, mocks = self._run([50, 100])
        self.assertEqual(mocks["create_agent"].call_count, 1)
        self.assertEqual(mocks["create_envs"].call_count, 1)
        self.assertEqual(mocks["create_buffer"].call_count, 1)

    def test_a_snapshot_exists_for_every_requested_step(self):
        trajectory, _agent, _env, _mocks = self._run([50, 100])
        self.assertEqual(set(trajectory.snapshots), {50, 100})

    def test_snapshots_share_the_same_underlying_env(self):
        trajectory, _agent, fake_env, _mocks = self._run([30, 90])
        snap_a = trajectory.at(30)
        snap_b = trajectory.at(90)
        self.assertIs(snap_a.train_env, snap_b.train_env)
        self.assertIs(snap_a.single_env, snap_b.single_env)
        self.assertIs(snap_a.train_env, fake_env)

    def test_snapshots_hold_distinct_frozen_agent_states(self):
        trajectory, live_agent, _env, _mocks = self._run([10, 40])
        snap_early = trajectory.at(10)
        snap_late = trajectory.at(40)

        # agent.sample_actions() is called once per interaction_step beyond
        # the first (see agent_runner._run_training_loop); a snapshot taken
        # at step N should have frozen the agent after exactly N-1 calls.
        self.assertEqual(snap_early.agent.calls, 9)
        self.assertEqual(snap_late.agent.calls, 39)
        self.assertEqual(live_agent.calls, 39)  # training ran to max(10, 40) = 40
        self.assertIsNot(snap_early.agent, snap_late.agent)
        self.assertIsNot(snap_early.agent, live_agent)

    def test_snapshot_probe_data_matches_its_own_step_not_the_final_step(self):
        trajectory, _agent, _env, _mocks = self._run([10, 40])
        self.assertEqual(len(trajectory.at(10).probe_capture), 10)
        self.assertEqual(len(trajectory.at(40).probe_capture), 40)

    def test_works_when_first_onset_is_later_than_second(self):
        # order of the snapshot_steps list must not matter (mirrors
        # "works when either onset is earlier than the other").
        trajectory, _agent, _env, _mocks = self._run([90, 20])
        self.assertEqual(set(trajectory.snapshots), {20, 90})
        self.assertEqual(len(trajectory.at(20).probe_capture), 20)
        self.assertEqual(len(trajectory.at(90).probe_capture), 90)

    def test_equal_onsets_reuse_the_same_snapshot_object(self):
        trajectory, _agent, _env, _mocks = self._run([50, 50])
        self.assertEqual(set(trajectory.snapshots), {50})

    def test_close_only_closes_the_shared_env_once(self):
        trajectory, _agent, fake_env, _mocks = self._run([10, 40])
        trajectory.close()
        self.assertTrue(fake_env.closed)

    def test_at_unknown_step_raises(self):
        trajectory, _agent, _env, _mocks = self._run([10, 40])
        with self.assertRaises(KeyError):
            trajectory.at(25)


# ---------------------------------------------------------------------------
# experiments.angle_2_a.run(): full orchestration
# ---------------------------------------------------------------------------
import hydra
from hydra.core.global_hydra import GlobalHydra

import experiments.angle_2_a as angle_2_a_module
from utils.onset_ledger import WandbIdentity, log_onset_event

FULL_OVERRIDES = [
    "angle_2_a.scaled_a.critic_num_blocks=5",
    "angle_2_a.scaled_a.critic_hidden_dim=768",
    "angle_2_a.scaled_b.critic_num_blocks=7",
    "angle_2_a.scaled_b.critic_hidden_dim=1024",
    "angle_2_a.reference.critic_num_blocks=2",
    "angle_2_a.reference.critic_hidden_dim=512",
    "angle_2_a.run_null_baseline=false",
]


class _FakeWandbRun:
    class _Summary(dict):
        def update(self, d):
            dict.update(self, d)

    def __init__(self, name, id_):
        self.name = name
        self.id = id_
        self.summary = self._Summary()


class _FakeReferenceTrajectory:
    def __init__(self, snapshot_steps):
        self.snapshot_steps_requested = list(snapshot_steps)
        self.snapshots = {step: mock.Mock(name=f"snapshot@{step}") for step in set(snapshot_steps)}
        self.closed = False

    def at(self, step):
        return self.snapshots[step]

    def close(self):
        self.closed = True


class TestAngle2ATopLevelOrchestration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ledger_root = os.path.join(self.tmpdir, "ledgers")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _log_onset(self, architecture, environment, seed, onset_step):
        run_key = f"angle1_{architecture}_{environment}_seed{seed}"
        row = {
            "run_key": run_key, "exact_run_name": run_key, "wandb_run_id": f"wid_{run_key}",
            "architecture": architecture, "environment": environment, "seed": seed,
            "critic_degradation_onset_step": onset_step,
            "critic_degradation_method": "td_variance_p95_sustained_v1",
            "propagation_onset_step": None, "propagation_method": None, "propagation_lag": None,
            "status": "success", "detection_notes": "fabricated for orchestration test",
        }
        identity = WandbIdentity(run_obj=_FakeWandbRun(run_key, row["wandb_run_id"]))
        log_onset_event("angle_1", architecture, row, identity=identity, root=self.ledger_root, mirror_to_wandb=False)

    def _run_angle_2_a(self, onset_5x768, onset_7x1024, seed=1, environment="reacher-hard"):
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        self._log_onset("D5W768", environment, seed, onset_5x768)
        self._log_onset("D7W1024", environment, seed, onset_7x1024)

        fake_trajectory_holder = {}

        def fake_train_reference_agent_with_snapshots(architecture, architecture_label, base_cfg, snapshot_steps, seed_context):
            trajectory = _FakeReferenceTrajectory(snapshot_steps)
            fake_trajectory_holder["trajectory"] = trajectory
            return trajectory

        run_matchup_calls = []

        def fake_run_matchup(**kwargs):
            run_matchup_calls.append(kwargs)
            return mock.Mock()

        with mock.patch.object(
            angle_2_a_module, "train_reference_agent_with_snapshots", side_effect=fake_train_reference_agent_with_snapshots
        ), mock.patch.object(angle_2_a_module, "run_matchup", side_effect=fake_run_matchup):
            angle_2_a_module.run(
                {
                    "experiment": "angle_2_a",
                    # angle_2_a_module.run() resolves this relative to the
                    # process CWD (via os.path.abspath), matching how a real
                    # `python run.py` invocation from the repo root works -
                    # unlike test_angle2a_config.py, which calls
                    # hydra.initialize() directly and needs a path relative
                    # to *this test file's* own directory instead.
                    "config_path": "./configs",
                    "config_name": "base_angle2a",
                    "overrides": FULL_OVERRIDES
                    + [
                        f"seed={seed}",
                        f"env_name={environment}",
                        f"angle_2_a.onset_ledger_root={self.ledger_root}",
                    ],
                    "checkpoint_dir": None,
                    "checkpoint_interval": 100_000,
                    "checkpoint_start_frac": 0.4,
                }
            )

        return fake_trajectory_holder["trajectory"], run_matchup_calls

    def test_exactly_one_reference_trajectory_is_trained_per_seed(self):
        trajectory, run_matchup_calls = self._run_angle_2_a(onset_5x768=120_000, onset_7x1024=180_000)
        # train_reference_agent_with_snapshots (the only path that
        # constructs a reference agent) is called exactly once.
        self.assertEqual(trajectory.snapshot_steps_requested, [120_000, 180_000])

    def test_both_matchups_use_snapshots_from_the_same_trajectory(self):
        trajectory, run_matchup_calls = self._run_angle_2_a(onset_5x768=120_000, onset_7x1024=180_000)
        self.assertEqual(len(run_matchup_calls), 2)
        for call in run_matchup_calls:
            self.assertIs(call["reference_handle"], trajectory.snapshots[call["onset_step"]])
            self.assertIsNotNone(call["reference_run_key"])
        # both calls reference the SAME trajectory identity string
        self.assertEqual(run_matchup_calls[0]["reference_run_key"], run_matchup_calls[1]["reference_run_key"])

    def test_matchup_1_uses_5x768_onset_matchup_2_uses_7x1024_onset(self):
        _trajectory, run_matchup_calls = self._run_angle_2_a(onset_5x768=120_000, onset_7x1024=180_000)
        by_name = {call["matchup_name"]: call for call in run_matchup_calls}
        self.assertEqual(by_name["matchup_1"]["onset_step"], 120_000)
        self.assertEqual(by_name["matchup_2"]["onset_step"], 180_000)

    def test_works_when_5x768_onset_is_later_than_7x1024_onset(self):
        # the reverse ordering from the spec's own example - matchup
        # correctness must not depend on which onset happens to be larger.
        trajectory, run_matchup_calls = self._run_angle_2_a(onset_5x768=180_000, onset_7x1024=120_000)
        by_name = {call["matchup_name"]: call for call in run_matchup_calls}

        self.assertEqual(by_name["matchup_1"]["onset_step"], 180_000)
        self.assertEqual(by_name["matchup_2"]["onset_step"], 120_000)
        self.assertEqual(sorted(trajectory.snapshot_steps_requested), [120_000, 180_000])
        self.assertIs(by_name["matchup_1"]["reference_handle"], trajectory.snapshots[180_000])
        self.assertIs(by_name["matchup_2"]["reference_handle"], trajectory.snapshots[120_000])

    def test_no_second_reference_agent_is_created_regardless_of_onset_order(self):
        # A regression guard: run_matchup is invoked with reference_handle
        # set (never None) for both real matchups, which is what stops
        # run_matchup from training its own second reference internally.
        _trajectory, run_matchup_calls = self._run_angle_2_a(onset_5x768=50_000, onset_7x1024=50_000)
        self.assertEqual(len(run_matchup_calls), 2)
        for call in run_matchup_calls:
            self.assertIsNotNone(call["reference_handle"])

    def test_reference_trajectory_is_closed_after_both_matchups(self):
        trajectory, _run_matchup_calls = self._run_angle_2_a(onset_5x768=120_000, onset_7x1024=180_000)
        self.assertTrue(trajectory.closed)


# ---------------------------------------------------------------------------
# RNG reproducibility: derive_rng_seed / seed_global_rng_for_agent
# ---------------------------------------------------------------------------
import random as _random_module

from experiments.angle_2a.agent_runner import seed_global_rng_for_agent


class TestRngReproducibility(unittest.TestCase):
    def test_derive_rng_seed_is_deterministic(self):
        from experiments.angle_2a.agent_runner import derive_rng_seed

        self.assertEqual(derive_rng_seed(3, "D:D5W768"), derive_rng_seed(3, "D:D5W768"))

    def test_derive_rng_seed_differs_per_context(self):
        from experiments.angle_2a.agent_runner import derive_rng_seed

        seeds = {
            derive_rng_seed(3, "matchup_1:D:D5W768"),
            derive_rng_seed(3, "matchup_1:R:D2W512"),
            derive_rng_seed(3, "matchup_2:D:D7W1024"),
            derive_rng_seed(3, "shared_reference:D2W512"),
            derive_rng_seed(3, "null_matchup_1:D:null_D2W512"),
            derive_rng_seed(3, "null_matchup_2:D:null_D2W512"),
        }
        self.assertEqual(len(seeds), 6, "every distinct agent context must get a distinct derived seed")

    def test_derive_rng_seed_differs_per_base_seed(self):
        from experiments.angle_2a.agent_runner import derive_rng_seed

        self.assertNotEqual(derive_rng_seed(3, "D:D5W768"), derive_rng_seed(4, "D:D5W768"))

    def test_seed_global_rng_for_agent_produces_reproducible_buffer_style_sampling(self):
        # simulates what NpyUniformBuffer.sample() actually does: draw from
        # the GLOBAL np.random state via np.random.randint(...).
        seed_global_rng_for_agent(3, "matchup_1:D:D5W768")
        first_draw = np.random.randint(0, 1_000_000, size=10)

        # perturb global state as if some OTHER agent had trained first
        np.random.seed(999)
        np.random.randint(0, 1_000_000, size=500)
        _random_module.random()

        # re-seeding for the SAME agent context must reproduce the exact
        # same buffer-sampling sequence, regardless of what happened before.
        seed_global_rng_for_agent(3, "matchup_1:D:D5W768")
        second_draw = np.random.randint(0, 1_000_000, size=10)

        np.testing.assert_array_equal(first_draw, second_draw)

    def test_seed_global_rng_for_agent_also_seeds_python_random(self):
        seed_global_rng_for_agent(3, "matchup_1:D:D5W768")
        first = _random_module.random()

        _random_module.seed(12345)
        _random_module.random()

        seed_global_rng_for_agent(3, "matchup_1:D:D5W768")
        second = _random_module.random()

        self.assertEqual(first, second)

    def test_different_agents_get_different_buffer_sampling_sequences(self):
        seed_global_rng_for_agent(3, "matchup_1:D:D5W768")
        d_draw = np.random.randint(0, 1_000_000, size=10)

        seed_global_rng_for_agent(3, "matchup_1:R:D2W512")
        r_draw = np.random.randint(0, 1_000_000, size=10)

        self.assertFalse(np.array_equal(d_draw, r_draw))

    def test_training_order_does_not_change_an_agents_own_stream(self):
        # A trains first, then B: A's own sequence...
        seed_global_rng_for_agent(3, "agent_A")
        sequence_when_a_is_first = np.random.randint(0, 1_000_000, size=5)
        seed_global_rng_for_agent(3, "agent_B")
        np.random.randint(0, 1_000_000, size=5)

        # ... versus B training first, then A: A's sequence must be identical.
        seed_global_rng_for_agent(3, "agent_B")
        np.random.randint(0, 1_000_000, size=5)
        seed_global_rng_for_agent(3, "agent_A")
        sequence_when_a_is_second = np.random.randint(0, 1_000_000, size=5)

        np.testing.assert_array_equal(sequence_when_a_is_first, sequence_when_a_is_second)

    def test_null_baseline_agents_get_distinct_seed_contexts_per_matchup(self):
        # mirrors exactly how experiments/angle_2_a.py and matchup.py build
        # seed_context strings for the null-baseline path (matchup_name is
        # "null_matchup_1"/"null_matchup_2", roles "D"/"R", architecture
        # labels "null_D2W512"/"D2W512" - see matchup.run_matchup and its
        # null-baseline call site in experiments/angle_2_a.py).
        from experiments.angle_2a.agent_runner import derive_rng_seed

        null_a_d = derive_rng_seed(3, "null_matchup_1:D:null_D2W512")
        null_a_r = derive_rng_seed(3, "null_matchup_1:R:D2W512")
        null_b_d = derive_rng_seed(3, "null_matchup_2:D:null_D2W512")
        null_b_r = derive_rng_seed(3, "null_matchup_2:R:D2W512")

        self.assertEqual(len({null_a_d, null_a_r, null_b_d, null_b_r}), 4)


if __name__ == "__main__":
    unittest.main()
