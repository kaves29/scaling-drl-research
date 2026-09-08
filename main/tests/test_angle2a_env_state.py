import unittest
from unittest import mock

import numpy as np
from gymnasium.wrappers import TimeLimit
import gymnasium as gym

from experiments.angle_2a.env_state import (
    assert_supported_env_type,
    capture_env_state,
    restore_env_state,
)
from experiments.angle_2a.errors import Angle2AEnvironmentError


class _FakePhysicsData:
    def __init__(self, size):
        self.qacc_warmstart = np.zeros(size, dtype=np.float64)


class _FakePhysics:
    def __init__(self, initial_state):
        self._state = np.array(initial_state, dtype=np.float64)
        self.data = _FakePhysicsData(len(initial_state))
        self.forward_called = False

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = np.array(state, dtype=np.float64)

    def forward(self):
        self.forward_called = True

    def step(self):
        self._state = self._state + 1.0
        self.data.qacc_warmstart = self.data.qacc_warmstart + 1.0


class _FakeDmEnv(gym.Env):
    """Minimal stand-in for shimmy's DmControlCompatibilityV0: exposes .physics directly."""

    def __init__(self, physics):
        self.physics = physics
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def reset(self, **kwargs):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.physics.step()
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}


class _FakeMujocoData:
    """Minimal stand-in for MyoSuite's mj_data: time/qpos/qvel/act/ctrl/
    qacc_warmstart as plain numpy arrays, mutated in place by set(...) the
    same way real mujoco.MjData fields are (d.qpos[:] = ...)."""

    def __init__(self, qpos, qvel, act, ctrl, time=0.0):
        self.time = time
        self.qpos = np.array(qpos, dtype=np.float64)
        self.qvel = np.array(qvel, dtype=np.float64)
        self.act = np.array(act, dtype=np.float64)
        self.ctrl = np.array(ctrl, dtype=np.float64)
        self.qacc_warmstart = np.zeros_like(self.qpos)


class _FakeMujocoModel:
    def __init__(self, na, nmocap=0):
        self.na = na
        self.nmocap = nmocap


class _FakeRobot:
    """Minimal stand-in for myosuite's Robot interface: owns the mj_data/
    mj_model where physics is actually stepped - see env_state.py's module
    docstring, note 2, on why this must be distinct from the env's own
    (unused-by-capture/restore) mj_data/mj_model mirror."""

    def __init__(self, mj_data, mj_model):
        self.mj_data = mj_data
        self.mj_model = mj_model


class _FakeMyosuiteEnv(gym.Env):
    """Minimal stand-in for MyoSuite's MujocoEnv: exposes .robot.mj_data/
    .robot.mj_model (not .mj_data/.mj_model directly - see env_state.py's
    module docstring, note 2), with no .physics attribute (so it must go
    through the myosuite-specific capture/restore path, never the dmc one).
    Also carries a `steps` counter, mirroring myoLegWalk-v0's own
    (confirmed real) non-physics phase-tracking attribute."""

    def __init__(self, mj_data, mj_model):
        self.robot = _FakeRobot(mj_data, mj_model)
        self.steps = 0
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def reset(self, **kwargs):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        # advance qpos by 1.0/element and time by a fixed dt, mimicking a
        # real integration step, and increment steps like myoLegWalk-v0 does.
        d = self.robot.mj_data
        d.qpos = d.qpos + 1.0
        d.time += 0.02
        self.steps += 1
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}


class TestAssertSupportedEnvType(unittest.TestCase):
    def test_dmc_is_accepted(self):
        assert_supported_env_type("dmc")  # must not raise

    def test_myosuite_is_accepted(self):
        assert_supported_env_type("myosuite")  # must not raise

    def test_other_env_types_are_rejected(self):
        with self.assertRaises(Angle2AEnvironmentError):
            assert_supported_env_type("gym")


class TestCaptureRestoreDmcEnvState(unittest.TestCase):
    def test_capture_reads_physics_state_and_elapsed_steps(self):
        physics = _FakePhysics([1.0, 2.0, 3.0])
        env = TimeLimit(_FakeDmEnv(physics), max_episode_steps=1000)
        env.reset()
        env.step(np.array([0.0]))
        env.step(np.array([0.0]))

        captured = capture_env_state(env, "dmc")

        # _FakePhysics.step() increments every element by 1.0 per step, and
        # capture happens after 2 steps.
        np.testing.assert_array_equal(captured["physics_state"], [3.0, 4.0, 5.0])
        np.testing.assert_array_equal(captured["qacc_warmstart"], [2.0, 2.0, 2.0])
        self.assertEqual(captured["elapsed_steps"], 2)
        self.assertEqual(captured["env_type"], "dmc")

    def test_restore_sets_physics_state_and_calls_forward(self):
        physics = _FakePhysics([0.0, 0.0])
        env = TimeLimit(_FakeDmEnv(physics), max_episode_steps=1000)
        env.reset()

        restore_env_state(
            env,
            {
                "physics_state": np.array([9.0, 9.0]),
                "qacc_warmstart": np.array([7.0, 7.0]),
                "elapsed_steps": 42,
                "env_type": "dmc",
            },
        )

        np.testing.assert_array_equal(physics.get_state(), [9.0, 9.0])
        np.testing.assert_array_equal(physics.data.qacc_warmstart, [7.0, 7.0])
        self.assertTrue(physics.forward_called)
        self.assertEqual(env._elapsed_steps, 42)

    def test_capture_then_restore_round_trip_is_exact(self):
        physics = _FakePhysics([5.0, 6.0, 7.0])
        env = TimeLimit(_FakeDmEnv(physics), max_episode_steps=1000)
        env.reset()
        env.step(np.array([0.0]))
        env.step(np.array([0.0]))
        env.step(np.array([0.0]))

        captured = capture_env_state(env, "dmc")

        # perturb the env further
        env.step(np.array([0.0]))
        env.step(np.array([0.0]))
        self.assertFalse(np.array_equal(physics.get_state(), captured["physics_state"]))

        restore_env_state(env, captured)

        np.testing.assert_array_equal(physics.get_state(), captured["physics_state"])
        self.assertEqual(env._elapsed_steps, captured["elapsed_steps"])

    def test_missing_physics_raises_clear_error(self):
        class _NoPhysicsEnv(gym.Env):
            observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
            action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

            def reset(self, **kwargs):
                return np.zeros(2), {}

            def step(self, action):
                return np.zeros(2), 0.0, False, False, {}

        env = _NoPhysicsEnv()
        with self.assertRaises(Angle2AEnvironmentError):
            capture_env_state(env, "dmc")


class TestCaptureRestoreMyosuiteEnvState(unittest.TestCase):
    """mujoco.mj_forward requires real mujoco._structs.MjModel/MjData -
    _FakeMujocoModel/_FakeMujocoData are plain Python stand-ins (mirroring
    _FakePhysics's role for the dmc tests above), so mj_forward is patched to
    a no-op: the fields it would recompute (sensor data, contact forces)
    aren't what these tests check - only that the state fields themselves
    (qpos/qvel/act/time) land exactly on the captured values, not the
    real-mujoco derived quantities forward() would additionally update."""

    def setUp(self):
        patcher = mock.patch("experiments.angle_2a.env_state.mujoco.mj_forward")
        self.mock_mj_forward = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_env(self, na=2, nmocap=0):
        data = _FakeMujocoData(qpos=[1.0, 2.0], qvel=[0.1, 0.2], act=[0.5] * na if na else [], ctrl=[0.0, 0.0])
        model = _FakeMujocoModel(na=na, nmocap=nmocap)
        env = TimeLimit(_FakeMyosuiteEnv(data, model), max_episode_steps=1000)
        env.reset()
        return env, data, model

    def test_capture_reads_mj_data_fields_and_elapsed_steps(self):
        env, data, _model = self._make_env()
        env.step(np.array([0.0]))
        env.step(np.array([0.0]))

        captured = capture_env_state(env, "myosuite")

        np.testing.assert_array_equal(captured["qpos"], [3.0, 4.0])
        self.assertEqual(captured["elapsed_steps"], 2)
        self.assertEqual(captured["env_type"], "myosuite")
        self.assertEqual(captured["steps"], 2)

    def test_capture_reads_from_robot_mj_data_not_env_mj_data(self):
        """The specific regression the robot.mj_data dispatch exists to
        prevent (confirmed real, 2026-09-07): a naive implementation reading
        env.mj_data directly would silently capture/restore the wrong
        (unstepped) object."""
        env, data, _model = self._make_env()
        self.assertFalse(hasattr(env.unwrapped, "mj_data"))
        captured = capture_env_state(env, "myosuite")
        np.testing.assert_array_equal(captured["qpos"], data.qpos)

    def test_restore_does_not_advance_time_or_qpos_past_captured_values(self):
        """The specific regression this dispatch exists to prevent: restoring
        must never look like an extra step (mimicking MyoSuite's own
        set_env_state(), which internally calls mj_step - see env_state.py's
        module docstring). This fake's step() advances qpos by 1.0 and time
        by 0.02 per call, so any accidental extra "step" during restore would
        show up here."""
        env, data, _model = self._make_env()
        env.step(np.array([0.0]))
        captured = capture_env_state(env, "myosuite")
        qpos_at_capture = captured["qpos"].copy()
        time_at_capture = captured["time"]

        env.step(np.array([0.0]))  # perturb further
        env.step(np.array([0.0]))

        restore_env_state(env, captured)

        np.testing.assert_array_equal(data.qpos, qpos_at_capture)
        self.assertEqual(data.time, time_at_capture)

    def test_capture_then_restore_round_trip_is_exact(self):
        env, data, _model = self._make_env(na=3)
        env.step(np.array([0.0]))
        env.step(np.array([0.0]))
        env.step(np.array([0.0]))

        captured = capture_env_state(env, "myosuite")

        env.step(np.array([0.0]))
        env.step(np.array([0.0]))
        self.assertFalse(np.array_equal(data.qpos, captured["qpos"]))

        restore_env_state(env, captured)

        np.testing.assert_array_equal(data.qpos, captured["qpos"])
        np.testing.assert_array_equal(data.qvel, captured["qvel"])
        np.testing.assert_array_equal(data.act, captured["act"])
        self.assertEqual(data.time, captured["time"])
        self.assertEqual(env._elapsed_steps, captured["elapsed_steps"])
        self.assertEqual(env.unwrapped.steps, captured["steps"])

    def test_zero_actuator_env_captures_none_act(self):
        env, _data, _model = self._make_env(na=0)
        captured = capture_env_state(env, "myosuite")
        self.assertIsNone(captured["act"])
        # must not raise on restore either
        restore_env_state(env, captured)

    def test_missing_mj_data_raises_clear_error(self):
        class _NoMjDataEnv(gym.Env):
            observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
            action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

            def reset(self, **kwargs):
                return np.zeros(2), {}

            def step(self, action):
                return np.zeros(2), 0.0, False, False, {}

        env = _NoMjDataEnv()
        with self.assertRaises(Angle2AEnvironmentError):
            capture_env_state(env, "myosuite")


if __name__ == "__main__":
    unittest.main()
