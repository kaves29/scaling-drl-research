import unittest

import numpy as np
from gymnasium.wrappers import TimeLimit
import gymnasium as gym

from experiments.angle_2a.env_state import (
    assert_dmc_env_type,
    capture_env_state,
    restore_env_state,
)
from experiments.angle_2a.errors import Angle2AEnvironmentError


class _FakePhysics:
    def __init__(self, initial_state):
        self._state = np.array(initial_state, dtype=np.float64)
        self.forward_called = False

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = np.array(state, dtype=np.float64)

    def forward(self):
        self.forward_called = True

    def step(self):
        self._state = self._state + 1.0


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


class TestAssertDmcEnvType(unittest.TestCase):
    def test_dmc_is_accepted(self):
        assert_dmc_env_type("dmc")  # must not raise

    def test_other_env_types_are_rejected(self):
        with self.assertRaises(Angle2AEnvironmentError):
            assert_dmc_env_type("myosuite")


class TestCaptureRestoreEnvState(unittest.TestCase):
    def test_capture_reads_physics_state_and_elapsed_steps(self):
        physics = _FakePhysics([1.0, 2.0, 3.0])
        env = TimeLimit(_FakeDmEnv(physics), max_episode_steps=1000)
        env.reset()
        env.step(np.array([0.0]))
        env.step(np.array([0.0]))

        captured = capture_env_state(env)

        # _FakePhysics.step() increments every element by 1.0 per step, and
        # capture happens after 2 steps.
        np.testing.assert_array_equal(captured["physics_state"], [3.0, 4.0, 5.0])
        self.assertEqual(captured["elapsed_steps"], 2)

    def test_restore_sets_physics_state_and_calls_forward(self):
        physics = _FakePhysics([0.0, 0.0])
        env = TimeLimit(_FakeDmEnv(physics), max_episode_steps=1000)
        env.reset()

        restore_env_state(env, {"physics_state": np.array([9.0, 9.0]), "elapsed_steps": 42})

        np.testing.assert_array_equal(physics.get_state(), [9.0, 9.0])
        self.assertTrue(physics.forward_called)
        self.assertEqual(env._elapsed_steps, 42)

    def test_capture_then_restore_round_trip_is_exact(self):
        physics = _FakePhysics([5.0, 6.0, 7.0])
        env = TimeLimit(_FakeDmEnv(physics), max_episode_steps=1000)
        env.reset()
        env.step(np.array([0.0]))
        env.step(np.array([0.0]))
        env.step(np.array([0.0]))

        captured = capture_env_state(env)

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
            capture_env_state(env)


if __name__ == "__main__":
    unittest.main()
