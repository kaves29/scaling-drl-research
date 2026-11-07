"""Standing, permanent determinism smoke test for Angle 2A's exact-state
capture/restore mechanism (experiments/angle_2a/env_state.py), run against
every environment in this study - not just the 6 core DMC ones Angle 2A
originally supported.

This exists to empirically verify (not just manually assess) the
"MyoSuite state-reset fidelity" Accepted Limitation in research-methodology.md:
capture a state mid-episode, then restore it and replay the identical action
sequence twice, and confirm the two resulting trajectories are bit-identical.
A future environment addition, myosuite/dm_control upgrade, or a regression
in env_state.py's dispatch would be caught here automatically, rather than
relying on the one-time manual investigation that motivated this test
(2026-09-07 - see research-methodology.md's Angle 2A section for the
MyoSuite set_env_state() finding this specifically guards against).

Real environments throughout (via scale_rl.envs.create_vec_env, the same
construction path Angle 2A itself uses) - no mocks. Slower than the rest of
the Angle 2A suite (14 real environments, each independently instantiated),
but this is exactly the kind of correctness property that must not be
assumed.
"""

import unittest

import numpy as np

from experiments.angle_2a.env_state import capture_env_state, restore_env_state
from scale_rl.envs import create_vec_env
from scale_rl.envs.dmc import DMC_HARD, DMC_HELDOUT2, DMC_MED
from scale_rl.envs.myosuite import MYOSUITE_CORE4, MYOSUITE_HELDOUT2

ALL_DMC_ENVS = DMC_MED + DMC_HARD + DMC_HELDOUT2
ALL_MYOSUITE_ENVS = MYOSUITE_CORE4 + MYOSUITE_HELDOUT2

NUM_WARMUP_STEPS = 5
NUM_ROLLOUT_STEPS = 5


def _rollout_from_restored_state(env, captured, actions):
    restore_env_state(env, captured)
    trajectory = []
    for action in actions:
        obs, reward, terminated, truncated, _info = env.step(action)
        trajectory.append((np.asarray(obs).copy(), float(reward), bool(terminated), bool(truncated)))
        if terminated or truncated:
            break
    return trajectory


def _assert_determinism(test_case, env_type: str, env_name: str) -> None:
    vec_env = create_vec_env(env_type=env_type, env_name=env_name, num_envs=1, seed=0)
    try:
        env = vec_env.envs[0]
        env.reset(seed=0)
        for _ in range(NUM_WARMUP_STEPS):
            env.step(env.action_space.sample())

        captured = capture_env_state(env, env_type)
        actions = [env.action_space.sample() for _ in range(NUM_ROLLOUT_STEPS)]

        trajectory_1 = _rollout_from_restored_state(env, captured, actions)
        trajectory_2 = _rollout_from_restored_state(env, captured, actions)

        test_case.assertEqual(
            len(trajectory_1), len(trajectory_2),
            f"{env_type}:{env_name} - two replays of the identical restored "
            f"state + action sequence terminated/truncated at different "
            f"points ({len(trajectory_1)} vs {len(trajectory_2)} steps).",
        )
        for step_idx, (step_1, step_2) in enumerate(zip(trajectory_1, trajectory_2)):
            obs_1, reward_1, terminated_1, truncated_1 = step_1
            obs_2, reward_2, terminated_2, truncated_2 = step_2
            np.testing.assert_array_equal(
                obs_1, obs_2,
                err_msg=f"{env_type}:{env_name} step {step_idx}: observation mismatch",
            )
            test_case.assertEqual(reward_1, reward_2, f"{env_type}:{env_name} step {step_idx}: reward mismatch")
            test_case.assertEqual(terminated_1, terminated_2, f"{env_type}:{env_name} step {step_idx}: terminated mismatch")
            test_case.assertEqual(truncated_1, truncated_2, f"{env_type}:{env_name} step {step_idx}: truncated mismatch")
    finally:
        vec_env.close()


class TestDmcDeterminism(unittest.TestCase):
    pass


class TestMyosuiteDeterminism(unittest.TestCase):
    pass


class TestBaodingCounterRestore(unittest.TestCase):
    def test_repeated_restore_replays_exactly(self):
        self._check_replays(fresh_instance=False)

    def test_fresh_instance_replays_exactly_with_same_episode_parameters(self):
        self._check_replays(fresh_instance=True)

    def _check_replays(self, fresh_instance):
        vectors = []
        try:
            def make():
                vector = create_vec_env(env_type="myosuite", env_name="myo-baoding-p1", num_envs=1, seed=0)
                vectors.append(vector)
                env = vector.envs[0]
                env.reset(seed=0)
                return env

            source = make()
            for _ in range(NUM_WARMUP_STEPS):
                source.step(source.action_space.sample())
            state = capture_env_state(source, "myosuite")
            self.assertEqual(state["counter"], NUM_WARMUP_STEPS)
            actions = [source.action_space.sample() for _ in range(NUM_ROLLOUT_STEPS)]
            expected = _rollout_from_restored_state(source, state, actions)
            target = make() if fresh_instance else source
            for _ in range(3):
                restore_env_state(target, state)
                self.assertEqual(target.unwrapped.counter, state["counter"])
                observed = _rollout_from_restored_state(target, state, actions)
                self.assertEqual(len(expected), len(observed))
                for first, second in zip(expected, observed):
                    np.testing.assert_array_equal(first[0], second[0])
                    self.assertEqual(first[1:], second[1:])
                self.assertEqual(target.unwrapped.counter, state["counter"] + len(actions))
        finally:
            for vector in vectors:
                vector.close()


def _make_test(env_type, env_name):
    def test(self):
        _assert_determinism(self, env_type, env_name)
    return test


for _env_name in ALL_DMC_ENVS:
    setattr(TestDmcDeterminism, f"test_{_env_name.replace('-', '_')}", _make_test("dmc", _env_name))

for _env_name in ALL_MYOSUITE_ENVS:
    setattr(TestMyosuiteDeterminism, f"test_{_env_name.replace('-', '_')}", _make_test("myosuite", _env_name))


if __name__ == "__main__":
    unittest.main()
