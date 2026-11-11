"""Real simulator restore retains each shortened action-repeat allowance."""

import copy
import unittest

import numpy as np

from experiments.exp12.envs import create_envs, env_restore_state, restore_env
from scale_rl.envs.wrappers.repeat_action import limit_raw_steps, repeat_wrapper
from tests.test_revised_exp2_runner import revised_cfg


class RepeatAwareEnvironmentRestoreTest(unittest.TestCase):
    def test_odd_boundary_and_legacy_state_restore(self):
        cfg = revised_cfg()
        cfg.env.action_repeat = 2
        cfg.env.max_episode_steps = 100
        reference, restored = create_envs(**cfg.env)
        try:
            reference.reset()
            action = np.zeros(reference.action_space.shape, dtype=np.float32)
            limit_raw_steps(reference, 3)
            reference.step(action)  # two physical steps
            limit_raw_steps(reference, 1)
            reference.step(action)  # odd physical boundary, three total
            state = env_restore_state(reference)
            self.assertEqual(state["env"]["repeat_limits"], [3, 1])
            restore_env(restored, state)
            for env in (reference, restored):
                limit_raw_steps(env, 2)
            a, b = reference.step(action), restored.step(action)
            for left, right in zip(a[:4], b[:4]):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(
                env_restore_state(reference)["env"]["repeat_limits"],
                env_restore_state(restored)["env"]["repeat_limits"],
            )

            # Older checkpoints had actions but no repeat metadata. They must
            # retain the original normal-repeat replay behavior.
            reference.reset()
            repeat_wrapper(reference).raw_step_limit = None
            reference.step(action)
            reference.step(action)
            legacy = copy.deepcopy(env_restore_state(reference))
            legacy["env"].pop("repeat_limits")
            legacy["env"].pop("repeat_limit")
            restore_env(restored, legacy)
            a, b = reference.step(action), restored.step(action)
            for left, right in zip(a[:4], b[:4]):
                np.testing.assert_array_equal(left, right)
        finally:
            reference.close()
            restored.close()
