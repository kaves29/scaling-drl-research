"""Covers the MyoSuite core-4/held-out-2 environment split added in the
2026-09-05 audit (see research-methodology.md's environment table):
Angle 1/2 may only ever select the core-4 set, and the other 2 are reserved
for Angle 3.
"""

import unittest

import gymnasium as gym

from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import check_single_env_type
from experiments.angle_2a.errors import Angle2AConfigError
from scale_rl.envs.myosuite import (
    MYOSUITE_CORE4,
    MYOSUITE_HELDOUT2,
    MYOSUITE_TASKS_DICT,
    validate_myosuite_core4,
)


class TestMyosuiteCore4HeldOut2Lists(unittest.TestCase):
    def test_core4_and_heldout2_are_disjoint(self):
        self.assertEqual(set(MYOSUITE_CORE4) & set(MYOSUITE_HELDOUT2), set())

    def test_core4_has_exactly_four_and_heldout2_has_exactly_two(self):
        self.assertEqual(len(MYOSUITE_CORE4), 4)
        self.assertEqual(len(MYOSUITE_HELDOUT2), 2)

    def test_every_core4_and_heldout2_alias_is_a_real_registered_myosuite_env(self):
        # Importing myosuite registers its gym envs as a side effect.
        from myosuite.utils import gym as myo_gym  # noqa: F401

        registered = set(gym.envs.registry.keys())
        for alias in MYOSUITE_CORE4 + MYOSUITE_HELDOUT2:
            self.assertIn(alias, MYOSUITE_TASKS_DICT, f"{alias} has no MYOSUITE_TASKS_DICT entry")
            task_id = MYOSUITE_TASKS_DICT[alias]
            self.assertIn(task_id, registered, f"{alias} -> {task_id} is not a registered myosuite env")


class TestMyosuiteCore4Validation(unittest.TestCase):
    def test_dmc_env_type_is_always_a_no_op(self):
        validate_myosuite_core4("dmc", "anything-at-all")  # must not raise

    def test_core4_myosuite_envs_are_accepted(self):
        for alias in MYOSUITE_CORE4:
            validate_myosuite_core4("myosuite", alias)  # must not raise

    def test_heldout2_myosuite_envs_are_rejected_with_a_clear_message(self):
        for alias in MYOSUITE_HELDOUT2:
            with self.assertRaises(ValueError) as ctx:
                validate_myosuite_core4("myosuite", alias)
            self.assertIn("held out for Angle 3", str(ctx.exception))

    def test_unknown_myosuite_env_name_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_myosuite_core4("myosuite", "myo-not-a-real-alias")
        self.assertIn("not one of the core-4", str(ctx.exception))


class TestAngle2AAcceptsMyosuiteEnvs(unittest.TestCase):
    """Angle 2A gained MyoSuite support 2026-09-07 (see env_state.py) - prior
    to this, check_single_env_type (formerly check_single_env_dmc) rejected
    every myosuite config outright via assert_supported_env_type."""

    def _base_cfg(self, env_name: str, num_train_envs: int = 1):
        return OmegaConf.create(
            {"env": {"env_type": "myosuite", "env_name": env_name, "num_train_envs": num_train_envs}}
        )

    def test_core4_myosuite_env_passes(self):
        for alias in MYOSUITE_CORE4:
            check_single_env_type(self._base_cfg(alias))  # must not raise

    def test_heldout2_myosuite_env_is_rejected(self):
        for alias in MYOSUITE_HELDOUT2:
            with self.assertRaises(ValueError) as ctx:
                check_single_env_type(self._base_cfg(alias))
            self.assertIn("held out for Angle 3", str(ctx.exception))

    def test_still_enforces_num_train_envs_one_after_the_core4_check(self):
        with self.assertRaises(Angle2AConfigError):
            check_single_env_type(self._base_cfg(MYOSUITE_CORE4[0], num_train_envs=2))


if __name__ == "__main__":
    unittest.main()
