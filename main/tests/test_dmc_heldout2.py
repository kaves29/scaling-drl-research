"""Covers the DMC held-out-2 environment split added in the 2026-09-05 audit
(see research-methodology.md's environment table): Angle 1/2 may only ever
select the core (DMC_MED + DMC_HARD) set, and DMC_HELDOUT2 is reserved for
Angle 3. Mirrors tests/test_myosuite_core4.py for the DMC side.
"""

import unittest

from dm_control import suite
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import check_single_env_type
from experiments.angle_2a.errors import Angle2AConfigError
from scale_rl.envs.dmc import DMC_HARD, DMC_HELDOUT2, DMC_MED, validate_dmc_not_heldout


class TestDmcHeldOut2List(unittest.TestCase):
    def test_core_and_heldout2_are_disjoint(self):
        core = set(DMC_MED) | set(DMC_HARD)
        self.assertEqual(core & set(DMC_HELDOUT2), set())

    def test_heldout2_has_exactly_two(self):
        self.assertEqual(len(DMC_HELDOUT2), 2)

    def test_every_heldout2_env_is_a_real_dm_control_task(self):
        for env_name in DMC_HELDOUT2:
            domain_name, task_name = env_name.split("-")
            self.assertIn(
                (domain_name, task_name),
                suite.ALL_TASKS,
                f"{env_name} is not a real dm_control (domain, task) pair",
            )


class TestDmcNotHeldoutValidation(unittest.TestCase):
    def test_myosuite_env_type_is_always_a_no_op(self):
        validate_dmc_not_heldout("myosuite", "anything-at-all")  # must not raise

    def test_core_dmc_envs_are_accepted(self):
        for env_name in DMC_MED + DMC_HARD:
            validate_dmc_not_heldout("dmc", env_name)  # must not raise

    def test_heldout2_dmc_envs_are_rejected_with_a_clear_message(self):
        for env_name in DMC_HELDOUT2:
            with self.assertRaises(ValueError) as ctx:
                validate_dmc_not_heldout("dmc", env_name)
            self.assertIn("held out for Angle 3", str(ctx.exception))


class TestAngle2ARejectsHeldoutDmcEnvs(unittest.TestCase):
    def _base_cfg(self, env_name: str, num_train_envs: int = 1):
        return OmegaConf.create(
            {"env": {"env_type": "dmc", "env_name": env_name, "num_train_envs": num_train_envs}}
        )

    def test_core_dmc_env_passes(self):
        check_single_env_type(self._base_cfg("cheetah-run"))  # must not raise

    def test_heldout2_dmc_env_is_rejected(self):
        for env_name in DMC_HELDOUT2:
            with self.assertRaises(ValueError) as ctx:
                check_single_env_type(self._base_cfg(env_name))
            self.assertIn("held out for Angle 3", str(ctx.exception))

    def test_still_enforces_num_train_envs_one_after_the_heldout_check(self):
        with self.assertRaises(Angle2AConfigError):
            check_single_env_type(self._base_cfg("cheetah-run", num_train_envs=2))


if __name__ == "__main__":
    unittest.main()
