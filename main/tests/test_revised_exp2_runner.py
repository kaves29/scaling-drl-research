"""Revised pilot integration with real SAC updates and interruption recovery."""

import contextlib
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv
from omegaconf import OmegaConf

from experiments.online_exp2 import OnlineExp2, validate
from experiments.exp12.online_exp2_pilot import CONDITIONS, _reference_state
from scale_rl.envs.wrappers.repeat_action import RepeatAction
from tests.test_online_exp2_driver import TinyEnv, cfg, tiny_envs


def revised_cfg():
    value = cfg()
    OmegaConf.set_struct(value, False)
    value.run_role = "pilot"
    value.online_exp2.update(
        {
            "pilot_enabled": True,
            "reference_regime": "reference_small",
            "reference_raw_steps": 8,
            "checkpoint_raw_steps": [4, 8],
            "branch_raw_steps": 10,
            "measurement_raw_steps": [0, 5, 10],
            "diagnostic_save_every": 2,
            "resource_cap_gpu_hours": 0,
        }
    )
    return value


class RestorableRepeat(RepeatAction):
    def restore_state(self):
        return self.env.restore_state()

    def restore(self, state):
        self.env.restore(state)


def wrapped_env(*args, **kwargs):
    return SyncVectorEnv([lambda: RestorableRepeat(TinyEnv(), 2)])


class BranchTinyEnv(TinyEnv):
    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        return obs, reward, term, self.step_count >= 7, info


def wrapped_branch_env(*args, **kwargs):
    return SyncVectorEnv([lambda: RestorableRepeat(BranchTinyEnv(), 2)])


def patches():
    stack = contextlib.ExitStack()
    stack.enter_context(
        patch("experiments.online_exp2.validate", return_value=(4, [4, 8], [4, 8]))
    )
    stack.enter_context(
        patch(
            "experiments.exp12.online_exp2_diagnostic.reference_roles",
            return_value=((1, 8), (1, 8), (1, 8)),
        )
    )
    stack.enter_context(
        patch(
            "experiments.exp12.trainer.create_envs",
            lambda **kwargs: (wrapped_env(), wrapped_env()),
        )
    )
    stack.enter_context(
        patch("experiments.exp12.online_exp2_pilot._branch_env", wrapped_branch_env)
    )
    return stack


class RevisedRunnerTest(unittest.TestCase):
    def test_full_four_conditions_raw_boundaries_and_frozen_state(self):
        with tempfile.TemporaryDirectory() as directory, patches():
            runner = OnlineExp2(revised_cfg(), directory)
            runner.run()
            self.assertEqual(runner.trainer.raw_env_steps, 8)
            self.assertGreater(
                runner.trainer.interaction_step, 4
            )  # early terminal repeats
            self.assertFalse(
                any(row["type"] == "pilot_evaluation" for row in runner.rows)
            )
            self.assertEqual(
                len([r for r in runner.rows if r["type"] == "pilot_diagnostic"]), 2
            )
            self.assertEqual(len(runner._pilot_analysis()["conditions"]), 4)
            self.assertTrue(
                (
                    Path(directory) / "diagnostics/raw_00000004/training_returns.png"
                ).exists()
            )
            for check in (4, 8):
                progress = pickle.loads(
                    (
                        Path(directory)
                        / "diagnostics"
                        / f"raw_{check:08d}"
                        / "progress.pkl"
                    ).read_bytes()
                )
                self.assertEqual(
                    set(progress["branches"]), {a + "_" + m for a, m in CONDITIONS}
                )
                actors = set()
                for condition, summary in progress["branches"].items():
                    self.assertEqual(summary["raw_env_steps"], 10)
                    self.assertEqual(set(summary["markers"]), {5, 10})
                    self.assertEqual(summary["markers"][5]["observed_raw_env_steps"], 6)
                    actors.add(summary["initial_actor_hash"])
                    state = pickle.loads(Path(summary["artifact"]).read_bytes())
                    self.assertTrue(state["complete"])
                    if condition.endswith("frozen"):
                        for name in ("critic", "_target_critic", "temperature"):
                            self.assertEqual(
                                state["initial_models"][name],
                                summary["final_models"][name],
                            )
                    else:
                        self.assertNotEqual(
                            state["initial_models"]["critic"],
                            summary["final_models"]["critic"],
                        )
                        self.assertNotEqual(
                            state["initial_models"]["temperature"],
                            summary["final_models"]["temperature"],
                        )
                        self.assertGreater(summary["final_replay_length"], check // 2)
                self.assertEqual(len(actors), 1)
            before = _reference_state(runner.trainer)
            resumed = OnlineExp2(revised_cfg(), directory)
            resumed.run()
            self.assertEqual(before, _reference_state(resumed.trainer))
            self.assertEqual(len(runner.rows), len(resumed.rows))

    def test_restart_learning_branch_matches_uninterrupted(self):
        original = RepeatAction.step
        interrupted = [False]

        def fail(env, action):
            # Four branches at checkpoint 4: inject on first learning branch.
            files = list(
                Path(directory).glob("diagnostics/raw_00000004/small_learning.pkl")
            )
            if files and not interrupted[0]:
                state = pickle.loads(files[0].read_bytes())
                if state["interactions"] >= 2:
                    interrupted[0] = True
                    raise RuntimeError("injected interruption")
            return original(env, action)

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as control, patches():
            with patch.object(RepeatAction, "step", fail):
                with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                    OnlineExp2(revised_cfg(), directory).run()
            resumed = OnlineExp2(revised_cfg(), directory)
            resumed.run()
            uninterrupted = OnlineExp2(revised_cfg(), control)
            uninterrupted.run()
            self.assertTrue(interrupted[0])
            for check in (4, 8):
                for architecture, mode in CONDITIONS:
                    relative = f"diagnostics/raw_{check:08d}/{architecture}_{mode}.pkl"
                    a = pickle.loads((Path(directory) / relative).read_bytes())
                    b = pickle.loads((Path(control) / relative).read_bytes())
                    self.assertEqual(
                        a["summary"]["final_models"], b["summary"]["final_models"]
                    )
                    self.assertEqual(a["episodes"], b["episodes"])
                    np.testing.assert_array_equal(a["rng"], b["rng"])
                    if mode == "learning":
                        self.assertEqual(
                            a["summary"]["final_replay_hash"],
                            b["summary"]["final_replay_hash"],
                        )

    def test_approved_profiles_validate_and_reject_old_horizon(self):
        from experiments.exp1 import compose_config

        for regime in ("small", "large"):
            value = compose_config(
                str(Path(__file__).parents[1] / "configs"),
                "online_exp2_pilot_reference_" + regime,
                [],
                allow_pilot=True,
            )
            self.assertEqual(len(validate(value)[1]), 15)
            value.online_exp2.branch_raw_steps = 20000
            with self.assertRaises(ValueError):
                validate(value)

    def test_real_dmc_four_branch_checkpoint_and_completed_restart(self):
        value = revised_cfg()
        value.env.action_repeat = 2
        value.env.max_episode_steps = 5
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "experiments.online_exp2.validate", return_value=(4, [4, 8], [4, 8])
            ), patch(
                "experiments.exp12.online_exp2_diagnostic.reference_roles",
                return_value=((1, 8), (1, 8), (1, 8)),
            ):
                runner = OnlineExp2(value, directory)
                runner.run()
                again = OnlineExp2(value, directory)
                again.run()
                self.assertEqual(runner.trainer.raw_env_steps, 8)
                self.assertEqual(again._pilot_analysis(), runner._pilot_analysis())


if __name__ == "__main__":
    unittest.main()
