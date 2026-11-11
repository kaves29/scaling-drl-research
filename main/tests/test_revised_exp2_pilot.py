"""CPU contracts for the revised four-branch online Exp2 pilot."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from gymnasium.vector import SyncVectorEnv
from omegaconf import OmegaConf
from scale_rl.envs.wrappers.repeat_action import RepeatAction

from experiments.exp1 import compose_config
from experiments.exp12.revised_exp2_pilot import (
    CONDITIONS, RevisedBranch, revised_analysis,
)
from experiments.online_exp2 import OnlineExp2
from tests.test_online_exp2_driver import TinyEnv, tiny_envs

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def revised_cfg():
    return compose_config(CONFIG_DIR, "online_exp2_revised_pilot", [
        "env_name=dog-run", "seed=1",
        "online_exp2.environments=[dog-run]", "online_exp2.training_seeds=[1]",
        "online_exp2.reference_regime=reference_small",
        "online_exp2.training_interactions=3",
        "online_exp2.checkpoint_steps=[2]",
        "online_exp2.probe_checkpoints=[2]",
        "online_exp2.diagnostic_raw_steps=20",
        "online_exp2.diagnostic_save_every=1",
        "online_exp2.resource_cap_gpu_hours=1",
        "online_exp2.dev_architectures=[[small,1,8],[large,1,8]]",
        "buffer.min_length=1", "buffer.sample_batch_size=4", "buffer.max_length=64",
        "probe.rounds=1", "probe.steps=2", "probe.pool_size=4",
        "probe.batch_size=4", "probe.eval_chunk=4",
        "logging_per_interaction_step=100", "evaluation_per_interaction_step=100",
        "num_eval_episodes=1", "actor_grad_cosine_every=100",
        "env.action_repeat=4", "action_repeat=4", "env.max_episode_steps=5",
        "wandb_enabled=false",
    ], allow_pilot=True)


def branch_env(*args):
    class RestorableRepeat(RepeatAction):
        def restore_state(self):
            return {"env": self.env.restore_state(), "repeat": self._action_repeat}

        def restore(self, state):
            self.env.restore(state["env"])
            self._action_repeat = state["repeat"]

    return SyncVectorEnv([lambda: RestorableRepeat(TinyEnv(), action_repeat=4)])


def _run(root, patch_validate=True):
    cfg = revised_cfg()
    context = [
        patch("experiments.exp12.trainer.create_envs", tiny_envs),
        patch("experiments.exp12.revised_exp2_pilot._branch_env", branch_env),
        patch("experiments.online_exp2.OnlineExp2._probe", lambda self, step: None),
    ]
    if patch_validate:
        context.append(patch("experiments.online_exp2.validate", return_value=(3, [2], [2])))
    for item in context:
        item.start()
    try:
        runner = OnlineExp2(cfg, root)
        runner.run()
        return runner
    finally:
        for item in reversed(context):
            item.stop()


class RevisedPilotTest(unittest.TestCase):
    def test_four_isolated_branches_with_raw_step_cap(self):
        with tempfile.TemporaryDirectory() as root:
            runner = _run(root)
            records = [row for row in runner.rows if row["type"] == "revised_pilot_checkpoint"]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(set(record["branches"]), set(CONDITIONS))
            for name, state in record["branches"].items():
                self.assertEqual(state["raw_steps"], 20, name)
                self.assertEqual(state["interactions"], sum(
                    episode["length"] for episode in record["branch_episodes"][name]))
                self.assertGreater(state["optimizer_updates"], 0)
            shadow_hashes = record["shadow_hashes"]
            for architecture in ("small", "large"):
                frozen = record["branches"][f"{architecture}_frozen"]
                learning = record["branches"][f"{architecture}_learning"]
                self.assertEqual(frozen["critic_hash"], shadow_hashes[architecture])
                self.assertNotEqual(learning["critic_hash"], shadow_hashes[architecture])
                self.assertNotEqual(frozen["actor_hash"], learning["actor_hash"])
            analysis = revised_analysis(runner)
            self.assertEqual(set(analysis["conditions"]), set(CONDITIONS))
            for condition in analysis["conditions"].values():
                self.assertEqual(set(condition["checkpoint_means"]), {2})
            self.assertFalse(any(row.get("type") == "pilot_evaluation" for row in runner.rows))
            # Reference replay and shadow critics stay untouched.
            self.assertEqual(len(runner.trainer.buffer), 3)
            self.assertEqual(runner.shadows["small"].updates, runner.trainer.update_step)
            self.assertEqual(runner.shadows["large"].updates, runner.trainer.update_step)

    def test_early_termination_counts_actual_raw_steps(self):
        with tempfile.TemporaryDirectory() as root:
            runner = _run(root)
            record = next(row for row in runner.rows if row["type"] == "revised_pilot_checkpoint")
            for name in CONDITIONS:
                episodes = record["branch_episodes"][name]
                self.assertEqual(sum(episode["raw_steps"] for episode in episodes), 20)
                self.assertTrue(any(episode["truncated"] for episode in episodes))

    def test_learning_branch_copies_reference_replay(self):
        with tempfile.TemporaryDirectory() as root:
            runner = _run(root)
            record = next(row for row in runner.rows if row["type"] == "revised_pilot_checkpoint")
            self.assertEqual(len(runner.trainer.buffer), 3)
            for name in ("small_learning", "large_learning"):
                state = record["branches"][name]
                self.assertGreater(state["interactions"], 0)
                self.assertEqual(state["raw_steps"], 20)

    def test_resume_mid_branch_is_bit_identical(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as control:
            original = RevisedBranch.observe_and_update
            interrupted = [False]

            def fail_once(branch, timestep, observation):
                if (branch.name == "small_learning" and branch.raw_steps == 10
                        and not interrupted[0]):
                    interrupted[0] = True
                    raise RuntimeError("simulated branch interruption")
                return original(branch, timestep, observation)

            with patch.object(RevisedBranch, "observe_and_update", fail_once):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    _run(root)
            resumed = _run(root)
            self.assertTrue(interrupted[0])
            control_runner = _run(control)
            resumed_record = next(row for row in resumed.rows
                                  if row["type"] == "revised_pilot_checkpoint")
            control_record = next(row for row in control_runner.rows
                                  if row["type"] == "revised_pilot_checkpoint")
            for name in CONDITIONS:
                self.assertEqual(resumed_record["branches"][name]["actor_hash"],
                                 control_record["branches"][name]["actor_hash"], name)
                self.assertEqual(resumed_record["branch_episodes"][name],
                                 control_record["branch_episodes"][name], name)
                self.assertEqual(resumed_record["branches"][name]["raw_steps"], 20)


if __name__ == "__main__":
    unittest.main()
