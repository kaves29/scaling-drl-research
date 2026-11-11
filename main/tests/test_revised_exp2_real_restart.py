"""Actual DMC environment restore inside a partially trained SAC branch."""

import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.online_exp2 import OnlineExp2
from experiments.exp12.online_exp2_pilot import CONDITIONS
from tests.test_revised_exp2_runner import revised_cfg


class RealBranchRestartTest(unittest.TestCase):
    def test_real_dmc_learning_branch_matches_uninterrupted(self):
        value = revised_cfg()
        value.env.action_repeat = 2
        value.online_exp2.checkpoint_raw_steps = [3, 8]
        value.env.max_episode_steps = 5
        original = OnlineExp2._check_resource_cap
        injected = [False]
        with tempfile.TemporaryDirectory() as output, tempfile.TemporaryDirectory() as control:

            def interrupt(runner):
                original(runner)
                path = Path(output) / "diagnostics/raw_00000003/small_learning.pkl"
                if path.exists() and not injected[0]:
                    state = pickle.loads(path.read_bytes())
                    if state["raw_env_steps"] >= 4 and not state["complete"]:
                        injected[0] = True
                        raise RuntimeError("real DMC interruption")

            with patch(
                "experiments.online_exp2.validate", return_value=(4, [3, 8], [3, 8])
            ), patch(
                "experiments.exp12.online_exp2_diagnostic.reference_roles",
                return_value=((1, 8), (1, 8), (1, 8)),
            ):
                with patch.object(OnlineExp2, "_check_resource_cap", interrupt):
                    with self.assertRaisesRegex(RuntimeError, "real DMC interruption"):
                        OnlineExp2(value, output).run()
                OnlineExp2(value, output).run()
                OnlineExp2(value, control).run()
            self.assertTrue(injected[0])
            for check in (3, 8):
                for architecture, mode in CONDITIONS:
                    relative = f"diagnostics/raw_{check:08d}/{architecture}_{mode}.pkl"
                    restored = pickle.loads((Path(output) / relative).read_bytes())[
                        "summary"
                    ]
                    expected = pickle.loads((Path(control) / relative).read_bytes())[
                        "summary"
                    ]
                    self.assertEqual(restored["final_models"], expected["final_models"])
                    self.assertEqual(
                        restored["final_replay_hash"], expected["final_replay_hash"]
                    )
                    self.assertEqual(restored["episodes"], expected["episodes"])
