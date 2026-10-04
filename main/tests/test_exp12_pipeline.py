"""Phase 7, requirement 9: the tiny end-to-end pipeline for each suite (DMC, MyoSuite, HumanoidBench).

Per suite: a default (D1W8) and a scaled (D1W16) dev run over all 20 checks; the scaled run's f*_run
is forced at check 5 (test-only hook), it forks, the control continues and the injected arm runs;
Checks 1-2, post-fork probes and evaluations, both ledgers, and both analysis scripts on the outputs.
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp12_helpers import patch_wandb, tiny_overrides  # noqa: E402
from experiments.exp12 import exp2_ledger, ledger  # noqa: E402
from test_exp12_fork import run_arm, run_exp1  # noqa: E402

HAVE_HB = importlib.util.find_spec("humanoid_bench") is not None


class PipelinePerSuiteTest(unittest.TestCase):
    def setUp(self):
        self.patchers = patch_wandb()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pipeline(self, env_name, group):
        from analysis import exp1_analysis, exp2_analysis

        results = os.path.join(self.tmp, "results")
        common = ["run_role=dev", "env.max_episode_steps=40", "fork.eval_episodes=1", f"results_root={results}",
                  "fork.architectures=[D1W16]"]
        default = tiny_overrides(env_name, group, seed=7, steps=400, extra=common)
        scaled = tiny_overrides(env_name, group, seed=7, steps=400,
                                extra=[*common, "critic_hidden_dim=16", "testing.force_trigger_check=5"])
        run_exp1(os.path.join(self.tmp, "default"), default)
        scaled_dir = os.path.join(self.tmp, "scaled")
        run_exp1(scaled_dir, scaled)
        run_arm(os.path.join(self.tmp, "arm"), scaled, scaled_dir, "injected", m="last")

        runs, checks = ledger.load(results, include_dev=True)
        self.assertEqual(sorted(runs.architecture), ["D1W16", "D1W8"])
        self.assertEqual(set(runs.status), {"complete"})
        for key in runs.run_key:
            self.assertEqual(sorted(checks[checks.run_key == key].check_index), list(range(21)))
        run = runs[runs.architecture == "D1W16"].iloc[0]
        self.assertEqual((run.f_star_check, run.fork_interaction_step), (5, 100))
        self.assertTrue(pd.isna(runs[runs.architecture == "D1W8"].iloc[0].fork_interaction_step))

        exp2 = exp2_ledger.run_root(run.run_key, results)
        with open(exp2 / "check1_injected.json") as f:
            self.assertTrue(json.load(f)["pass"])
        with open(exp2 / "check2.json") as f:
            self.assertEqual(len(json.load(f)["paired_difference_rounds"]), 5)
        for arm, last in (("control", 20), ("injected", 10)):
            self.assertEqual(sorted(pd.read_csv(exp2 / f"arm_{arm}" / "checks.csv").check_index),
                             list(range(5, last + 1)))
            self.assertEqual(pd.read_csv(exp2 / f"arm_{arm}" / "eval_episodes.csv").eval_index.nunique(), 26)

        out1 = os.path.join(self.tmp, "exp1_analysis")
        primary = exp1_analysis.run_analysis(out1, results, include_dev=True, default="D1W8",
                                             scaled=("D1W16",))["primary"]
        self.assertEqual(list(primary.comparison), ["D1W16 - D1W8"])
        self.assertEqual((primary.runs_scaled.iloc[0], primary.runs_default.iloc[0]), (1, 1))
        out2 = os.path.join(self.tmp, "exp2_analysis")
        outputs = exp2_analysis.run_analysis(out2, results_root=results, include_dev=True, scaled=("D1W16",))
        self.assertEqual(len(outputs["paired"]), 26)
        for d, names in ((out1, ("primary_endpoint.csv", "trajectories.png", "every_seed.png", "f_star_table.csv")),
                         (out2, ("paired_returns_D1W16.png", "check1_table.csv", "check2_table.csv",
                                 f"shared_time_axis_D1W16_{env_name}.png", "plasticity_post_fork_D1W16.png"))):
            for name in names:
                self.assertTrue(os.path.exists(os.path.join(d, name)), name)

    def test_dmc(self):
        self._pipeline("hopper-hop", "dmc_medium")

    def test_myosuite(self):
        self._pipeline("myo-reach", "myosuite_simba")

    @unittest.skipUnless(HAVE_HB, "needs humanoid_bench (venv_hb; MUJOCO_GL=PYOPENGL_PLATFORM=egl or osmesa)")
    def test_humanoid_bench(self):
        self._pipeline("h1-reach-v0", "humanoid_bench")


if __name__ == "__main__":
    unittest.main()
