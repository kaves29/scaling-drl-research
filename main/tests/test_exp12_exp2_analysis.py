"""Phase 5: Experiment 2 analysis on synthetic results with known effects."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from analysis import exp2_analysis as ea
from experiments.exp12 import exp2_ledger, ledger

PLAN = {"fork_step": 100, "fork_check_index": 5, "num_interaction_steps": 400, "horizon_steps": 100,
        "arm_end_step": 200, "control_end_step": 400, "eval_every_steps": 50, "eval_episodes": 2}
EFFECT = {"dog-run": 30.0, "hopper-hop": -5.0}


def _identity(arch, env, seed, role="confirmatory"):
    return {"run_key": f"exp1_{arch}_{env}_seed{seed}", "run_role": role, "architecture": arch, "environment": env,
            "seed": seed, "budget_env_steps": 800, "num_interaction_steps": 400, "num_checks": 20,
            "code_commit": "test"}


def _write_run(root, arch, env, seed, role="confirmatory", check2_pass=True, complete=True, identity_arm=False):
    ident = _identity(arch, env, seed, role)
    records = [{"check_index": k, "interaction_step": 20 * k, "loss_iqm": 0.01 * k, "triggered": k in (4, 5)}
               for k in range(0, 21)]
    metrics = [{"env_step": 40 * i, "train/policy_kl": 0.1 * i, "train/actor_gnorm": 1.0,
                "train/actor_gnorm_std": 0.1, "train/actor_saturation": 0.2, "train/actor_action": 0.5,
                "eval/avg_return": 10.0 * i} for i in range(1, 11)]
    probe_dir = Path(tempfile.mkdtemp())
    ledger.write_run(ident, records, {"check_index": 5, "interaction_step": 100}, "complete", probe_dir, root,
                     fork_step=100, metrics_rows=metrics, action_repeat=2)
    key = ident["run_key"]
    exp2_ledger.write_json(key, "fork.json", PLAN, root)
    exp2_ledger.write_json(key, "check1_injected.json", {"pass": True, "max_eps_units": 1.5},
                           root)
    exp2_ledger.write_json(key, "check2.json", {"pass": check2_pass, "paired_difference_iqm": 0.1,
                                                "paired_difference_ci_low": 0.05 if check2_pass else -0.1,
                                                "paired_difference_ci_high": 0.2}, root)
    arms = ("control", "injected") + (("identity",) if identity_arm else ())
    for arm in arms:
        offset = EFFECT[env] + seed if arm == "injected" else 0.0
        last = 2 if (not complete and arm == "injected") else 3
        evals = [{"arm": arm, "eval_index": k, "steps_since_fork": 50 * k, "interaction_step": 100 + 50 * k,
                  "episode": ep, "return": 100.0 + 10 * k + offset + (1.0 if ep else -1.0), "length": 1000,
                  "success": 0.0} for k in range(last) for ep in range(2)]
        probe = [{"check_index": k, "interaction_step": 20 * k, "loss_iqm": 0.05 if arm == "injected" else 0.1}
                 for k in range(5, 11)]
        mrows = [{"env_step": 2 * (100 + 20 * i), "train/policy_kl": 0.3} for i in range(5)]
        exp2_ledger.write_arm(key, arm, PLAN, probe, evals, mrows, 2, root)
    return key


class BandTest(unittest.TestCase):
    def test_constant_seeds_give_a_degenerate_band(self):
        b = ea.bootstrap_band(np.full((5, 3), 2.0), "mean")
        for k in ("point", "low", "high"):
            np.testing.assert_array_equal(b[k], [2.0, 2.0, 2.0])

    def test_point_is_the_statistic_over_seeds_and_the_band_contains_it(self):
        m = np.random.default_rng(0).normal(size=(5, 4))
        for stat, fn in (("mean", lambda x: x.mean(0)), ("iqm", lambda x: trim_mean(x, 0.25, axis=0))):
            b = ea.bootstrap_band(m, stat)
            np.testing.assert_allclose(b["point"], fn(m))
            self.assertTrue(np.all(b["low"] <= b["point"]) and np.all(b["point"] <= b["high"]))

    def test_resampled_seed_sets_are_shared_across_points(self):
        m = np.random.default_rng(1).normal(size=(5, 1))
        b = ea.bootstrap_band(np.hstack([m, m + 1.0]), "mean")
        self.assertAlmostEqual(b["low"][1] - b["low"][0], 1.0)
        self.assertAlmostEqual(b["high"][1] - b["high"][0], 1.0)

    def test_reproducible_and_iqm_resists_an_outlier(self):
        m = np.array([[1.0], [1.1], [0.9], [1.0], [50.0]])
        self.assertEqual(ea.bootstrap_band(m, "iqm")["high"][0], ea.bootstrap_band(m, "iqm")["high"][0])
        self.assertLess(ea.bootstrap_band(m, "iqm")["point"][0], ea.bootstrap_band(m, "mean")["point"][0])

    def test_iqm_is_the_default_and_unknown_statistics_are_refused(self):
        import inspect

        self.assertEqual(inspect.signature(ea.run_analysis).parameters["statistic"].default, "iqm")
        with self.assertRaises(ValueError):
            ea.run_analysis(tempfile.mkdtemp(), "median")


class SyntheticResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.keys = {}
        for env in EFFECT:
            for seed in (1, 2, 3):
                cls.keys[(env, seed)] = _write_run(cls.root, "D4W1536", env, seed, check2_pass=seed != 3,
                                                   identity_arm=seed == 1)
        _write_run(cls.root, "D4W1536", "dog-run", 4, complete=False)  # injected arm unfinished
        _write_run(cls.root, "D4W1536", "dog-run", 101, role="dev")
        cls.out = os.path.join(cls.root, "analysis")
        cls.outputs = ea.run_analysis(cls.out, "mean", cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_paired_difference_is_injected_minus_control_mean_return(self):
        paired = self.outputs["paired"]
        for (env, seed), key in self.keys.items():
            rows = paired[paired.run_key == key]
            self.assertEqual(list(rows.eval_index), [0, 1, 2])
            np.testing.assert_allclose(rows.difference, EFFECT[env] + seed)
            np.testing.assert_allclose(rows.control, 100.0 + 10 * rows.eval_index)

    def test_band_per_environment_over_complete_confirmatory_forks(self):
        bands = self.outputs["bands"]
        for env, effect in EFFECT.items():
            b = bands[bands.environment == env]
            self.assertEqual(set(b.n_seeds), {3})  # seed 4 (unfinished) and the dev run are excluded
            np.testing.assert_allclose(b.point, effect + 2.0)  # mean over seeds 1, 2, 3
            self.assertTrue(np.all(b.ci_low >= effect + 1.0) and np.all(b.ci_high <= effect + 3.0))

    def test_secondary_is_the_check2_success_subset(self):
        bands = self.outputs["bands_check2_success_only"]
        self.assertEqual(set(bands.n_seeds), {2})
        np.testing.assert_allclose(bands[bands.environment == "dog-run"].point, EFFECT["dog-run"] + 1.5)

    def test_tables_and_files(self):
        forks = pd.read_csv(os.path.join(self.out, "forks.csv"))
        self.assertEqual(len(forks), 7)  # 6 complete + the unfinished one; dev excluded
        self.assertEqual(int(forks.complete.sum()), 6)
        self.assertNotIn(101, set(forks.seed))
        for name in ("check1_table.csv", "check2_table.csv", "paired_returns.csv", "paired_bands.csv",
                     "paired_returns_D4W1536.png", "paired_returns_D4W1536_check2_success_only.png",
                     "plasticity_post_fork_D4W1536.png", "diagnostics_policy_kl_D4W1536.png",
                     "shared_time_axis_D4W1536_dog-run.png", "plasticity_post_fork.csv"):
            self.assertTrue(os.path.exists(os.path.join(self.out, name)), name)
        post = pd.read_csv(os.path.join(self.out, "plasticity_post_fork.csv"))
        self.assertEqual(set(post.arm), {"control", "injected"})  # the identity arm is validation only

    def test_normalize_hook_is_applied_per_environment(self):
        halve_dog = lambda env, v: v / 2 if env == "dog-run" else v
        data = ea.load(self.root)
        paired = ea.paired_returns(data["evals"], data["forks"], halve_dog)
        dog = paired[paired.environment == "dog-run"]
        hop = paired[paired.environment == "hopper-hop"]
        np.testing.assert_allclose(dog.difference, (EFFECT["dog-run"] + dog.seed) / 2)
        np.testing.assert_allclose(hop.difference, EFFECT["hopper-hop"] + hop.seed)

    def test_dev_included_on_request(self):
        data = ea.load(self.root, include_dev=True)
        self.assertIn(101, set(data["forks"].seed))


if __name__ == "__main__":
    unittest.main()
