"""Phase 3: trigger, f*_run, structured ledger and the Experiment 1 analysis."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp12_helpers import CONFIG_PATH, patch_wandb, tiny_overrides  # noqa: E402
from experiments.exp12 import ledger  # noqa: E402
from experiments.exp12.trigger import bootstrap_interval, f_star, last_eligible_check, triggered  # noqa: E402


def false_trigger_rate(n_checks=4000, rounds=5, seed=0):
    """Share of null checks (symmetric per-round noise around 0) whose 95% lower bound is > 0."""
    rng = np.random.default_rng(seed)
    hits = [triggered(bootstrap_interval(rng.normal(size=rounds), seed=1, check_index=i, reps=2_000)[0])
            for i in range(n_checks)]
    return float(np.mean(hits))


class TriggerTest(unittest.TestCase):
    def test_current_worse_triggers_and_reverse_never_does(self):
        loss = np.array([0.30, 0.25, 0.28, 0.35, 0.27])  # L = P(fresh) - P(current) > 0: plasticity lost
        low, high = bootstrap_interval(loss, seed=1, check_index=3)
        self.assertGreater(low, 0)
        self.assertTrue(triggered(low))
        low, _ = bootstrap_interval(-loss, seed=1, check_index=3)
        self.assertFalse(triggered(low))

    def test_known_loss_is_detected(self):
        rng = np.random.default_rng(0)
        hits = [triggered(bootstrap_interval(rng.normal(1.0, 0.2, 5), 1, i)[0]) for i in range(200)]
        self.assertEqual(np.mean(hits), 1.0)

    def test_null_false_trigger_rate_is_measured(self):
        rate = false_trigger_rate()
        per_run = 1 - (1 - rate) ** last_eligible_check(20)
        print(f"\n[trigger] synthetic null, 5 rounds: per-check false-trigger rate {rate:.4f}; "
              f"implied chance a null run triggers at least once in 19 independent checks {per_run:.3f}")
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 0.25)

    def test_edge_cases(self):
        self.assertFalse(triggered(bootstrap_interval(np.zeros(5), 1, 1)[0]))  # all zero: lower bound == 0
        low, high = bootstrap_interval(np.full(5, 0.5), 1, 1)  # ties: degenerate interval at the value
        self.assertEqual((low, high), (0.5, 0.5))
        self.assertTrue(triggered(low))
        low, high = bootstrap_interval(np.array([0.3, np.nan, 0.3, 0.3, 0.3]), 1, 1)
        self.assertTrue(np.isnan(low) and np.isnan(high))
        self.assertFalse(triggered(low))

    def test_reproducible_from_seed_and_check(self):
        x = np.array([0.1, -0.05, 0.2, 0.15, 0.02])
        self.assertEqual(bootstrap_interval(x, 7, 4), bootstrap_interval(x, 7, 4))
        # Distinct (seed, check) streams; with 5 rounds the IQM's bootstrap quantiles are
        # discrete, so two streams can legitimately give the same interval.
        streams = [np.random.default_rng([7, 0x424F4F54, k]).integers(0, 5, 20) for k in (4, 5)]
        self.assertFalse(np.array_equal(*streams))

    def test_statistic_is_iqm_not_mean(self):
        # An outlier pulls the mean's interval below zero, but not the IQM's.
        x = np.array([0.2, 0.21, 0.22, 0.23, -5.0])
        low, _ = bootstrap_interval(x, 1, 1)
        rng = np.random.default_rng([1, 0x424F4F54, 1])
        means = x[rng.integers(0, 5, size=(10_000, 5))].mean(1)
        self.assertLess(np.percentile(means, 2.5), 0)
        self.assertGreater(np.percentile(means, 97.5), low)
        self.assertNotAlmostEqual(low, np.percentile(means, 2.5))

    def test_f_star_first_eligible_check_only(self):
        rec = lambda k, t, valid=True: {"check_index": k, "interaction_step": 100 * k, "triggered": t, "valid": valid}
        self.assertIsNone(f_star([rec(k, k == 20) for k in range(1, 21)], 20))  # 20/20 is past 95%
        self.assertEqual(f_star([rec(k, k in (7, 9)) for k in range(1, 21)], 20)["check_index"], 7)
        self.assertEqual(f_star([rec(k, k == 19) for k in range(1, 21)], 20)["interaction_step"], 1900)
        self.assertIsNone(f_star([rec(0, True)] + [rec(k, False) for k in range(1, 21)], 20))  # fresh check 0
        self.assertEqual(last_eligible_check(20), 19)


def _identity(run_key, arch, env, seed, role="confirmatory"):
    return {"run_key": run_key, "run_role": role, "architecture": arch, "environment": env, "seed": seed,
            "budget_env_steps": 1000, "num_interaction_steps": 500, "num_checks": 20, "code_commit": "abc"}


def _records(final_loss, rng, noise=0.02):
    recs = [{"check_index": 0, "interaction_step": 10, "score_fresh_iqm": 0.3,
             **{f"score_fresh_r{r}": 0.3 for r in range(5)}}]
    for k in range(1, 21):
        loss = final_loss * k / 20 + rng.normal(0, noise, 5)
        recs.append({"check_index": k, "interaction_step": 25 * k,
                     **{f"score_current_r{r}": 0.3 - loss[r] for r in range(5)},
                     **{f"score_fresh_r{r}": 0.3 for r in range(5)}, **{f"loss_r{r}": loss[r] for r in range(5)},
                     "score_current_iqm": 0.3 - float(np.median(loss)), "score_fresh_iqm": 0.3,
                     "loss_iqm": float(np.mean(np.sort(loss)[1:4])), "ci_low": 0.0, "ci_high": 0.0,
                     "triggered": False, "valid": True})
    return recs


def write_synthetic_grid(root, effects, envs=("e1", "e2", "e3"), seeds=(1, 2, 3, 4, 5), seed=0, dev_runs=()):
    rng = np.random.default_rng(seed)
    probe_dir = Path(root) / "no_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    for arch, effect in effects.items():
        for env in envs:
            for s in seeds:
                key = f"exp1_{arch}_{env}_seed{s}"
                ledger.write_run(_identity(key, arch, env, s), _records(0.1 + effect + rng.normal(0, 0.02), rng),
                                 None, "complete", probe_dir, results_root=root)
    for arch, env, s in dev_runs:
        key = f"exp1_{arch}_{env}_seed{s}"
        ledger.write_run(_identity(key, arch, env, s, role="dev"), _records(5.0, rng), None, "complete",
                         probe_dir, results_root=root)


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_round_trip_and_dev_excluded_by_default(self):
        write_synthetic_grid(self.root, {"D2W512": 0.0}, dev_runs=[("D6W1536", "e1", 1001)])
        runs, checks = ledger.load(self.root)
        self.assertEqual(len(runs), 15)
        self.assertEqual(set(runs.run_role), {"confirmatory"})
        self.assertEqual(len(checks), 15 * 21)
        runs_all, _ = ledger.load(self.root, include_dev=True)
        self.assertEqual(len(runs_all), 16)

    def test_incomplete_and_foreign_rows_are_rejected(self):
        rng = np.random.default_rng(0)
        probe_dir = Path(self.root) / "p"
        probe_dir.mkdir()
        ledger.write_run(_identity("exp1_D2W512_e1_seed1", "D2W512", "e1", 1), _records(0.1, rng), None, "running",
                         probe_dir, results_root=self.root)
        self.assertEqual(len(ledger.load(self.root)[0]), 0)
        self.assertEqual(len(ledger.load(self.root, require_complete=False)[0]), 1)
        run_csv = ledger.ledger_root(self.root) / "exp1_D2W512_e1_seed1" / "run.csv"
        df = pd.read_csv(run_csv)
        df["experiment"] = "angle_1"
        df.to_csv(run_csv, index=False)
        with self.assertRaises(ledger.LedgerSchemaError):
            ledger.load(self.root, require_complete=False)
        df.drop(columns=["f_star_check"]).to_csv(run_csv, index=False)
        with self.assertRaises(ledger.LedgerSchemaError):
            ledger.load(self.root, require_complete=False)


class Exp1AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_known_effect_is_recovered_and_null_effect_is_not(self):
        from analysis.exp1_analysis import primary_endpoint

        write_synthetic_grid(self.root, {"D2W512": 0.0, "D4W1024": 0.5, "D6W1536": 0.0})
        runs, _ = ledger.load(self.root)
        res = primary_endpoint(runs, reps=5_000).set_index("comparison")
        big, null = res.loc["D4W1024 - D2W512"], res.loc["D6W1536 - D2W512"]
        self.assertGreater(big.ci_low, 0)
        self.assertLess(big.ci_low, 0.5)
        self.assertGreater(big.ci_high, 0.5)
        self.assertLess(null.ci_low, 0)
        self.assertGreater(null.ci_high, 0)
        self.assertEqual(int(big.runs_scaled), 15)

    def test_missing_run_is_refused(self):
        from analysis.exp1_analysis import primary_endpoint

        write_synthetic_grid(self.root, {"D2W512": 0.0, "D4W1024": 0.5, "D6W1536": 0.0})
        shutil.rmtree(ledger.ledger_root(self.root) / "exp1_D6W1536_e2_seed3")
        runs, _ = ledger.load(self.root)
        with self.assertRaisesRegex(ValueError, "D6W1536"):
            primary_endpoint(runs, reps=1_000)

    def test_analysis_is_reproducible(self):
        from analysis.exp1_analysis import primary_endpoint

        write_synthetic_grid(self.root, {"D2W512": 0.0, "D4W1024": 0.5, "D6W1536": 0.0})
        runs, _ = ledger.load(self.root)
        pd.testing.assert_frame_equal(primary_endpoint(runs, reps=2_000), primary_endpoint(runs, reps=2_000))

    def test_dev_runs_do_not_change_the_confirmatory_result(self):
        from analysis.exp1_analysis import primary_endpoint

        write_synthetic_grid(self.root, {"D2W512": 0.0, "D4W1024": 0.5, "D6W1536": 0.0},
                             dev_runs=[("D6W1536", "e1", 1001)])
        runs, _ = ledger.load(self.root)
        self.assertNotIn(1001, set(runs.seed))
        a = primary_endpoint(runs, reps=2_000)
        shutil.rmtree(ledger.ledger_root(self.root) / "exp1_D6W1536_e1_seed1001")
        b = primary_endpoint(ledger.load(self.root)[0], reps=2_000)
        pd.testing.assert_frame_equal(a, b)

    def test_full_analysis_writes_outputs(self):
        from analysis.exp1_analysis import run_analysis

        write_synthetic_grid(self.root, {"D2W512": 0.0, "D4W1024": 0.5, "D6W1536": 0.2})
        out = os.path.join(self.root, "analysis")
        run_analysis(out, self.root)
        for name in ("primary_endpoint.csv", "final_loss_by_architecture.csv", "trajectories.csv",
                     "f_star_table.csv", "every_seed_checks.csv", "trajectories.png", "every_seed.png"):
            self.assertTrue((Path(out) / name).exists(), name)
        traj = pd.read_csv(Path(out) / "trajectories.csv")
        self.assertEqual(len(traj), 3 * 20)
        self.assertAlmostEqual(traj.budget_fraction.max(), 1.0)


class Exp1LedgerEndToEndTest(unittest.TestCase):
    """A real (tiny) exp1 run writes a complete ledger whose rows match its probe records."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patchers = patch_wandb()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_writes_ledger(self):
        from experiments import exp1

        results = os.path.join(self.tmp, "results")
        exp1.run({"experiment": "exp1", "config_path": CONFIG_PATH, "config_name": "base_exp12",
                  "overrides": tiny_overrides(steps=400, extra=[f"results_root={results}"]),
                  "checkpoint_dir": os.path.join(self.tmp, "run"), "checkpoint_interval": 10**9,
                  "checkpoint_start_frac": 0.0})
        runs, checks = ledger.load(results)
        self.assertEqual(len(runs), 1)
        run = runs.iloc[0]
        self.assertEqual(run.status, "complete")
        self.assertEqual(list(checks.check_index), list(range(21)))
        eligible = checks[(checks.check_index >= 1) & (checks.check_index <= 19) & checks.triggered.astype(bool)]
        if eligible.empty:
            self.assertTrue(pd.isna(run.f_star_check))
        else:
            self.assertEqual(run.f_star_check, eligible.check_index.min())
        for _, row in checks[checks.check_index > 0].iterrows():
            low, high = bootstrap_interval([row[f"loss_r{r}"] for r in range(5)], 1, int(row.check_index))
            self.assertAlmostEqual(row.ci_low, low)
            self.assertEqual(bool(row.triggered), triggered(low))
        curves = np.load(ledger.ledger_root(results) / run.run_key / "probe_curves.npz")
        self.assertEqual(curves["check_20/current_losses"].shape, (5, 20))


if __name__ == "__main__":
    unittest.main()
