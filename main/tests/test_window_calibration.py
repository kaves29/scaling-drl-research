import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from analysis.metrics_store import METRIC_COLUMNS, RunIdentity, metrics_path
from analysis.window_calibration import (
    calibrate_window_parameters,
    load_or_calibrate_window_parameters,
    load_window_calibration,
)

LOGGING_INTERVAL = 1  # so recorded-point index == interaction_step, for simple assertions


class TestWindowCalibration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.metrics_root = os.path.join(self.tmpdir, "metrics")
        self.window_root = os.path.join(self.tmpdir, "baselines")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _identities(self, environment: str, n_seeds: int = 5):
        return [
            RunIdentity(experiment="angle_1", architecture="D2W512", environment=environment, seed=s)
            for s in range(1, n_seeds + 1)
        ]

    def _write(self, identity, td_var, actor_grad_cosine):
        path = metrics_path(identity, root=self.metrics_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        steps = list(range(1, len(td_var) + 1))
        df = pd.DataFrame(
            {
                "interaction_step": steps,
                "env_step": [s * 2 for s in steps],
                "td_error_variance": td_var,
                "actor_grad_cosine": actor_grad_cosine,
            },
            columns=METRIC_COLUMNS,
        )
        df.to_csv(path, index=False)

    def _ar1(self, phi: float, n: int, seed: int) -> np.ndarray:
        """AR(1): x[t] = phi*x[t-1] + noise[t]; ACF decays as phi^k."""
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1.0, size=n)
        x = np.empty(n)
        x[0] = noise[0]
        for t in range(1, n):
            x[t] = phi * x[t - 1] + noise[t]
        return x

    def test_n_is_genuinely_different_between_a_fast_and_slow_decorrelating_environment(self):
        # fast-env: phi=0.3 -> theoretical decorrelation lag ~= -1/ln(0.3) ~= 0.83 (~1)
        # slow-env: phi=0.95 -> theoretical decorrelation lag ~= -1/ln(0.95) ~= 19.5
        n_points = 400
        fast_ids = self._identities("fast-env")
        slow_ids = self._identities("slow-env")
        for i, ident in enumerate(fast_ids):
            td = self._ar1(0.3, n_points, seed=100 + i)
            self._write(ident, td, np.random.default_rng(i).normal(0, 0.01, size=n_points))
        for i, ident in enumerate(slow_ids):
            td = self._ar1(0.95, n_points, seed=200 + i)
            self._write(ident, td, np.random.default_rng(i).normal(0, 0.01, size=n_points))

        fast_result = calibrate_window_parameters(fast_ids, logging_per_interaction_step=LOGGING_INTERVAL, metrics_root=self.metrics_root)
        slow_result = calibrate_window_parameters(slow_ids, logging_per_interaction_step=LOGGING_INTERVAL, metrics_root=self.metrics_root)

        self.assertGreater(slow_result.n_sustain_window_points, fast_result.n_sustain_window_points * 3)
        self.assertGreater(fast_result.n_sustain_window_points, 0)
        self.assertLess(fast_result.n_sustain_window_points, 5)
        self.assertTrue(10 <= slow_result.n_sustain_window_points <= 35)

    def test_w_recovers_an_injected_lag_and_differs_between_environments(self):
        n_points = 300
        true_lag_short = 3
        true_lag_long = 10
        short_ids = self._identities("short-lag-env")
        long_ids = self._identities("long-lag-env")

        for i, ident in enumerate(short_ids):
            rng = np.random.default_rng(300 + i)
            td = self._ar1(0.5, n_points, seed=300 + i)
            actor = np.empty(n_points)
            actor[:true_lag_short] = rng.normal(0, 0.1, size=true_lag_short)
            actor[true_lag_short:] = td[: n_points - true_lag_short] + rng.normal(0, 0.05, size=n_points - true_lag_short)
            self._write(ident, td, actor)

        for i, ident in enumerate(long_ids):
            rng = np.random.default_rng(400 + i)
            td = self._ar1(0.5, n_points, seed=400 + i)
            actor = np.empty(n_points)
            actor[:true_lag_long] = rng.normal(0, 0.1, size=true_lag_long)
            actor[true_lag_long:] = td[: n_points - true_lag_long] + rng.normal(0, 0.05, size=n_points - true_lag_long)
            self._write(ident, td, actor)

        short_result = calibrate_window_parameters(short_ids, logging_per_interaction_step=LOGGING_INTERVAL, metrics_root=self.metrics_root)
        long_result = calibrate_window_parameters(long_ids, logging_per_interaction_step=LOGGING_INTERVAL, metrics_root=self.metrics_root)

        # LOGGING_INTERVAL=1, so recorded-points units == raw-step units here.
        self.assertNotEqual(short_result.w_propagation_window_steps, long_result.w_propagation_window_steps)
        self.assertAlmostEqual(short_result.w_propagation_window_steps, true_lag_short, delta=2)
        self.assertAlmostEqual(long_result.w_propagation_window_steps, true_lag_long, delta=2)
        self.assertEqual(short_result.excluded_seeds_for_w, [])
        self.assertEqual(long_result.excluded_seeds_for_w, [])

    def test_negative_lag_seeds_are_excluded_not_folded_in(self):
        # actor leads critic here, so corr(td[t], actor[t+k]) peaks at
        # k = -true_lag; must be excluded, not folded in via abs().
        n_points = 200
        ids = self._identities("negative-lag-env")
        true_lag = 4
        for i, ident in enumerate(ids):
            rng = np.random.default_rng(500 + i)
            actor = self._ar1(0.5, n_points, seed=500 + i)
            td = np.empty(n_points)
            td[:true_lag] = rng.normal(0, 0.1, size=true_lag)  # filler: no valid actor reference yet
            td[true_lag:] = actor[: n_points - true_lag] + rng.normal(0, 0.05, size=n_points - true_lag)
            self._write(ident, td, actor)

        with self.assertRaises(ValueError) as ctx:
            calibrate_window_parameters(ids, logging_per_interaction_step=LOGGING_INTERVAL, metrics_root=self.metrics_root)
        self.assertIn("negative-lag", str(ctx.exception).lower())

    def test_acf_non_decay_is_capped_and_logged_not_raised(self):
        # _decorrelation_time directly, not calibrate_window_parameters: a
        # zero-variance segment (needed to force non-decay) also makes W
        # uncalibratable, which would raise first. A constant signal triggers
        # non-decay reliably; white noise and near-unit-root AR(1) don't.
        from analysis.window_calibration import _decorrelation_time

        n_points = 60
        burn_in_start = int(np.floor(n_points * 0.25))
        segment = np.full(n_points - burn_in_start, 5.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dc_time, was_capped = _decorrelation_time(segment, "test-seed")

        acf_warnings = [w for w in caught if "never decayed below 1/e" in str(w.message)]
        self.assertGreater(len(acf_warnings), 0, "expected a non-decay warning for a constant signal")
        self.assertTrue(was_capped)
        # segment length = 60 - burn_in(25%)=15 -> 45 points; capped at
        # max_lag = len(segment)-1 = 44.
        self.assertEqual(dc_time, 44.0)

    def test_accepts_the_expanded_10_seed_pool_and_calibrates_correctly(self):
        """analysis/baseline_calibration_pool.py's expanded shared pool
        (2026-09-08): 10 baseline seeds must be accepted, not just the
        original 5 - and must still produce a real, correct calibration
        (not merely pass the entry-count check)."""
        n_points = 400
        ids = self._identities("ten-seed-env", n_seeds=10)
        for i, ident in enumerate(ids):
            td = self._ar1(0.5, n_points, seed=500 + i)
            self._write(ident, td, np.random.default_rng(i).normal(0, 0.01, size=n_points))

        result = calibrate_window_parameters(ids, logging_per_interaction_step=LOGGING_INTERVAL, metrics_root=self.metrics_root)

        self.assertEqual(len(result.per_seed_decorrelation_points), 10)
        self.assertGreater(result.n_sustain_window_points, 0)

    def test_rejects_a_seed_count_that_is_neither_5_nor_10(self):
        ids = self._identities("bad-count-env", n_seeds=7)
        for i, ident in enumerate(ids):
            self._write(ident, self._ar1(0.5, 100, seed=1), np.zeros(100))
        with self.assertRaises(ValueError) as ctx:
            calibrate_window_parameters(ids, logging_per_interaction_step=LOGGING_INTERVAL, metrics_root=self.metrics_root)
        self.assertIn("5", str(ctx.exception))
        self.assertIn("10", str(ctx.exception))

    def test_calibrate_rejects_mixed_environments(self):
        mixed = self._identities("env-a")[:4] + [
            RunIdentity(experiment="angle_1", architecture="D2W512", environment="env-b", seed=5)
        ]
        for ident in mixed:
            self._write(ident, self._ar1(0.5, 100, seed=1), np.zeros(100))
        with self.assertRaises(ValueError) as ctx:
            calibrate_window_parameters(mixed, logging_per_interaction_step=LOGGING_INTERVAL, metrics_root=self.metrics_root)
        self.assertIn("environment", str(ctx.exception).lower())

    def test_load_or_calibrate_caches_and_detects_staleness(self):
        ids = self._identities("cache-env")
        for i, ident in enumerate(ids):
            self._write(
                ident, self._ar1(0.5, 200, seed=700 + i),
                np.random.default_rng(800 + i).normal(0, 0.01, size=200),
            )

        first = load_or_calibrate_window_parameters(
            ids, logging_per_interaction_step=LOGGING_INTERVAL, metrics_root=self.metrics_root, window_root=self.window_root,
        )

        cached = load_window_calibration("D2W512", "cache-env", root=self.window_root)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.n_sustain_window_points, first.n_sustain_window_points)

        import analysis.window_calibration as wc_module
        real = wc_module.calibrate_window_parameters
        calls = {"n": 0}

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        wc_module.calibrate_window_parameters = counting
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                second = load_or_calibrate_window_parameters(
                    ids, logging_per_interaction_step=LOGGING_INTERVAL, metrics_root=self.metrics_root, window_root=self.window_root,
                )
        finally:
            wc_module.calibrate_window_parameters = real

        self.assertEqual(calls["n"], 0, "cache hit must not recalibrate")
        self.assertEqual(second.n_sustain_window_points, first.n_sustain_window_points)


if __name__ == "__main__":
    unittest.main()
