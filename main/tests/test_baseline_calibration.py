import os
import shutil
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

from analysis.baseline_calibration import (
    calibrate_baseline,
    load_baseline,
    load_or_calibrate_baseline,
    save_baseline,
)
from analysis.metrics_store import METRIC_COLUMNS, RunIdentity, metrics_path


class TestBaselineCalibration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.metrics_root = os.path.join(self.tmpdir, "metrics")
        self.baseline_root = os.path.join(self.tmpdir, "baselines")
        self.identities = [
            RunIdentity(experiment="angle_1", architecture="D2W512", environment="reacher-hard", seed=s)
            for s in range(1, 6)
        ]

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_metrics(self, identity, steps, td_var_values):
        path = metrics_path(identity, root=self.metrics_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "interaction_step": steps,
                "env_step": [s * 2 for s in steps],
                "td_error_variance": td_var_values,
                "actor_grad_cosine": [0.0] * len(steps),
            },
            columns=METRIC_COLUMNS,
        )
        df.to_csv(path, index=False)

    def test_calibrate_computes_percentile_at_each_aligned_step(self):
        steps = [1000, 2000, 3000]
        # seed i contributes value i at every step -> values across seeds at
        # each step are [1,2,3,4,5]
        for i, identity in enumerate(self.identities, start=1):
            self._write_metrics(identity, steps, [float(i)] * len(steps))

        thresholds = calibrate_baseline(self.identities, "td_error_variance", percentile=95, metrics_root=self.metrics_root)

        expected = np.quantile([1, 2, 3, 4, 5], 0.95)
        self.assertEqual(thresholds.num_seeds, 5)
        np.testing.assert_array_equal(thresholds.steps, np.array(steps))
        for t in thresholds.thresholds:
            self.assertAlmostEqual(t, expected)

    def test_calibrate_aligns_on_intersection_of_steps_only(self):
        # seed 5 stopped early / logged fewer points; alignment must only use
        # the common interaction_step values, not fabricate/interpolate extra ones.
        common_steps = [1000, 2000, 3000]
        for identity in self.identities[:4]:
            self._write_metrics(identity, common_steps + [4000], [1.0, 1.0, 1.0, 1.0])
        self._write_metrics(self.identities[4], common_steps, [1.0, 1.0, 1.0])

        thresholds = calibrate_baseline(self.identities, "td_error_variance", percentile=95, metrics_root=self.metrics_root)

        self.assertEqual(list(thresholds.steps), common_steps)

    def test_calibrate_requires_exactly_five_seeds_rejects_one(self):
        with self.assertRaises(ValueError):
            calibrate_baseline(self.identities[:1], "td_error_variance", metrics_root=self.metrics_root)

    def test_calibrate_requires_exactly_five_seeds_rejects_four(self):
        # four is not an error under the old ">= 2" rule but must be
        # rejected under the current "exactly 5" scientific requirement.
        for identity in self.identities[:4]:
            self._write_metrics(identity, [1000, 2000], [1.0, 1.0])
        with self.assertRaises(ValueError):
            calibrate_baseline(self.identities[:4], "td_error_variance", metrics_root=self.metrics_root)

    def test_calibrate_requires_exactly_five_seeds_rejects_six(self):
        sixth = RunIdentity(experiment="angle_1", architecture="D2W512", environment="reacher-hard", seed=6)
        with self.assertRaises(ValueError):
            calibrate_baseline(self.identities + [sixth], "td_error_variance", metrics_root=self.metrics_root)

    def test_calibrate_rejects_mismatched_logging_interval(self):
        # baseline seeds recorded every 1000 steps, but the analyzed run
        # (represented here by expected_logging_interval) uses 3000.
        steps = [1000, 2000, 3000, 4000]
        for identity in self.identities:
            self._write_metrics(identity, steps, [1.0] * len(steps))

        with self.assertRaises(ValueError) as ctx:
            calibrate_baseline(
                self.identities, "td_error_variance", metrics_root=self.metrics_root,
                expected_logging_interval=3000,
            )
        self.assertIn("logging interval", str(ctx.exception).lower())

    def test_calibrate_accepts_matching_logging_interval(self):
        steps = [1000, 2000, 3000, 4000]
        for identity in self.identities:
            self._write_metrics(identity, steps, [1.0] * len(steps))

        # must not raise
        thresholds = calibrate_baseline(
            self.identities, "td_error_variance", metrics_root=self.metrics_root,
            expected_logging_interval=1000,
        )
        self.assertEqual(len(thresholds.steps), 4)

    def test_calibrate_raises_clear_error_when_a_seed_has_no_metrics(self):
        for identity in self.identities[:4]:
            self._write_metrics(identity, [1000], [1.0])
        # 5th seed never trained with tracking enabled -> no metrics file

        with self.assertRaises(ValueError):
            calibrate_baseline(self.identities, "td_error_variance", metrics_root=self.metrics_root)

    def test_save_and_load_baseline_round_trip(self):
        steps = [1000, 2000]
        for i, identity in enumerate(self.identities, start=1):
            self._write_metrics(identity, steps, [float(i)] * len(steps))

        thresholds = calibrate_baseline(self.identities, "td_error_variance", percentile=95, metrics_root=self.metrics_root)
        save_baseline(thresholds, root=self.baseline_root)

        loaded = load_baseline("D2W512", "reacher-hard", "td_error_variance", 95, root=self.baseline_root)

        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded.steps, thresholds.steps)
        np.testing.assert_allclose(loaded.thresholds, thresholds.thresholds)
        self.assertEqual(loaded.num_seeds, 5)

    def test_load_baseline_returns_none_when_not_yet_calibrated(self):
        loaded = load_baseline("D2W512", "reacher-hard", "td_error_variance", 95, root=self.baseline_root)
        self.assertIsNone(loaded)

    def test_load_or_calibrate_uses_cache_when_source_unchanged(self):
        steps = [1000, 2000]
        for i, identity in enumerate(self.identities, start=1):
            self._write_metrics(identity, steps, [float(i)] * len(steps))

        first = load_or_calibrate_baseline(
            self.identities, "td_error_variance", metrics_root=self.metrics_root, baseline_root=self.baseline_root,
        )

        call_count = {"n": 0}
        import analysis.baseline_calibration as bc_module
        real_calibrate = bc_module.calibrate_baseline

        def counting_calibrate(*args, **kwargs):
            call_count["n"] += 1
            return real_calibrate(*args, **kwargs)

        bc_module.calibrate_baseline = counting_calibrate
        try:
            second = load_or_calibrate_baseline(
                self.identities, "td_error_variance", metrics_root=self.metrics_root, baseline_root=self.baseline_root,
            )
        finally:
            bc_module.calibrate_baseline = real_calibrate

        self.assertEqual(call_count["n"], 0, "cache hit must not recalibrate")
        np.testing.assert_allclose(second.thresholds, first.thresholds)

    def test_load_or_calibrate_recomputes_when_source_metrics_change(self):
        steps = [1000, 2000]
        for i, identity in enumerate(self.identities, start=1):
            self._write_metrics(identity, steps, [float(i)] * len(steps))

        first = load_or_calibrate_baseline(
            self.identities, "td_error_variance", metrics_root=self.metrics_root, baseline_root=self.baseline_root,
        )

        # mutate one baseline seed's underlying metrics (simulating a
        # corrected/re-run baseline seed) - must be picked up automatically,
        # without force_recompute, since the source fingerprint changes.
        time.sleep(0.01)
        self._write_metrics(self.identities[0], steps, [999.0] * len(steps))

        second = load_or_calibrate_baseline(
            self.identities, "td_error_variance", metrics_root=self.metrics_root, baseline_root=self.baseline_root,
        )

        self.assertFalse(
            np.allclose(second.thresholds, first.thresholds),
            "stale cache was reused despite changed baseline source data",
        )

    def test_force_recompute_bypasses_cache_even_if_fresh(self):
        steps = [1000, 2000]
        for i, identity in enumerate(self.identities, start=1):
            self._write_metrics(identity, steps, [float(i)] * len(steps))

        load_or_calibrate_baseline(
            self.identities, "td_error_variance", metrics_root=self.metrics_root, baseline_root=self.baseline_root,
        )

        import analysis.baseline_calibration as bc_module
        real_calibrate = bc_module.calibrate_baseline
        call_count = {"n": 0}

        def counting_calibrate(*args, **kwargs):
            call_count["n"] += 1
            return real_calibrate(*args, **kwargs)

        bc_module.calibrate_baseline = counting_calibrate
        try:
            load_or_calibrate_baseline(
                self.identities, "td_error_variance", metrics_root=self.metrics_root,
                baseline_root=self.baseline_root, force_recompute=True,
            )
        finally:
            bc_module.calibrate_baseline = real_calibrate

        self.assertEqual(call_count["n"], 1, "force_recompute=True must recalibrate even with a fresh cache")


if __name__ == "__main__":
    unittest.main()
