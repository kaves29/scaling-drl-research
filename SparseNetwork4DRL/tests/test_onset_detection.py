import unittest

import numpy as np

from analysis.baseline_calibration import BaselineThresholds
from analysis.onset_detection import (
    STATUS_NEEDS_REVIEW,
    STATUS_NO_ONSET,
    STATUS_SUCCESS,
    detect_critic_degradation_onset,
    detect_propagation_onset,
)


def const_baseline(steps, threshold_value, metric="td_error_variance"):
    steps = np.array(steps)
    thresholds = np.full_like(steps, fill_value=threshold_value, dtype=float)
    return BaselineThresholds(
        architecture="D2W512",
        environment="reacher-hard",
        metric=metric,
        percentile=95,
        num_seeds=5,
        steps=steps,
        thresholds=thresholds,
    )


# Steps logged every 1000 interaction steps, as this repo's
# `logging_per_interaction_step` cadence would produce.
STEPS_10 = [1000 * i for i in range(1, 11)]  # 1000..10000


class TestCriticDegradationOnset(unittest.TestCase):
    def test_no_degradation_never_exceeds_baseline(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0)
        values = [0.5] * len(STEPS_10)

        result = detect_critic_degradation_onset(STEPS_10, values, baseline, sustain_window_points=3)

        self.assertIsNone(result.onset_step)
        self.assertEqual(result.status, STATUS_NO_ONSET)

    def test_degradation_that_does_not_persist_long_enough(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0)
        # exceeds for exactly 3 consecutive points; sustain_window_points=3 requires
        # a run STRICTLY GREATER than 3, so this must not trigger.
        values = [0.5, 0.5, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5]

        result = detect_critic_degradation_onset(STEPS_10, values, baseline, sustain_window_points=3)

        self.assertIsNone(result.onset_step)
        self.assertEqual(result.status, STATUS_NO_ONSET)

    def test_degradation_sustained_for_more_than_n_steps_is_detected(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0)
        # exceeds for 4 consecutive points (> N=3), starting at index 2 (step 3000)
        values = [0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5]

        result = detect_critic_degradation_onset(STEPS_10, values, baseline, sustain_window_points=3)

        self.assertEqual(result.status, STATUS_SUCCESS)
        self.assertEqual(result.onset_step, 3000)

    def test_onset_is_first_step_of_run_not_last(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0)
        values = [2.0, 2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5]

        result = detect_critic_degradation_onset(STEPS_10, values, baseline, sustain_window_points=2)

        self.assertEqual(result.onset_step, STEPS_10[0])  # first point, not the last qualifying one

    def test_earliest_qualifying_run_wins_over_a_later_longer_run(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0)
        # first run: indices 1-3 (len 3, qualifies for N=2); second run: indices 6-9 (len 4)
        values = [0.5, 2.0, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0]

        result = detect_critic_degradation_onset(STEPS_10, values, baseline, sustain_window_points=2)

        self.assertEqual(result.onset_step, STEPS_10[1], "must report the first qualifying run, not a later one")

    def test_no_recorded_points_needs_manual_review(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0)
        result = detect_critic_degradation_onset([], [], baseline, sustain_window_points=3)
        self.assertEqual(result.status, STATUS_NEEDS_REVIEW)

    def test_no_overlap_with_baseline_coverage_needs_manual_review(self):
        # baseline calibrated on a completely disjoint step grid
        baseline = const_baseline([50, 51, 52], threshold_value=1.0)
        result = detect_critic_degradation_onset(STEPS_10, [2.0] * len(STEPS_10), baseline, sustain_window_points=1)
        self.assertEqual(result.status, STATUS_NEEDS_REVIEW)

    def test_alignment_with_interval_logged_steps(self):
        # baseline only has coverage for every other step (simulating a
        # baseline run that logged at a coarser cadence / stopped early);
        # detection must still work over the intersection.
        sparse_steps = STEPS_10[::2]
        baseline = const_baseline(sparse_steps, threshold_value=1.0)
        values = [2.0] * len(STEPS_10)

        result = detect_critic_degradation_onset(STEPS_10, values, baseline, sustain_window_points=2)

        # only 5 of 10 points are covered by baseline (indices 0,2,4,6,8 ->
        # steps 1000,3000,5000,7000,9000), all exceed, run length 5 > N=2
        self.assertEqual(result.status, STATUS_SUCCESS)
        self.assertEqual(result.onset_step, 1000)


class TestPropagationOnset(unittest.TestCase):
    def test_no_critic_degradation_means_no_onset_detected(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0, metric="actor_grad_cosine")
        values = [2.0] * len(STEPS_10)

        result = detect_propagation_onset(STEPS_10, values, baseline, degradation_onset_step=None, propagation_window=5000)

        self.assertIsNone(result.onset_step)
        self.assertEqual(result.status, STATUS_NO_ONSET)
        self.assertIn("critic degradation", result.notes.lower())

    def test_degradation_with_no_propagation_in_window(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0, metric="actor_grad_cosine")
        values = [0.5] * len(STEPS_10)  # never exceeds

        result = detect_propagation_onset(STEPS_10, values, baseline, degradation_onset_step=3000, propagation_window=5000)

        self.assertIsNone(result.onset_step)
        self.assertEqual(result.status, STATUS_NO_ONSET)

    def test_propagation_detected_inside_window(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0, metric="actor_grad_cosine")
        # degradation onset at 3000, window [3000, 8000]; exceedance first at step 6000
        values_by_step = {s: 0.5 for s in STEPS_10}
        values_by_step[6000] = 2.0
        values_by_step[7000] = 2.0
        values = [values_by_step[s] for s in STEPS_10]

        result = detect_propagation_onset(STEPS_10, values, baseline, degradation_onset_step=3000, propagation_window=5000)

        self.assertEqual(result.status, STATUS_SUCCESS)
        self.assertEqual(result.onset_step, 6000)

    def test_exceedance_outside_window_is_not_detected(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0, metric="actor_grad_cosine")
        # degradation onset at 3000, window [3000, 4000] (W=1000); exceedance
        # only happens at step 6000, well outside the window.
        values_by_step = {s: 0.5 for s in STEPS_10}
        values_by_step[6000] = 2.0
        values = [values_by_step[s] for s in STEPS_10]

        result = detect_propagation_onset(STEPS_10, values, baseline, degradation_onset_step=3000, propagation_window=1000)

        self.assertIsNone(result.onset_step)
        self.assertEqual(result.status, STATUS_NO_ONSET)

    def test_window_bounds_are_inclusive(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0, metric="actor_grad_cosine")
        # degradation onset at 3000, W=3000 -> window is [3000, 6000] inclusive.
        # only qualifying point sits exactly at the right edge, step 6000.
        values_by_step = {s: 0.5 for s in STEPS_10}
        values_by_step[6000] = 2.0
        values = [values_by_step[s] for s in STEPS_10]

        result = detect_propagation_onset(STEPS_10, values, baseline, degradation_onset_step=3000, propagation_window=3000)

        self.assertEqual(result.status, STATUS_SUCCESS)
        self.assertEqual(result.onset_step, 6000)

    def test_propagation_lag_calculation(self):
        baseline = const_baseline(STEPS_10, threshold_value=1.0, metric="actor_grad_cosine")
        values_by_step = {s: 0.5 for s in STEPS_10}
        values_by_step[6000] = 2.0
        values = [values_by_step[s] for s in STEPS_10]
        degradation_onset_step = 3000

        result = detect_propagation_onset(STEPS_10, values, baseline, degradation_onset_step, propagation_window=5000)

        propagation_lag = result.onset_step - degradation_onset_step
        self.assertEqual(propagation_lag, 3000)


if __name__ == "__main__":
    unittest.main()
