"""Covers experiments/angle_2b/statistics.py's (mean + 2*std) null-exceedance
rule.

Note: an is_marginal safeguard (flagging a result close to, not just beyond,
the threshold) was added 2026-09-08 and removed 2026-09-09 once the shared
10-agent pool refactor gave every null distribution 45 points instead of
<=5 - the thin-null justification that motivated it no longer applied. See
research-methodology.md's Pending/Open Items for the full history.
"""

import unittest

from experiments.angle_2b.statistics import compare_to_null

NULL_VALUES = [1.0, 1.0, 1.0, 1.0, 3.0]  # mean=1.4, std=0.8, threshold=1.4+2*0.8=3.0


class TestCompareToNull(unittest.TestCase):
    def test_threshold_is_mean_plus_two_std(self):
        result = compare_to_null("metric", observed_value=0.0, null_values=NULL_VALUES)
        self.assertAlmostEqual(result.null_mean, 1.4)
        self.assertAlmostEqual(result.null_std, 0.8, places=6)
        self.assertAlmostEqual(result.threshold, 3.0, places=6)

    def test_value_below_threshold_does_not_exceed_null(self):
        result = compare_to_null("metric", observed_value=0.0, null_values=NULL_VALUES)
        self.assertFalse(result.exceeds_null)

    def test_value_above_threshold_exceeds_null(self):
        result = compare_to_null("metric", observed_value=10.0, null_values=NULL_VALUES)
        self.assertTrue(result.exceeds_null)

    def test_value_exactly_at_threshold_does_not_exceed_null(self):
        result = compare_to_null("metric", observed_value=3.0, null_values=NULL_VALUES)
        self.assertFalse(result.exceeds_null)  # strict >, not >=


if __name__ == "__main__":
    unittest.main()
