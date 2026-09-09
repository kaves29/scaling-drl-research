"""Covers experiments/angle_2b/statistics.py's (mean + 2*std) null-exceedance
rule and the is_marginal safeguard added 2026-09-08 (reporting a result close
to its threshold with reduced confidence, rather than the same implied
confidence as a clearly-beyond/clearly-below result)."""

import unittest

from experiments.angle_2b.statistics import MARGINAL_FRACTION_OF_MARGIN, compare_to_null

NULL_VALUES = [1.0, 1.0, 1.0, 1.0, 3.0]  # mean=1.4, std=0.8, threshold=1.4+2*0.8=3.0


class TestCompareToNull(unittest.TestCase):
    def test_threshold_is_mean_plus_two_std(self):
        result = compare_to_null("metric", observed_value=0.0, null_values=NULL_VALUES)
        self.assertAlmostEqual(result.null_mean, 1.4)
        self.assertAlmostEqual(result.null_std, 0.8, places=6)
        self.assertAlmostEqual(result.threshold, 3.0, places=6)

    def test_clearly_below_threshold_is_not_marginal(self):
        # margin = threshold - null_mean = 1.6; 20% of that = 0.32.
        # threshold - 0.32 = 2.68 is the marginal band's lower edge.
        result = compare_to_null("metric", observed_value=0.0, null_values=NULL_VALUES)
        self.assertFalse(result.exceeds_null)
        self.assertFalse(result.is_marginal)

    def test_clearly_above_threshold_is_not_marginal(self):
        result = compare_to_null("metric", observed_value=10.0, null_values=NULL_VALUES)
        self.assertTrue(result.exceeds_null)
        self.assertFalse(result.is_marginal)

    def test_just_below_threshold_is_marginal(self):
        result = compare_to_null("metric", observed_value=2.9, null_values=NULL_VALUES)
        self.assertFalse(result.exceeds_null)
        self.assertTrue(result.is_marginal)

    def test_just_above_threshold_is_marginal_not_just_exceeds(self):
        """The safeguard is symmetric: a result that JUST clears the
        threshold is still flagged as marginal, not reported with the same
        confidence as a result that clears it by several margins."""
        result = compare_to_null("metric", observed_value=3.1, null_values=NULL_VALUES)
        self.assertTrue(result.exceeds_null)
        self.assertTrue(result.is_marginal)

    def test_exactly_at_threshold_is_marginal(self):
        result = compare_to_null("metric", observed_value=3.0, null_values=NULL_VALUES)
        self.assertFalse(result.exceeds_null)  # strict >, not >=
        self.assertTrue(result.is_marginal)

    def test_marginal_band_edge_matches_configured_fraction(self):
        margin = 3.0 - 1.4
        edge = 3.0 - MARGINAL_FRACTION_OF_MARGIN * margin
        just_inside = compare_to_null("metric", observed_value=edge + 1e-9, null_values=NULL_VALUES)
        just_outside = compare_to_null("metric", observed_value=edge - 1e-9, null_values=NULL_VALUES)
        self.assertTrue(just_inside.is_marginal)
        self.assertFalse(just_outside.is_marginal)


if __name__ == "__main__":
    unittest.main()
