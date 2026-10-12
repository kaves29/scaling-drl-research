"""Reference cases for isolated, CPU-only D1 decision replay."""

import unittest

from scripts.sci_investigation.compare_probe_decisions import (
    exact_binomial_interval, null_decisions, percentile95, replay_checks,
)


class CompareProbeDecisionsTest(unittest.TestCase):
    def test_linear_p95_not_nearest_rank(self):
        self.assertAlmostEqual(percentile95(range(200)), 189.05)
        with self.assertRaises(ValueError):
            percentile95([float("nan")])

    def test_null_uses_lower_bound_and_strict_comparison(self):
        rows = [{"valid": True, "ci_low": .2}, {"valid": True, "ci_low": .3},
                {"valid": False, "ci_low": 2.0}]
        self.assertEqual(null_decisions(rows, .2), [False, True, False])
        self.assertAlmostEqual(exact_binomial_interval(0, 100)[1], .036216692645, places=9)

    def test_two_consecutive_and_eligibility(self):
        def row(k, low):
            return {"check_index": k, "interaction_step": k * 50000,
                    "ci_low": low, "valid": True}
        rows = [row(1, .1), row(2, -.1), row(3, .15), row(4, .2)]
        self.assertEqual(replay_checks(rows, 0), {"check_index": 4, "interaction_step": 200000})
        self.assertIsNone(replay_checks(rows, .15))
        self.assertIsNone(replay_checks([row(19, .2), row(20, .3)], 0))
        self.assertIsNone(replay_checks([row(1, .2), row(3, .2)], 0))
        with self.assertRaisesRegex(ValueError, "forced"):
            replay_checks([dict(row(1, .2), forced=True)], 0)


if __name__ == "__main__":
    unittest.main()
