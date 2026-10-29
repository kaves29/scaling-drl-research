"""Array-only reporting checks; no natural positive-control evidence or fits."""

import unittest

import numpy as np

from experiments.exp12.m_selection import evaluate
from experiments.exp12.probe import summarize


def losses(degraded, last, half, full):
    return {
        name: np.full(5, value, dtype=np.float32)
        for name, value in zip(
            ("degraded", "injected_last", "injected_half", "injected_all"),
            (degraded, last, half, full),
        )
    }


class RawPlasticityContrastTest(unittest.TestCase):
    def test_direction_matches_the_actual_paired_score_definition(self):
        fresh = np.ones(5, dtype=np.float32)
        loss = {}
        for name, score in (
            ("degraded", 0.5),
            ("injected_last", 0.75),
            ("injected_half", 0.5),
            ("injected_all", 0.25),
        ):
            result = summarize(
                {
                    "current": {"score": np.full(5, score, np.float32)},
                    "fresh": {"score": fresh},
                }
            )
            loss[name] = result["loss_rounds"]
        out = evaluate(loss, 0.10, 0.10)
        self.assertEqual(out["chosen_m"], "last")
        for label, expected in (("last", 0.25), ("half", 0.0), ("all", -0.25)):
            row = out["plasticity_loss_contrasts"][label]
            self.assertEqual(row["unrescued_loss_iqm"], 0.5)
            self.assertEqual(row["unrescued_minus_injected"], expected)
            self.assertEqual(row["injected_loss_iqm"], 0.5 - expected)
            self.assertEqual(out["recovery"][label], expected / 0.5)

    def test_negative_recoveries_and_overshoot_do_not_change_ranking(self):
        for series, expected in (
            (losses(1.0, 2.0, 2.5, 3.0), "last"),
            (losses(1.0, -0.25, -0.5, -1.0), "all"),
        ):
            out = evaluate(series, 0.10, 0.10)
            self.assertIsNone(out["stop"])
            self.assertEqual(out["chosen_m"], expected)
            self.assertNotIn("rescue_success", out)
            for label, row in out["plasticity_loss_contrasts"].items():
                self.assertEqual(
                    row["unrescued_minus_injected"],
                    out["l_trigger"] - out["l_injected"][label],
                )

    def test_raw_values_are_retained_when_normalized_recovery_is_undefined(self):
        out = evaluate(losses(0.0, 0.25, 0.5, 1.0), 0.10, 0.10)
        self.assertIsNone(out["chosen_m"])
        self.assertIn("undefined", out["stop"])
        self.assertNotIn("recovery", out)
        self.assertEqual(
            out["plasticity_loss_contrasts"]["last"]["unrescued_minus_injected"],
            -0.25,
        )

    def test_noise_stop_and_nonfinite_rejection_remain_unchanged(self):
        series = losses(1.0, 0.5, 0.25, 0.0)
        series["degraded"][0] = 10.0
        out = evaluate(series, 0.10, 0.10)
        self.assertIsNone(out["chosen_m"])
        self.assertIn("noise", out["stop"])
        self.assertEqual(len(out["plasticity_loss_contrasts"]), 3)
        series["degraded"][0] = np.nan
        out = evaluate(series, 0.10, 0.10)
        self.assertIsNone(out["chosen_m"])
        self.assertIn("finite", out["stop"])
        self.assertNotIn("plasticity_loss_contrasts", out)
