"""Known-answer accounting checks; no model creation, training or probe fits."""

import copy
import pickle
import unittest

import numpy as np

from audits.probe_difficulty.accounting import (
    VARIANCE_FLOOR,
    component_bounds,
    grouped_intervals,
    constant_fresh_witness,
    precision_and_curves,
    ratio,
    rounds_and_points,
    seed_interval,
    source_predicted_sampling,
)
from audits.echocritic_forensic.evidence import source_iqm


def example_archive():
    result = {}
    for kind, mse in (("fresh", 0.1), ("current", 0.3)):
        result[kind + "_b"] = np.full(10, 0.5, np.float32)
        result[kind + "_final_loss"] = np.full(10, mse, np.float32)
        result[kind + "_score"] = result[kind + "_b"] - result[kind + "_final_loss"]
        result[kind + "_offset"] = np.full(
            10, 2000 if kind == "current" else -0.4, np.float32
        )
        result[kind + "_losses"] = np.tile(
            np.arange(1, 2001, dtype=np.float32), (10, 1)
        )
    return result


class DifficultyAccountingTest(unittest.TestCase):
    def test_ratio_rule_no_hidden_denominator_epsilon(self):
        self.assertEqual(ratio(1, 0.5), 2)
        for b in (None, 0, VARIANCE_FLOOR):
            self.assertIsNone(ratio(1, b))
        self.assertEqual(ratio(1, 2 * VARIANCE_FLOOR), 1 / (2 * VARIANCE_FLOOR))

    def test_joint_bounds_use_both_scores(self):
        bounds = component_bounds(0.3, 0.5)
        self.assertEqual(bounds["b_lower_bound"], 0.5)
        self.assertAlmostEqual(bounds["fresh_mse_lower_bound"], 0.2)
        self.assertEqual(bounds["current_mse_lower_bound"], 0)
        self.assertEqual(bounds["current_mse_upper_bound"], 0.5)
        with self.assertRaises(ValueError):
            component_bounds(1.1)

    def test_two_component_histories_produce_same_joint_scores(self):
        rows = [
            {"fresh_score": 0.5, "current_score": 0.45},
            {"fresh_score": 0.2, "current_score": 0.48},
        ]
        witness = constant_fresh_witness(rows)
        self.assertTrue(witness["feasible"])
        mf = witness["witness_fresh_mse"]
        for row in rows:
            variable_b = row["fresh_score"] + mf
            mc = variable_b - row["current_score"]
            self.assertTrue(0 <= variable_b <= 1 and mc >= 0)
            self.assertAlmostEqual(variable_b - mf, row["fresh_score"])
            self.assertAlmostEqual(variable_b - mc, row["current_score"])
            self.assertAlmostEqual(
                0.75 - (0.75 - row["fresh_score"]), row["fresh_score"]
            )
        # Same falling fresh score: one witness changes only variance; the
        # constant-b alternative changes fresh error. Neither is observed data.
        self.assertLess(rows[-1]["fresh_score"] + mf, rows[0]["fresh_score"] + mf)
        self.assertGreater(
            0.75 - rows[-1]["fresh_score"], 0.75 - rows[0]["fresh_score"]
        )

    def test_some_score_histories_reject_constant_mse_not_a_forced_hypothesis(self):
        self.assertFalse(
            constant_fresh_witness(
                [
                    {"fresh_score": 0.9, "current_score": 0.9},
                    {"fresh_score": -0.9, "current_score": -0.9},
                ]
            )["feasible"]
        )

    def test_relative_error_difference_independent_of_variance(self):
        for b in (0.5, 0.9):
            mf, mc = b - 0.4, b + 7
            self.assertAlmostEqual(mc - mf, 7.4)
        bounds = component_bounds(0.4, -7)
        self.assertGreaterEqual(bounds["current_mse_lower_bound"], 7)

    def test_curves_do_not_invent_intermediate_pool_values(self):
        precision, points, summaries = precision_and_curves(example_archive())
        self.assertEqual(len(points), 100)
        early = next(
            p
            for p in points
            if p["kind"] == "fresh"
            and p["round"] == 0
            and p["fit_updates_completed"] == 1000
        )
        self.assertEqual(early["mse"], 1001)
        self.assertIsNone(early["initial_or_intermediate_pool_mse"])
        final = next(
            p
            for p in points
            if p["kind"] == "fresh"
            and p["round"] == 0
            and p["fit_updates_completed"] == 2000
        )
        self.assertAlmostEqual(final["mse"], 0.1, places=7)
        self.assertTrue(
            all(s["initial_to_final_pool_improvement"] is None for s in summaries)
        )
        current = next(p for p in precision if p["kind"] == "current")
        self.assertEqual(current["max_offset_plus_base_rounding_abs"], 2**-14)
        self.assertIn("optimization-path", current["does_not_bound"])
        synthetic = example_archive()
        synthetic["current_losses"][:, 0] = 100
        bounds, _, _ = precision_and_curves(synthetic)
        lower = next(p for p in bounds if p["kind"] == "current")[
            "max_abs_initial_minibatch_Q_minus_stored_offset_lower_bound"
        ]
        self.assertGreater(lower, 8.99)
        self.assertLess(lower, 9)

    def test_unobserved_components_remain_missing_and_archive_binds_exactly(self):
        arrays = example_archive()
        init = {
            "probe/check_index": 0,
            "probe/interaction_step": 5000,
            "_step": 8000,
            **{f"probe/score_fresh_r{i}": 0.49 for i in range(10)},
        }
        last = {
            "probe/check_index": 50,
            "probe/interaction_step": 500000,
            "_step": 1000000,
            **{
                f"probe/score_{kind}_r{i}": float(arrays[kind + "_score"][i])
                for kind in ("fresh", "current")
                for i in range(10)
            },
            **{
                f"probe/loss_r{i}": float(
                    arrays["fresh_score"][i] - arrays["current_score"][i]
                )
                for i in range(10)
            },
        }
        run = {
            "id": "r",
            "environment": "myo-pen-twirl",
            "architecture": "D2W512",
            "seed": 5,
            "cohort": "exp96",
            "history": [init, last],
        }
        source_points = [
            {"id": "r", "check": 0},
            {"id": "r", "check": 50, "paired_deficit": 0.2},
        ]
        rows, points = rounds_and_points([run], "r", arrays, source_points)
        self.assertTrue(
            all(
                r["b"] is None and r["fresh_final_pool_mse"] is None
                for r in rows
                if r["check"] == 0
            )
        )
        self.assertTrue(
            all(
                r["ratio_status"] == "variance_unobserved"
                for r in rows
                if r["check"] == 0
            )
        )
        self.assertTrue(
            all(
                r["b"] == 0.5 and r["ratio_status"] == "defined"
                for r in rows
                if r["check"] == 50
            )
        )
        corrupted = copy.deepcopy(arrays)
        corrupted["fresh_score"][0] += 1
        with self.assertRaisesRegex(ValueError, "binding"):
            rounds_and_points([run], "r", corrupted, source_points)
        self.assertAlmostEqual(points[-1]["fresh_final_pool_mse_iqm"], 0.1, places=7)
        grouped = grouped_intervals(points)
        missing = next(
            row
            for row in grouped
            if row["check"] == 0 and row["metric"] == "current_mse_lower_iqm_bound"
        )
        self.assertEqual(missing["n_seeds"], 0)
        self.assertIsNone(missing["mean"])

    def test_seed_uncertainty_not_round_uncertainty_and_local_rng_only(self):
        before = pickle.dumps(np.random.get_state())
        self.assertIsNone(seed_interval([3], 1)["ci_low"])
        summary = seed_interval([0, 1, 2, 3], 1)
        self.assertEqual(summary["n_seeds"], 4)
        self.assertEqual(summary["mean"], 1.5)
        self.assertEqual(before, pickle.dumps(np.random.get_state()))

    def test_sampling_prediction_is_not_claimed_observed_coverage(self):
        rows = source_predicted_sampling(
            [
                {"id": "r", "check": 0, "interaction_step": 5000},
                {"id": "r", "check": 35, "interaction_step": 350000},
            ]
        )
        self.assertGreater(
            rows[0]["expected_duplicate_index_fraction"],
            rows[1]["expected_duplicate_index_fraction"],
        )
        self.assertTrue(all(r["physical_state_coverage"] is None for r in rows))
        self.assertIn("NOT saved", rows[0]["evidence_type"])

    def test_iqm_is_not_assumed_additive_for_component_changes(self):
        fresh = -np.arange(10, dtype=np.float32)
        current = np.roll(fresh, 3)
        self.assertNotEqual(
            source_iqm(fresh - current, 10),
            source_iqm(fresh, 10) - source_iqm(current, 10),
        )
