"""Canonical per-Q fields and plotted curves retain the separate measurements."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from analysis.exp1_analysis import _plot_setup, plot_learning_curves
from experiments.exp12 import ledger


class TwinReportingTest(unittest.TestCase):
    def test_per_q_csv_and_individual_curve_values(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            probe = root / "source/probes"
            probe.mkdir(parents=True)
            arrays = {
                f"{name}_q{q}_losses": np.full((5, 1000), value, np.float32)
                for value, (name, q) in enumerate(
                    [("fresh", 1), ("fresh", 2), ("current", 1), ("current", 2)], 1
                )
            }
            np.savez(probe / "check_01.npz", **arrays)
            identity = dict(
                run_key="exp1_D4W1024_h1-run-v0_seed1",
                run_role="confirmatory",
                architecture="D4W1024",
                environment="h1-run-v0",
                seed=1,
                budget_env_steps=2000000,
                num_interaction_steps=1000000,
                num_checks=20,
            )
            record = dict(
                check_index=1,
                interaction_step=50000,
                loss_iqm=-1.0,
                **{k: i + 0.25 for i, k in enumerate(ledger.TWIN_CHECK_COLUMNS)},
            )
            directory = ledger.write_run(identity, [record], None, "running", probe, d)
            actual = pd.read_csv(directory / "checks.csv").iloc[0]
            for column in ledger.TWIN_CHECK_COLUMNS:
                self.assertEqual(actual[column], record[column])
            self.assertEqual(actual.loss_iqm, -1.0)
            plt = _plot_setup()
            with mock.patch.object(plt, "close"):
                plot_learning_curves(
                    identity["run_key"], root / "plot.png", d, check_indices=(1,)
                )
                fig = plt.gcf()
                lines = fig.axes[0].lines
                self.assertEqual(
                    [line.get_label() for line in lines],
                    ["fresh_q1", "fresh_q2", "current_q1", "current_q2"],
                )
                for i, line in enumerate(lines, 1):
                    np.testing.assert_array_equal(line.get_ydata(), np.full(1000, i))
            plt.close(fig)
            self.assertGreater((root / "plot.png").stat().st_size, 0)

    def test_legacy_single_header_loads_without_invented_q_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            probe = root / "probes"
            probe.mkdir()
            identity = dict(
                run_key="single",
                run_role="confirmatory",
                architecture="D2W512",
                environment="dog-run",
                seed=1,
                budget_env_steps=1000000,
                num_interaction_steps=500000,
                num_checks=20,
            )
            directory = ledger.write_run(
                identity,
                [dict(check_index=1, interaction_step=25000)],
                None,
                "complete",
                probe,
                d,
            )
            frame = pd.read_csv(directory / "checks.csv")
            frame[ledger.LEGACY_CHECK_COLUMNS].to_csv(
                directory / "checks.csv", index=False
            )
            _, checks = ledger.load(d)
            self.assertTrue(checks[ledger.TWIN_CHECK_COLUMNS].isna().all().all())
