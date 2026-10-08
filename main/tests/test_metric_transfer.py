"""Exact host replay parity for the metric transfer engineering change."""

import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import numpy as np

from experiments.angle_1 import ACTOR_GRAD_COSINE_KEY, PendingUpdateMetrics
from experiments.exp12.diagnostics import (
    ActorDiagnostics,
    DiagnosticPendingUpdateMetrics,
)
from scale_rl.common.logger import WandbTrainerLogger


def make_pending(diagnostic=True, every=7):
    logger = WandbTrainerLogger.__new__(WandbTrainerLogger)
    logger.reset()
    if diagnostic:
        cfg = SimpleNamespace(
            diagnostics=SimpleNamespace(
                enabled=True, kl_reference_size=2, saturation_threshold=0.99
            ),
            seed=1,
            logging_per_interaction_step=2000,
        )
        pending = DiagnosticPendingUpdateMetrics(logger, every, ActorDiagnostics(cfg))
    else:
        pending = PendingUpdateMetrics(logger, every)
    return logger, pending


def legacy_flush(pending):
    host_infos = jax.device_get([info for _, info in pending._pending])
    if isinstance(pending, DiagnosticPendingUpdateMetrics):
        for info in host_infos:
            pending._diagnostics.collect(info)
        pending._pending = [
            (step, host) for (step, _), host in zip(pending._pending, host_infos)
        ]
        host_infos = jax.device_get([info for _, info in pending._pending])
    for (first, _), info in zip(pending._pending, host_infos):
        for index in range(len(info[ACTOR_GRAD_COSINE_KEY])):
            row = {key: float(values[index]) for key, values in info.items()}
            if (first + index) % pending._actor_grad_cosine_every != 0:
                del row[ACTOR_GRAD_COSINE_KEY]
            pending._logger.update_metric(**row)
    pending._pending = []


class MetricTransferTest(unittest.TestCase):
    def test_full_payload_matches_legacy_rows_meters_and_diagnostics(self):
        log, pending = make_pending()
        reference_log, reference = make_pending()
        rng = np.random.default_rng(123)
        for group in range(1001):
            first = 2 * group + 3
            values = rng.standard_normal((19, 2)).astype(np.float32)
            info = {
                f"train/metric_{key}": jax.device_put(values[key]) for key in range(17)
            }
            info["train/actor_gnorm"] = jax.device_put(np.abs(values[17]))
            cosine = np.where((first + np.arange(2)) % 7 == 0, values[18], np.nan)
            info[ACTOR_GRAD_COSINE_KEY] = jax.device_put(cosine)
            pending.add(first, info)
            reference.add(first, info)
        leaves = jax.tree_util.tree_leaves([info for _, info in pending._pending])
        self.assertEqual(len(leaves), 19019)
        self.assertEqual(sum(value.nbytes for value in leaves), 152152)
        with mock.patch.object(log, "update_metric", wraps=log.update_metric) as rows:
            with mock.patch("jax.device_get", wraps=jax.device_get) as transfer:
                pending.flush()
            transfer.assert_called_once()
        with mock.patch.object(
            reference_log, "update_metric", wraps=reference_log.update_metric
        ) as reference_rows:
            legacy_flush(reference)
        self.assertEqual(rows.call_args_list, reference_rows.call_args_list)
        self.assertEqual(len(rows.call_args_list), 2002)
        self.assertEqual(
            {key: vars(meter) for key, meter in log.average_meter_dict.meters.items()},
            {
                key: vars(meter)
                for key, meter in reference_log.average_meter_dict.meters.items()
            },
        )
        self.assertEqual(pending._diagnostics.state(), reference._diagnostics.state())
        self.assertEqual(
            pending._diagnostics.window_metrics(),
            reference._diagnostics.window_metrics(),
        )
        self.assertEqual(pending._pending, [])

    def test_variable_group_lengths_and_host_precision_preserve_order(self):
        for diagnostic in (False, True):
            with self.subTest(diagnostic=diagnostic):
                log, pending = make_pending(diagnostic, every=1)
                reference_log, reference = make_pending(diagnostic, every=1)
                for first, values in ((0, [1e20, 1.0]), (2, [-1e20]), (3, [3.0])):
                    info = {
                        ACTOR_GRAD_COSINE_KEY: np.zeros(len(values), np.float32),
                        "train/actor_gnorm": np.asarray(values, np.float64),
                    }
                    pending.add(first, info)
                    reference.add(first, info)
                pending.flush()
                legacy_flush(reference)
                self.assertEqual(
                    log.average_meter_dict.averages(),
                    reference_log.average_meter_dict.averages(),
                )
                self.assertEqual(log.average_meter_dict["train/actor_gnorm"].sum, 3.0)
                if diagnostic:
                    self.assertEqual(
                        pending._diagnostics.state(), reference._diagnostics.state()
                    )

    def test_transfer_failure_retains_original_pending_and_diagnostics(self):
        for diagnostic in (False, True):
            with self.subTest(diagnostic=diagnostic):
                log, pending = make_pending(diagnostic)
                info = {
                    ACTOR_GRAD_COSINE_KEY: jax.device_put(np.zeros(2, np.float32)),
                    "train/actor_gnorm": jax.device_put(np.ones(2, np.float32)),
                }
                pending.add(0, info)
                original = pending._pending
                with mock.patch("jax.device_get", side_effect=OSError("copy failed")):
                    with self.assertRaisesRegex(OSError, "copy failed"):
                        pending.flush()
                self.assertIs(pending._pending, original)
                self.assertIs(pending._pending[0][1], info)
                self.assertEqual(log.average_meter_dict.averages(), {})
                if diagnostic:
                    self.assertEqual(pending._diagnostics.gnorm, [])

    def test_empty_and_repeated_flushes_do_not_duplicate_rows(self):
        log, pending = make_pending()
        pending.flush()
        self.assertEqual(log.average_meter_dict.averages(), {})
        pending.add(
            0,
            {
                ACTOR_GRAD_COSINE_KEY: np.array([0.5], np.float32),
                "train/actor_gnorm": np.array([2.0], np.float32),
            },
        )
        pending.flush()
        pending.flush()
        self.assertEqual(log.average_meter_dict["train/actor_gnorm"].count, 1)
        self.assertEqual(pending._diagnostics.gnorm, [2.0])


if __name__ == "__main__":
    unittest.main()
