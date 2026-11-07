"""Metric-transfer adoption contract at actual complete-state checkpoint boundaries."""

import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import numpy as np

from experiments.angle_1 import ACTOR_GRAD_COSINE_KEY
from experiments.exp12.diagnostics import DiagnosticPendingUpdateMetrics
from experiments.exp12.state import state_differences
from experiments.exp12.trainer import Exp12Trainer
from tests.exp12_helpers import compose, patch_wandb, tiny_overrides
from tests.test_metric_transfer import legacy_flush


class MetricCheckpointContractTest(unittest.TestCase):
    def trainer(self, root, extra=()):
        cfg = compose(tiny_overrides(extra=["env.max_episode_steps=20", *extra]))
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        trainer = Exp12Trainer(cfg, str(root))
        self.addCleanup(trainer.close)
        return trainer

    def setUp(self):
        patchers = patch_wandb()
        self.addCleanup(lambda: [patch.stop() for patch in patchers])

    def test_training_and_resumed_continuation_match_bulk_for_single_twin_and_injected(
        self,
    ):
        for twin in (False, True):
            for injected in (False, True):
                with self.subTest(
                    twin=twin, injected=injected
                ), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    extra = [
                        "critic_num_blocks=2",
                        f"agent.critic_use_cdq={str(twin).lower()}",
                    ]
                    original = self.trainer(root / "original", extra)
                    original.start()
                    original.train(16)
                    if injected:
                        original.inject("last", 990)
                    fork_state = original.save(root / "fork")

                    grouped = self.trainer(root / "grouped", extra)
                    grouped.restore(fork_state)
                    grouped.train(23)
                    midpoint = grouped.save(root / "midpoint")
                    resumed = self.trainer(root / "resumed", extra)
                    resumed.restore(midpoint)
                    resumed.train(36)
                    actual = resumed.save(root / "grouped_final")

                    bulk = self.trainer(root / "bulk", extra)
                    bulk.restore(fork_state)
                    with mock.patch.object(
                        DiagnosticPendingUpdateMetrics, "flush", legacy_flush
                    ):
                        bulk.train(23)
                        bulk.save(root / "bulk_midpoint")
                        bulk.train(36)
                        expected = bulk.save(root / "bulk_final")
                    self.assertEqual(state_differences(expected, actual), [])

    def test_checkpoint_flushes_full_payload_once_and_restores_window_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.trainer(root / "original")
            original.start()
            expected = []
            for group in range(1001):
                values = np.array([group / 8, -group / 16], np.float32)
                expected.extend(map(float, values))
                info = {
                    f"train/metric_{key}": jax.device_put(values) for key in range(17)
                }
                info["train/actor_gnorm"] = jax.device_put(values)
                info[ACTOR_GRAD_COSINE_KEY] = jax.device_put(values)
                original.pending.add(2 * group, info)
            with mock.patch.object(
                original.logger, "update_metric", wraps=original.logger.update_metric
            ) as rows:
                checkpoint = original.save(root / "state")
            self.assertEqual(rows.call_count, 2002)
            self.assertEqual(original.pending._pending, [])
            self.assertEqual(original.diagnostics.gnorm, expected)
            meter = original.logger.average_meter_dict["train/actor_gnorm"]
            self.assertEqual(meter.sum, sum(expected))
            self.assertEqual(meter.count, 2002)
            restored = self.trainer(root / "restored")
            restored.restore(checkpoint)
            self.assertEqual(restored.diagnostics.gnorm, expected)
            self.assertEqual(
                vars(restored.logger.average_meter_dict["train/actor_gnorm"]),
                vars(meter),
            )
            state = original.diagnostics.state()
            rng = np.asarray(original._sac_agent._rng).copy()
            original.save(root / "state_again")
            self.assertEqual(original.diagnostics.state(), state)
            np.testing.assert_array_equal(original._sac_agent._rng, rng)
            with mock.patch.object(
                restored.logger, "update_metric", wraps=restored.logger.update_metric
            ) as replay:
                restored.save(root / "restored_state")
            self.assertEqual(replay.call_count, 0)
            self.assertEqual(restored.diagnostics.gnorm, expected)

    def test_failed_transfer_does_not_publish_a_checkpoint_or_consume_update_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = self.trainer(root / "run")
            trainer.start()
            first = trainer.save(root / "state")
            pointer = (root / "state/LATEST").read_bytes()
            trainer.pending.add(
                0,
                {
                    ACTOR_GRAD_COSINE_KEY: jax.device_put(np.ones(2, np.float32)),
                    "train/actor_gnorm": jax.device_put(np.ones(2, np.float32)),
                },
            )
            with mock.patch("jax.device_get", side_effect=OSError("transfer failed")):
                with self.assertRaisesRegex(OSError, "transfer failed"):
                    trainer.save(root / "state")
            self.assertEqual((root / "state/LATEST").read_bytes(), pointer)
            self.assertEqual(list((root / "state").glob("step_*")), [first])
            self.assertEqual(len(trainer.pending._pending), 1)
            self.assertEqual(trainer.diagnostics.gnorm, [])
            second = trainer.save(root / "state")
            meta = pickle.loads((second / "meta.pkl").read_bytes())
            self.assertEqual(meta["actor_diagnostics"]["gnorm"], [1.0, 1.0])
            self.assertEqual(meta["meters"]["train/actor_gnorm"][3], 2)
            self.assertEqual(trainer.pending._pending, [])


if __name__ == "__main__":
    unittest.main()
