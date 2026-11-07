"""CPU regression for both precision paths of the recording-isolation harness."""

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax

from scripts import exp12_recording_isolation_qualification as qualification
from tests import exp12_helpers
from experiments import exp1
from experiments.exp12 import state


class RecordingQualificationPrecisionTest(unittest.TestCase):
    def setUp(self):
        previous = jax.config.jax_default_matmul_precision
        self.addCleanup(jax.config.update, "jax_default_matmul_precision", previous)

    def run_harness(self, out, precision, steps=41, probes=False, extra=()):
        repo = Path(__file__).resolve().parents[2]
        commit = qualification.git_head(repo)
        argv = [
            "qualification",
            "--out",
            str(out),
            "--source-commit",
            commit,
            "--harness-commit",
            commit,
            "--precision",
            precision,
            *extra,
        ]
        original_overrides = exp12_helpers.tiny_overrides
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(qualification, "require_a100_backend"),
            mock.patch.object(qualification, "STEPS", steps),
            mock.patch.object(
                exp1,
                "set_matmul_precision",
                side_effect=lambda: jax.config.update(
                    "jax_default_matmul_precision", "tensorfloat32"
                ),
            ),
            mock.patch.object(
                exp12_helpers,
                "tiny_overrides",
                side_effect=lambda steps, extra: original_overrides(
                    steps=steps, extra=[*extra, f"probe.enabled={str(probes).lower()}"]
                ),
            ),
            contextlib.chdir(repo / "main"),
        ):
            return qualification.main()

    def test_cpu_is_rejected_by_production_backend_guard(self):
        if jax.default_backend() != "cpu":
            self.skipTest("CPU backend required for this rejection test")
        with self.assertRaisesRegex(ValueError, "A100 GPU backend required"):
            qualification.require_a100_backend()

    def test_only_declared_recording_meters_are_excluded(self):
        common = {
            "train/critic_loss": (1.0, 1.0, 1.0, 1),
            "train/policy_kl": (0.1, 0.1, 0.1, 1),
        }
        with (
            mock.patch.object(state, "state_differences", return_value=[]),
            mock.patch.object(
                state,
                "load_meta",
                side_effect=[
                    {"meters": common},
                    {"meters": {"train/critic_loss": common["train/critic_loss"]}},
                ],
            ),
        ):
            self.assertEqual(qualification.checkpoint_differences("a", "b"), [])
        with (
            mock.patch.object(state, "state_differences", return_value=[]),
            mock.patch.object(
                state,
                "load_meta",
                side_effect=[
                    {"meters": common},
                    {"meters": {"train/critic_loss": (2.0, 2.0, 2.0, 1)}},
                ],
            ),
        ):
            self.assertEqual(
                qualification.checkpoint_differences("a", "b"),
                ["meta:meters/train/critic_loss"],
            )

    def test_both_precision_triples_reach_comparison(self):
        for precision in ("highest", "production_tf32"):
            with self.subTest(
                precision=precision
            ), tempfile.TemporaryDirectory() as temp:
                out = Path(temp) / "evidence"
                code = self.run_harness(out, precision)
                receipt = json.loads((out / "receipt.json").read_text())
                self.assertEqual(code, 0)
                self.assertEqual(
                    receipt["verdict"], "PASS_EXACT_ISOLATION_AT_TESTED_CONFIGURATION"
                )
                self.assertTrue(
                    all(
                        receipt[name]["status"] == "COMPLETE"
                        for name in ("off_a", "on", "off_b")
                    )
                )
                self.assertIsNone(receipt["pairs"]["off_off"]["first_divergence"])
                self.assertIsNone(receipt["pairs"]["on_off"]["first_divergence"])
                self.assertEqual(
                    receipt["pairs"]["on_off"]["complete_checkpoint_differences"][
                        "training"
                    ],
                    [],
                )
                self.assertEqual(
                    receipt["pairs"]["on_off"]["complete_checkpoint_differences"][
                        "ancillary"
                    ],
                    [],
                )
                self.assertEqual(receipt["gpu"], "cpu")
                self.assertEqual(receipt["precision"], precision)
                self.assertEqual(receipt["backend"], "cpu")
                self.assertIn("41-interaction", receipt["scope"])
                self.assertNotIn("A100", receipt["scope"])

    def test_precision_drift_fails_and_restores_training_hook(self):
        from experiments.exp12.trainer import Exp12Trainer

        original_train = Exp12Trainer.train
        original_fingerprint = qualification.fingerprint

        def drift(trainer):
            result = original_fingerprint(trainer)
            jax.config.update("jax_default_matmul_precision", "highest")
            return result

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "evidence"
            with (
                mock.patch.object(qualification, "fingerprint", side_effect=drift),
                self.assertRaisesRegex(
                    ValueError, "precision changed during the training run"
                ),
            ):
                self.run_harness(out, "production_tf32", steps=12)
            receipt = json.loads((out / "receipt.json").read_text())
            self.assertEqual(receipt["verdict"], "INCOMPLETE")
            self.assertEqual(receipt["off_a"]["status"], "FAILED")
            self.assertEqual(receipt["off_a"]["error"]["type"], "ValueError")
            self.assertIs(Exp12Trainer.train, original_train)

    def test_stale_checkpoint_cannot_pass_complete_fingerprints(self):
        rows = [
            {"interaction_step": step, "training": {}, "ancillary": {}}
            for step in range(1, 3)
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for arm, _ in qualification.ARMS:
                (root / arm).mkdir()
                (root / arm / "fingerprints.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows)
                )
            with (
                mock.patch.object(qualification, "STEPS", 2),
                mock.patch.object(state, "latest_state_dir", return_value=root),
                mock.patch.object(
                    state, "load_meta", return_value={"interaction_step": 1}
                ),
                self.assertRaisesRegex(
                    ValueError, "checkpoint is not at interaction 2"
                ),
            ):
                qualification.compare(root)


if __name__ == "__main__":
    unittest.main()
