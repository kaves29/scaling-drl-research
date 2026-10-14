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
    def test_only_declared_recording_meters_are_excluded(self):
        common = {"train/critic_loss": (1.0, 1.0, 1.0, 1),
                  "train/policy_kl": (0.1, 0.1, 0.1, 1)}
        with (mock.patch.object(state, "state_differences", return_value=[]),
              mock.patch.object(state, "load_meta", side_effect=[
                  {"meters": common}, {"meters": {"train/critic_loss": common["train/critic_loss"]}}])):
            self.assertEqual(qualification.checkpoint_differences("a", "b"), [])
        with (mock.patch.object(state, "state_differences", return_value=[]),
              mock.patch.object(state, "load_meta", side_effect=[
                  {"meters": common}, {"meters": {"train/critic_loss": (2.0, 2.0, 2.0, 1)}}])):
            self.assertEqual(qualification.checkpoint_differences("a", "b"),
                             ["meta:meters/train/critic_loss"])

    def test_both_precision_triples_reach_comparison(self):
        repo = Path(__file__).resolve().parents[2]
        source = repo / "main"
        commit = qualification.git_head(repo)
        original_overrides = exp12_helpers.tiny_overrides
        for precision in ("highest", "production_tf32"):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temp:
                out = Path(temp) / "evidence"
                argv = ["qualification", "--out", str(out), "--source-commit", commit,
                        "--harness-commit", commit, "--precision", precision]
                with (mock.patch.object(sys, "argv", argv),
                      mock.patch.object(qualification, "require_a100_backend"),
                      mock.patch.object(qualification, "STEPS", 12),
                      mock.patch.object(exp1, "set_matmul_precision",
                                        side_effect=lambda: jax.config.update(
                                            "jax_default_matmul_precision", "tensorfloat32")),
                      mock.patch.object(exp12_helpers, "tiny_overrides",
                                        side_effect=lambda steps, extra: original_overrides(
                                            steps=steps, extra=[*extra, "probe.enabled=false"])),
                      contextlib.chdir(source)):
                    code = qualification.main()
                receipt = json.loads((out / "receipt.json").read_text())
                self.assertEqual(code, 0)
                self.assertEqual(receipt["verdict"], "PASS_EXACT_ISOLATION_AT_TESTED_CONFIGURATION")
                self.assertTrue(all(receipt[name]["status"] == "COMPLETE"
                                    for name in ("off_a", "on", "off_b")))
                self.assertIsNone(receipt["pairs"]["off_off"]["first_divergence"])
                self.assertIsNone(receipt["pairs"]["on_off"]["first_divergence"])
                self.assertEqual(receipt["pairs"]["on_off"]["complete_checkpoint_differences"]["training"], [])
                self.assertEqual(receipt["pairs"]["on_off"]["complete_checkpoint_differences"]["ancillary"], [])
                self.assertEqual(receipt["gpu"], "cpu")
                self.assertEqual(receipt["precision"], precision)


if __name__ == "__main__":
    unittest.main()
