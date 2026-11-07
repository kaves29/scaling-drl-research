"""CPU checks of the separate, fail-closed numerical-isolation harness."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import exp12_diagnostic_isolation as harness


class IsolationHarnessTest(unittest.TestCase):
    def test_typed_fingerprints_and_named_component_differences(self):
        self.assertNotEqual(harness._digest(np.array([0.0], np.float32)),
                            harness._digest(np.array([-0.0], np.float32)))
        self.assertNotEqual(harness._digest(np.array([1], np.float32)),
                            harness._digest(np.array([1], np.float64)))
        a = {"training": {"actor/kernel": "a"}, "ancillary": {"replay/index": "a"}}
        b = {"training": {"actor/kernel": "b"}, "ancillary": {"replay/index": "a"}}
        self.assertEqual(harness.differences(a, b),
                         {"training": ["actor/kernel"], "ancillary": []})

    def test_agent_deviation_is_named_and_uses_reference_leaf_scale(self):
        trees = ({"actor": {"kernel": np.array([2.0, 0.0], np.float32)},
                  "rng": np.array([1], np.uint32)},
                 {"actor": {"kernel": np.array([1.0, 0.0], np.float32)},
                  "rng": np.array([1], np.uint32)})
        with mock.patch("experiments.exp12.state.load_agent_tree", side_effect=trees):
            result = harness.agent_deviations("on", "off")
        self.assertEqual(result["max_relative_deviation"], 1.0)
        self.assertEqual(result["differing_leaf_count"], 1)
        self.assertIn("kernel", result["largest_leaves"][0]["path"])

    def test_first_divergence_and_off_off_baseline_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_steps = harness.STEPS
            harness.STEPS = 3
            try:
                for name, _ in harness.ARMS:
                    arm = root / name
                    (arm / "state").mkdir(parents=True)
                    rows = [{"interaction_step": i, "update_step": i,
                             "training": {"actor/kernel": "common" if i < 2 or name != "on" else "on"},
                             "ancillary": {"replay/index": "common"}}
                            for i in range(1, 4)]
                    (arm / "fingerprints.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows))
                with mock.patch("experiments.exp12.state.latest_state_dir", side_effect=lambda p: p), \
                     mock.patch("experiments.exp12.state.state_differences", return_value=[]), \
                     mock.patch.object(harness, "agent_deviations", return_value={"max_relative_deviation": 0}):
                    result = harness.compare(root)
                self.assertEqual(result["verdict"], "BLOCKED_DIAGNOSTICS_ASSOCIATED_DIVERGENCE")
                self.assertEqual(result["pairs"]["on_off"]["first_divergence"]["interaction_step"], 2)
                self.assertIsNone(result["pairs"]["off_off"]["first_divergence"])
                path = root / "off_b" / "fingerprints.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["ancillary"]["replay/index"] = "different"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with mock.patch("experiments.exp12.state.latest_state_dir", side_effect=lambda p: p), \
                     mock.patch("experiments.exp12.state.state_differences", return_value=[]), \
                     mock.patch.object(harness, "agent_deviations", return_value={"max_relative_deviation": 0}):
                    result = harness.compare(root)
                self.assertEqual(result["verdict"], "BLOCKED_OFF_OFF_BASELINE_DIVERGES")
                self.assertEqual(result["pairs"]["off_off"]["first_divergence"]["interaction_step"], 1)
            finally:
                harness.STEPS = original_steps


if __name__ == "__main__":
    unittest.main()
