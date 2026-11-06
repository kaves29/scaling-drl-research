"""Small CPU checks for the proposed common-replay admission contract."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp12.exp2_shared_data import make_plan, paired_indices


def parent(root, architecture, *, complete=True, protocol="same", rows=8):
    run = root / architecture
    state = run / "state" / "step_000000008_test"
    (state / "agent_ckpt").mkdir(parents=True)
    if complete:
        (run / "DONE").write_text(json.dumps({"interaction_step": 8}))
    (run / "state" / "LATEST").write_text(state.name)
    (state / "meta.pkl").write_bytes(pickle.dumps({"interaction_step": 8}))
    (state / "obs_rms.pkl").write_bytes(b"test-only")
    arrays = {
        "observations": np.zeros((rows, 3), np.float32),
        "actions": np.zeros((rows, 2), np.float32),
        "rewards": np.zeros(rows, np.float32),
        "terminateds": np.zeros(rows, np.bool_),
        "truncateds": np.zeros(rows, np.bool_),
        "next_observations": np.zeros((rows, 3), np.float32),
    }
    np.savez(state / "buffer.npz", **arrays)
    metadata = {
        "identity": {
            "experiment": "exp1",
            "architecture": architecture,
            "environment": "dog-run",
            "seed": 1,
        },
        "resolved_config": {
            "num_interaction_steps": 8,
            "probe": {"rounds": 10, "steps": 2000, "checks": 50},
        },
        "code": {"commit": "test"},
        "config_hash": "config-" + architecture,
        "protocol_hash": protocol,
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata))
    return run


class SharedDataTest(unittest.TestCase):
    def test_same_plan_and_sample_stream_for_both_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small = parent(root, "D2W512")
            large = parent(root, "D4W1536", rows=11)
            plan = make_plan(small, large, 17, 256)
            self.assertEqual(plan["status"], "data-plan-only")
            self.assertFalse(plan["launch_approved"])
            self.assertEqual((plan["small"]["rows"], plan["large"]["rows"]), (8, 11))
            a = paired_indices(17, 42, 256, 8, 11)
            b = paired_indices(17, 42, 256, 8, 11)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            self.assertEqual(np.bincount(a[0]).tolist(), [128, 128])
            self.assertTrue(np.all(a[1][a[0] == 0] < 8))
            self.assertTrue(np.all(a[1][a[0] == 1] < 11))

    def test_partial_or_mismatched_parents_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small = parent(root, "D2W512")
            large = parent(root, "D4W1536", complete=False)
            with self.assertRaisesRegex(ValueError, "missing DONE"):
                make_plan(small, large, 17, 256)
            (large / "DONE").write_text(json.dumps({"interaction_step": 7}))
            with self.assertRaisesRegex(ValueError, "final steps differ"):
                make_plan(small, large, 17, 256)
            (large / "DONE").write_text(json.dumps({"interaction_step": 8}))
            metadata_path = large / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["protocol_hash"] = "different"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "different training protocols"):
                make_plan(small, large, 17, 256)
            metadata["protocol_hash"] = "same"
            metadata["resolved_config"]["probe"]["steps"] = 1000
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "ten-target"):
                make_plan(small, large, 17, 256)

    def test_invalid_sampler_rejected(self):
        with self.assertRaises(ValueError):
            paired_indices(17, 0, 255, 8, 8)
        with self.assertRaises(ValueError):
            paired_indices(17, 0, 256, 0, 8)


if __name__ == "__main__":
    unittest.main()
