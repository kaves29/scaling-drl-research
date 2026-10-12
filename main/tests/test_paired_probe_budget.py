"""Focused CPU equivalence check for the exploratory staged optimizer."""

import unittest
import json
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.exp12.probe import _fit
from scripts.sci_investigation.paired_probe_budget import _save_state, _summary, staged_fit


class _LinearCritic:
    def apply(self, variables, obs, act):
        return obs @ variables["params"]["w"] + variables["params"]["bias"]


class PairedProbeBudgetTest(unittest.TestCase):
    def test_selected_optimizer_snapshot_is_complete_and_hashed(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            receipt = _save_state(path, ({"w": jnp.array([1., 2.])}, {"count": jnp.array(3)}))
            self.assertEqual(receipt["leaf_count"], 2)
            self.assertEqual(receipt["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with np.load(path) as archive:
                np.testing.assert_array_equal(archive["leaf_0000"], [1., 2.])
                np.testing.assert_array_equal(archive["leaf_0001"], 3)

    def test_partial_summary_keeps_three_null_rules_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"null_1000": {"fired_zero": True, "fired_pooled_p95": False},
                 "null_2000": {"fired_zero": False}},
                {"null_1000": {"fired_zero": False, "fired_pooled_p95": False},
                 "null_2000": {"fired_zero": True}},
            ]
            for i, row in enumerate(rows):
                (root / f"pair_{i:03d}.json").write_text(json.dumps(row))
            _summary(root, "null", 100, .0009080017606417335, time.monotonic())
            result = json.loads((root / "summary.json").read_text())
            self.assertFalse(result["complete"])
            self.assertEqual(result["completed"], 2)
            self.assertEqual([result["methods"][name]["fires"] for name in
                              ("A_zero_1000", "B_pooled_p95_1000", "C_zero_2000")], [1, 0, 1])

    def test_first_stage_matches_production_fit_and_continuation(self):
        network = _LinearCritic()
        tx = optax.adamw(learning_rate=.01, weight_decay=.001)
        params = {"w": jnp.array([.2, -.1]), "bias": jnp.array(.1)}
        obs = jnp.arange(16, dtype=jnp.float32).reshape(8, 2) / 10
        act = jnp.zeros((8, 1))
        base = jnp.sin(jnp.arange(8, dtype=jnp.float32))
        idx1 = jax.random.randint(jax.random.PRNGKey(3), (7, 3), 0, 8)
        idx2 = jax.random.randint(jax.random.PRNGKey(4), (5, 3), 0, 8)
        production_loss, production_final, production_offset = _fit(
            network, tx, params, obs, act, base, idx1, 4)
        first, second, mse1, mse2, offset = staged_fit(
            network, tx, params, obs, act, base, idx1, idx2, 4)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(production_loss))
        np.testing.assert_array_equal(np.asarray(mse1), np.asarray(production_final))
        np.testing.assert_array_equal(np.asarray(offset), np.asarray(production_offset))
        full_loss, full_final, _ = _fit(network, tx, params, obs, act, base,
                                       jnp.concatenate((idx1, idx2)), 4)
        np.testing.assert_array_equal(np.asarray(jnp.concatenate((first, second))),
                                      np.asarray(full_loss))
        np.testing.assert_array_equal(np.asarray(mse2), np.asarray(full_final))
        self.assertEqual(second.shape, (5,))
        self.assertTrue(np.isfinite(np.asarray(mse2)))


if __name__ == "__main__":
    unittest.main()
