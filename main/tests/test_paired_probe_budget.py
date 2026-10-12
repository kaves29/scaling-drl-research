"""Focused CPU equivalence check for the exploratory staged optimizer."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.exp12.probe import _fit
from scripts.sci_investigation.paired_probe_budget import staged_fit


class _LinearCritic:
    def apply(self, variables, obs, act):
        return obs @ variables["params"]["w"] + variables["params"]["bias"]


class PairedProbeBudgetTest(unittest.TestCase):
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
