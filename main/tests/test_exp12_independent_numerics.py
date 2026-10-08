"""Independent CPU equations for fitting and entropy; fixture tolerances are not gates."""

import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.exp12 import probe
from scale_rl.agents.sac.sac_network import SACTemperature
from scale_rl.agents.sac.sac_update import update_temperature, update_target_network
from scale_rl.networks.policies import NormalTanhPolicy
from scale_rl.networks.trainer import Trainer


class AffineCritic(nn.Module):
    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate((observations, actions), axis=1)
        w = self.param("w", nn.initializers.zeros, (x.shape[1],))
        b = self.param("b", nn.initializers.zeros, ())
        return (x @ w + b)[:, None]


def reference_adamw_fit(x, base, indices, weights, bias, lr, wd):
    """Analytic gradients and explicit bias-corrected AdamW; no production helpers."""
    p = np.r_[weights, bias].astype(np.float64)
    design = np.column_stack((np.asarray(x, np.float64), np.ones(len(x))))
    offset = float((design @ p).mean())
    target = np.asarray(base, np.float64) + offset
    m, v = np.zeros_like(p), np.zeros_like(p)
    losses = []
    for t, idx in enumerate(indices, 1):
        errors = design[idx] @ p - target[idx]
        losses.append(float(np.mean(errors**2)))
        gradient = 2 * design[idx].T @ errors / len(idx)
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient**2
        p -= lr * ((m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8) + wd * p)
    return np.asarray(losses), float(np.mean((design @ p - target) ** 2)), offset


class IndependentFittingTest(unittest.TestCase):
    def test_every_minibatch_loss_final_full_pool_and_offset_match_analytic_adamw(self):
        rng = np.random.default_rng(991)
        x = rng.normal(size=(24, 5)).astype(np.float32)
        base = np.sin(x[:, 0] * 1.3).astype(np.float32)
        indices = rng.integers(0, 24, (25, 8), dtype=np.int32)
        weights = np.array([0.2, -0.3, 0.1, 0.4, -0.2], np.float32)
        params = {"w": jnp.asarray(weights), "b": jnp.asarray(0.7, jnp.float32)}
        original = jax.tree.map(np.asarray, params)
        expected = reference_adamw_fit(x, base, indices, weights, 0.7, 1e-3, 0.01)
        for chunk in (4, 8, 24):
            args = (
                AffineCritic(),
                optax.adamw(1e-3, weight_decay=0.01),
                params,
                jnp.asarray(x[:, :3]),
                jnp.asarray(x[:, 3:]),
                jnp.asarray(base),
                jnp.asarray(indices),
                chunk,
            )
            actual = probe._fit(*args)
            with jax.disable_jit():
                eager = probe._fit(*args)
            for a, e, ref in zip(actual, eager, expected):
                np.testing.assert_allclose(a, ref, rtol=3e-6, atol=2e-6)
                np.testing.assert_allclose(a, e, rtol=3e-6, atol=2e-6)
        for key in params:
            np.testing.assert_array_equal(params[key], original[key])

    def test_offset_is_fixed_prefit_mean_not_a_learned_or_recentered_target(self):
        obs = jnp.arange(12, dtype=jnp.float32).reshape(6, 2) / 10
        act = jnp.ones((6, 1))
        params = {"w": jnp.asarray([0.1, 0.2, -0.3]), "b": jnp.asarray(4.0)}
        base = jnp.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        idx = jnp.tile(jnp.arange(6), (3, 1))
        losses, final, offset = probe._fit(
            AffineCritic(), optax.sgd(0), params, obs, act, base, idx, 3
        )
        q = np.c_[obs, act] @ np.array([0.1, 0.2, -0.3]) + 4
        target = q.mean() + np.asarray(base)
        self.assertAlmostEqual(float(offset), float(q.mean()), places=6)
        np.testing.assert_allclose(final, np.mean((q - target) ** 2), atol=2e-7)
        np.testing.assert_allclose(losses, float(final), atol=2e-7)

    def test_probe_streams_are_distinct_and_reconstructible(self):
        keys = set()
        for seed in range(1, 6):
            for check in range(21):
                for round_index in range(5):
                    key = np.asarray(
                        probe.probe_key(seed, check, round_index)
                    ).tobytes()
                    self.assertNotIn(key, keys)
                    keys.add(key)
                    np.testing.assert_array_equal(
                        probe.probe_rng(seed, check, round_index).integers(0, 5000, 8),
                        probe.probe_rng(seed, check, round_index).integers(0, 5000, 8),
                    )
        self.assertEqual(len(keys), 525)


class IndependentEntropyTest(unittest.TestCase):
    def test_literal_positive_and_negative_entropy_targets_have_opposite_updates(self):
        net = SACTemperature(0.01)
        tx = optax.sgd(1e-3)
        temperature = Trainer.create(net, {"rngs": jax.random.PRNGKey(0)}, tx)
        initial = float(temperature.params["log_temp"])
        alpha = float(temperature())
        for target in (-1.0, 1.0):
            updated, info = update_temperature(temperature, jnp.array(0.0), target)
            gradient = alpha * (0 - target)
            self.assertAlmostEqual(
                float(updated.params["log_temp"]), initial - 1e-3 * gradient, places=6
            )
            self.assertAlmostEqual(
                float(info["train/temperature_loss"]), gradient, places=7
            )
            self.assertEqual(
                np.sign(float(updated.params["log_temp"]) - initial), np.sign(target)
            )

    def test_tanh_log_density_matches_independent_change_of_variables(self):
        net = NormalTanhPolicy(2)
        inputs = jnp.asarray([[0.2, -0.3], [0.1, 0.7]])
        variables = net.init(jax.random.PRNGKey(11), inputs)
        # Set an ordinary-width Gaussian using the unchanged [-10, 2] mapping.
        variables["params"]["Dense_1"]["kernel"] = jnp.zeros((2, 2))
        variables["params"]["Dense_1"]["bias"] = jnp.full((2,), np.arctanh(2 / 3))
        dist = net.apply(variables, inputs)
        actions = np.asarray(dist.sample(seed=jax.random.PRNGKey(12)))
        z = np.arctanh(actions.astype(np.float64))
        mu = np.asarray(dist.distribution.mean(), np.float64)
        sd = np.asarray(dist.distribution.stddev(), np.float64)
        log_density = np.sum(
            -0.5 * ((z - mu) / sd) ** 2
            - np.log(sd)
            - 0.5 * np.log(2 * np.pi)
            - np.log1p(-actions.astype(np.float64) ** 2),
            axis=1,
        )
        np.testing.assert_allclose(
            dist.log_prob(jnp.asarray(actions)), log_density, atol=2e-6, rtol=2e-6
        )
        self.assertAlmostEqual(
            -float(np.asarray(dist.log_prob(jnp.asarray(actions))).mean()),
            -float(log_density.mean()),
            places=5,
        )

    def test_polyak_update_matches_independent_equation_and_preserves_optimizer(self):
        net = SACTemperature(0.01)
        online = Trainer.create(net, {"rngs": jax.random.PRNGKey(0)}, optax.adamw(1e-4))
        target = online.replace(params={"log_temp": jnp.array(2.0)})
        updated, _ = update_target_network(online, target, 0.005)
        expected = 0.005 * np.asarray(online.params["log_temp"]) + 0.995 * 2
        self.assertAlmostEqual(
            float(updated.params["log_temp"]), float(expected), places=6
        )
        self.assertIs(updated.opt_state, target.opt_state)


if __name__ == "__main__":
    unittest.main()
