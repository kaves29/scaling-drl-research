"""Method C production fitting stream and continuous AdamW regression."""

import unittest
from dataclasses import replace
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.exp12 import probe
from experiments.exp1 import compose_config


class LinearCritic(nn.Module):
    @nn.compact
    def __call__(self, observations, actions):
        return nn.Dense(1)(jnp.concatenate((observations, actions), axis=1))


class MethodCProductionTest(unittest.TestCase):
    def test_canonical_configuration_changes_only_fitting_length(self):
        cfg = compose_config(str(Path(__file__).resolve().parents[1] / "configs"),
                             "base_exp12", ["env_name=hopper-hop", "env=dmc_medium"])
        self.assertEqual((cfg.probe.rounds, cfg.probe.steps, cfg.probe.pool_size,
                          cfg.probe.batch_size, cfg.trigger.null_threshold,
                          cfg.trigger.confidence, cfg.trigger.consecutive_checks),
                         (5, 2000, 25_600, 256, 0, .95, 2))

    def test_five_targets_and_first_stage_indices_are_preserved(self):
        net = LinearCritic()
        obs = jnp.arange(32, dtype=jnp.float32).reshape(16, 2) / 17
        act = jnp.zeros((16, 1), dtype=jnp.float32)
        cfg = probe.ProbeConfig(5, 2000, 20, 16, 4, 1e5, 8)
        old = replace(cfg, steps=1000)
        params = net.init(jax.random.PRNGKey(123), obs[:1], act[:1])["params"]
        keys = [jax.random.split(probe.probe_key(990, 1, r))[0] for r in range(5)]
        self.assertEqual(len({tuple(np.asarray(k)) for k in keys}), 5)
        targets = [np.asarray(probe._base_targets(net, k, obs, act, 8, 1e5)) for k in keys]
        self.assertTrue(all(not np.array_equal(targets[i], targets[j])
                            for i in range(5) for j in range(i + 1, 5)))

        from unittest import mock
        indices = []
        real_fit = probe._fit

        def capture(*args, **kwargs):
            indices.append(np.asarray(args[6]))
            return real_fit(*args, **kwargs)

        with mock.patch.object(probe, "_fit", capture):
            result_c = probe.probe_round(net, {"current": (net, params), "fresh": (net, params)},
                                         optax.adamw(.01), obs, act, probe.probe_key(990, 1, 0), cfg)
            probe.probe_round(net, {"current": (net, params)}, optax.adamw(.01),
                              obs, act, probe.probe_key(990, 1, 0), old)
        self.assertEqual([x.shape for x in indices], [(2000, 4), (2000, 4), (1000, 4)])
        np.testing.assert_array_equal(indices[0], indices[1])
        np.testing.assert_array_equal(indices[0][:1000], indices[2])
        self.assertEqual(result_c["current"]["losses"].shape, (2000,))

    def test_one_scan_matches_two_stages_with_live_adamw_state(self):
        net = LinearCritic()
        obs = jnp.arange(32, dtype=jnp.float32).reshape(16, 2) / 17
        act = jnp.zeros((16, 1), dtype=jnp.float32)
        params = net.init(jax.random.PRNGKey(3), obs[:1], act[:1])["params"]
        base = jnp.sin(jnp.arange(16, dtype=jnp.float32))
        tx = optax.adamw(.01, weight_decay=.001)
        idx_key = jax.random.PRNGKey(4)
        first = jax.random.randint(idx_key, (1000, 4), 0, 16)
        second = jax.random.randint(jax.random.fold_in(idx_key, 1), (1000, 4), 0, 16)
        offset = probe._mean_prediction(net, params, obs, act, 8)
        full_losses, full_mse, _ = probe._fit(net, tx, params, obs, act, base,
                                              jnp.concatenate((first, second)), 8, offset)

        def step(carry, ids):
            p, state = carry
            def loss(q):
                pred = net.apply({"params": q}, obs[ids], act[ids]).reshape(-1)
                return jnp.mean((pred - (offset + base[ids])) ** 2)
            value, gradient = jax.value_and_grad(loss)(p)
            updates, state = tx.update(gradient, state, p)
            return (optax.apply_updates(p, updates), state), value

        middle, a = jax.lax.scan(step, (params, tx.init(params)), first)
        end, b = jax.lax.scan(step, middle, second)
        uninterrupted, _ = jax.lax.scan(step, (params, tx.init(params)),
                                        jnp.concatenate((first, second)))
        for staged, single in zip(jax.tree_util.tree_leaves(end),
                                  jax.tree_util.tree_leaves(uninterrupted)):
            np.testing.assert_array_equal(np.asarray(staged), np.asarray(single))
        np.testing.assert_array_equal(np.asarray(full_losses), np.asarray(jnp.concatenate((a, b))))
        prediction = probe._chunked(lambda o, x: net.apply({"params": end[0]}, o, x).reshape(-1),
                                    obs, act, 8)
        np.testing.assert_allclose(np.asarray(full_mse),
                                   np.asarray(jnp.mean((prediction - (offset + base)) ** 2)),
                                   rtol=0, atol=np.finfo(np.float32).eps)
        self.assertEqual(int(middle[1][0].count), 1000)
        self.assertEqual(int(end[1][0].count), 2000)


if __name__ == "__main__":
    unittest.main()
