"""Focused CPU equivalence for the observational pilot probe path."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from audits.probe_discrepancy.adapter import Capture, fit_values, teacher_values
from experiments.exp12 import probe


@dataclass(frozen=True)
class TinyCritic:
    def init(self, key, observations, actions):
        width = observations.shape[-1] + actions.shape[-1]
        return {"params": {"w": jax.random.normal(key, (width,)), "b": jnp.zeros(())}}

    def apply(self, variables, observations, actions):
        x = jnp.concatenate((observations, actions), axis=-1)
        return (x @ variables["params"]["w"] + variables["params"]["b"])[..., None]


def test_original_target_and_continuous_fit_are_bitwise_equal():
    network = TinyCritic()
    obs = jnp.arange(16, dtype=jnp.float32).reshape(8, 2) / 10
    act = jnp.arange(8, dtype=jnp.float32).reshape(8, 1) / 20
    key = jax.random.PRNGKey(13)
    target_key, index_key = jax.random.split(key)
    base = probe._base_targets(network, target_key, obs, act, 4, 100000)
    observed_base, prescale, teacher = teacher_values(network, target_key, obs, act, 4, 100000)
    np.testing.assert_array_equal(observed_base, base)
    np.testing.assert_array_equal(jnp.sin(100000 * prescale), base)
    assert set(teacher) == {"w", "b"}

    first = jax.random.randint(index_key, (1000, 2), 0, 8)
    second = jax.random.randint(jax.random.fold_in(index_key, 1), (1000, 2), 0, 8)
    idx = jnp.concatenate((first, second), axis=0)
    params = network.init(jax.random.PRNGKey(7), obs[:1], act[:1])["params"]
    tx = optax.adamw(learning_rate=1e-4, weight_decay=0.01)
    losses, final, offset = probe._fit(network, tx, params, obs, act, base, idx, 4)
    traced_losses, traced_final, traced_offset, q, targets = fit_values(
        network, tx, params, obs, act, base, idx, 4
    )
    np.testing.assert_array_equal(traced_losses, losses)
    np.testing.assert_array_equal(traced_final, final)
    np.testing.assert_array_equal(traced_offset, offset)
    np.testing.assert_array_equal(targets, offset + base)
    np.testing.assert_array_equal(jnp.mean((q[-1] - (offset + base)) ** 2), final)
    assert q.shape == (5, 8)
    cfg = probe.ProbeConfig(rounds=10, steps=2000, checks=50, pool_size=8,
                            batch_size=2, target_scale=100000, eval_chunk=4)
    original_round = probe.probe_round(network, {"fresh": (network, params)}, tx,
                                       obs, act, key, cfg)["fresh"]
    for field, value in (("losses", traced_losses), ("final_loss", traced_final),
                         ("offset", traced_offset), ("b", jnp.var(base)),
                         ("score", jnp.var(base) - traced_final)):
        np.testing.assert_array_equal(original_round[field], value)


def test_selected_round_writes_literal_targets_and_full_pool_errors(tmp_path):
    class Buffer:
        _num_in_buffer = 8
        _max_length = 8
        _current_idx = 0
        _observations = np.arange(16, dtype=np.float32).reshape(8, 2) / 10
        _actions = np.arange(8, dtype=np.float32).reshape(8, 1) / 20

    class Agent:
        @staticmethod
        def _normalize(obs):
            return obs

    network = TinyCritic()
    params = network.init(jax.random.PRNGKey(7), jnp.zeros((1, 2)), jnp.zeros((1, 1)))["params"]
    tx = optax.adamw(learning_rate=1e-4, weight_decay=0.01)
    cfg = probe.ProbeConfig(rounds=10, steps=2000, checks=50, pool_size=8,
                            batch_size=2, target_scale=100000, eval_chunk=4)
    capture = Capture(tmp_path, {"retained_checks": [0]}, "0" * 40, "0" * 64)
    result = capture.run_probe(Agent(), Buffer(), network, {"fresh": (network, params)},
                               tx, 3, 0, cfg)
    assert result["fresh"]["losses"].shape == (10, 2000)
    with np.load(tmp_path / "diagnostic/check_00/round_00.npz") as data:
        q = data["fresh_full_pool_q_0_100_500_1000_2000"]
        target = data["fresh_literal_targets"]
        mse = data["fresh_full_pool_mse_0_100_500_1000_2000"]
        np.testing.assert_allclose(mse, np.mean((q - target) ** 2, axis=1))
        np.testing.assert_array_equal(data["fresh_losses_preupdate"], result["fresh"]["losses"][0])
