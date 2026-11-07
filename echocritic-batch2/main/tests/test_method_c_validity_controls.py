"""Focused CPU checks for the separate Method C construct-validation harness."""

import json
import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from omegaconf import OmegaConf

from experiments.exp12.probe import ProbeConfig, _base_targets, probe_key, probe_rng, probe_round
from scripts.method_c_validity_controls import (
    PILOT_COMMIT,
    diagnostic_fit,
    exact_binomial_interval,
    followup_indices,
    init_fresh,
    paired_decision,
    pool,
    resolve_state,
    source_metadata,
)


class TinyCritic(nn.Module):
    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate((observations, actions), axis=-1)
        x = nn.tanh(nn.Dense(8)(x))
        return nn.Dense(1)(x)


def test_first_2000_indices_are_exact_method_c_prefix():
    key = jax.random.PRNGKey(17)
    first = jax.random.randint(key, (1000, 4), 0, 16)
    second = jax.random.randint(jax.random.fold_in(key, 1), (1000, 4), 0, 16)
    chunks_4000 = followup_indices(key, 4, 16, 4000)
    chunks_8000 = followup_indices(key, 4, 16, 8000)
    np.testing.assert_array_equal(chunks_4000[0], first)
    np.testing.assert_array_equal(chunks_4000[1], second)
    np.testing.assert_array_equal(jnp.concatenate(chunks_4000), jnp.concatenate(chunks_8000)[:4000])
    assert [len(chunk) for chunk in chunks_8000] == [1000, 1000, 2000, 4000]
    with pytest.raises(ValueError):
        followup_indices(key, 4, 16, 3000)


def test_diagnostic_scan_preserves_fit_and_adamw_state():
    network = TinyCritic()
    obs = jnp.arange(32, dtype=jnp.float32).reshape(16, 2) / 20
    act = jnp.arange(16, dtype=jnp.float32).reshape(16, 1) / 10
    params = init_fresh(network, jax.random.PRNGKey(8), obs, act)
    key = probe_key(1, 20, 0)
    target_key, idx_key = jax.random.split(key)
    cfg = ProbeConfig(5, 2000, 20, 16, 4, 100.0, 8)
    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    official = probe_round(network, {"fresh": (network, params)}, tx, obs, act, key, cfg)["fresh"]
    base = _base_targets(network, target_key, obs, act, cfg.eval_chunk, cfg.target_scale)
    first_2k = followup_indices(idx_key, cfg.batch_size, cfg.pool_size, 2000)
    diagnostic = diagnostic_fit(network, tx, params, obs, act, base, first_2k, cfg.eval_chunk)
    np.testing.assert_allclose(np.concatenate([x["losses"] for x in diagnostic]),
                               official["losses"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(diagnostic[-1]["pool_mse"], official["final_loss"],
                               rtol=1e-5, atol=1e-6)
    assert diagnostic[0]["optimizer_step"] == 1000
    assert diagnostic[1]["optimizer_step"] == 2000
    extended = diagnostic_fit(network, tx, params, obs, act, base,
                              followup_indices(idx_key, 4, 16, 4000), 8)
    np.testing.assert_array_equal(extended[0]["losses"], diagnostic[0]["losses"])
    np.testing.assert_array_equal(extended[1]["losses"], diagnostic[1]["losses"])
    assert extended[-1]["optimizer_step"] == 4000
    assert np.isfinite(np.concatenate([x["gradient_norms"] for x in extended])).all()


def test_fresh_pair_is_independent_and_pool_draws_match_official_stream():
    network = TinyCritic()
    obs = jnp.ones((16, 2), dtype=jnp.float32)
    act = jnp.zeros((16, 1), dtype=jnp.float32)
    a_key, b_key = jax.random.split(jax.random.PRNGKey(81))
    a, b = init_fresh(network, a_key, obs, act), init_fresh(network, b_key, obs, act)
    leaves_a, leaves_b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert any(not np.array_equal(x, y) for x, y in zip(leaves_a, leaves_b))
    raw_obs = np.arange(32, dtype=np.float32).reshape(16, 2)
    raw_act = np.arange(16, dtype=np.float32).reshape(16, 1)
    normalizer = {"mean": np.array([1.0, 2.0]), "var": np.array([4.0, 9.0]),
                  "epsilon": 1e-8}
    sampled_obs, sampled_act = pool(raw_obs, raw_act, normalizer, 92, 20, 3, 16)
    indices = probe_rng(92, 20, 3).integers(0, 16, size=16)
    np.testing.assert_array_equal(sampled_act, raw_act[indices])
    np.testing.assert_allclose(sampled_obs,
                               (raw_obs[indices] - normalizer["mean"]) /
                               np.sqrt(normalizer["var"] + normalizer["epsilon"]))


def test_pair_order_is_recorded_without_doubling_sample_size():
    cfg = OmegaConf.create({"trigger": {"resamples": 10000, "confidence": 0.95,
                                         "null_threshold": 0.0}})
    result = {"fresh_a": {"score": np.array([0.1, 0.12, 0.11, 0.13, 0.14])},
              "fresh_b": {"score": np.array([0.2, 0.21, 0.22, 0.23, 0.24])}}
    forward = paired_decision(result, "fresh_a", "fresh_b", 99, 20, cfg)
    backward = paired_decision(result, "fresh_b", "fresh_a", 99, 20, cfg)
    assert forward["fires"] and not backward["fires"]
    np.testing.assert_array_equal(forward["loss_rounds"], -np.array(backward["loss_rounds"]))
    assert exact_binomial_interval(0, 4)[0] == 0.0
    assert exact_binomial_interval(4, 4)[1] == 1.0


def test_foreign_or_dirty_parent_is_rejected(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    meta = {"code": {"commit": "a" * 40, "dirty": False},
            "identity": {"run_role": "confirmatory"}}
    (run / "run_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="expected source commit"):
        source_metadata(run)
    meta["code"] = {"commit": PILOT_COMMIT, "dirty": True}
    (run / "run_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="dirty"):
        source_metadata(run)


def test_extra_crossover_arm_does_not_change_official_probe_pair():
    network = TinyCritic()
    obs = jnp.arange(32, dtype=jnp.float32).reshape(16, 2) / 20
    act = jnp.arange(16, dtype=jnp.float32).reshape(16, 1) / 10
    current = init_fresh(network, jax.random.PRNGKey(3), obs, act)
    fresh = init_fresh(network, jax.random.PRNGKey(4), obs, act)
    other = init_fresh(network, jax.random.PRNGKey(5), obs, act)
    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    cfg = ProbeConfig(5, 20, 20, 16, 4, 100.0, 8)
    key = probe_key(1, 2, 0)
    original = probe_round(network, {"current": (network, current),
                                     "fresh": (network, fresh)}, tx, obs, act, key, cfg)
    extended = probe_round(network, {"current": (network, current),
                                     "other": (network, other),
                                     "fresh": (network, fresh)}, tx, obs, act, key, cfg)
    for arm in ("current", "fresh"):
        for field in original[arm]:
            np.testing.assert_array_equal(original[arm][field], extended[arm][field])


def test_checkpoint_point_must_match_50k_or_500k(tmp_path):
    root = tmp_path / "fork" / "state"
    state = root / "step_000049999_bad"
    state.mkdir(parents=True)
    (root / "LATEST").write_text(state.name)
    with (state / "meta.pkl").open("wb") as stream:
        pickle.dump({"interaction_step": 49_999}, stream)
    with pytest.raises(ValueError, match="expected interaction step 50000"):
        resolve_state(tmp_path, "early")
