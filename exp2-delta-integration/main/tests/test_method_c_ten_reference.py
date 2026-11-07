"""Independent CPU oracles for the unactivated ten-function proposal."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from omegaconf import OmegaConf

from experiments.exp12.probe import ProbeConfig, probe_key, probe_round
from experiments.exp12.trigger import BOOT_STREAM
from scripts import method_c_validity_controls as controls
from scripts.method_c_ten_reference import bootstrap_iqm10, empirical_iqm10, target_round_key


def empirical_quantile_integral(x):
    """Generic overlap calculation, intentionally unlike production weights."""
    ordered = np.sort(np.asarray(x, dtype=float))
    n = len(ordered)
    return sum(value * max(0, min((i + 1) / n, .75) - max(i / n, .25))
               for i, value in enumerate(ordered)) / .5


class TinyCritic(nn.Module):
    @nn.compact
    def __call__(self, observations, actions):
        return nn.Dense(1)(jnp.concatenate([observations, actions], axis=-1))


def test_iqm10_fractional_boundaries_and_nonfinite_rejection():
    x = np.arange(1., 11.)
    assert empirical_iqm10(x) == pytest.approx(5.5)
    assert empirical_iqm10(np.array([0., 0., 0., 0., 0., 0., 0., 0., 10., 100.])) == 0
    for seed in range(10):
        sample = np.random.default_rng(seed).normal(size=10)
        assert empirical_iqm10(sample) == pytest.approx(empirical_quantile_integral(sample))
    for bad in (np.zeros(5), np.r_[np.ones(9), np.nan]):
        with pytest.raises(ValueError):
            empirical_iqm10(bad)


def test_bootstrap_uses_paired_ten_function_units_and_linear_percentiles():
    x = np.array([.12, -.02, .03, .2, -.04, .09, .1, .21, -.08, .17])
    observed = bootstrap_iqm10(x, 7, 13, resamples=1000)
    rng = np.random.default_rng([7, BOOT_STREAM, 13])
    samples = x[rng.integers(0, 10, size=(1000, 10))]
    oracle = np.array([empirical_quantile_integral(sample) for sample in samples])
    low, high = np.quantile(oracle, [.025, .975], method="linear")
    assert observed.loss_iqm == pytest.approx(empirical_quantile_integral(x))
    assert (observed.ci_low, observed.ci_high) == pytest.approx((low, high))
    assert observed == bootstrap_iqm10(x, 7, 13, resamples=1000)
    assert bootstrap_iqm10(np.ones(10), 7, 13).fires
    assert not bootstrap_iqm10(np.zeros(10), 7, 13).fires


def test_first_five_teacher_keys_are_original_and_ten_are_distinct():
    keys = [target_round_key(17, 3, 0, r) for r in range(10)]
    for r in range(5):
        np.testing.assert_array_equal(keys[r], probe_key(17, 3, r))
    later = [target_round_key(17, 3, 1, r) for r in range(10)]
    assert len({tuple(np.asarray(k)) for k in keys + later}) == 20
    with pytest.raises(ValueError):
        target_round_key(17, 3, 0, 10)


def test_shared_offset_gives_identical_full_targets_without_changing_method_c():
    net = TinyCritic()
    obs = jnp.arange(32, dtype=jnp.float32).reshape(16, 2) / 16
    act = jnp.zeros((16, 1), dtype=jnp.float32)
    current = net.init(jax.random.PRNGKey(1), obs[:1], act[:1])["params"]
    fresh = net.init(jax.random.PRNGKey(2), obs[:1], act[:1])["params"]
    cfg = ProbeConfig(5, 20, 20, 16, 4, 100_000., 8)
    critics = {"current": (net, current), "fresh": (net, fresh)}
    tx = optax.adamw(1e-3)
    own = probe_round(net, critics, tx, obs, act, probe_key(1, 2, 0), cfg)
    assert float(own["current"]["offset"]) != float(own["fresh"]["offset"])
    for anchor in ("current", "fresh", "mean"):
        shared = probe_round(net, critics, tx, obs, act, probe_key(1, 2, 0), cfg,
                             shared_offset=anchor)
        np.testing.assert_array_equal(shared["current"]["offset"], shared["fresh"]["offset"])
        np.testing.assert_array_equal(own["current"]["b"], shared["fresh"]["b"])
        if anchor != "mean":
            np.testing.assert_array_equal(own[anchor]["losses"], shared[anchor]["losses"])


def test_five_ten_runner_reuses_original_five_and_keeps_sets_separate(tmp_path, monkeypatch):
    run = tmp_path / "pilot"
    (run / "probes").mkdir(parents=True)

    def fields(round_index, arm, set_index=0):
        score = np.float32(round_index + set_index * 100 + (0.1 if arm == "fresh" else 0))
        return {"score": score, "losses": np.array([score, score + 1]),
                "final_loss": score + 2, "offset": score + 3, "b": np.float32(5)}

    original = {f"{arm}_{field}": np.stack([fields(r, arm)[field] for r in range(5)])
                for arm in ("current", "fresh") for field in fields(0, arm)}
    np.savez_compressed(run / "probes/check_02.npz", **original)
    monkeypatch.setattr(controls, "original_critic_def", lambda cfg: object())
    monkeypatch.setattr(controls, "critic_optimizer", lambda agent: object())
    monkeypatch.setattr(controls, "pool", lambda raw, act, norm, seed, check, round_index, size:
                        (round_index, None))
    monkeypatch.setattr(controls, "target_round_key", lambda seed, check, set_index, r:
                        (set_index, r))
    monkeypatch.setattr(controls, "probe_config", lambda cfg: object())
    monkeypatch.setattr(controls, "probe_round", lambda net, critics, tx, obs, act, key, cfg:
                        {arm: fields(key[1], arm, key[0]) for arm in critics})
    cfg = OmegaConf.create({"seed": 1, "agent": {}, "probe": {"pool_size": 16},
                            "trigger": {"resamples": 100, "confidence": .95,
                                        "null_threshold": 0.0}})
    out = tmp_path / "out"
    states = {"early": (tmp_path / "state", object(), None, None, None)}
    controls.run_five_ten(run, out, cfg, states, object(), 2)
    import json
    first = json.loads((out / "five_ten_early_set0/summary.json").read_text())
    second = json.loads((out / "five_ten_early_set1/summary.json").read_text())
    assert all(all(flags.values()) for flags in first["original_first_five_bitwise_replay"].values())
    assert first["target_round_indices"] == list(range(10))
    assert second["target_round_indices"] == list(range(10, 20))
    assert second["original_first_five_bitwise_replay"] is None
    assert len(first["five"]["loss_rounds"]) == 5
    assert len(first["ten"]["loss_rounds"]) == 10
