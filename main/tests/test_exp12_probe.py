"""Phase 2: the plasticity probe (Lyle et al. 2023, adapted).

Known-answer checks on the probe arithmetic, the pairing invariant (current and
fresh critics see identical inputs, target function and minibatch order), the
P/L sign convention, and that probes never change the training trajectory.
"""

import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp12_helpers import CONFIG_PATH, compose, patch_wandb, state_differences, tiny_overrides  # noqa: E402
from experiments.exp12 import probe as probe_module  # noqa: E402
from experiments.exp12.probe import ProbeConfig, iqm, probe_round, run_probe, summarize  # noqa: E402
from experiments.exp12.state import latest_state_dir  # noqa: E402

CFG = ProbeConfig(rounds=5, steps=50, checks=20, pool_size=64, batch_size=16, target_scale=1e5, eval_chunk=32)


class ConstantCritic(nn.Module):
    """Q(s, a) = c for every input: a critic whose fit we can compute by hand."""

    @nn.compact
    def __call__(self, observations, actions):
        c = self.param("c", nn.initializers.zeros, (1,))
        return jnp.broadcast_to(c, (observations.shape[0], 1))


class LinearCritic(nn.Module):
    @nn.compact
    def __call__(self, observations, actions):
        return nn.Dense(1)(jnp.concatenate([observations, actions], axis=1))


def _pool(n=64, obs_dim=3, act_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(n, obs_dim)), jnp.float32), jnp.asarray(rng.uniform(-1, 1, (n, act_dim)), jnp.float32)


class ProbeArithmeticTest(unittest.TestCase):
    def test_iqm_matches_rliable_definition(self):
        self.assertAlmostEqual(iqm([1, 2, 3, 4, 100]), 3.0)  # n=5 trims one from each end
        self.assertAlmostEqual(iqm([4, 4, 4, 4]), 4.0)

    def test_frozen_constant_critic_has_known_score(self):
        # lr=0: nothing is learned and a = c, so the final pool loss is mean(base^2) and
        # P = Var(base) - mean(base^2) = -mean(base)^2.
        obs, act = _pool()
        net = ConstantCritic()
        params = {"c": jnp.array([3.7])}
        out = probe_round(LinearCritic(), {"x": (net, params)}, optax.sgd(0.0), obs, act, jax.random.PRNGKey(0), CFG)["x"]
        base = probe_module._base_targets(LinearCritic(), jax.random.split(jax.random.PRNGKey(0))[0], obs, act, 32, 1e5)
        self.assertAlmostEqual(float(out["offset"]), 3.7, places=5)
        np.testing.assert_allclose(float(out["b"]), float(jnp.var(base)), rtol=1e-6)
        np.testing.assert_allclose(float(out["score"]), -float(jnp.mean(base)) ** 2, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(float(out["final_loss"]), float(jnp.mean(base ** 2)), rtol=1e-5)

    def test_a_critic_that_learns_scores_higher_than_one_that_cannot(self):
        obs, act = _pool()
        net = LinearCritic()
        params = net.init(jax.random.PRNGKey(1), obs[:1], act[:1])["params"]
        key = jax.random.PRNGKey(2)
        learns = probe_round(net, {"x": (net, params)}, optax.adam(1e-2), obs, act, key, CFG)["x"]
        frozen = probe_round(net, {"x": (net, params)}, optax.sgd(0.0), obs, act, key, CFG)["x"]
        self.assertGreater(float(learns["score"]), float(frozen["score"]))
        self.assertLess(float(learns["losses"][-1]), float(learns["losses"][0]))
        self.assertLessEqual(float(learns["score"]), float(learns["b"]) + 1e-6)  # P <= b since loss >= 0

    def test_sign_convention_plasticity_loss_positive_when_current_is_worse(self):
        fresh = {"score": np.array([0.40, 0.42, 0.41, 0.39, 0.40])}
        worse = {"score": np.array([0.10, 0.12, 0.11, 0.13, 0.10])}
        s = summarize({"current": worse, "fresh": fresh})
        self.assertTrue(np.all(s["loss_rounds"] > 0))
        self.assertGreater(s["loss_iqm"], 0)
        s = summarize({"current": fresh, "fresh": worse})
        self.assertLess(s["loss_iqm"], 0)

    def test_non_finite_round_marks_check_invalid(self):
        s = summarize({"current": {"score": np.array([0.1, np.nan, 0.1, 0.1, 0.1])},
                       "fresh": {"score": np.array([0.4] * 5)}})
        self.assertFalse(s["valid"])
        self.assertTrue(np.isnan(s["loss_iqm"]))


class ProbePairingTest(unittest.TestCase):
    """Current and fresh critics are fitted on identical inputs, targets and minibatch order."""

    def _agent_and_buffer(self):
        from experiments.exp12.trainer import Exp12Trainer

        self.patchers = patch_wandb()
        self.tmp = tempfile.mkdtemp()
        cfg = compose(tiny_overrides(steps=200))
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        t = Exp12Trainer(cfg, self.tmp)
        t.start()
        t.train(60)
        return t

    def tearDown(self):
        for p in getattr(self, "patchers", []):
            p.stop()
        shutil.rmtree(getattr(self, "tmp", ""), ignore_errors=True)

    def test_identical_inputs_targets_and_minibatches(self):
        t = self._agent_and_buffer()
        critic = t._sac_agent.critic
        seen = []
        real_fit = probe_module._fit

        def spy(network_def, tx, params, obs, act, base, idx, chunk):
            seen.append((np.asarray(obs), np.asarray(act), np.asarray(base), np.asarray(idx)))
            return real_fit(network_def, tx, params, obs, act, base, idx, chunk)

        fresh = critic.network_def.init(jax.random.PRNGKey(9), jnp.zeros((1, 15)), jnp.zeros((1, 4)))["params"]
        with mock.patch.object(probe_module, "_fit", spy):
            result = run_probe(t.agent, t.buffer, critic.network_def,
                               {"current": (critic.network_def, critic.params), "fresh": (critic.network_def, fresh)},
                               optax.adamw(1e-4, weight_decay=1e-2), 1, 3, CFG)
        self.assertEqual(len(seen), 2 * CFG.rounds)
        for r in range(CFG.rounds):
            cur, fr = seen[2 * r], seen[2 * r + 1]
            for x, y in zip(cur, fr):
                np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(seen[0][2], seen[2][2]), "rounds should draw different target functions")
        self.assertEqual(result["current"]["losses"].shape, (CFG.rounds, CFG.steps))
        self.assertEqual(result["fresh"]["final_loss"].shape, (CFG.rounds,))
        # Same critic under two names gives bit-identical results: nothing but the params differs.
        same = run_probe(t.agent, t.buffer, critic.network_def,
                         {"a": (critic.network_def, critic.params), "b": (critic.network_def, critic.params)},
                         optax.adamw(1e-4, weight_decay=1e-2), 1, 3, CFG)
        np.testing.assert_array_equal(same["a"]["losses"], same["b"]["losses"])
        # Reproducible from (seed, check index) alone.
        again = run_probe(t.agent, t.buffer, critic.network_def,
                          {"current": (critic.network_def, critic.params), "fresh": (critic.network_def, fresh)},
                          optax.adamw(1e-4, weight_decay=1e-2), 1, 3, CFG)
        np.testing.assert_array_equal(again["current"]["losses"], result["current"]["losses"])

    def test_per_critic_offset_is_each_critics_own_mean(self):
        obs, act = _pool()
        net = LinearCritic()
        params = net.init(jax.random.PRNGKey(1), obs[:1], act[:1])["params"]
        shifted = {"Dense_0": {**params["Dense_0"], "bias": params["Dense_0"]["bias"] + 5.0}}
        out = probe_round(net, {"a": (net, params), "b": (net, shifted)}, optax.adam(1e-2), obs, act,
                          jax.random.PRNGKey(3), CFG)
        self.assertAlmostEqual(float(out["b"]["offset"]) - float(out["a"]["offset"]), 5.0, places=4)
        np.testing.assert_allclose(out["a"]["losses"], out["b"]["losses"], rtol=1e-4, atol=1e-6)
        self.assertEqual(float(out["a"]["b"]), float(out["b"]["b"]))


class ProbeDoesNotChangeTrainingTest(unittest.TestCase):
    """exp1 with probes on has a bit-identical training trajectory to probes off."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patchers = patch_wandb()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, name, probe_on):
        from experiments import exp1

        run_dir = os.path.join(self.tmp, name)
        exp1.run({"experiment": "exp1", "config_path": CONFIG_PATH, "config_name": "base_exp12",
                  "overrides": tiny_overrides(steps=400, extra=[f"probe.enabled={str(probe_on).lower()}"]),
                  "checkpoint_dir": run_dir, "checkpoint_interval": 10**9, "checkpoint_start_frac": 0.0})
        return run_dir, latest_state_dir(Path(run_dir) / "state")

    def test_probes_on_equals_probes_off(self):
        off_dir, off = self._run("off", False)
        on_dir, on = self._run("on", True)
        self.assertEqual(state_differences(off, on, ignore_meta=("wandb_run_id", "extra_state")), [])
        records = __import__("pickle").load(open(on / "meta.pkl", "rb"))["extra_state"]["probe_records"]
        self.assertEqual([r["check_index"] for r in records], list(range(0, 21)))
        self.assertTrue(all(r["valid"] for r in records[1:]))
        self.assertTrue((Path(on_dir) / "fresh_critic").exists())
        self.assertEqual(len(list((Path(on_dir) / "probes").glob("check_*.npz"))), 21)

    def test_breaking_probe_isolation_is_detected(self):
        off_dir, off = self._run("off", False)
        real_sample_pool = probe_module.sample_pool

        def leaky_sample_pool(agent, buffer, rng, pool_size):
            np.random.randint(0, 10)  # consumes the global stream the replay sampler uses
            return real_sample_pool(agent, buffer, rng, pool_size)

        with mock.patch.object(probe_module, "sample_pool", leaky_sample_pool):
            _, on = self._run("leaky_rng", True)
        self.assertTrue(any(d.startswith("agent:") for d in
                            state_differences(off, on, ignore_meta=("wandb_run_id", "extra_state"))))

        def mutating_sample_pool(agent, buffer, rng, pool_size):
            obs, act = real_sample_pool(agent, buffer, rng, pool_size)
            agent.obs_rms.update(np.asarray(obs))  # mutates normaliser statistics
            return obs, act

        with mock.patch.object(probe_module, "sample_pool", mutating_sample_pool):
            _, on = self._run("mutating_rms", True)
        self.assertIn("obs_rms:mean", state_differences(off, on, ignore_meta=("wandb_run_id", "extra_state")))


if __name__ == "__main__":
    unittest.main()
