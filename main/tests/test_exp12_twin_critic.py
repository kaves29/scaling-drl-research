"""Twin critics (clipped double Q, HumanoidBench only; amendment (z)): defects found by the audit and the
SACClippedDoubleCritic path end to end, on tiny networks."""

import sys
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp12_helpers import precision  # noqa: E402
from scale_rl.agents.sac import sac_network, sac_update  # noqa: E402
from scale_rl.networks.metrics import flatten_dict, get_dormant_ratio, get_feature_norm, get_srank  # noqa: E402
from test_exp12_diagnostics import _agent, _batches  # noqa: E402

TWIN = ["agent.critic_use_cdq=true", "agent.critic_num_blocks=2", "agent.critic_hidden_dim=8"]


def _twin_agent(seed=0):
    a = _agent(seed=seed, extra=TWIN)
    return getattr(a, "agent", a)


def _batch():
    return {k: jnp.asarray(v[0]) for k, v in _batches(1).items()}


def _slice(tree, k):
    return jax.tree_util.tree_map(lambda x: x[k], tree)


class TwinCriticDefectTest(unittest.TestCase):
    """Each test failed on the code before the fix (tests/exp12_break_checks.py restores the old code)."""

    def test_twin_critic_initialises_with_two_independent_networks(self):
        net = sac_network.SACClippedDoubleCritic("residual", 2, 8, jnp.float32)
        obs, act = jnp.ones((4, 6)), jnp.ones((4, 3)) * 0.5
        params = net.init(jax.random.PRNGKey(0), observations=obs, actions=act)["params"]
        leaves = jax.tree_util.tree_leaves(params)
        self.assertTrue(all(x.shape[0] == 2 for x in leaves))
        self.assertTrue(any(not np.array_equal(x[0], x[1]) for x in leaves))
        q = net.apply({"params": params}, obs, act)
        single = sac_network.SACCritic("residual", 2, 8, jnp.float32)
        for k in (0, 1):
            np.testing.assert_array_equal(q[k], single.apply({"params": _slice(params["VmapSACCritic_0"], k)}, obs, act))

    def test_td_error_var_is_the_mean_of_the_two_networks(self):
        g, b, key = _twin_agent(), _batch(), jax.random.PRNGKey(7)
        with precision("highest"):
            _, info = sac_update.update_critic(key, g.actor, g.critic, g._target_critic, g.temperature, b,
                                               g._cfg.gamma, 1, True)
            nd = g.actor(observations=b["next_observation"])
            na = nd.sample(seed=key)
            tq = g._target_critic(observations=b["next_observation"], actions=na)
            t = b["reward"] + g._cfg.gamma * (jnp.minimum(tq[0], tq[1]).reshape(-1) - g.temperature() * nd.log_prob(na))
            q = g.critic(observations=b["observation"], actions=b["action"])
        v1, v2 = (float(jnp.var(q[k].reshape(-1) - t)) for k in (0, 1))
        self.assertGreater(abs(v1 - v2), 1e-3 * max(v1, v2))  # the two networks really differ
        np.testing.assert_allclose(float(info["train/td_error_q1_var"]), v1, rtol=1e-5)
        np.testing.assert_allclose(float(info["train/td_error_q2_var"]), v2, rtol=1e-5)
        np.testing.assert_allclose(float(info["train/td_error_var"]), (v1 + v2) / 2, rtol=1e-5)

    def test_actor_grad_cosine_uses_min_of_the_two_networks(self):
        g, b, key = _twin_agent(), _batch(), jax.random.PRNGKey(3)
        with precision("highest"):
            got = float(sac_update.compute_actor_gradient_cosine(key, g.actor, g.critic, g.temperature, b, True))
            keys = jax.random.split(key, b["observation"].shape[0])

            def loss(p, o, k):
                d = g.actor.apply(variables={"params": p}, observations=o[None])
                a = d.sample(seed=k)
                q = g.critic(observations=o[None], actions=a)
                return jnp.squeeze(d.log_prob(a)) * g.temperature() - jnp.squeeze(jnp.minimum(q[0], q[1]))

            grads = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))(g.actor.params, b["observation"], keys)
            flat = jnp.concatenate([x.reshape(x.shape[0], -1) for x in jax.tree_util.tree_leaves(grads)], axis=1)
            n = jnp.linalg.norm(flat, axis=1, keepdims=True)
            cos = (flat @ flat.T) / (n * n.T + 1e-8)
            ref = float((cos.sum() - jnp.trace(cos)) / (cos.shape[0] * (cos.shape[0] - 1)))
        np.testing.assert_allclose(got, ref, rtol=1e-5)

    def test_update_many_runs_with_twin_critics(self):
        g = _twin_agent()
        info = g.update_many(0, _batches(3), actor_grad_cosine_every=2)
        self.assertTrue(np.isfinite(np.asarray(info["train/td_error_var"])).all())
        cos = np.asarray(info["train/actor_grad_cosine"])
        self.assertTrue(np.isfinite(cos[[0, 2]]).all() and np.isnan(cos[1]))


class TwinCriticPathTest(unittest.TestCase):
    """The rest of the SACClippedDoubleCritic path against independent references (no defect found)."""

    @classmethod
    def setUpClass(cls):
        cls.g, cls.b = _twin_agent(), _batch()

    def test_critic_loss_uses_the_shared_min_target_and_each_network_its_own_term(self):
        g, b, key = self.g, self.b, jax.random.PRNGKey(7)
        with precision("highest"):
            _, info = sac_update.update_critic(key, g.actor, g.critic, g._target_critic, g.temperature, b,
                                               g._cfg.gamma, 1, True)
            nd = g.actor(observations=b["next_observation"])
            na = nd.sample(seed=key)
            tq = g._target_critic(observations=b["next_observation"], actions=na)
            t = b["reward"] + g._cfg.gamma * (jnp.minimum(tq[0], tq[1]).reshape(-1) - g.temperature() * nd.log_prob(na))
            q = g.critic(observations=b["observation"], actions=b["action"])
            ref = float(((q[0].reshape(-1) - t) ** 2 + (q[1].reshape(-1) - t) ** 2).mean())
            only_q1 = jax.grad(lambda p: ((g.critic.network_def.apply({"params": p}, b["observation"], b["action"])[0]
                                           .reshape(-1) - t) ** 2).mean())(g.critic.params)
        np.testing.assert_allclose(float(info["train/critic_loss"]), ref, rtol=1e-6)
        self.assertTrue(all(float(jnp.abs(x[1]).max()) == 0 for x in jax.tree_util.tree_leaves(only_q1)))

    def test_actor_loss_uses_min(self):
        g, b, key = self.g, self.b, jax.random.PRNGKey(3)
        with precision("highest"):
            _, info = sac_update.update_actor(key, g.actor, g.critic, g.temperature, b, True)
            d = g.actor(observations=b["observation"])
            a = d.sample(seed=key)
            q = g.critic(observations=b["observation"], actions=a)
            ref = float((d.log_prob(a) * g.temperature() - jnp.minimum(q[0], q[1]).reshape(-1)).mean())
        np.testing.assert_allclose(float(info["train/actor_loss"]), ref, rtol=1e-6)

    def test_polyak_update_covers_both_target_networks(self):
        g, tau = self.g, 0.25
        new, _ = sac_update.update_target_network(g.critic, g._target_critic, tau)
        for n_, p_, t_ in zip(*(jax.tree_util.tree_leaves(x) for x in (new.params, g.critic.params,
                                                                         g._target_critic.params))):
            np.testing.assert_allclose(n_, tau * p_ + (1 - tau) * t_, rtol=0, atol=1e-7)

    def test_structural_metrics_are_per_network(self):
        """critic_q1_* and critic_q2_* come from network 0 and 1 respectively (Q2 made distinguishable)."""
        from flax.core import freeze, unfreeze

        g, b = self.g, self.b
        p = unfreeze(g.critic.params)
        enc = p["VmapSACCritic_0"]["encoder"]
        enc["LayerNorm_0"]["scale"] = enc["LayerNorm_0"]["scale"].at[1].multiply(3.0)
        d = enc["ResidualBlock_0"]["Dense_0"]
        d["kernel"] = d["kernel"].at[1, :, : d["kernel"].shape[-1] // 2].set(0.0)
        d["bias"] = d["bias"].at[1, : d["bias"].shape[-1] // 2].set(0.0)
        critic = g.critic.replace(params=p if isinstance(g.critic.params, dict) else freeze(p))
        m = sac_update.get_critic_with_metrics(None, g.actor, critic, b, True)
        single = sac_network.SACCritic("residual", 2, 8, jnp.float32)
        for k in (0, 1):
            _, inter = single.apply({"params": _slice(p["VmapSACCritic_0"], k)}, b["observation"], b["action"],
                                    capture_intermediates=True)
            fi = flatten_dict(inter)
            e = fi["intermediates_encoder___call__"]
            np.testing.assert_allclose(float(m[f"train/critic_q{k + 1}_fnorm"]), float(get_feature_norm(e)), rtol=1e-6)
            np.testing.assert_allclose(float(m[f"train/critic_q{k + 1}_DR0.1"]),
                                       float(get_dormant_ratio(fi, prefix="critic", tau=0.1)["critic/dormant_total"]))
            np.testing.assert_allclose(float(m[f"train/critic_q{k + 1}_srank"]), float(get_srank(e, thershold=0.01)))
        self.assertNotAlmostEqual(float(m["train/critic_q1_fnorm"]), float(m["train/critic_q2_fnorm"]), places=3)
        leaves = jax.tree_util.tree_leaves(p)
        joint = float(jnp.sqrt(sum((x ** 2).sum() for x in leaves)))
        np.testing.assert_allclose(float(m["train/critic_wnorm"]), joint, rtol=1e-6)  # joint L2 norm, not a sum

    def test_checkpoint_round_trip_keeps_both_networks(self):
        g = self.g
        with tempfile.TemporaryDirectory() as d:
            g.save_checkpoint(d)
            g2 = _twin_agent(seed=5)
            g2.load_checkpoint(d)
        for x, y in zip(*(jax.tree_util.tree_leaves((a.critic.params, a._target_critic.params, a.critic.opt_state))
                          for a in (g, g2))):
            np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
