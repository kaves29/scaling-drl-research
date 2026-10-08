"""Twin critics (clipped double Q, HumanoidBench only; amendment (z)): defects found by the audit, the
SACClippedDoubleCritic path end to end, and the Exp 1/2 machinery (probe, injection, Checks 1-2, fork, identity
fork, diagnostics) with two Q networks, on tiny networks."""

import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp12_helpers import (check_gpu_tolerance, gpu_mode, max_relative_deviation, on_gpu, patch_wandb,  # noqa: E402
                           precision, state_differences, tiny_overrides)
from experiments.exp12 import exp2_ledger, fork, injection, twin  # noqa: E402
from experiments.exp12.state import latest_state_dir  # noqa: E402
from scale_rl.agents.sac import sac_network, sac_update  # noqa: E402
from scale_rl.networks.metrics import flatten_dict, get_dormant_ratio, get_feature_norm, get_srank  # noqa: E402
from test_exp12_diagnostics import OBS, _agent, _batches  # noqa: E402
from test_exp12_fork import fork_overrides, run_arm, run_exp1  # noqa: E402

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


def _leaves_equal(a, b):
    return all(np.array_equal(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


class TwinProbeTest(unittest.TestCase):
    """Part 4(a): each network against its own fresh copy; per-round L = mean of the two networks' L."""

    def test_expand_gives_each_network_its_own_slice(self):
        g = _twin_agent()
        fresh = _twin_agent(seed=3).critic.params
        tx = object()
        critics, twins = twin.expand({"current": (g.critic.network_def, g.critic.params), "fresh": ("single", fresh)},
                                     injected_tx=tx)
        self.assertEqual(twins, ["current", "fresh"])
        self.assertEqual(list(critics), ["current_q1", "current_q2", "fresh_q1", "fresh_q2"])
        for k in (0, 1):
            self.assertTrue(_leaves_equal(critics[f"current_q{k + 1}"][1], _slice(g.critic.params["VmapSACCritic_0"], k)))
            self.assertTrue(_leaves_equal(critics[f"fresh_q{k + 1}"][1], _slice(fresh["VmapSACCritic_0"], k)))
        self.assertIsInstance(critics["current_q1"][0], sac_network.SACCritic)
        self.assertEqual(critics["fresh_q1"][0], "single")  # the fresh reference's def is already one network
        ic, _ = injection.inject_twin(g.critic, g._target_critic, "last", injection.injection_key(7), 1e-3, 1e-2)
        inj, _ = twin.expand({"current": (ic.network_def, ic.params)}, injected_tx=tx)
        self.assertIsInstance(inj["current_q1"][0], injection.InjectedSACCritic)
        self.assertIs(inj["current_q2"][2], tx)  # an injected network is probed with the injected optimizer
        single, none = twin.expand({"current": ("def", {"encoder": {}})}, injected_tx=tx)
        self.assertEqual((list(single), none), (["current"], []))

    def test_combined_loss_is_the_mean_of_the_two_networks(self):
        from experiments.exp12.probe import summarize
        from experiments.exp12.run_probes import per_network_fields

        rng = np.random.default_rng(0)
        result = {f"{n}_q{k}": {"score": rng.normal(size=5), "final_loss": rng.normal(size=5), "b": np.ones(5)}
                  for n in ("current", "fresh") for k in (1, 2)}
        combined = twin.combine(result, ["current", "fresh"])
        # P remains a mean-score estimand (notably in Check 2), independently of L.
        for name in ("current", "fresh"):
            np.testing.assert_array_equal(combined[name]["score"],
                                          np.mean([result[f"{name}_q{q}"]["score"] for q in (1, 2)], axis=0))
        loss = summarize(combined)["loss_rounds"]
        l1 = result["fresh_q1"]["score"] - result["current_q1"]["score"]
        l2 = result["fresh_q2"]["score"] - result["current_q2"]["score"]
        np.testing.assert_allclose(loss, (l1 + l2) / 2, rtol=1e-12)
        fields = per_network_fields(combined)
        np.testing.assert_allclose([fields[f"loss_q1_r{r}"] for r in range(5)], l1)
        np.testing.assert_allclose([fields[f"loss_q2_r{r}"] for r in range(5)], l2)


class TwinInjectionTest(unittest.TestCase):
    """Part 4(b): the same head in both networks and both targets, same m."""

    @classmethod
    def setUpClass(cls):
        cls.g, cls.b = _twin_agent(), _batch()
        cls.g.update_many(0, _batches(2), 10**9)  # trained a little: target lags online, Adam moments non-zero
        assert not _leaves_equal(cls.g.critic.params, cls.g._target_critic.params)

    def _inject(self, label):
        return injection.inject_twin(self.g.critic, self.g._target_critic, label, injection.injection_key(7), 1e-3, 1e-2)

    def test_predictions_unchanged_and_action_gradients_within_tolerance(self):
        g, b = self.g, self.b
        eps = float(np.finfo(np.float32).eps)
        for label in injection.M_LABELS:
            with self.subTest(m=label):
                ic, it = self._inject(label)
                with precision("highest"):
                    for before, after in ((g.critic, ic), (g._target_critic, it)):
                        q0 = before(observations=b["observation"], actions=b["action"])
                        q1 = after(observations=b["observation"], actions=b["action"])
                        np.testing.assert_array_equal(q0, q1)  # both networks, bit for bit
                    grad = lambda c: jax.grad(lambda a: c(observations=b["observation"], actions=a).sum(axis=(1, 2)).sum())
                    g0, g1 = grad(g.critic)(b["action"]), grad(ic)(b["action"])
                self.assertLessEqual(float(jnp.abs(g0 - g1).max()), injection.CHECK1_TOLERANCE_EPS * eps * float(jnp.abs(g0).max()))

    def test_same_head_in_both_networks_and_targets(self):
        for label, m in (("last", 1), ("half", 1), ("all", 2)):
            with self.subTest(m=label):
                ic, it = self._inject(label)
                self.assertEqual(ic.network_def.head_blocks, m)
                for t in (ic, it):
                    p = t.params[injection.INJECTED_TWIN]
                    self.assertEqual(sorted(p), ["copy", "new", "old", "trunk"])
                    self.assertEqual(sum(k.startswith("ResidualBlock") for k in p["old"]), m)
                    self.assertTrue(all(x.shape[0] == 2 for x in jax.tree_util.tree_leaves(p)))
                p, tp = ic.params[injection.INJECTED_TWIN], it.params[injection.INJECTED_TWIN]
                self.assertTrue(_leaves_equal(p["new"], p["copy"]) and _leaves_equal(tp["new"], p["new"])
                                and _leaves_equal(tp["copy"], p["new"]))
                self.assertTrue(any(not np.array_equal(x[0], x[1]) for x in jax.tree_util.tree_leaves(p["new"])))
                trunk, old = injection.split_params(self.g.critic.params["VmapSACCritic_0"], 2, m)
                self.assertTrue(_leaves_equal(p["trunk"], trunk) and _leaves_equal(p["old"], old))
                ttrunk, told = injection.split_params(self.g._target_critic.params["VmapSACCritic_0"], 2, m)
                self.assertTrue(_leaves_equal(tp["trunk"], ttrunk) and _leaves_equal(tp["old"], told))

    def test_training_keeps_frozen_heads_and_optimizer_rules(self):
        g, b = self.g, self.b
        ic, _ = self._inject("half")
        mu = g.critic.opt_state[0].mu["VmapSACCritic_0"]
        trunk_mu, _ = injection.split_params(mu, 2, 1)
        self.assertTrue(_leaves_equal(ic.opt_state["trunk"][0].mu, trunk_mu))
        self.assertEqual(int(ic.opt_state["trunk"][0].count), int(g.critic.opt_state[0].count))
        self.assertEqual(int(ic.opt_state["new"][0].count), 0)
        loss = lambda p: (jnp.mean(ic.network_def.apply({"params": p}, b["observation"], b["action"]) ** 2), {})
        after, _ = ic.apply_gradient(loss)
        p0, p1 = ic.params[injection.INJECTED_TWIN], after.params[injection.INJECTED_TWIN]
        self.assertTrue(_leaves_equal(p0["old"], p1["old"]) and _leaves_equal(p0["copy"], p1["copy"]))
        self.assertFalse(_leaves_equal(p0["new"], p1["new"]))
        self.assertFalse(_leaves_equal(p0["trunk"], p1["trunk"]))


class TwinCheck1Test(unittest.TestCase):
    """Part 4(c): Q of both networks and dQ/da of min(Q1, Q2)."""

    def test_panel_values(self):
        g = _twin_agent()
        rng = np.random.default_rng(0)
        panel = {"observation": rng.normal(size=(256, OBS)).astype(np.float32),
                 "action": rng.uniform(-1, 1, (256, 3)).astype(np.float32)}
        out = fork.panel_q_and_grad(g.critic, panel)
        obs, act = jnp.asarray(panel["observation"]), jnp.asarray(panel["action"])
        with precision("highest"):
            q = g.critic(observations=obs, actions=act)
            dq = jax.grad(lambda a: jnp.minimum(*g.critic(observations=obs, actions=a)).sum())(act)
        np.testing.assert_array_equal(out["q"], np.asarray(q).reshape(-1))
        self.assertEqual(out["q"].shape, (512,))
        np.testing.assert_allclose(out["dq_da"], np.asarray(dq), rtol=1e-5, atol=1e-7)  # jit vs eager: ULPs


class TwinForkEndToEndTest(unittest.TestCase):
    """Part 4(d): fork state, restore, identity fork, injected arm and Checks 1-2 with twin critics."""

    @classmethod
    def setUpClass(cls):
        cls.patchers = patch_wandb()
        cls.tmp = tempfile.mkdtemp()
        cls.results = os.path.join(cls.tmp, "results")
        cls.overrides = fork_overrides(cls.results, ["agent.critic_use_cdq=true"])
        cls.run_dir = os.path.join(cls.tmp, "run")
        run_exp1(cls.run_dir, cls.overrides)
        cls.arm_dirs = {}
        for arm in ("identity", "injected"):
            cls.arm_dirs[arm] = os.path.join(cls.tmp, f"arm_{arm}")
            run_arm(cls.arm_dirs[arm], cls.overrides, cls.run_dir, arm)
        cls.exp2 = exp2_ledger.run_root(fork.read_fork(cls.run_dir)["run_key"], cls.results)

    @classmethod
    def tearDownClass(cls):
        for p in cls.patchers:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _meta(self, state_dir):
        with open(Path(state_dir) / "meta.pkl", "rb") as f:
            return pickle.load(f)

    def test_runs_are_twin_and_record_the_critic_count(self):
        for d in (self.run_dir, *self.arm_dirs.values()):
            with open(Path(d) / "run_metadata.json") as f:
                meta = json.load(f)
            self.assertEqual({launch["critic_count"] for launch in meta["launches"]}, {2}, d)
            self.assertTrue(meta["resolved_config"]["agent"]["critic_use_cdq"])
        self.assertIn("VmapSACCritic_0",
                      load_tree(latest_state_dir(fork.fork_dir(self.run_dir) / "state"))["critic"]["params"])

    def test_identity_fork_is_bit_identical(self):
        control = latest_state_dir(fork.fork_dir(self.run_dir) / "identity_control")
        identity = latest_state_dir(Path(self.arm_dirs["identity"]) / "identity_identity")
        self.assertEqual(state_differences(control, identity, ignore_meta=("wandb_run_id", "extra_state")), [])
        self.assertEqual(self._meta(control)["extra_state"]["probe_records"],
                         self._meta(identity)["extra_state"]["probe_records"])

    def test_check1_both_networks(self):
        with open(self.exp2 / "check1_identity.json") as f:
            identity = json.load(f)
        with open(self.exp2 / "check1_injected.json") as f:
            injected = json.load(f)
        self.assertTrue(identity["pass"] and injected["pass"])
        self.assertEqual(identity["max_eps_units"], 0.0)
        self.assertEqual(injected["pairs"]["pre_vs_after"]["max_abs_dq"], 0.0)
        self.assertEqual(fork.load_npz(Path(self.arm_dirs["injected"]) / "check1_after.npz")["q"].shape, (512,))

    def test_probe_records_are_the_mean_of_both_networks(self):
        records = self._meta(latest_state_dir(Path(self.run_dir) / "state"))["extra_state"]["probe_records"]
        self.assertIn("score_fresh_q1_r0", records[0])
        for row in records[1:]:
            for r in range(5):
                l1, l2 = row[f"loss_q1_r{r}"], row[f"loss_q2_r{r}"]
                self.assertAlmostEqual(row[f"loss_r{r}"], (l1 + l2) / 2, places=6)
                self.assertAlmostEqual(l1, row[f"score_fresh_q1_r{r}"] - row[f"score_current_q1_r{r}"], places=6)

    def test_check2_uses_the_mean_and_records_both_networks(self):
        with open(self.exp2 / "check2.json") as f:
            c2 = json.load(f)
        for n in ("injected", "control", "fresh"):
            mean = (np.array(c2[f"score_{n}_q1_rounds"]) + np.array(c2[f"score_{n}_q2_rounds"])) / 2
            np.testing.assert_allclose(c2[f"score_{n}_rounds"], mean, rtol=1e-6)
        diff = np.array(c2["score_injected_rounds"]) - np.array(c2["score_control_rounds"])
        np.testing.assert_array_equal(c2["paired_difference_rounds"], diff)

    def test_injected_arm_both_networks(self):
        final = load_tree(latest_state_dir(Path(self.arm_dirs["injected"]) / "state"))["critic"]["params"]
        pre = load_tree(latest_state_dir(fork.fork_dir(self.run_dir) / "state"))["critic"]["params"]
        p = final[injection.INJECTED_TWIN]
        np.testing.assert_array_equal(p["old"]["LinearCritic_0"]["Dense_0"]["kernel"],
                                      pre["VmapSACCritic_0"]["predictor"]["Dense_0"]["kernel"])
        for a, b in zip(jax.tree_util.tree_leaves(p["new"]), jax.tree_util.tree_leaves(p["copy"])):
            self.assertFalse(np.array_equal(a[0], b[0]) or np.array_equal(a[1], b[1]))


def load_tree(state_dir):
    from experiments.exp12.state import load_agent_tree

    return load_agent_tree(state_dir)


class TwinProbeIsolationTest(unittest.TestCase):
    """Part 4(d): probes on = probes off with twin critics."""

    def test_probes_on_equals_probes_off(self):
        from experiments import exp1

        patchers = patch_wandb()
        tmp = tempfile.mkdtemp()
        try:
            states = {}
            for name, on in (("off", "false"), ("on", "true")):
                run_dir = os.path.join(tmp, name)
                exp1.run({"experiment": "exp1", "config_path": str(Path(__file__).resolve().parents[1] / "configs"),
                          "config_name": "base_exp12",
                          "overrides": tiny_overrides(steps=400, extra=["agent.critic_use_cdq=true", f"probe.enabled={on}",
                                                                        f"results_root={tmp}/results_{name}"]),
                          "checkpoint_dir": run_dir, "checkpoint_interval": 10**9, "checkpoint_start_frac": 0.0})
                states[name] = latest_state_dir(Path(run_dir) / "state")
            self.assertEqual(state_differences(states["off"], states["on"], ignore_meta=("wandb_run_id", "extra_state")), [])
        finally:
            for p in patchers:
                p.stop()
            shutil.rmtree(tmp, ignore_errors=True)


class TwinDiagnosticsTest(unittest.TestCase):
    """Part 4(d): the actor diagnostics never change a twin-critic update (CPU bit-exact, GPU as in
    test_exp12_diagnostics)."""

    def test_diagnostics_never_change_the_update(self):
        mode = gpu_mode() if on_gpu() else "cpu"
        on_precision, off_precision = {"cpu": (None, None), "highest_deterministic": ("highest", "highest"),
                                       "tf32": ("tensorfloat32", "highest")}[mode]
        a, b = _twin_agent(), _twin_agent()
        ref = np.random.default_rng(1).normal(size=(64, OBS)).astype(np.float32)
        with precision(off_precision):
            info_a = a.update_many(0, _batches(4), 2)
        with precision(on_precision):
            info_b = b.update_many(0, _batches(4), 2, saturation_threshold=0.99, kl_ref_observations=ref)
        state = lambda g: (g._actor, g._critic, g._target_critic, g._temperature, g._rng)
        leaves_a, leaves_b = jax.tree_util.tree_leaves(state(a)), jax.tree_util.tree_leaves(state(b))
        if mode != "cpu":
            deviation = max_relative_deviation(leaves_b + [info_b[k] for k in info_a],
                                               leaves_a + [info_a[k] for k in info_a],
                                               allow_nan_indices=(len(leaves_a) + list(info_a).index("train/actor_grad_cosine"),))
            return check_gpu_tolerance(self, f"diagnostics_update/{mode}", deviation, updates=4, twin=True)
        for x, y in zip(leaves_a, leaves_b):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        for k in info_a:
            np.testing.assert_array_equal(np.asarray(info_a[k]), np.asarray(info_b[k]), err_msg=k)


if __name__ == "__main__":
    unittest.main()
