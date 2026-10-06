"""Phase 5: Exp 1/2 actor diagnostics I1-I4: known answers, isolation from training, persistence."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp12_helpers import (CONFIG_PATH, check_gpu_tolerance, compose, gpu_mode, load_agent_tree,  # noqa: E402
                           max_relative_deviation, on_gpu, patch_wandb, precision, record_numerics,
                           state_differences, tiny_overrides)
from experiments.exp12.diagnostics import KL_STREAM, ActorDiagnostics  # noqa: E402
from experiments.exp12.state import latest_state_dir, load_meta  # noqa: E402

OBS, ACT, N = 6, 3, 32
DIAGNOSTIC_KEYS = ("train/policy_kl", "train/actor_saturation", "train/actor_gnorm_std")


def _agent(seed=0, extra=()):
    from scale_rl.agents import create_agent

    cfg = compose(tiny_overrides(seed=seed, extra=list(extra)))
    obs_space = gym.spaces.Box(-np.inf, np.inf, (1, OBS), np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, (1, ACT), np.float32)
    return create_agent(observation_space=obs_space, action_space=act_space, cfg=cfg.agent)


def _batches(n=1, seed=0):
    rng = np.random.default_rng(seed)
    return {"observation": rng.normal(size=(n, N, OBS)).astype(np.float32),
            "action": rng.uniform(-1, 1, (n, N, ACT)).astype(np.float32),
            "reward": rng.normal(size=(n, N)).astype(np.float32),
            "terminated": np.zeros((n, N), np.float32), "truncated": np.zeros((n, N), np.float32),
            "next_observation": rng.normal(size=(n, N, OBS)).astype(np.float32)}


def _gaussian(agent, params, obs):
    """Mean and SD of the pre-tanh Gaussian, in full FP32 (never TF32), returned as float64."""
    with precision("highest"):
        dist = agent.actor.network_def.apply({"params": params}, observations=jnp.asarray(obs)).distribution
        return np.asarray(dist.mean(), np.float64), np.asarray(dist.stddev(), np.float64)


def _kl64(m0, s0, m1, s1):
    """KL(N(m1, s1) || N(m0, s0)) of diagonal Gaussians in float64, mean over states."""
    return float(np.sum(np.log(s0 / s1) + (s1 ** 2 + (m1 - m0) ** 2) / (2 * s0 ** 2) - 0.5, axis=-1).mean())


def _well_conditioned(agent):
    """sigma between ~0.1 and ~1: the log-std head's bias at 0.55 (log sigma = -10 + 6 (1 + tanh 0.55) = -1),
    its kernel scaled by 0.1 so sigma still varies with the state."""
    params = dict(agent.actor.params)
    predictor = dict(params["predictor"])
    head = dict(predictor["Dense_1"])
    head["kernel"], head["bias"] = head["kernel"] * 0.1, jnp.full_like(head["bias"], 0.55)
    predictor["Dense_1"], params["predictor"] = head, predictor
    agent._actor = agent._actor.replace(params=params)


def _diagnostic_and_closed_form_kl(agent, ref):
    before = agent.actor.params
    info = agent.update_many(0, _batches(), 10**9, saturation_threshold=0.99, kl_ref_observations=ref)
    (m0, s0), (m1, s1) = _gaussian(agent, before, ref), _gaussian(agent, agent.actor.params, ref)
    return float(info["train/policy_kl"][0]), _kl64(m0, s0, m1, s1), _kl64(m1, s1, m0, s0), s0


class KnownAnswerTest(unittest.TestCase):
    def test_policy_kl_is_the_closed_form_gaussian_kl_new_vs_old(self):
        # Well conditioned: sigma ~0.1-1 and a learning rate (1e-3) that gives a per-update KL ~1e-2, so the
        # float32 KL is not dominated by cancellation. The reference is float64 numpy on FP32 forward passes.
        agent = _agent(extra=["agent.actor_learning_rate=1e-3"])
        agent = getattr(agent, "agent", agent)
        _well_conditioned(agent)
        ref = np.random.default_rng(1).normal(size=(64, OBS)).astype(np.float32)
        diagnostic, kl, reverse, s0 = _diagnostic_and_closed_form_kl(agent, ref)
        self.assertTrue(0.05 < s0.min() and s0.max() < 1.5, (s0.min(), s0.max()))
        self.assertGreater(kl, 1e-3)
        record_numerics("policy_kl/well_conditioned", diagnostic=diagnostic, closed_form=kl,
                        rel_diff=abs(diagnostic - kl) / kl, sigma_min=float(s0.min()), sigma_max=float(s0.max()))
        np.testing.assert_allclose(diagnostic, kl, rtol=1e-4)
        self.assertGreater(abs(kl - reverse) / kl, 10 * 1e-4)  # the direction KL(pi_t || pi_{t-1}) is visible at rtol

    def test_policy_kl_near_deterministic_actor_is_reported_not_gated(self):
        # The freshly initialised test actor has sigma down to ~5e-5: KL ~ (d mu)^2 / sigma^2 amplifies any rounding
        # difference between two computations of mu. Reported for information only.
        agent = _agent()
        agent = getattr(agent, "agent", agent)
        ref = np.random.default_rng(1).normal(size=(64, OBS)).astype(np.float32)
        diagnostic, kl, _, s0 = _diagnostic_and_closed_form_kl(agent, ref)
        record_numerics("policy_kl/near_deterministic (information only)", diagnostic=diagnostic, closed_form=kl,
                        rel_diff=abs(diagnostic - kl) / kl, sigma_min=float(s0.min()))
        self.assertTrue(np.isfinite(diagnostic))

    def test_saturation_is_the_fraction_of_sampled_components_beyond_the_threshold(self):
        from scale_rl.agents.sac.sac_update import update_actor

        agent = getattr(_agent(), "agent", _agent())
        batch = {k: jnp.asarray(v[0]) for k, v in _batches().items()}
        key = jax.random.PRNGKey(3)
        for threshold in (0.0, 0.5, 0.99):
            _, info = update_actor(key=key, actor=agent.actor, critic=agent.critic, temperature=agent.temperature,
                                   batch=batch, critic_use_cdq=False, saturation_threshold=threshold)
            dist = agent.actor.network_def.apply({"params": agent.actor.params}, observations=batch["observation"])
            actions = np.asarray(dist.sample(seed=key))
            self.assertAlmostEqual(float(info["train/actor_saturation"]), float(np.mean(np.abs(actions) > threshold)),
                                   places=6)
        _, info = update_actor(key=key, actor=agent.actor, critic=agent.critic, temperature=agent.temperature,
                               batch=batch, critic_use_cdq=False)
        self.assertNotIn("train/actor_saturation", info)  # off by default: the Angle 1 path is unchanged

    def test_existing_path_has_no_new_outputs(self):
        agent = getattr(_agent(), "agent", _agent())
        info = agent.update_many(0, _batches(), 10**9)
        for k in DIAGNOSTIC_KEYS:
            self.assertNotIn(k, info)

    def test_diagnostics_never_change_the_update(self):
        # CPU: bit-exact. GPU: diagnostics off and on compile to different programs, so the comparison runs in the
        # two GPU modes of exp12_helpers (tolerances from measured deviations).
        mode = gpu_mode() if on_gpu() else "cpu"
        on_precision, off_precision = {"cpu": (None, None), "highest_deterministic": ("highest", "highest"),
                                       "tf32": ("tensorfloat32", "highest")}[mode]  # (on, off): the run setting vs FP32
        a, b = _agent(), _agent()
        a, b = getattr(a, "agent", a), getattr(b, "agent", b)
        ref = np.random.default_rng(1).normal(size=(64, OBS)).astype(np.float32)
        with precision(off_precision):
            info_a = a.update_many(0, _batches(4), 2)
        with precision(on_precision):
            info_b = b.update_many(0, _batches(4), 2, saturation_threshold=0.99, kl_ref_observations=ref)
        state = lambda g: (g._actor, g._critic, g._target_critic, g._temperature, g._rng)
        leaves_a, leaves_b = jax.tree_util.tree_leaves(state(a)), jax.tree_util.tree_leaves(state(b))
        if mode != "cpu":
            deviation = max_relative_deviation(leaves_b + [info_b[k] for k in info_a],
                                               leaves_a + [info_a[k] for k in info_a])
            return check_gpu_tolerance(self, f"diagnostics_update/{mode}", deviation, updates=4)
        for x, y in zip(leaves_a, leaves_b):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        for k in info_a:
            np.testing.assert_array_equal(np.asarray(info_a[k]), np.asarray(info_b[k]), err_msg=k)

    def test_gnorm_std_is_the_population_sd_of_the_window(self):
        cfg = compose(tiny_overrides())
        d = ActorDiagnostics(cfg)
        d.collect({"train/actor_gnorm": np.array([1.0, 2.0])})
        d.collect({"train/actor_gnorm": np.array([4.0])})
        self.assertAlmostEqual(d.window_metrics()["train/actor_gnorm_std"], float(np.std([1.0, 2.0, 4.0])))
        self.assertEqual(d.window_metrics(), {})  # a new window starts empty


class DiagnosticsPrecisionTest(unittest.TestCase):
    """Amendment (x): the churn and KL forward passes run under 'highest' when the Exp 1/2 diagnostics are on;
    every training matmul keeps the job's setting (TF32), and the Angle 1 path is unchanged."""

    def _lowered(self, agent, diagnostics):
        from scale_rl.agents.sac.sac_agent import _update_sac_networks_scan

        batches = {k: jnp.asarray(v) for k, v in _batches(2).items()}
        churn_ref = {k: v[0] for k, v in batches.items()}
        kw = {"saturation_threshold": 0.99, "kl_ref_observations": jnp.asarray(
            np.random.default_rng(1).normal(size=(64, OBS)).astype(np.float32))} if diagnostics else {}
        c = agent._cfg
        return _update_sac_networks_scan.lower(
            agent._rng, agent._actor, agent._critic, agent._target_critic, agent._temperature, batches,
            first_update_step=0, gamma=c.gamma, n_step=c.n_step, critic_use_cdq=c.critic_use_cdq,
            target_tau=c.target_tau, temp_target_entropy=c.temp_target_entropy, churn_ref_batch=churn_ref,
            actor_grad_cosine_every=2, **kw).as_text()

    def test_diagnostic_forward_passes_are_highest_and_training_matmuls_keep_the_run_setting(self):
        agent = _agent()
        agent = getattr(agent, "agent", agent)
        before = jax.config.jax_default_matmul_precision
        jax.config.update("jax_default_matmul_precision", "tensorfloat32")  # the run setting on the GPU
        try:
            def actor_dots(fn):
                return jax.jit(lambda p, o: fn(agent.actor.network_def.apply({"params": p}, observations=o).distribution)
                               ).lower(agent.actor.params, jnp.zeros((4, OBS))).as_text().count("dot_general")

            n_mean = actor_dots(lambda d: d.mean())                    # churn uses the mean only
            n_full = actor_dots(lambda d: (d.mean(), d.stddev()))      # the KL also uses the log-std head
            on, off = self._lowered(agent, True), self._lowered(agent, False)
        finally:
            jax.config.update("jax_default_matmul_precision", before)
        highest, high = "precision = [HIGHEST, HIGHEST]", "precision = [HIGH, HIGH]"
        self.assertGreater(n_full, n_mean)
        self.assertEqual(off.count(highest), 0)                   # diagnostics off (the Angle 1 path): nothing changes
        self.assertEqual(on.count(highest), 2 * n_mean + 2 * n_full)   # churn before/after, KL new/old
        self.assertEqual(on.count(high), off.count(high) - 2 * n_mean)  # only the two churn passes left the run setting
        self.assertGreater(on.count(high), 0)                 # training matmuls stay at the run setting


class ReferenceBatchTest(unittest.TestCase):
    def test_reference_batch_per_window_from_a_dedicated_stream(self):
        from experiments.exp12.trainer import Exp12Trainer

        cfg = compose(tiny_overrides(extra=["env.max_episode_steps=20"]))
        t = Exp12Trainer(cfg, tempfile.mkdtemp())
        t.buffer.reset()
        for i in range(30):
            t.buffer.add({"observation": np.full((1, t.buffer._observations.shape[-1]), i, np.float32),
                          "action": np.zeros((1, t.buffer._actions.shape[-1]), np.float32),
                          "reward": np.zeros(1), "terminated": np.zeros(1), "truncated": np.zeros(1),
                          "next_observation": np.zeros((1, t.buffer._observations.shape[-1]), np.float32)})
        np.random.seed(0)
        np_state = np.random.get_state()
        w = int(cfg.logging_per_interaction_step)
        t.interaction_step = 1
        first = t.diagnostics.update_kwargs(t)["kl_ref_observations"]
        t.interaction_step = w
        self.assertIs(t.diagnostics.update_kwargs(t)["kl_ref_observations"], first)  # same window
        t.interaction_step = w + 1
        second = t.diagnostics.update_kwargs(t)["kl_ref_observations"]
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(first.shape, (int(cfg.diagnostics.kl_reference_size), t.buffer._observations.shape[-1]))
        idx = np.random.default_rng([int(cfg.seed), KL_STREAM, 1]).integers(0, 30, size=first.shape[0])
        np.testing.assert_array_equal(second, np.asarray(t.agent._normalize(t.buffer._observations[idx]), np.float32))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state())))
        t.close()


class TrainingRunTest(unittest.TestCase):
    """Tiny exp1 runs with diagnostics on and off: identical training, new metrics present."""

    @classmethod
    def setUpClass(cls):
        cls.patchers = patch_wandb()
        cls.tmp = tempfile.mkdtemp()
        cls.dirs = {}
        cls.mode = gpu_mode() if on_gpu() else "cpu"
        side_precision = {"cpu": {"on": None, "off": None},
                          "highest_deterministic": {"on": "highest", "off": "highest"},
                          "tf32": {"on": "tensorfloat32", "off": "highest"}}[cls.mode]  # the run setting vs FP32
        for on in (True, False):
            name = "on" if on else "off"
            from experiments import exp1

            cls.dirs[name] = os.path.join(cls.tmp, name)
            with precision(side_precision[name]):
                exp1.run({"experiment": "exp1", "config_path": CONFIG_PATH, "config_name": "base_exp12",
                          "overrides": tiny_overrides(steps=400, extra=[f"diagnostics.enabled={str(on).lower()}",
                                                                        f"results_root={cls.tmp}/results_{name}"]),
                          "checkpoint_dir": cls.dirs[name], "checkpoint_interval": 10**9, "checkpoint_start_frac": 0.0})
        cls.meta = {n: load_meta(latest_state_dir(Path(d) / "state")) for n, d in cls.dirs.items()}

    @classmethod
    def tearDownClass(cls):
        for p in cls.patchers:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_training_is_identical_with_diagnostics_on_and_off(self):
        on, off = (latest_state_dir(Path(self.dirs[n]) / "state") for n in ("on", "off"))
        rows_on, rows_off = self.meta["on"]["metrics_rows"], self.meta["off"]["metrics_rows"]
        self.assertEqual(len(rows_on), len(rows_off))
        for a, b in zip(rows_on, rows_off):
            self.assertEqual(set(a) - set(b), {k for k in DIAGNOSTIC_KEYS if k in a})
        if self.mode != "cpu":  # GPU: the two runs compile different programs (see test_diagnostics_never_change_the_update)
            tree_on, tree_off = load_agent_tree(on), load_agent_tree(off)
            deviation = max_relative_deviation(jax.tree_util.tree_leaves(tree_on), jax.tree_util.tree_leaves(tree_off))
            return check_gpu_tolerance(self, f"diagnostics_training/{self.mode}", deviation, interaction_steps=400,
                                       differing_components=len(state_differences(
                                           on, off, ignore_meta=("wandb_run_id", "metrics_rows", "actor_diagnostics"))))
        self.assertEqual(state_differences(on, off, ignore_meta=("wandb_run_id", "metrics_rows",
                                                                 "actor_diagnostics")), [])
        for a, b in zip(rows_on, rows_off):
            for k in b:
                self.assertTrue(a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), k)

    def test_metrics_logged_every_window_once_learning_starts(self):
        learning = [r for r in self.meta["on"]["metrics_rows"] if "train/actor_gnorm" in r]
        self.assertGreater(len(learning), 5)
        for r in learning:
            for k in DIAGNOSTIC_KEYS:
                self.assertTrue(np.isfinite(r[k]), k)
            self.assertGreaterEqual(r["train/policy_kl"], 0)
            self.assertTrue(0 <= r["train/actor_saturation"] <= 1)


if __name__ == "__main__":
    unittest.main()
