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

from exp12_helpers import CONFIG_PATH, compose, patch_wandb, state_differences, tiny_overrides  # noqa: E402
from experiments.exp12.diagnostics import KL_STREAM, ActorDiagnostics  # noqa: E402
from experiments.exp12.state import latest_state_dir, load_meta  # noqa: E402

OBS, ACT, N = 6, 3, 32
DIAGNOSTIC_KEYS = ("train/policy_kl", "train/actor_saturation", "train/actor_gnorm_std")


def _agent(seed=0):
    from scale_rl.agents import create_agent

    cfg = compose(tiny_overrides(seed=seed))
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
    dist = agent.actor.network_def.apply({"params": params}, observations=jnp.asarray(obs)).distribution
    return np.asarray(dist.mean(), np.float64), np.asarray(dist.stddev(), np.float64)


class KnownAnswerTest(unittest.TestCase):
    def test_policy_kl_is_the_closed_form_gaussian_kl_new_vs_old(self):
        agent = getattr(_agent(), "agent", _agent())
        ref = np.random.default_rng(1).normal(size=(64, OBS)).astype(np.float32)
        before = agent.actor.params
        info = agent.update_many(0, _batches(), 10**9, saturation_threshold=0.99, kl_ref_observations=ref)
        (m0, s0), (m1, s1) = _gaussian(agent, before, ref), _gaussian(agent, agent.actor.params, ref)
        kl = np.sum(np.log(s0 / s1) + (s1 ** 2 + (m1 - m0) ** 2) / (2 * s0 ** 2) - 0.5, axis=-1).mean()
        reverse = np.sum(np.log(s1 / s0) + (s0 ** 2 + (m0 - m1) ** 2) / (2 * s1 ** 2) - 0.5, axis=-1).mean()
        self.assertGreater(kl, 0)
        np.testing.assert_allclose(float(info["train/policy_kl"][0]), kl, rtol=1e-4)
        self.assertNotAlmostEqual(kl, reverse, places=12)  # the direction is KL(pi_t || pi_{t-1})

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
        a, b = _agent(), _agent()
        a, b = getattr(a, "agent", a), getattr(b, "agent", b)
        ref = np.random.default_rng(1).normal(size=(64, OBS)).astype(np.float32)
        info_a = a.update_many(0, _batches(4), 2)
        info_b = b.update_many(0, _batches(4), 2, saturation_threshold=0.99, kl_ref_observations=ref)
        state = lambda g: (g._actor, g._critic, g._target_critic, g._temperature, g._rng)
        for x, y in zip(jax.tree_util.tree_leaves(state(a)), jax.tree_util.tree_leaves(state(b))):
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
        for on in (True, False):
            name = "on" if on else "off"
            from experiments import exp1

            cls.dirs[name] = os.path.join(cls.tmp, name)
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
        self.assertEqual(state_differences(on, off, ignore_meta=("wandb_run_id", "metrics_rows",
                                                                 "actor_diagnostics")), [])
        rows_on, rows_off = self.meta["on"]["metrics_rows"], self.meta["off"]["metrics_rows"]
        self.assertEqual(len(rows_on), len(rows_off))
        for a, b in zip(rows_on, rows_off):
            self.assertEqual(set(a) - set(b), {k for k in DIAGNOSTIC_KEYS if k in a})
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
