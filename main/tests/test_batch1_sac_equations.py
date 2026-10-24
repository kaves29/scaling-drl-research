"""Independent equations for production SAC operators, without simulator training."""

import unittest

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp

from experiments.exp12.fixed_task import tree_sha256
from scale_rl.agents.sac import sac_agent, sac_update
from scale_rl.agents.sac.sac_network import SACTemperature
from scale_rl.networks.trainer import Trainer
from tests.test_batch1_reproducibility import agent
from tests.test_sac_agent_checkpoint import _make_random_batch


class Actor(nn.Module):
    @nn.compact
    def __call__(self, observations, temperature=1.0):
        theta = self.param("theta", nn.initializers.ones, (2,))
        normal = tfp.distributions.MultivariateNormalDiag(
            loc=observations[:, :2] * theta,
            scale_diag=jnp.ones((2,)) * 0.3 * temperature,
        )
        return tfp.distributions.TransformedDistribution(normal, tfp.bijectors.Tanh())


class Critic(nn.Module):
    twin: bool = False

    @nn.compact
    def __call__(self, observations, actions):
        heads = 2 if self.twin else 1
        wo = self.param("wo", nn.initializers.zeros, (heads, 4))
        wa = self.param("wa", nn.initializers.zeros, (heads, 2))
        b = self.param("b", nn.initializers.zeros, (heads,))
        q = observations @ wo.T + actions @ wa.T + b
        return q.T[:, :, None] if self.twin else q


def trainers(twin=False):
    a = Trainer.create(
        Actor(),
        {"rngs": jax.random.PRNGKey(1), "observations": jnp.zeros((1, 4))},
        optax.sgd(0.01),
    )
    a = a.replace(params={"theta": jnp.array([0.2, -0.3])})
    c = Trainer.create(
        Critic(twin),
        {
            "rngs": jax.random.PRNGKey(2),
            "observations": jnp.zeros((1, 4)),
            "actions": jnp.zeros((1, 2)),
        },
        optax.sgd(0.01),
    )
    p = {
        "wo": jnp.array(
            [[0.2, 0.1, -0.3, 0.4], [0.1, 0.4, 0.2, -0.2]][: 2 if twin else 1]
        ),
        "wa": jnp.array([[0.7, -0.4], [-0.2, 0.6]][: 2 if twin else 1]),
        "b": jnp.array([0.3, 3.0][: 2 if twin else 1]),
    }
    c = c.replace(params=p)
    target = c.replace(
        tx=None, opt_state=None, params=jax.tree_util.tree_map(lambda x: x + 0.2, p)
    )
    t = Trainer.create(
        SACTemperature(0.2), {"rngs": jax.random.PRNGKey(3)}, optax.sgd(0.01)
    )
    return a, c, target, t


def batch():
    return {
        "observation": jnp.array(
            [[1.0, 0.5, -0.5, 1.0], [-1.0, 0.3, 0.4, -0.2], [0.2, -0.8, 0.9, 0.1]]
        ),
        "action": jnp.array([[0.1, 0.2], [-0.2, 0.5], [0.3, -0.1]]),
        "next_observation": jnp.array(
            [[0.3, 0.4, 0.5, 0.6], [0.5, -0.2, 0.1, 0.4], [-0.5, 0.3, -0.2, 0.1]]
        ),
        "reward": jnp.array([1.0, 2.0, -1.0]),
        "terminated": jnp.array([1.0, 0.0, 0.0]),
        "truncated": jnp.array([0.0, 1.0, 0.0]),
    }


def q_numpy(params, obs, act):
    return (
        np.asarray(obs, dtype=np.float64) @ np.asarray(params["wo"], dtype=np.float64).T
        + np.asarray(act, dtype=np.float64)
        @ np.asarray(params["wa"], dtype=np.float64).T
        + np.asarray(params["b"], dtype=np.float64)
    )


def log_pi_numpy(actions, means):
    action = np.asarray(actions, dtype=np.float64)
    pre = np.arctanh(action)
    return np.sum(
        -0.5 * ((pre - np.asarray(means)) / 0.3) ** 2
        - np.log(0.3)
        - 0.5 * np.log(2 * np.pi)
        - np.log1p(-(action**2)),
        axis=1,
    )


class SacEquationTest(unittest.TestCase):
    def test_actor_loss_and_gradient_follow_tanh_change_of_variables_and_q_actions(
        self,
    ):
        b = batch()
        key = jax.random.PRNGKey(31)
        for twin in (False, True):
            with self.subTest(twin=twin):
                actor, critic, target, temp = trainers(twin)
                frozen = tree_sha256((actor, critic, target, temp))
                actions = actor(observations=b["observation"]).sample(seed=key)
                means = np.asarray(b["observation"][:, :2]) * np.asarray(
                    actor.params["theta"]
                )
                logs = log_pi_numpy(actions, means)
                q = q_numpy(critic.params, b["observation"], actions)
                selected = np.argmin(q, axis=1)
                w = np.asarray(critic.params["wa"])[selected]
                a = np.asarray(actions, dtype=np.float64)
                alpha = float(temp())
                expected_gradient = np.mean(
                    (2 * alpha * a - w * (1 - a * a))
                    * np.asarray(b["observation"][:, :2]),
                    axis=0,
                )
                expected_loss = np.mean(alpha * logs - q.min(axis=1))
                updated, info = sac_update.update_actor(
                    key, actor, critic, temp, b, twin
                )
                np.testing.assert_allclose(
                    info["train/actor_loss"], expected_loss, rtol=2e-6, atol=2e-6
                )
                np.testing.assert_allclose(
                    updated.params["theta"],
                    np.asarray(actor.params["theta"]) - 0.01 * expected_gradient,
                    rtol=2e-6,
                    atol=2e-7,
                )
                self.assertEqual(tree_sha256((actor, critic, target, temp)), frozen)
                self.assertEqual(int(updated.update_step), 1)
                self.assertNotEqual(
                    tree_sha256(actor.params), tree_sha256(updated.params)
                )

    def test_bellman_targets_and_critic_gradients_terminal_truncated_and_continuing(
        self,
    ):
        b = batch()
        key = jax.random.PRNGKey(32)
        for twin in (False, True):
            for gamma in (0.0, 0.95, 1.0):
                with self.subTest(twin=twin, gamma=gamma):
                    actor, critic, target, temp = trainers(twin)
                    frozen = tree_sha256((actor, critic, target, temp))
                    next_actions = actor(observations=b["next_observation"]).sample(
                        seed=key
                    )
                    means = np.asarray(b["next_observation"][:, :2]) * np.asarray(
                        actor.params["theta"]
                    )
                    logs = log_pi_numpy(next_actions, means)
                    tq = q_numpy(
                        target.params, b["next_observation"], next_actions
                    ).min(axis=1)
                    y = np.asarray(b["reward"]) + gamma * (
                        1 - np.asarray(b["terminated"])
                    ) * (tq - float(temp()) * logs)
                    residual = (
                        q_numpy(critic.params, b["observation"], b["action"])
                        - y[:, None]
                    )
                    expected_gradient = {
                        "wo": 2 * residual.T @ np.asarray(b["observation"]) / 3,
                        "wa": 2 * residual.T @ np.asarray(b["action"]) / 3,
                        "b": 2 * residual.mean(axis=0),
                    }
                    updated, info = sac_update.update_critic(
                        key, actor, critic, target, temp, b, gamma, 1, twin
                    )
                    np.testing.assert_allclose(
                        info["train/critic_loss"],
                        np.mean(np.sum(residual**2, axis=1)),
                        rtol=2e-6,
                        atol=2e-6,
                    )
                    for name in critic.params:
                        np.testing.assert_allclose(
                            updated.params[name],
                            np.asarray(critic.params[name])
                            - 0.01 * expected_gradient[name],
                            rtol=2e-6,
                            atol=2e-7,
                        )
                    self.assertEqual(y[0], float(b["reward"][0]))
                    self.assertEqual(tree_sha256((actor, critic, target, temp)), frozen)

    def test_target_update_and_initialization_are_functionally_isolated(self):
        for twin in (False, True):
            g = agent(critic_use_cdq=twin)
            self.assertEqual(
                tree_sha256(g.critic.params), tree_sha256(g._target_critic.params)
            )
            self.assertIsNone(g._target_critic.tx)
            self.assertIsNone(g._target_critic.opt_state)
            old = tree_sha256((g.critic, g._target_critic))
            changed = g.critic.replace(
                params=jax.tree_util.tree_map(lambda x: x + 0.1, g.critic.params)
            )
            for tau in (0.0, 0.005, 1.0):
                new, _ = sac_update.update_target_network(
                    changed, g._target_critic, tau
                )
                for actual, online, target in zip(
                    jax.tree_util.tree_leaves(new.params),
                    jax.tree_util.tree_leaves(changed.params),
                    jax.tree_util.tree_leaves(g._target_critic.params),
                ):
                    np.testing.assert_array_equal(
                        actual, online * tau + target * (1 - tau)
                    )
                self.assertIsNone(new.opt_state)
            self.assertEqual(tree_sha256((g.critic, g._target_critic)), old)

    def test_temperature_gradient_matches_documented_alpha_times_entropy_gap(self):
        for entropy, target_entropy in ((-2.0, -1.0), (0.0, -1.0), (-1.0, -1.0)):
            _, _, _, temp = trainers()
            old = float(temp.params["log_temp"])
            alpha = float(temp())
            updated, _ = sac_update.update_temperature(
                temp, jnp.array(entropy), target_entropy
            )
            expected = old - 0.01 * alpha * (entropy - target_entropy)
            self.assertAlmostEqual(
                float(updated.params["log_temp"]), expected, places=6
            )

    def test_evaluation_is_mean_policy_training_sampling_and_prng_round_trip_are_exact(
        self,
    ):
        g = agent()
        obs = np.full((3, 4), 0.3, np.float32)
        original = tree_sha256((g.actor, g.critic, g.temperature))
        x = g.sample_actions(0, {"next_observation": obs.copy()}, False)
        y = g.sample_actions(0, {"next_observation": obs.copy()}, False)
        np.testing.assert_array_equal(x, y)
        expected = jnp.tanh(g.actor(observations=jnp.asarray(obs)).distribution.mean())
        np.testing.assert_array_equal(x, expected)
        key = g._rng
        x = g.sample_actions(1, {"next_observation": obs.copy()}, True)
        self.assertNotEqual(tree_sha256(key), tree_sha256(g._rng))
        g._rng = key
        y = g.sample_actions(1, {"next_observation": obs.copy()}, True)
        np.testing.assert_array_equal(x, y)
        self.assertEqual(tree_sha256((g.actor, g.critic, g.temperature)), original)

    def test_sac_scan_continuation_restores_all_dynamic_state_and_counter_phase(self):
        batches = {
            k: jnp.asarray(
                np.stack(
                    [_make_random_batch(np.random.default_rng(i))[k] for i in range(6)]
                )
            )
            for k in _make_random_batch(np.random.default_rng(0))
        }
        for twin in (False, True):
            g = agent(critic_use_cdq=twin)
            initial = (g._rng, g.actor, g.critic, g._target_critic, g.temperature)
            ref = {k: v[0] for k, v in batches.items()}

            def run(carry, bs, first):
                return sac_agent._update_sac_networks_scan(
                    *carry,
                    bs,
                    first,
                    g._cfg.gamma,
                    1,
                    twin,
                    g._cfg.target_tau,
                    g._cfg.temp_target_entropy,
                    ref,
                    2
                )

            whole = run(initial, batches, 0)
            half = run(initial, {k: v[:3] for k, v in batches.items()}, 0)
            restored = serialization.from_bytes(
                initial, serialization.to_bytes(half[:5])
            )
            self.assertEqual(tree_sha256(restored), tree_sha256(half[:5]))
            end = run(restored, {k: v[3:] for k, v in batches.items()}, 3)
            self.assertEqual(tree_sha256(whole[:5]), tree_sha256(end[:5]))
            for name in whole[-1]:
                np.testing.assert_array_equal(
                    whole[-1][name], jnp.concatenate([half[-1][name], end[-1][name]])
                )
            self.assertEqual(int(end[2].opt_state[0].count), 6)
            self.assertEqual(int(end[2].update_step), 6)


if __name__ == "__main__":
    unittest.main()
