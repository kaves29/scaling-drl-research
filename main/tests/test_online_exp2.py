"""CPU contracts for the opt-in online Exp2 primitives."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.exp12.online_exp2 import (
    Shadow, gradient_cosine, paired_actor_intervention, paired_shadow_update,
)
from scale_rl.agents.sac.sac_network import SACActor, SACCritic, SACTemperature
from scale_rl.agents.sac.sac_update import update_critic, update_target_network
from scale_rl.networks.trainer import Trainer


def make_trainer(module, key, observations=None, actions=None, train=True):
    kwargs = {"rngs": key}
    if observations is not None:
        kwargs["observations"] = observations
    if actions is not None:
        kwargs["actions"] = actions
    return Trainer.create(module, kwargs, tx=optax.adamw(1e-4, weight_decay=.01) if train else None)


def equal_tree(left, right):
    return all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)))


class OnlineExp2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        obs = jnp.zeros((4, 3), jnp.float32)
        act = jnp.zeros((4, 2), jnp.float32)
        cls.actor = make_trainer(SACActor("residual", 1, 8, 2, jnp.float32), jax.random.PRNGKey(1), obs)
        cls.temperature = make_trainer(SACTemperature(.01), jax.random.PRNGKey(2))
        critic = make_trainer(SACCritic("residual", 1, 8, jnp.float32), jax.random.PRNGKey(3), obs, act)
        cls.shadows = {"small": Shadow(critic, critic.replace(tx=None, opt_state=None), critic.params),
                       "large": Shadow(critic, critic.replace(tx=None, opt_state=None), critic.params)}
        cls.batch = {"observation": np.ones((4, 3), np.float32),
                     "next_observation": np.ones((4, 3), np.float32) * .5,
                     "action": np.zeros((4, 2), np.float32),
                     "reward": np.array([1, 1, 1, 1], np.float32),
                     "terminated": np.array([0, 1, 0, 1], np.float32)}

    def test_shared_samples_independent_targets_and_reference_unchanged(self):
        before = jax.device_get((self.actor.params, self.actor.opt_state,
                                 self.temperature.params, self.batch["observation"].copy()))
        shadows, record = paired_shadow_update(self.actor, self.temperature, self.shadows,
                                               self.batch, jax.random.PRNGKey(4), .99, 1, .005)
        self.assertEqual(record["next_actions"].shape, (4, 2))
        self.assertTrue(equal_tree(shadows["small"].critic.params, shadows["large"].critic.params))
        self.assertEqual(shadows["small"].updates, 1)
        self.assertTrue(equal_tree(before[0], self.actor.params))
        self.assertTrue(equal_tree(before[1], self.actor.opt_state))
        self.assertTrue(equal_tree(before[2], self.temperature.params))
        np.testing.assert_array_equal(before[3], self.batch["observation"])
        changed_target = self.shadows["large"].target.replace(
            params=jax.tree.map(lambda x: x + .1, self.shadows["large"].target.params))
        _, other = paired_shadow_update(self.actor, self.temperature,
                                        {**self.shadows, "large": self.shadows["large"].__class__(
                                            self.shadows["large"].critic, changed_target,
                                            self.shadows["large"].fresh_params)},
                                        self.batch, jax.random.PRNGKey(4), .99, 1, .005)
        np.testing.assert_array_equal(record["next_actions"], other["next_actions"])
        self.assertNotEqual(float(other["critics"]["small"]["q_target_mean"]),
                            float(other["critics"]["large"]["q_target_mean"]))

    def test_actor_clone_and_label_swap(self):
        clones, records = paired_actor_intervention(self.actor, self.temperature,
                                                     self.shadows, self.batch["observation"],
                                                     jax.random.PRNGKey(5))
        self.assertTrue(equal_tree(clones["small"].params, clones["large"].params))
        self.assertTrue(equal_tree(clones["small"].opt_state, clones["large"].opt_state))
        self.assertTrue(equal_tree(self.actor.opt_state, self.shadows["small"].critic.opt_state) is False)
        self.assertGreater(float(records["small"]["gradient_norm"]), 0)
        self.assertGreater(float(records["small"]["delta_norm"]), 0)
        self.assertAlmostEqual(gradient_cosine(records["small"]["gradient"], records["large"]["gradient"]), 1.0, places=5)

    def test_invalid_batch_fails_closed(self):
        bad = {**self.batch, "reward": np.array([np.nan] * 4, np.float32)}
        with self.assertRaises(ValueError):
            paired_shadow_update(self.actor, self.temperature, self.shadows, bad,
                                 jax.random.PRNGKey(6), .99, 1, .005)
        wrong = {**self.batch, "reward": np.ones((4, 1), np.float32)}
        with self.assertRaisesRegex(ValueError, "shapes"):
            paired_shadow_update(self.actor, self.temperature, self.shadows, wrong,
                                 jax.random.PRNGKey(6), .99, 1, .005)

    def test_shadow_step_matches_source_sac_critic(self):
        shadow = self.shadows["small"]
        key = jax.random.PRNGKey(7)
        updated, _ = paired_shadow_update(self.actor, self.temperature, {"small": shadow},
                                          self.batch, key, .99, 1, .005)
        source, _ = update_critic(key, self.actor, shadow.critic, shadow.target,
                                  self.temperature, {k: jnp.asarray(v) for k, v in self.batch.items()},
                                  .99, 1, False)
        target, _ = update_target_network(source, shadow.target, .005)
        self.assertTrue(equal_tree(updated["small"].critic.params, source.params))
        self.assertTrue(equal_tree(updated["small"].critic.opt_state, source.opt_state))
        self.assertTrue(equal_tree(updated["small"].target.params, target.params))

    def test_termination_blocks_bootstrap_but_truncation_does_not(self):
        target = self.shadows["small"].target.replace(
            params=jax.tree.map(lambda x: x + .5, self.shadows["small"].target.params))
        changed = Shadow(self.shadows["small"].critic, target, self.shadows["small"].fresh_params)
        all_terminal = {**self.batch, "terminated": np.ones(4, np.float32)}
        all_truncated = {**self.batch, "terminated": np.zeros(4, np.float32),
                         "truncated": np.ones(4, np.float32)}
        _, terminal_a = paired_shadow_update(self.actor, self.temperature,
                                             {"one": self.shadows["small"]}, all_terminal,
                                             jax.random.PRNGKey(8), .99, 1, .005)
        _, terminal_b = paired_shadow_update(self.actor, self.temperature,
                                             {"one": changed}, all_terminal,
                                             jax.random.PRNGKey(8), .99, 1, .005)
        self.assertEqual(float(terminal_a["critics"]["one"]["q_target_mean"]),
                         float(terminal_b["critics"]["one"]["q_target_mean"]))
        _, truncated_a = paired_shadow_update(self.actor, self.temperature,
                                              {"one": self.shadows["small"]}, all_truncated,
                                              jax.random.PRNGKey(8), .99, 1, .005)
        _, truncated_b = paired_shadow_update(self.actor, self.temperature,
                                              {"one": changed}, all_truncated,
                                              jax.random.PRNGKey(8), .99, 1, .005)
        self.assertNotEqual(float(truncated_a["critics"]["one"]["q_target_mean"]),
                            float(truncated_b["critics"]["one"]["q_target_mean"]))

    def test_actor_gradient_depends_on_critic_and_labels_swap(self):
        first = self.shadows["small"]
        changed_params = jax.tree.map(lambda x: x * 1.1, first.critic.params)
        second = Shadow(first.critic.replace(params=changed_params), first.target, first.fresh_params)
        key = jax.random.PRNGKey(9)
        _, normal = paired_actor_intervention(self.actor, self.temperature,
                                              {"small": first, "large": second},
                                              self.batch["observation"], key)
        _, swapped = paired_actor_intervention(self.actor, self.temperature,
                                               {"small": second, "large": first},
                                               self.batch["observation"], key)
        self.assertFalse(equal_tree(normal["small"]["gradient"], normal["large"]["gradient"]))
        self.assertTrue(equal_tree(normal["small"]["gradient"], swapped["large"]["gradient"]))
        self.assertTrue(equal_tree(normal["large"]["parameter_delta"], swapped["small"]["parameter_delta"]))
