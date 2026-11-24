import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.exp12.exp3_mechanisms import (
    actor_gradient_components,
    actor_policy_kl,
    gradient_balance,
    matched_entropy_weight,
    propose_actor_update,
    scale_actor_displacement,
    widen_actor,
)
from scale_rl.agents.sac.sac_network import SACActor, SACClippedDoubleCritic
from scale_rl.agents.sac.sac_update import update_actor
from scale_rl.networks.trainer import Trainer


class MechanismTest(unittest.TestCase):
    def setUp(self):
        self.obs = jax.random.normal(jax.random.PRNGKey(1), (16, 3))
        self.key = jax.random.PRNGKey(2)
        self.actor = self.actor_for("residual")
        self.critic = Trainer.create(
            SACClippedDoubleCritic("residual", 1, 8, jnp.float32),
            {"rngs": self.key, "observations": self.obs, "actions": jnp.zeros((16, 2))},
        )
        self.alpha = lambda: jnp.asarray(0.2)

    def actor_for(self, kind):
        return Trainer.create(
            SACActor(kind, 1, 8, 2, jnp.float32),
            {"rngs": self.key, "observations": self.obs},
            tx=optax.adamw(1e-3),
        )

    def proposal(self, **kwargs):
        return propose_actor_update(
            self.actor,
            self.critic,
            self.alpha,
            self.obs,
            self.key,
            critic_use_cdq=True,
            **kwargs,
        )[0]

    def test_baseline_proposal_reproduces_source_update(self):
        baseline, _ = update_actor(
            self.key,
            self.actor,
            self.critic,
            self.alpha,
            {"observation": self.obs},
            True,
        )
        proposal = self.proposal()
        for actual, expected in zip(
            jax.tree.leaves(proposal), jax.tree.leaves(baseline)
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_entropy_balance_match_and_scaled_policy(self):
        q, h = actor_gradient_components(
            self.actor, self.critic, self.obs, self.key, critic_use_cdq=True
        )
        stronger_q = jax.tree.map(lambda x: 4 * x, q)
        alpha = matched_entropy_weight(q, stronger_q, self.alpha())
        self.assertAlmostEqual(alpha, 4 * float(self.alpha()))
        self.assertAlmostEqual(
            float(gradient_balance(q, h, self.alpha())["entropy_to_critic_norm"]),
            float(gradient_balance(stronger_q, h, alpha)["entropy_to_critic_norm"]),
        )
        proposal = self.proposal(entropy_weight=alpha)
        half = scale_actor_displacement(self.actor, proposal, 0.5)
        self.assertEqual(int(half.update_step), int(self.actor.update_step))
        self.assertLess(
            float(actor_policy_kl(self.actor, half, self.obs)),
            float(actor_policy_kl(self.actor, proposal, self.obs)),
        )

    def test_saturation_penalty_modifies_proposal(self):
        normal, treated = self.proposal(), self.proposal(
            saturation_penalty=1.0, saturation_threshold=0.05
        )
        self.assertTrue(
            any(
                not np.array_equal(a, b)
                for a, b in zip(
                    jax.tree.leaves(normal.params), jax.tree.leaves(treated.params)
                )
            )
        )

    def test_widening_preserves_distribution_and_breaks_replica_symmetry(self):
        for kind in ("mlp", "residual"):
            with self.subTest(kind=kind):
                actor = self.actor_for(kind)
                wide = widen_actor(actor)
                self.assertEqual(wide.network_def.hidden_dim, 16)
                a, b = actor(self.obs).distribution, wide(self.obs).distribution
                np.testing.assert_allclose(a.mean(), b.mean(), atol=2e-6)
                np.testing.assert_allclose(a.stddev(), b.stddev(), atol=2e-6, rtol=2e-5)
                self.assertLess(
                    abs(float(actor_policy_kl(actor, wide, self.obs))), 1e-6
                )
                updated, _ = propose_actor_update(
                    wide,
                    self.critic,
                    self.alpha,
                    self.obs,
                    self.key,
                    critic_use_cdq=True,
                )
                if kind == "mlp":
                    outgoing = updated.params["encoder"]["MLPBlock_0"]["Dense_1"][
                        "bias"
                    ]
                else:
                    outgoing = updated.params["encoder"]["Dense_0"]["bias"]
                self.assertTrue(
                    np.any(np.asarray(outgoing[::2]) != np.asarray(outgoing[1::2]))
                )
