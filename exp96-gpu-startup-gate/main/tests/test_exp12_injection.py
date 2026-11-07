"""Phase 4: plasticity injection invariants (Nikishin et al. 2023, as in the Methodology)."""

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.exp12 import injection  # noqa: E402
from scale_rl.agents.sac.sac_network import SACCritic  # noqa: E402
from scale_rl.agents.sac.sac_update import update_target_network  # noqa: E402
from scale_rl.networks.trainer import Trainer  # noqa: E402

OBS, ACT, LR, WD = 6, 3, 1e-3, 1e-2


def _critic(blocks, width=16, seed=0, warm_steps=5):
    """A trained-for-a-few-steps SACCritic Trainer and its (lagged) target."""
    net = SACCritic("residual", blocks, width, jnp.float32)
    inputs = {"rngs": jax.random.PRNGKey(seed), "observations": jnp.zeros((1, OBS)), "actions": jnp.zeros((1, ACT))}
    critic = Trainer.create(net, inputs, tx=optax.adamw(LR, weight_decay=WD))
    target = Trainer.create(net, inputs, tx=None)
    for i in range(warm_steps):
        critic, _ = critic.apply_gradient(_loss(critic, _batch(100 + i)))
        target, _ = update_target_network(critic, target, 0.005)
    return critic, target


def _batch(seed, n=32):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.normal(size=(n, OBS)), jnp.float32), jnp.asarray(rng.uniform(-1, 1, (n, ACT)), jnp.float32),
            jnp.asarray(rng.normal(size=(n,)), jnp.float32))


def _loss(trainer, batch):
    obs, act, y = batch

    def loss_fn(params):
        q = trainer.network_def.apply({"params": params}, obs, act).reshape(-1)
        loss = jnp.mean((q - y) ** 2)
        return loss, {"loss": loss}

    return loss_fn


def _q_and_grad(trainer, obs, act):
    # Full FP32, as Check 1 (fork.CHECK1_PRECISION): the construction is exact, but TF32 input rounding in
    # the backward matmuls turns the ULP-level cotangent summation difference into TF32-level noise.
    with jax.default_matmul_precision("highest"):
        q = trainer.network_def.apply({"params": trainer.params}, obs, act)
        dq = jax.grad(lambda a: trainer.network_def.apply({"params": trainer.params}, obs, a).sum())(act)
    return np.asarray(q), np.asarray(dq)


def _inject(critic, target, label, seed=7):
    return injection.inject(critic, target, label, injection.injection_key(seed), LR, WD)


class InjectionInvariantsTest(unittest.TestCase):
    def test_predictions_unchanged_bit_for_bit_and_action_gradients_within_dtype_tolerance(self):
        # new(z) - copy(z) is exactly 0, so Q is bit-identical. Reverse mode sums the three
        # heads' cotangents in its own order, so dQ/da can differ at the ULP level (measured
        # <= 7.2 eps * max|dQ/da| at D6W1536 on CPU); Check 1 uses the dtype tolerance.
        obs, act, _ = _batch(1, 256)
        tol = injection.CHECK1_TOLERANCE_EPS * np.finfo(np.float32).eps
        for blocks in (2, 4):
            critic, target = _critic(blocks)
            for label in injection.M_LABELS:
                with self.subTest(blocks=blocks, m=label):
                    inj, inj_target = _inject(critic, target, label)
                    for before, after in ((critic, inj), (target, inj_target)):
                        q0, g0 = _q_and_grad(before, obs, act)
                        q1, g1 = _q_and_grad(after, obs, act)
                        np.testing.assert_array_equal(q0, q1)
                        self.assertLessEqual(np.abs(g0 - g1).max(), tol * np.abs(g0).max())

    def test_parameter_counts(self):
        critic, target = _critic(4)
        for label in injection.M_LABELS:
            with self.subTest(m=label):
                inj, _ = _inject(critic, target, label)
                original = injection.count_params(critic.params)
                head = injection.count_params(inj.params["new"])
                self.assertEqual(injection.trainable_count(inj.params), original)
                self.assertEqual(injection.count_params(inj.params), original + 2 * head)
                self.assertEqual(injection.count_params(inj.params["old"]), head)

    def test_head_boundary_per_label(self):
        self.assertEqual([injection.head_blocks(m, 4) for m in injection.M_LABELS], [1, 2, 4])
        self.assertEqual([injection.head_blocks(m, 6) for m in injection.M_LABELS], [1, 3, 6])
        self.assertEqual([injection.head_blocks(m, 2) for m in injection.M_LABELS], [1, 1, 2])
        critic, target = _critic(4)
        inj, _ = _inject(critic, target, "half")
        self.assertEqual(sorted(inj.params["trunk"]), ["Dense_0", "ResidualBlock_0", "ResidualBlock_1"])
        np.testing.assert_array_equal(inj.params["old"]["ResidualBlock_0"]["Dense_0"]["kernel"],
                                      critic.params["encoder"]["ResidualBlock_2"]["Dense_0"]["kernel"])
        np.testing.assert_array_equal(inj.params["old"]["LinearCritic_0"]["Dense_0"]["kernel"],
                                      critic.params["predictor"]["Dense_0"]["kernel"])

    def test_new_copies_identical_and_freshly_initialised(self):
        critic, target = _critic(2)
        inj, inj_target = _inject(critic, target, "last", seed=7)
        other, _ = _inject(critic, target, "last", seed=8)
        for a, b in zip(jax.tree_util.tree_leaves(inj.params["new"]), jax.tree_util.tree_leaves(inj.params["copy"])):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(jax.tree_util.tree_leaves(inj.params["new"]), jax.tree_util.tree_leaves(inj_target.params["new"])):
            np.testing.assert_array_equal(a, b)
        differs = [not np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(inj.params["new"]),
                                                            jax.tree_util.tree_leaves(other.params["new"]))]
        self.assertTrue(any(differs), "the injection key should determine the new head")
        self.assertFalse(np.array_equal(inj.params["new"]["LinearCritic_0"]["Dense_0"]["kernel"],
                                        inj.params["old"]["LinearCritic_0"]["Dense_0"]["kernel"]))

    def test_training_keeps_frozen_heads_bit_identical_and_moves_trainable_parts(self):
        critic, target = _critic(4)
        inj, _ = _inject(critic, target, "half")
        start = inj.params
        for i in range(50):
            inj, _ = inj.apply_gradient(_loss(inj, _batch(500 + i)))
        for part in ("old", "copy"):
            for a, b in zip(jax.tree_util.tree_leaves(start[part]), jax.tree_util.tree_leaves(inj.params[part])):
                np.testing.assert_array_equal(a, b, err_msg=f"{part} head changed")
        for part in ("new", "trunk"):
            moved = [not np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(start[part]),
                                                              jax.tree_util.tree_leaves(inj.params[part]))]
            self.assertTrue(all(moved), f"every {part} leaf should be updated")

    def test_optimizer_state_rules(self):
        critic, target = _critic(4, warm_steps=5)
        inj, _ = _inject(critic, target, "half")
        trunk_state, new_state = inj.opt_state["trunk"][0], inj.opt_state["new"][0]
        self.assertEqual(int(trunk_state.count), int(critic.opt_state[0].count))  # existing state kept
        self.assertEqual(int(new_state.count), 0)  # newly created
        np.testing.assert_array_equal(trunk_state.mu["ResidualBlock_1"]["Dense_0"]["kernel"],
                                      critic.opt_state[0].mu["encoder"]["ResidualBlock_1"]["Dense_0"]["kernel"])
        np.testing.assert_array_equal(trunk_state.nu["Dense_0"]["kernel"],
                                      critic.opt_state[0].nu["encoder"]["Dense_0"]["kernel"])
        for leaf in jax.tree_util.tree_leaves(new_state.mu) + jax.tree_util.tree_leaves(new_state.nu):
            self.assertFalse(np.any(np.asarray(leaf)))
        self.assertEqual(set(inj.opt_state), {"trunk", "new"})  # frozen parameters have no optimizer state
        inj, _ = inj.apply_gradient(_loss(inj, _batch(9)))
        self.assertEqual(int(inj.opt_state["trunk"][0].count), int(critic.opt_state[0].count) + 1)
        self.assertEqual(int(inj.opt_state["new"][0].count), 1)

    def test_gradients_reach_earlier_blocks_through_the_frozen_head(self):
        for label in ("half", "all"):
            critic, target = _critic(4)
            inj, _ = _inject(critic, target, label)
            grads = jax.grad(lambda p: _loss(inj.replace(params=p), _batch(3))(p)[0])(inj.params)
            for leaf in jax.tree_util.tree_leaves(grads["trunk"]):
                self.assertTrue(np.any(np.asarray(leaf) != 0), f"{label}: trunk gradient is zero")
            for part in ("old", "copy"):
                for leaf in jax.tree_util.tree_leaves(grads[part]):
                    self.assertFalse(np.any(np.asarray(leaf)), f"{part} parameters should get no gradient")

    def test_target_injection_and_polyak(self):
        critic, target = _critic(4)
        inj, inj_target = _inject(critic, target, "half")
        target_trunk, target_old = injection.split_params(target.params, 4, 2)
        for a, b in zip(jax.tree_util.tree_leaves(target_old), jax.tree_util.tree_leaves(inj_target.params["old"])):
            np.testing.assert_array_equal(a, b)
        copy0 = inj_target.params["copy"]
        for i in range(20):
            inj, _ = inj.apply_gradient(_loss(inj, _batch(700 + i)))
            prev = inj_target.params
            inj_target, _ = update_target_network(inj, inj_target, 0.005)
            for a, b in zip(jax.tree_util.tree_leaves(copy0), jax.tree_util.tree_leaves(inj_target.params["copy"])):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)  # stays at theta' (Polyak to an equal value)
            expected = jax.tree_util.tree_map(lambda p, tp: p * 0.005 + tp * 0.995, inj.params, prev)
            for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(inj_target.params)):
                np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
