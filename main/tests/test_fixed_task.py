"""Small real-JAX contracts; no environment or production data mutation."""

import unittest
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from experiments.exp12 import fixed_task as ft
from experiments.exp12.probe import _fit
from experiments.exp12.diagnostic_artifacts import load_critic, replay_array


class Tiny(nn.Module):
    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=1)
        return nn.Dense(1)(nn.tanh(nn.Dense(8)(x)))


class FixedTaskTests(unittest.TestCase):
    def setUp(self):
        self.net = Tiny()
        self.o = jax.random.normal(jax.random.PRNGKey(1), (32, 3))
        self.a = jax.random.normal(jax.random.PRNGKey(2), (32, 2))
        self.p = self.net.init(jax.random.PRNGKey(3), self.o[:1], self.a[:1])["params"]
        self.z = jnp.sin(jnp.arange(32))
        self.tx = optax.adamw(1e-4, weight_decay=0.01)
        self.idx = ft.extended_indices(jax.random.PRNGKey(4), 8000, 8, 32)

    def test_frozen_indices_prefix(self):
        key = jax.random.PRNGKey(4)
        expected = jnp.concatenate(
            [
                jax.random.randint(key, (1000, 8), 0, 32),
                jax.random.randint(jax.random.fold_in(key, 1), (1000, 8), 0, 32),
            ]
        )
        np.testing.assert_array_equal(self.idx[:2000], expected)

    def test_aot_signature_and_original_repeat(self):
        exe = _fit.lower(
            self.net, self.tx, self.p, self.o, self.a, self.z, self.idx[:2000], 16
        ).compile()
        a = exe(self.p, self.o, self.a, self.z, self.idx[:2000])
        b = _fit(self.net, self.tx, self.p, self.o, self.a, self.z, self.idx[:2000], 16)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_original_segmented_and_identity(self):
        before = ft.tree_sha256(self.p)
        losses, final, offset = _fit(
            self.net, self.tx, self.p, self.o, self.a, self.z, self.idx[:2000], 16
        )
        target = offset + self.z
        st = ft.FitState(self.p, self.tx.init(self.p))
        st, l1 = ft.fit_segment(
            self.net, self.tx, st, self.o, self.a, target, self.idx[:1000]
        )
        st, l2 = ft.fit_segment(
            self.net, self.tx, st, self.o, self.a, target, self.idx[1000:2000]
        )
        metrics = ft.pool_metrics(self.net, st.params, self.o, self.a, target, 16)
        np.testing.assert_allclose(
            jnp.concatenate([l1, l2]), losses, rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(metrics["mse"], final, rtol=2e-5, atol=2e-6)
        st2 = ft.FitState(self.p, self.tx.init(self.p))
        st2, _ = ft.fit_segment(
            self.net, self.tx, st2, self.o, self.a, target, self.idx[:1000]
        )
        st2, _ = ft.fit_segment(
            self.net, self.tx, st2, self.o, self.a, target, self.idx[1000:2000]
        )
        self.assertEqual(ft.tree_sha256(st.params), ft.tree_sha256(st2.params))
        self.assertEqual(ft.tree_sha256(self.p), before)
        self.assertEqual(int(st.optimizer[0].count), 2000)
        initial = ft.FitState(self.p, self.tx.init(self.p))
        exe = ft.fit_segment.lower(
            self.net, self.tx, initial, self.o, self.a, target, self.idx[:1000]
        ).compile()
        carry, _ = exe(initial, self.o, self.a, target, self.idx[:1000])
        carry, _ = exe(carry, self.o, self.a, target, self.idx[1000:2000])
        self.assertEqual(ft.tree_sha256(carry.params), ft.tree_sha256(st.params))

    def test_pool_error_decomposition(self):
        m = ft.pool_metrics(self.net, self.p, self.o, self.a, self.z, 16)
        np.testing.assert_allclose(
            m["mse"],
            m["centered_error_variance"] + m["mean_error"] ** 2,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_own_offset_translation_cancels(self):
        q = ft.predictions(self.net, self.p, self.o, self.a, 16)
        np.testing.assert_allclose(
            q - (q.mean() + self.z), (q + 100) - (q.mean() + 100 + self.z), atol=2e-5
        )

    def test_continuation_keeps_adam_count(self):
        st = ft.FitState(self.p, self.tx.init(self.p))
        for k in range(8):
            st, _ = ft.fit_segment(
                self.net,
                self.tx,
                st,
                self.o,
                self.a,
                self.z,
                self.idx[k * 1000 : (k + 1) * 1000],
            )
        self.assertEqual(int(st.optimizer[0].count), 8000)

    def test_partial_checkpoint_and_readonly_replay(self):
        import orbax.checkpoint as ocp

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ocp.PyTreeCheckpointer().save(
                str(root / "fresh_critic"), {"params": self.p}
            )
            tree, provenance = load_critic(root, "fresh")
            self.assertEqual(ft.tree_sha256(tree), ft.tree_sha256(self.p))
            values = np.arange(96, dtype=np.float32).reshape(32, 3)
            np.savez(root / "buffer.npz", observations=values)
            mm = replay_array(root, "observations")
            np.testing.assert_array_equal(mm, values)
            self.assertFalse(mm.flags.writeable)
            self.assertTrue(provenance["leaf_sha256"])


if __name__ == "__main__":
    unittest.main()
