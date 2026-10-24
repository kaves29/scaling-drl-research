"""Known-answer pilot measurement and diagnostic state contracts (CPU only)."""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from experiments.exp12 import fixed_task as ft, probe
from experiments.exp12.probe_tasks import fixed_task_paired_summary
from experiments.exp12.trainer import evaluate_episodes
from tests.test_fixed_task import Tiny


class PilotContracts(unittest.TestCase):
    def test_terminal_success_belongs_to_finished_episode(self):
        # SyncVectorEnv auto-resets: top-level info is the NEXT episode's reset.
        for terminal, reset in ((True, False), (False, True)):
            env = mock.Mock()
            env.reset.return_value = (np.zeros((1, 2)), {})
            env.step.return_value = (
                np.zeros((1, 2)),
                np.array([7.0]),
                np.array([False]),
                np.array([True]),
                {
                    "success": np.array([reset]),
                    "final_info": np.array([{"success": terminal}], object),
                },
            )
            agent = mock.Mock()
            info, returns, lengths, successes = evaluate_episodes(agent, env, 1)
            self.assertEqual(successes, [float(terminal)])
            self.assertEqual(returns, [7.0])
            self.assertEqual(lengths, [1])
            self.assertEqual(info["eval/avg_return"], 7.0)
            self.assertFalse(agent.sample_actions.call_args.kwargs["training"])

    def test_pairing_rejects_nonpositive_or_noninteger_completed_horizons(self):
        args = [np.ones(10), np.ones(10), np.ones(10), list(range(10)), list(range(10))]
        for horizon in (0, -2000, 2000.5, True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                fixed_task_paired_summary(*args, horizon, horizon)

    def test_first_update_parameter_and_function_changes_have_known_values(self):
        from tests.test_exp12_probe import ConstantCritic

        net = ConstantCritic()
        params = {"c": jnp.array([2.0])}
        tx = optax.sgd(0.1)
        out = ft.first_update(
            net,
            tx,
            ft.FitState(params, tx.init(params)),
            jnp.zeros((4, 2)),
            jnp.zeros((4, 1)),
            jnp.zeros(4),
            jnp.arange(4),
            2,
        )
        self.assertAlmostEqual(float(out["parameter_change_l2"]), 0.4, places=6)
        self.assertAlmostEqual(float(out["parameter_change_rms"]), 0.4, places=6)
        self.assertAlmostEqual(float(out["prediction_change_rms"]), 0.4, places=6)
        self.assertAlmostEqual(float(out["minibatch_mse_before"]), 4.0)

    def test_own_offsets_are_distinct_targets_but_shared_base_and_samples(self):
        from tests.test_exp12_probe import LinearCritic

        o = jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
        a = jnp.zeros((8, 1))
        net = LinearCritic()
        p = net.init(jax.random.PRNGKey(1), o[:1], a[:1])["params"]
        q = jax.tree_util.tree_map(lambda x: x.copy(), p)
        q["Dense_0"]["bias"] = q["Dense_0"]["bias"] + 100
        cfg = probe.ProbeConfig(5, 2, 20, 8, 4, 1e5, 4)
        with mock.patch.object(probe, "_fit", wraps=probe._fit) as fits:
            result = probe.probe_round(
                net,
                {"fresh": (net, p), "current": (net, q)},
                optax.sgd(0),
                o,
                a,
                probe.probe_key(1, 1, 0),
                cfg,
            )
        x, y = [call.args for call in fits.call_args_list]
        for i in (3, 4, 5, 6):
            np.testing.assert_array_equal(x[i], y[i])
        self.assertAlmostEqual(
            float(result["current"]["offset"] - result["fresh"]["offset"]),
            100,
            places=4,
        )
        np.testing.assert_allclose(
            result["current"]["losses"], result["fresh"]["losses"], atol=2e-4
        )

    def test_minibatch_loss_changes_without_any_update(self):
        from tests.test_exp12_probe import ConstantCritic

        net = ConstantCritic()
        loss, final, _ = probe._fit(
            net,
            optax.sgd(0),
            {"c": jnp.zeros(1)},
            jnp.zeros((2, 2)),
            jnp.zeros((2, 1)),
            jnp.array([1.0, 10.0]),
            jnp.array([[0], [1]]),
            1,
        )
        np.testing.assert_array_equal(loss, [1, 100])
        self.assertEqual(float(final), 50.5)

    def test_segment_serialization_preserves_all_state_and_next_update(self):
        net = Tiny()
        o = jax.random.normal(jax.random.PRNGKey(7), (32, 3))
        a = jnp.zeros((32, 2))
        p = net.init(jax.random.PRNGKey(8), o[:1], a[:1])["params"]
        tx = optax.adamw(1e-4, weight_decay=0.01)
        initial = ft.FitState(p, tx.init(p))
        idx = ft.extended_indices(probe.probe_key(1, 0, 0), 2000, 8, 32)
        y = jnp.sin(jnp.arange(32))
        halfway, _ = ft.fit_segment(net, tx, initial, o, a, y, idx[:1000])
        restored = serialization.from_bytes(initial, serialization.to_bytes(halfway))
        self.assertEqual(ft.tree_sha256(restored), ft.tree_sha256(halfway))
        direct, l1 = ft.fit_segment(net, tx, halfway, o, a, y, idx[1000:])
        resumed, l2 = ft.fit_segment(net, tx, restored, o, a, y, idx[1000:])
        self.assertEqual(ft.tree_sha256(direct), ft.tree_sha256(resumed))
        np.testing.assert_array_equal(l1, l2)
        self.assertEqual(int(resumed.optimizer[0].count), 2000)
        reset = ft.FitState(halfway.params, tx.init(halfway.params))
        broken, _ = ft.fit_segment(net, tx, reset, o, a, y, idx[1000:])
        self.assertNotEqual(ft.tree_sha256(broken), ft.tree_sha256(resumed))

    def test_score_baseline_and_iqm_are_known_answers(self):
        result = {
            "fresh": {"score": np.array([0.2, 0.3, 0.4, 0.5, 0.6])},
            "current": {"score": np.array([0.1, 0.1, 0.1, 0.1, 0.1])},
        }
        summary = probe.summarize(result)
        self.assertAlmostEqual(summary["loss_iqm"], 0.3)
        self.assertAlmostEqual(summary["score_fresh_iqm"], 0.4)
        self.assertAlmostEqual(probe.iqm(np.arange(10)), 4.5)
        self.assertTrue(summary["valid"])


if __name__ == "__main__":
    unittest.main()
