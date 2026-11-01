"""CPU coverage only; tiny synthetic fixtures are not campaign evidence."""

import dataclasses
from pathlib import Path
import unittest
import jax
import numpy as np
import optax
from experiments.exp1 import compose_config
from experiments.exp12.m_selection import evaluate
from experiments.exp12.probe import probe_round, run_probe
from scripts.m_control_pilot import LIMITS, reserve
from scripts.positive_control import _setting_errors
from tests.test_exp12_probe import CFG, LinearCritic, _pool


class PilotTest(unittest.TestCase):
    def test_canonical_config_and_rejections(self):
        from omegaconf import OmegaConf

        cfg = compose_config(
            str(Path(__file__).resolve().parents[1] / "configs"),
            "base_exp12_m_control_ten",
            [],
        )
        self.assertEqual(
            (cfg.seed, cfg.run_role, cfg.critic_num_blocks, cfg.critic_hidden_dim),
            (9, "dev", 4, 1536),
        )
        self.assertEqual(
            (cfg.probe.rounds, cfg.probe.steps, cfg.probe.checks), (10, 2000, 50)
        )
        self.assertEqual(cfg.num_interaction_steps, 500000)
        self.assertEqual(_setting_errors(cfg, False, "m-control-ten"), [])
        for field, value in [
            ("seed", 8),
            ("seed", 10),
            ("probe.steps", 1000),
            ("probe.rounds", 5),
            ("probe.checks", 20),
            ("testing.force_trigger_check", 5),
        ]:
            other = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            OmegaConf.update(other, field, value)
            self.assertTrue(_setting_errors(other, False, "m-control-ten"), field)

    def test_ten_round_selection(self):
        loss = dict(
            zip(
                ["degraded", "injected_last", "injected_half", "injected_all"],
                [np.full(10, x) for x in [0.8, 0.7, 0.1, 0.05]],
            )
        )
        self.assertEqual(evaluate(loss, 0.1, 0.1, rounds=10)["chosen_m"], "half")
        self.assertIsNotNone(evaluate(loss, 0.1, 0.1)["stop"])
        for n in [5, 9, 11]:
            bad = {k: np.full(n, v[0]) for k, v in loss.items()}
            self.assertIsNone(evaluate(bad, 0.1, 0.1, rounds=10)["chosen_m"])
        loss["injected_all"][0] = np.nan
        self.assertIsNone(evaluate(loss, 0.1, 0.1, rounds=10)["chosen_m"])

    def test_real_2000_step_cpu_fit(self):
        obs, act = _pool()
        net = LinearCritic()
        params = net.init(jax.random.PRNGKey(1), obs[:1], act[:1])["params"]
        before = jax.tree_util.tree_map(np.array, params)
        out = probe_round(
            net,
            {"a": (net, params), "b": (net, params)},
            optax.adamw(1e-4),
            obs,
            act,
            jax.random.PRNGKey(3),
            dataclasses.replace(CFG, steps=2000),
        )
        from types import SimpleNamespace

        retained = []
        buffer = SimpleNamespace(
            _observations=np.asarray(obs),
            _actions=np.asarray(act),
            _num_in_buffer=len(obs),
        )
        cfg = dataclasses.replace(CFG, steps=2000, rounds=10)
        curves = run_probe(
            object(),
            buffer,
            net,
            {"a": (net, params)},
            optax.adamw(1e-4),
            9,
            2,
            cfg,
            on_round=lambda r, x: retained.append((r, x)),
        )
        plain = run_probe(
            object(), buffer, net, {"a": (net, params)}, optax.adamw(1e-4), 9, 2, cfg
        )
        self.assertEqual(curves["a"]["losses"].shape, (10, 2000))
        self.assertEqual([r for r, _ in retained], list(range(10)))
        for key in curves["a"]:
            np.testing.assert_array_equal(curves["a"][key], plain["a"][key])
        self.assertEqual(out["a"]["losses"].shape, (2000,))
        np.testing.assert_array_equal(out["a"]["losses"], out["b"]["losses"])
        for a, b in zip(
            jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(params)
        ):
            np.testing.assert_array_equal(a, b)

    def test_budget_fail_closed(self):
        ledger = reserve({}, "startup", "source")
        with self.assertRaises(ValueError):
            reserve(ledger, "calibrate", "source")
        ledger["stages"]["startup"]["status"] = "complete"
        ledger = reserve(ledger, "generate", "source")
        with self.assertRaises(ValueError):
            reserve(ledger, "calibrate", "source")
        ledger["stages"]["generate"]["status"] = "complete"
        ledger = reserve(ledger, "calibrate", "source")
        self.assertEqual(
            sum(r["reserved_seconds"] for r in ledger["stages"].values()), 36000
        )
        for stage in LIMITS:
            with self.assertRaises(ValueError):
                reserve(ledger, stage, "source")
        with self.assertRaises(ValueError):
            reserve(ledger, "calibrate", "changed-source")


if __name__ == "__main__":
    unittest.main()
