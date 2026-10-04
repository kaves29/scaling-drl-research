"""Positive control and m-selection (amendments (c)-(e), (q)): arithmetic, shared offset, end to end."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from exp12_helpers import CONFIG_PATH, patch_wandb, tiny_overrides  # noqa: E402
from experiments.exp12.m_selection import evaluate, loss_rounds, pooled_sd, recovery, select_m  # noqa: E402
from experiments.exp12.probe import iqm, probe_round  # noqa: E402
from test_exp12_probe import CFG, LinearCritic, _pool  # noqa: E402


class MSelectionArithmeticTest(unittest.TestCase):
    def test_loss_rounds_ordered_numerically(self):
        r = {"check_index": 1, **{f"loss_r{i}": float(i) for i in range(12)}}
        np.testing.assert_array_equal(loss_rounds(r), np.arange(12.0))

    def test_recovery_toward_the_fresh_critic(self):
        self.assertEqual(recovery(0.8, 0.8), 0.0)
        self.assertEqual(recovery(0.8, 0.0), 1.0)  # back to the fresh critic's level, L = 0
        self.assertAlmostEqual(recovery(0.8, 0.2), 0.75)
        self.assertLess(recovery(0.8, 1.0), 0)  # worse than at the trigger

    def test_select_smallest_m_within_tolerance(self):
        self.assertEqual(select_m({"last": 0.5, "half": 0.85, "all": 0.9}, 0.10), "half")
        self.assertEqual(select_m({"last": 0.81, "half": 0.85, "all": 0.9}, 0.10), "last")
        self.assertEqual(select_m({"last": 0.5, "half": 0.6, "all": 0.9}, 0.10), "all")
        self.assertEqual(select_m({"last": 0.9, "half": 0.2, "all": 0.1}, 0.10), "last")
        self.assertEqual(select_m({"last": 0.78, "half": 0.89, "all": 0.89}, 0.10), "half")

    def test_pooled_sd(self):
        a, b = np.array([1.0, 2.0, 3.0]), np.array([10.0, 10.0, 13.0, 15.0])
        expect = np.sqrt((np.sum((a - a.mean()) ** 2) + np.sum((b - b.mean()) ** 2)) / (2 + 3))
        self.assertAlmostEqual(pooled_sd({"a": a, "b": b}), expect)
        self.assertAlmostEqual(pooled_sd({"a": a}), np.std(a, ddof=1))
        self.assertEqual(pooled_sd({"a": a, "b": a + 100}), pooled_sd({"a": a}))  # means do not mix

    def _loss(self, degraded, last, half, full):
        return {"degraded": np.array(degraded), "injected_last": np.array(last),
                "injected_half": np.array(half), "injected_all": np.array(full)}

    def test_evaluate_chooses_m(self):
        loss = self._loss([0.8] * 5, [0.7] * 5, [0.1] * 5, [0.05, 0.05, 0.05, 0.05, 0.06])
        out = evaluate(loss, 0.10, 0.10)
        self.assertIsNone(out["stop"])
        self.assertEqual(out["chosen_m"], "half")
        self.assertAlmostEqual(out["recovery"]["half"], 0.875)
        self.assertEqual(out["noise_series"], sorted(loss))  # the four real-settings evaluations
        self.assertAlmostEqual(out["noise_sd"], pooled_sd(loss))
        self.assertAlmostEqual(out["noise"], pooled_sd(loss) / 0.8)

    def test_evaluate_stops_on_noise(self):
        loss = self._loss([0.8, 0.8, 0.8, 0.8, 1.8], [0.7] * 5, [0.1] * 5, [0.05] * 5)
        out = evaluate(loss, 0.10, 0.10)  # pooled SD sqrt(0.8 / 16) = 0.224; / 0.8 = 0.28
        self.assertIsNone(out["chosen_m"])
        self.assertIn("noise", out["stop"])
        self.assertAlmostEqual(out["noise"], pooled_sd(loss) / 0.8)

    def test_noise_exactly_at_the_threshold_stops(self):
        offsets = np.array([-0.5, 0.5, 0.0, -0.5, 0.5])  # IQM 0; within-series sum of squares 1.0
        for l_trigger, stops in ((2.5, True), (0.25 / 0.0999, False)):  # pooled SD sqrt(1 / 16) = 0.25
            out = evaluate(self._loss(l_trigger + offsets, [0.0] * 5, [0.0] * 5, [0.0] * 5), 0.10, 0.10)
            self.assertAlmostEqual(out["noise_sd"], 0.25)
            self.assertAlmostEqual(out["l_trigger"], l_trigger)
            self.assertEqual(out["stop"] is not None, stops, out["noise"])

    def test_evaluate_stops_when_l_trigger_is_not_positive(self):
        for degraded in ([0.0] * 5, [-0.2] * 5):
            out = evaluate(self._loss(degraded, [0.1] * 5, [0.1] * 5, [0.1] * 5), 0.10, 0.10)
            self.assertIsNone(out["chosen_m"])
            self.assertIn("undefined", out["stop"])


class SharedOffsetTest(unittest.TestCase):
    def setUp(self):
        self.obs, self.act = _pool()
        self.net = LinearCritic()
        self.a = self.net.init(jax.random.PRNGKey(1), self.obs[:1], self.act[:1])["params"]
        self.b = {"Dense_0": {**self.a["Dense_0"], "bias": self.a["Dense_0"]["bias"] + 5.0}}
        self.critics = {"a": (self.net, self.a), "b": (self.net, self.b)}

    def _round(self, shared):
        return probe_round(self.net, self.critics, optax.adam(1e-2), self.obs, self.act, jax.random.PRNGKey(3),
                           CFG, shared_offset=shared)

    def test_default_is_each_critics_own_offset(self):
        own, default = self._round(None), probe_round(self.net, self.critics, optax.adam(1e-2), self.obs,
                                                      self.act, jax.random.PRNGKey(3), CFG)
        for n in ("a", "b"):
            for f in ("losses", "final_loss", "offset", "score"):
                np.testing.assert_array_equal(own[n][f], default[n][f])
        self.assertAlmostEqual(float(own["b"]["offset"]) - float(own["a"]["offset"]), 5.0, places=4)

    def test_shared_offset_modes(self):
        own = self._round(None)
        for mode, expect in (("a", own["a"]["offset"]), ("b", own["b"]["offset"]),
                             ("mean", (own["a"]["offset"] + own["b"]["offset"]) / 2)):
            out = self._round(mode)
            self.assertEqual(float(out["a"]["offset"]), float(out["b"]["offset"]))
            self.assertAlmostEqual(float(out["a"]["offset"]), float(expect), places=5)
            np.testing.assert_array_equal(out["a"]["b"], own["a"]["b"])  # b = Var(targets) ignores the offset
        # The critic whose own mean is the shared offset fits exactly as with its own offset.
        np.testing.assert_array_equal(self._round("a")["a"]["score"], own["a"]["score"])


def _run_exp1(run_dir, overrides):
    from experiments import exp1

    exp1.run({"experiment": "exp1", "config_path": CONFIG_PATH, "config_name": "base_exp12",
              "overrides": overrides, "checkpoint_dir": run_dir, "checkpoint_interval": 10**9,
              "checkpoint_start_frac": 0.0})


class PositiveControlEndToEndTest(unittest.TestCase):
    """A tiny dev run whose f*_run is forced at check 5 (checks 4 and 5 fire; test-only hook), then the script."""

    @classmethod
    def setUpClass(cls):
        cls.patchers = patch_wandb()
        cls.tmp = tempfile.mkdtemp()
        common = ["run_role=dev", "env.max_episode_steps=40", "fork.eval_episodes=1",
                  f"results_root={cls.tmp}/results"]
        cls.run_dir = os.path.join(cls.tmp, "dev_run")
        _run_exp1(cls.run_dir, tiny_overrides(seed=7, steps=400, extra=[
            *common, "testing.force_trigger_check=5", "fork.architectures=[D1W8]"]))
        cls.no_fork_dir = os.path.join(cls.tmp, "dev_run_no_fork")
        _run_exp1(cls.no_fork_dir, tiny_overrides(seed=7, steps=400, extra=[*common, "fork.architectures=[]"]))

    @classmethod
    def tearDownClass(cls):
        for p in cls.patchers:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _script(self, run_dir, out_name, allow_any=True):
        import positive_control

        out = os.path.join(self.tmp, out_name)
        code = positive_control.run(positive_control_args(run_dir, out, allow_any))
        with open(os.path.join(out, "positive_control.json")) as f:
            return code, json.load(f)

    def test_full_report(self):
        code, report = self._script(self.run_dir, "pc")
        self.assertEqual(code, 0 if report["status"] == "m_chosen" else 3)
        self.assertEqual(report["trigger_check"]["check_index"], 5)
        self.assertTrue(report["trigger_check"]["forced"])
        # The degraded critic on the trigger check's own streams reproduces the run's recorded probe exactly.
        self.assertEqual(report["trigger_reproduction_max_abs_diff"], 0.0)
        self.assertEqual(set(report["loss_rounds"]), {"degraded", "injected_last", "injected_half", "injected_all"})
        self.assertEqual(report["healthy_reference"], "fresh critic (check 0), L_healthy = 0")
        self.assertEqual(set(report["shared_offset_sensitivity"]), {"degraded", "fresh", "mean"})
        sel = report["selection"]
        # Noise: pooled SD over the four real-settings evaluations only (shared-offset repeats excluded).
        self.assertEqual(sel["noise_series"], ["degraded", "injected_all", "injected_half", "injected_last"])
        self.assertAlmostEqual(sel["noise_sd"], pooled_sd({n: np.array(v) for n, v in report["loss_rounds"].items()}))
        self.assertAlmostEqual(sel["l_trigger"], iqm(report["loss_rounds"]["degraded"]))
        if "recovery" in sel:
            for m in ("last", "half", "all"):
                self.assertAlmostEqual(sel["recovery"][m], recovery(sel["l_trigger"], sel["l_injected"][m]))
        if report["status"] == "m_chosen":
            self.assertEqual(report["chosen_m"], select_m(sel["recovery"], 0.10))
            self.assertLess(sel["noise"], 0.10)
        else:
            self.assertIsNone(report["chosen_m"])
            self.assertTrue(report["stop"])
        print(f"\n[positive control, tiny forced run] status={report['status']} stop={report['stop']}")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "pc", "positive_control_curves.npz")))

    def test_injection_does_not_change_predictions_before_the_probe(self):
        # D1W8: m = last = half = all = 1 block, so all three injected critics start from the same head.
        _, report = self._script(self.run_dir, "pc_offsets")
        offsets = {n: report["probe"][n]["offset_rounds"] for n in ("degraded", "injected_last")}
        np.testing.assert_allclose(offsets["degraded"], offsets["injected_last"], rtol=0, atol=1e-6)

    def test_never_triggered_run_stops(self):
        code, report = self._script(self.no_fork_dir, "pc_nofork")
        self.assertEqual((code, report["status"], report["chosen_m"]), (3, "stop", None))
        self.assertIn("never triggered", report["stop"])

    def test_refuses_wrong_setting(self):
        with self.assertRaises(SystemExit) as e:  # D1W8 hopper-hop is not D6W1536 dog-run
            self._script(self.run_dir, "x", allow_any=False)
        self.assertIn("D6W1536", str(e.exception))

    def test_cli_needs_no_definition_flags(self):
        import positive_control

        with self.assertRaises(SystemExit):
            positive_control.main(["--run_dir", "/a", "--out_dir", "/b", "--healthy_reference", "last_pre_trigger"])


def positive_control_args(run_dir, out_dir, allow_any=True):
    import argparse

    return argparse.Namespace(run_dir=run_dir, out_dir=out_dir, allow_any_setting=allow_any)


if __name__ == "__main__":
    unittest.main()
