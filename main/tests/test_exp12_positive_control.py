"""Phase 4: positive control and m-selection (amendments (c)-(f)): arithmetic, shared offset, end to end."""

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
from experiments.exp12.m_selection import (  # noqa: E402
    Stop, evaluate, healthy_reference, loss_rounds, recovery, select_m, spread)
from experiments.exp12.probe import iqm, probe_round  # noqa: E402
from test_exp12_probe import CFG, LinearCritic, _pool  # noqa: E402


def _record(k, rounds):
    return {"check_index": k, "loss_iqm": iqm(rounds), **{f"loss_r{r}": v for r, v in enumerate(rounds)}}


class MSelectionArithmeticTest(unittest.TestCase):
    RECORDS = [_record(0, [0.0] * 5), _record(1, [0.1, 0.2, 0.3, 0.4, 0.5]),
               _record(2, [0.0, 0.1, 0.1, 0.1, 0.9]), _record(3, [1.0] * 5), _record(4, [2.0] * 5)]

    def test_loss_rounds_ordered_numerically(self):
        r = {"check_index": 1, **{f"loss_r{i}": float(i) for i in range(12)}}
        np.testing.assert_array_equal(loss_rounds(r), np.arange(12.0))

    def test_healthy_reference_definitions(self):
        value, used = healthy_reference(self.RECORDS, 3, "last_pre_trigger")
        self.assertEqual((value, used), (iqm([0.0, 0.1, 0.1, 0.1, 0.9]), [2]))
        value, used = healthy_reference(self.RECORDS, 3, "iqm_pre_trigger")
        pooled = [0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.1, 0.1, 0.1, 0.9]
        self.assertAlmostEqual(value, iqm(pooled))
        self.assertEqual(used, [1, 2])  # check 0 (the fresh probe) and the trigger check and later are excluded

    def test_no_pre_trigger_check_stops(self):
        with self.assertRaises(Stop):
            healthy_reference(self.RECORDS, 1, "last_pre_trigger")
        with self.assertRaises(ValueError):
            healthy_reference(self.RECORDS, 3, "median")

    def test_recovery_endpoints(self):
        self.assertEqual(recovery(1.0, 1.0, 0.2), 0.0)
        self.assertEqual(recovery(1.0, 0.2, 0.2), 1.0)
        self.assertAlmostEqual(recovery(1.0, 0.6, 0.2), 0.5)
        self.assertLess(recovery(1.0, 1.4, 0.2), 0)  # worse than at the trigger

    def test_select_smallest_m_within_tolerance(self):
        self.assertEqual(select_m({"last": 0.5, "half": 0.85, "all": 0.9}, 0.10), "half")
        self.assertEqual(select_m({"last": 0.81, "half": 0.85, "all": 0.9}, 0.10), "last")
        self.assertEqual(select_m({"last": 0.5, "half": 0.6, "all": 0.9}, 0.10), "all")
        self.assertEqual(select_m({"last": 0.9, "half": 0.2, "all": 0.1}, 0.10), "last")
        self.assertEqual(select_m({"last": 0.78, "half": 0.89, "all": 0.89}, 0.10), "half")

    def test_spread(self):
        self.assertEqual(spread([1.0, 3.0, 2.0], "range"), 2.0)
        self.assertAlmostEqual(spread([1.0, 3.0, 2.0], "std"), 1.0)

    def _loss(self, degraded, last, half, full):
        return {"degraded": np.array(degraded), "injected_last": np.array(last),
                "injected_half": np.array(half), "injected_all": np.array(full)}

    def test_evaluate_chooses_m(self):
        loss = self._loss([1.0] * 5, [0.9] * 5, [0.3] * 5, [0.25, 0.25, 0.25, 0.25, 0.26])
        out = evaluate(loss, 0.2, "range", 0.10, 0.10)
        self.assertIsNone(out["stop"])
        self.assertEqual(out["chosen_m"], "half")
        self.assertAlmostEqual(out["recovery"]["half"], 0.875)
        self.assertAlmostEqual(out["noise_recovery_units"]["injected_all"], 0.01 / 0.8)

    def test_evaluate_stops_on_noise(self):
        loss = self._loss([1.0, 1.0, 1.0, 1.0, 1.2], [0.9] * 5, [0.3] * 5, [0.25] * 5)
        out = evaluate(loss, 0.2, "range", 0.10, 0.10)  # 0.2 / 0.8 = 0.25 recovery units
        self.assertIsNone(out["chosen_m"])
        self.assertIn("noise", out["stop"])
        self.assertAlmostEqual(out["noise"], 0.25)

    def test_evaluate_stops_when_trigger_not_above_healthy(self):
        loss = self._loss([0.2] * 5, [0.1] * 5, [0.1] * 5, [0.1] * 5)
        out = evaluate(loss, 0.3, "std", 0.10, 0.10)
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
    """A tiny dev run whose trigger is forced at check 5 (test-only hook), then the script."""

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

    def _script(self, run_dir, out_name, *extra):
        import positive_control

        out = os.path.join(self.tmp, out_name)
        code = positive_control.run(positive_control_args(run_dir, out, *extra))
        with open(os.path.join(out, "positive_control.json")) as f:
            return code, json.load(f)

    def test_full_report(self):
        code, report = self._script(self.run_dir, "pc", "last_pre_trigger", "range")
        self.assertEqual(code, 0 if report["status"] == "m_chosen" else 3)
        self.assertEqual(report["trigger_check"]["check_index"], 5)
        self.assertTrue(report["trigger_check"]["forced"])
        # The degraded critic on the trigger check's own streams reproduces the run's recorded probe exactly.
        self.assertEqual(report["trigger_reproduction_max_abs_diff"], 0.0)
        self.assertEqual(set(report["loss_rounds"]), {"degraded", "injected_last", "injected_half", "injected_all"})
        self.assertEqual(report["healthy_reference_checks"], [4])
        self.assertEqual([c["check_index"] for c in report["pre_trigger_checks"]], [1, 2, 3, 4])
        self.assertEqual(set(report["shared_offset_sensitivity"]), {"degraded", "fresh", "mean"})
        sel = report["selection"]
        if "recovery" in sel:
            for m in ("last", "half", "all"):
                self.assertAlmostEqual(sel["recovery"][m], recovery(sel["l_trigger"], sel["l_injected"][m],
                                                                    sel["l_healthy"]))
        if report["status"] == "m_chosen":
            self.assertEqual(report["chosen_m"], select_m(sel["recovery"], 0.10))
        else:
            self.assertIsNone(report["chosen_m"])
            self.assertTrue(report["stop"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "pc", "positive_control_curves.npz")))

    def test_injection_does_not_change_predictions_before_the_probe(self):
        # D1W8: m = last = half = all = 1 block, so all three injected critics start from the same head.
        _, report = self._script(self.run_dir, "pc_iqm", "iqm_pre_trigger", "std")
        self.assertEqual(report["healthy_reference_checks"], [1, 2, 3, 4])
        offsets = {n: report["probe"][n]["offset_rounds"] for n in ("degraded", "injected_last")}
        np.testing.assert_allclose(offsets["degraded"], offsets["injected_last"], rtol=0, atol=1e-6)

    def test_never_triggered_run_stops(self):
        code, report = self._script(self.no_fork_dir, "pc_nofork", "last_pre_trigger", "range")
        self.assertEqual((code, report["status"], report["chosen_m"]), (3, "stop", None))
        self.assertIn("never triggered", report["stop"])

    def test_refuses_wrong_setting(self):
        import positive_control

        with self.assertRaises(SystemExit) as e:  # D1W8 hopper-hop is not D6W1536 dog-run
            positive_control.run(positive_control_args(self.run_dir, os.path.join(self.tmp, "x"), "last_pre_trigger",
                                                       "range", allow_any=False))
        self.assertIn("D6W1536", str(e.exception))

    def test_requires_both_pending_definitions(self):
        import positive_control

        for argv in (["--run_dir", "/a", "--out_dir", "/b", "--noise_statistic", "range"],
                     ["--run_dir", "/a", "--out_dir", "/b", "--healthy_reference", "last_pre_trigger"]):
            with self.assertRaises(SystemExit):
                positive_control.main(argv)


def positive_control_args(run_dir, out_dir, healthy, noise, allow_any=True):
    import argparse

    return argparse.Namespace(run_dir=run_dir, out_dir=out_dir, healthy_reference=healthy, noise_statistic=noise,
                              allow_any_setting=allow_any)


if __name__ == "__main__":
    unittest.main()
