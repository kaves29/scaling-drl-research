"""Phase 4: fork, control restart, identity fork, injected arm, Checks 1-3, kill-and-resume around the fork."""

import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp12_helpers import CONFIG_PATH, load_agent_tree, patch_wandb, state_differences, tiny_overrides  # noqa: E402
from experiments.exp12 import exp2_ledger, fork, ledger  # noqa: E402
from experiments.exp12.probe import iqm  # noqa: E402
from experiments.exp12.state import latest_state_dir  # noqa: E402

RUNNER = str(Path(__file__).resolve().parent / "exp12_subprocess_runner.py")
STEPS, FORCED_CHECK = 400, 5  # checks every 20 steps -> fork at 100; horizon 100 -> arm end 200


def fork_overrides(results_root, extra=()):
    return tiny_overrides("hopper-hop", "dmc_medium", steps=STEPS, extra=[
        "run_role=dev", f"testing.force_trigger_check={FORCED_CHECK}", "fork.architectures=[D1W8]",
        "env.max_episode_steps=40", "fork.eval_episodes=1", "fork.identity_snapshot_steps=50",
        f"results_root={results_root}", *extra])


def run_exp1(run_dir, overrides, interval=10**9):
    from experiments import exp1

    exp1.run({"experiment": "exp1", "config_path": CONFIG_PATH, "config_name": "base_exp12",
              "overrides": overrides, "checkpoint_dir": run_dir, "checkpoint_interval": interval,
              "checkpoint_start_frac": 0.0})


def run_arm(arm_dir, overrides, source, arm, m="half", interval=10**9):
    from experiments import exp2_arm

    exp2_arm.run({"experiment": "exp2_arm", "config_path": CONFIG_PATH, "config_name": "base_exp12",
                  "overrides": list(overrides) + [f"fork.source={source}", f"fork.arm={arm}", f"injection.m={m}"],
                  "checkpoint_dir": arm_dir, "checkpoint_interval": interval, "checkpoint_start_frac": 0.0})


def _meta(state_dir):
    with open(Path(state_dir) / "meta.pkl", "rb") as f:
        return pickle.load(f)


def _without_arm(rows):
    return [{k: v for k, v in r.items() if k != "arm"} for r in rows]


class ForkEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patchers = patch_wandb()
        cls.tmp = tempfile.mkdtemp()
        cls.results = os.path.join(cls.tmp, "results")
        cls.overrides = fork_overrides(cls.results)
        cls.run_dir = os.path.join(cls.tmp, "run")
        run_exp1(cls.run_dir, cls.overrides)
        cls.arm_dirs = {}
        for arm in ("identity", "injected"):
            cls.arm_dirs[arm] = os.path.join(cls.tmp, f"arm_{arm}")
            run_arm(cls.arm_dirs[arm], cls.overrides, cls.run_dir, arm)
        cls.plan = fork.read_fork(cls.run_dir)
        cls.run_key = cls.plan["run_key"]
        cls.exp2 = exp2_ledger.run_root(cls.run_key, cls.results)

    @classmethod
    def tearDownClass(cls):
        for p in cls.patchers:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_fork_plan_and_files(self):
        p = self.plan
        self.assertEqual((p["fork_step"], p["fork_check_index"]), (100, FORCED_CHECK))
        self.assertEqual((p["horizon_steps"], p["arm_end_step"], p["control_end_step"]), (100, 200, 400))
        self.assertEqual(p["eval_every_steps"], 4)
        d = fork.fork_dir(self.run_dir)
        for name in ("FORK_READY", "fork.json", "panel.npz", "check1_pre.npz", "check1_control.npz"):
            self.assertTrue((d / name).exists(), name)
        self.assertEqual(fork.load_npz(d / "panel.npz")["observation"].shape[0], 256)
        self.assertEqual(_meta(latest_state_dir(d / "state"))["interaction_step"], 100)

    def test_exp1_ledger_records_fork_and_control_completes(self):
        runs, checks = ledger.load(self.results, include_dev=True)
        run = runs.iloc[0]
        self.assertEqual(run.fork_interaction_step, 100)
        self.assertEqual(run.f_star_check, FORCED_CHECK)
        self.assertEqual(run.status, "complete")
        self.assertEqual(list(checks.check_index), list(range(21)))
        self.assertEqual(_meta(latest_state_dir(Path(self.run_dir) / "state"))["interaction_step"], 400)

    def test_identity_fork_is_bit_identical(self):
        control = latest_state_dir(fork.fork_dir(self.run_dir) / "identity_control")
        identity = latest_state_dir(Path(self.arm_dirs["identity"]) / "identity_identity")
        self.assertEqual(_meta(control)["interaction_step"], 150)
        self.assertEqual(state_differences(control, identity, ignore_meta=("wandb_run_id", "extra_state")), [])
        a, b = _meta(control)["extra_state"], _meta(identity)["extra_state"]
        self.assertEqual(a["probe_records"], b["probe_records"])
        self.assertEqual(_without_arm(a["post_fork_evals"]), _without_arm(b["post_fork_evals"]))

    def test_check1(self):
        with open(self.exp2 / "check1_identity.json") as f:
            identity = json.load(f)
        with open(self.exp2 / "check1_injected.json") as f:
            injected = json.load(f)
        self.assertTrue(identity["pass"] and injected["pass"])
        for pair in identity["pairs"].values():
            self.assertEqual((pair["max_abs_dq"], pair["max_abs_d_dq_da"], pair["tolerance_eps"]), (0.0, 0.0, 0.0))
        self.assertEqual(injected["pairs"]["pre_vs_after"]["max_abs_dq"], 0.0)
        self.assertEqual(injected["pairs"]["pre_vs_after"]["tolerance_eps"], 64)
        self.assertEqual(injected["pairs"]["pre_vs_control"], {**injected["pairs"]["pre_vs_control"],
                                                               "max_abs_dq": 0.0, "max_abs_d_dq_da": 0.0,
                                                               "tolerance_eps": 0.0})
        self.assertLessEqual(injected["max_eps_units"], 64)
        for c in (identity, injected):
            self.assertFalse(c["tf32_detected"])
            self.assertIn("float32_matmul_max_rel_error", c["precision"])
            self.assertIn("device_kind", c["precision"])

    def test_check2_written_and_paired_with_the_exp1_check(self):
        with open(self.exp2 / "check2.json") as f:
            c2 = json.load(f)
        self.assertEqual(c2["check_index"], FORCED_CHECK)
        diff = np.array(c2["score_injected_rounds"]) - np.array(c2["score_control_rounds"])
        np.testing.assert_array_equal(c2["paired_difference_rounds"], diff)
        self.assertAlmostEqual(c2["paired_difference_iqm"], iqm(diff))
        self.assertLessEqual(c2["paired_difference_ci_low"], c2["paired_difference_ci_high"])
        self.assertEqual(c2["pass"], bool(c2["paired_difference_ci_low"] > 0))
        # The control critic at the fork is the Exp 1 run's current critic at that check, on the same pool.
        _, checks = ledger.load(self.results, include_dev=True)
        row = checks[checks.check_index == FORCED_CHECK].iloc[0]
        np.testing.assert_allclose([row[f"score_current_r{r}"] for r in range(5)], c2["score_control_rounds"])

    def test_arm_records(self):
        for arm, last_check in (("control", 20), ("injected", 10)):
            d = self.exp2 / f"arm_{arm}"
            evals = pd.read_csv(d / "eval_episodes.csv")
            self.assertEqual(sorted(evals.eval_index.unique()), list(range(26)))
            self.assertEqual(evals.steps_since_fork.max(), 100)
            checks = pd.read_csv(d / "checks.csv")
            self.assertEqual(list(checks.check_index), list(range(FORCED_CHECK, last_check + 1)))
            metrics = pd.read_csv(d / "metrics.csv")
            self.assertIn("train/q1_mean", metrics.columns)
            self.assertIn("train/actor_gnorm", metrics.columns)
        ctrl = pd.read_csv(self.exp2 / "arm_control" / "eval_episodes.csv")
        inj = pd.read_csv(self.exp2 / "arm_injected" / "eval_episodes.csv")
        self.assertEqual(ctrl[ctrl.eval_index == 0]["return"].tolist(), inj[inj.eval_index == 0]["return"].tolist())

    def test_injected_arm_frozen_head_unchanged_and_new_head_trained(self):
        final = load_agent_tree(latest_state_dir(Path(self.arm_dirs["injected"]) / "state"))["critic"]["params"]
        pre = load_agent_tree(latest_state_dir(fork.fork_dir(self.run_dir) / "state"))["critic"]["params"]
        np.testing.assert_array_equal(final["old"]["LinearCritic_0"]["Dense_0"]["kernel"],
                                      pre["predictor"]["Dense_0"]["kernel"])
        for a, b in zip(jax.tree_util.tree_leaves(final["new"]), jax.tree_util.tree_leaves(final["copy"])):
            self.assertFalse(np.array_equal(a, b), "the trainable new head should have moved away from its copy")
        self.assertEqual(_meta(latest_state_dir(Path(self.arm_dirs["injected"]) / "state"))["interaction_step"], 200)

    def test_fork_is_invisible_to_the_exp1_trajectory(self):
        # Same run with forking disabled: the control's continuation must be bit-identical, which shows the
        # save/restore, the post-fork evaluations and the extra probe checks do not touch training.
        plain = os.path.join(self.tmp, "no_fork")
        run_exp1(plain, fork_overrides(os.path.join(self.tmp, "results_no_fork"), ["fork.enabled=false"]))
        self.assertEqual(state_differences(latest_state_dir(Path(plain) / "state"),
                                           latest_state_dir(Path(self.run_dir) / "state"),
                                           ignore_meta=("wandb_run_id", "extra_state")), [])

    def test_arm_refuses_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "differs from the parent"):
            run_arm(os.path.join(self.tmp, "bad_cfg"), self.overrides + ["critic_hidden_dim=16"], self.run_dir,
                    "injected")
        with self.assertRaisesRegex(ValueError, "injection.m"):
            run_arm(os.path.join(self.tmp, "bad_m"), self.overrides, self.run_dir, "injected", m="null")
        with self.assertRaisesRegex(ValueError, "FORK_READY"):
            run_arm(os.path.join(self.tmp, "no_fork_arm"), self.overrides, os.path.join(self.tmp, "nowhere"),
                    "injected")
        with self.assertRaisesRegex(ValueError, "run_role=dev"):
            run_exp1(os.path.join(self.tmp, "forced_confirmatory"),
                     fork_overrides(self.results, ["run_role=confirmatory"]))
        other = {**fork.device_info(), "device_kind": "Some other GPU"}
        with mock.patch.object(fork, "device_info", return_value=other):
            with self.assertRaisesRegex(RuntimeError, "same device model"):
                run_arm(os.path.join(self.tmp, "other_gpu"), self.overrides, self.run_dir, "injected")

    def test_arm_stops_before_training_when_check1_fails(self):
        from experiments.exp2_arm import Check1Failed

        record = self.exp2 / "check1_identity.json"
        kept = record.read_text()
        arm_dir = os.path.join(self.tmp, "tf32_arm")
        tf32 = {**fork.matmul_precision_report(), "tf32_detected": True}
        try:
            with mock.patch.object(fork, "matmul_precision_report", return_value=tf32):
                with self.assertRaisesRegex(Check1Failed, "stop and ask"):
                    run_arm(arm_dir, self.overrides, self.run_dir, "identity")
            with open(record) as f:
                self.assertTrue(json.load(f)["tf32_detected"])
        finally:
            record.write_text(kept)
        self.assertIsNone(latest_state_dir(Path(arm_dir) / "state"))  # stopped before its first save


class ForkUnitTest(unittest.TestCase):
    def _cfg(self, n=1000, horizon=0.25, every=0.01, max_total=1.2):
        from omegaconf import OmegaConf

        return OmegaConf.create({"num_interaction_steps": n, "fork": {
            "horizon_fraction": horizon, "eval_every_fraction": every, "max_total_fraction": max_total,
            "eval_episodes": 10}})

    def test_fork_plan(self):
        p = fork.fork_plan(self._cfg(), 300, 6)
        self.assertEqual((p["arm_end_step"], p["control_end_step"], p["horizon_steps"]), (550, 1000, 250))
        p = fork.fork_plan(self._cfg(), 950, 19)  # the control runs past 100% to fork + 25%
        self.assertEqual((p["arm_end_step"], p["control_end_step"]), (1200, 1200))
        with self.assertRaises(ValueError):
            fork.fork_plan(self._cfg(), 960, 19)  # beyond the 120% maximum
        self.assertEqual([fork.eval_due(p, s) for s in (949, 950, 960, 1200, 1210)], [None, 0, 1, 25, None])

    def _shifted(self, pre, field, eps_units):
        off = {k: v.copy() for k, v in pre.items()}
        off[field] = off[field] + eps_units * float(np.finfo(np.float32).eps) * np.abs(pre[field]).max()
        return off

    def test_check1_compares_values_and_action_gradients(self):
        rng = np.random.default_rng(0)
        pre = {"q": rng.normal(size=8).astype(np.float32), "dq_da": rng.normal(size=(8, 2)).astype(np.float32)}
        same = {k: v.copy() for k, v in pre.items()}
        self.assertTrue(fork.check1(pre, same, same, 64, injected=True)["pass"])
        for field in ("q", "dq_da"):
            result = fork.check1(pre, self._shifted(pre, field, 100), same, 64, injected=True)
            self.assertFalse(result["pass"], field)
            self.assertFalse(result["pairs"]["pre_vs_after"]["pass"])
            self.assertTrue(result["pairs"]["pre_vs_control"]["pass"])
            self.assertGreater(result["max_eps_units"], 64)
            self.assertTrue(fork.check1(pre, self._shifted(pre, field, 10), same, 64, injected=True)["pass"])

    def test_check1_control_and_identity_must_be_bit_exact(self):
        rng = np.random.default_rng(1)
        pre = {"q": rng.normal(size=8).astype(np.float32), "dq_da": rng.normal(size=(8, 2)).astype(np.float32)}
        one_ulp = {"q": np.nextafter(pre["q"], np.float32(np.inf)), "dq_da": pre["dq_da"].copy()}
        result = fork.check1(pre, pre, one_ulp, 64, injected=True)  # control off by one ulp
        self.assertFalse(result["pass"])
        self.assertFalse(result["pairs"]["pre_vs_control"]["pass"])
        self.assertFalse(fork.check1(pre, one_ulp, pre, 64, injected=False)["pass"])  # identity arm
        self.assertTrue(fork.check1(pre, one_ulp, pre, 64, injected=True)["pass"])

    def test_check1_tf32_stops(self):
        rng = np.random.default_rng(2)
        pre = {"q": rng.normal(size=8).astype(np.float32), "dq_da": rng.normal(size=(8, 2)).astype(np.float32)}
        report = fork.matmul_precision_report()
        self.assertFalse(report["tf32_detected"])  # CPU float32
        self.assertLess(report["float32_matmul_max_rel_error"], 1e-5)
        self.assertTrue(fork.check1(pre, pre, pre, 64, injected=True, precision=report)["pass"])
        result = fork.check1(pre, pre, pre, 64, injected=True, precision={**report, "tf32_detected": True})
        self.assertFalse(result["pass"])
        self.assertTrue(result["tf32_detected"])

    def test_both_arms_must_run_on_the_fork_device_model(self):
        d = Path(tempfile.mkdtemp())
        (d / "fork").mkdir()
        (d / "fork" / "fork.json").write_text(json.dumps({"fork_step": 1, "device": fork.device_info()}))
        fork.check_same_device(d)
        self.assertEqual(fork.read_fork(d), {"fork_step": 1})
        (d / "fork" / "fork.json").write_text(json.dumps({"fork_step": 1, "device": {
            **fork.device_info(), "device_kind": "NVIDIA A100-SXM4-80GB"}}))
        with self.assertRaises(RuntimeError):
            fork.check_same_device(d)
        shutil.rmtree(d)

    def test_post_fork_eval_leaves_training_state_untouched(self):
        import random

        from exp12_helpers import compose
        from experiments.exp12.trainer import Exp12Trainer

        cfg = compose(tiny_overrides(extra=["env.max_episode_steps=20"]))
        trainer = Exp12Trainer(cfg, tempfile.mkdtemp())
        real_create = fork.create_eval_env
        draws = []

        def global_rng_env(**kwargs):
            # Like HumanoidBench's Reach, which draws its goal from the global np.random at reset.
            env = real_create(**kwargs)
            reset = env.reset
            env.reset = lambda *a, **k: (draws.append((np.random.random(), random.random())), reset(*a, **k))[-1]
            return env

        np.random.seed(5)
        random.seed(5)
        key, np_state, py_state = trainer._sac_agent._rng, np.random.get_state(), random.getstate()
        plan = {"fork_step": 0, "eval_every_steps": 4, "eval_episodes": 2}
        with mock.patch.object(fork, "create_eval_env", global_rng_env):
            rows = fork.post_fork_eval(trainer, plan, 3, "control")
        np.testing.assert_array_equal(np.asarray(trainer._sac_agent._rng), np.asarray(key))
        self.assertEqual(random.getstate(), py_state)
        after = np.random.get_state()
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(after, np_state)))
        self.assertEqual([(r["eval_index"], r["steps_since_fork"], r["episode"]) for r in rows],
                         [(3, 12, 0), (3, 12, 1)])
        first = list(draws)
        np.random.seed(6)  # a different training-process state must not change the evaluation
        random.seed(6)
        draws.clear()
        with mock.patch.object(fork, "create_eval_env", global_rng_env):
            again = fork.post_fork_eval(trainer, plan, 3, "injected")
        self.assertEqual([r["return"] for r in again], [r["return"] for r in rows])  # seeded by (seed, index)
        self.assertEqual(draws, first)
        self.assertEqual(len(first), 2)
        trainer.close()


@unittest.skipUnless(importlib.util.find_spec("humanoid_bench") is not None, "needs humanoid_bench (venv_hb)")
class HumanoidBenchReachEvalSeedingTest(unittest.TestCase):
    """h1-reach-v0 draws its goal from the global np.random at reset; post-fork evaluations seed the
    global RNGs per (seed, eval index) and restore the training process's states afterwards."""

    def _rng_states(self, trainer):
        import random

        return np.random.get_state(), random.getstate(), np.asarray(trainer._sac_agent._rng)

    def _assert_same_states(self, a, b):
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a[0], b[0])), "numpy global RNG changed")
        self.assertEqual(a[1], b[1], "Python random changed")
        np.testing.assert_array_equal(a[2], b[2], "agent key changed")

    def test_reach_eval_seeding_saves_and_restores_the_training_rng(self):
        import random

        from exp12_helpers import compose
        from experiments.exp12.envs import create_eval_env
        from experiments.exp12.trainer import Exp12Trainer

        cfg = compose(tiny_overrides("h1-reach-v0", "humanoid_bench", extra=["env.max_episode_steps=30"]))
        env_cfg = {**dict(cfg.env), "seed": 123}
        first_obs = []
        for global_seed in (1, 2):  # precondition: Reach really reads the global RNG at reset
            np.random.seed(global_seed)
            env = create_eval_env(**env_cfg)
            first_obs.append(np.asarray(env.reset()[0]).copy())
            env.close()
        self.assertFalse(np.array_equal(*first_obs), "Reach no longer reads np.random; this test is vacuous")

        trainer = Exp12Trainer(cfg, tempfile.mkdtemp())
        plan = {"fork_step": 0, "eval_every_steps": 4, "eval_episodes": 2}
        returns = []
        for training_seed in (5, 6):
            np.random.seed(training_seed)
            random.seed(training_seed)
            np.random.random(7)  # some training-time draws
            before = self._rng_states(trainer)
            rows = fork.post_fork_eval(trainer, plan, 2, "injected" if training_seed == 5 else "control")
            self._assert_same_states(before, self._rng_states(trainer))
            returns.append([r["return"] for r in rows])
        self.assertEqual(returns[0], returns[1])  # the same goals whatever the training process's RNG state
        trainer.close()


class IdentityValidationTest(unittest.TestCase):
    """The CUDA identity-fork procedure end to end: both jobs stop at the snapshot, then the comparison script."""

    @classmethod
    def setUpClass(cls):
        cls.patchers = patch_wandb()
        cls.tmp = tempfile.mkdtemp()
        cls.overrides = fork_overrides(os.path.join(cls.tmp, "results"), extra=[
            "seed=101", "fork.identity_snapshot_steps=30", "testing.stop_after_identity_snapshot=true"])
        cls.run_dir = os.path.join(cls.tmp, "parent")
        cls.arm_dir = os.path.join(cls.tmp, "identity")
        run_exp1(cls.run_dir, cls.overrides, interval=20)
        run_arm(cls.arm_dir, cls.overrides, cls.run_dir, "identity", interval=20)

    @classmethod
    def tearDownClass(cls):
        for p in cls.patchers:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_jobs_stop_at_the_snapshot(self):
        from experiments.angle_1 import DONE_MARKER

        for d in (self.run_dir, self.arm_dir):
            self.assertFalse((Path(d) / DONE_MARKER).exists())
        self.assertLess(_meta(latest_state_dir(Path(self.run_dir) / "state"))["interaction_step"], 130)
        snap = latest_state_dir(fork.fork_dir(self.run_dir) / "identity_control")
        self.assertEqual(_meta(snap)["interaction_step"], 130)

    def test_compare_script_passes_and_detects_a_difference(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import compare_identity_fork

        result = compare_identity_fork.compare(Path(self.run_dir), Path(self.arm_dir))
        self.assertTrue(result["pass"], result)
        self.assertEqual((result["interaction_step"], result["fork_step"]), (130, 100))
        broken = os.path.join(self.tmp, "identity_broken")
        shutil.copytree(self.arm_dir, broken)
        panel = fork.load_npz(Path(broken) / "check1_after.npz")
        panel["q"][0] = np.nextafter(panel["q"][0], np.float32(np.inf))
        fork.save_npz(Path(broken) / "check1_after.npz", panel)
        meta_path = latest_state_dir(Path(broken) / "identity_identity") / "meta.pkl"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta["update_step"] += 1
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        result = compare_identity_fork.compare(Path(self.run_dir), Path(broken))
        self.assertFalse(result["pass"])
        self.assertEqual(set(result["differences"]), {"meta:update_step", "check1_panel:q"})

    def test_flag_refused_outside_dev(self):
        from exp12_helpers import compose

        cfg = compose(fork_overrides(self.tmp, extra=["run_role=confirmatory", "fork.identity_snapshot_steps=30",
                                                      "testing.stop_after_identity_snapshot=true",
                                                      "testing.force_trigger_check=null"]))
        with self.assertRaises(ValueError):
            fork.check_validation_flags(cfg)


class KillAroundForkTest(unittest.TestCase):
    """Interrupted at the fork or after it, exp1 and the injected arm resume to bit-identical results."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, run_dir, experiment="exp1", source=None, **crash):
        overrides = fork_overrides(run_dir + "_results")
        if experiment == "exp2_arm":
            overrides += [f"fork.source={source}", "fork.arm=injected", "injection.m=half"]
        spec = {"experiment": experiment, "overrides": overrides, "checkpoint_dir": run_dir,
                "checkpoint_interval": 20, **crash}
        path = os.path.join(self.tmp, "spec.json")
        with open(path, "w") as f:
            json.dump(spec, f)
        env = dict(os.environ, JAX_PLATFORMS="cpu", MUJOCO_GL=os.environ.get("MUJOCO_GL", "disable"))
        return subprocess.run([sys.executable, RUNNER, "exp1", path], env=env, capture_output=True, text=True)

    def test_kill_points_around_the_fork(self):
        ref = os.path.join(self.tmp, "ref")
        r = self._run(ref)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        final = latest_state_dir(Path(ref) / "state")
        cases = {
            "inside the fork write (state saved, FORK_READY missing)": dict(crash_inside_fork_write=True),
            "right after the fork, before the control's first save": dict(crash_step=101),
            "mid post-fork window": dict(crash_step=137),
        }
        for name, crash in cases.items():
            with self.subTest(kill=name):
                run_dir = os.path.join(self.tmp, name.split()[0] + str(len(name)))
                r = self._run(run_dir, **crash)
                self.assertNotEqual(r.returncode, 0)
                r = self._run(run_dir)
                self.assertEqual(r.returncode, 0, r.stderr[-3000:])
                self.assertEqual(state_differences(final, latest_state_dir(Path(run_dir) / "state"),
                                                   ignore_meta=("wandb_run_id",)), [])
        arm_ref = os.path.join(self.tmp, "arm_ref")
        r = self._run(arm_ref, "exp2_arm", source=ref)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        arm = os.path.join(self.tmp, "arm_killed")
        r = self._run(arm, "exp2_arm", source=ref, crash_step=153)
        self.assertNotEqual(r.returncode, 0)
        r = self._run(arm, "exp2_arm", source=ref)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertEqual(state_differences(latest_state_dir(Path(arm_ref) / "state"),
                                           latest_state_dir(Path(arm) / "state"), ignore_meta=("wandb_run_id",)), [])


if __name__ == "__main__":
    unittest.main()
