"""Phase 1 (Exp 1/2 foundations): exact env restore, seeding, bit-exact resume,
kill-and-resume through the real entry point, and parity with angle_1's loop."""

import json
import os
import pickle
import random
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

from exp12_helpers import (  # noqa: E402
    CONFIG_PATH,
    compose,
    load_agent_tree,
    patch_wandb,
    state_differences,
    tiny_overrides,
)
from experiments.exp12 import envs as exp12_envs  # noqa: E402
from experiments.exp12 import trainer as trainer_module  # noqa: E402
from experiments.exp12.state import latest_state_dir  # noqa: E402
from experiments.exp12.trainer import Exp12Trainer  # noqa: E402

RUNNER = str(Path(__file__).resolve().parent / "exp12_subprocess_runner.py")
ALL_ENVS = [
    ("dmc", "dog-run"), ("dmc", "dog-trot"), ("dmc", "humanoid-run"), ("dmc", "humanoid-walk"),
    ("dmc", "humanoid-stand"), ("dmc", "swimmer-swimmer15"), ("dmc", "hopper-hop"),
    ("myosuite", "myo-key-turn"), ("myosuite", "myo-pen-twirl"), ("myosuite", "myo-pose-hard"),
    ("myosuite", "myo-reach"),
]
SUITES = [("hopper-hop", "dmc_medium"), ("myo-reach", "myosuite_hard")]


def _subprocess(*args):
    env = dict(os.environ, JAX_PLATFORMS="cpu", MUJOCO_GL=os.environ.get("MUJOCO_GL", "disable"))
    return subprocess.run([sys.executable, RUNNER, *args], env=env, capture_output=True, text=True)


def _env(env_type, env_name, seed=3):
    return exp12_envs.create_envs(
        env_type=env_type, seed=seed, env_name=env_name, num_train_envs=1, num_eval_envs=1,
        rescale_action=True, no_termination=False, action_repeat=2, reward_scale=1.0, max_episode_steps=1000,
    )[0]


def _mismatches(env_a, env_b, actions):
    count = 0
    for a in actions:
        oa, ob = env_a.step(a), env_b.step(a)
        count += int(not all(np.array_equal(x, y) for x, y in zip(oa[:4], ob[:4])))
    return count


class EnvRestoreAcrossProcessesTest(unittest.TestCase):
    """D1: a mid-episode state restored in a new process reproduces the original trajectory."""

    def test_every_environment_replays_bit_exactly_in_a_new_process(self):
        tmp = tempfile.mkdtemp()
        try:
            rec, rep = os.path.join(tmp, "rec.pkl"), os.path.join(tmp, "rep.pkl")
            r = _subprocess("env_record", rec, *[f"{t}:{n}" for t, n in ALL_ENVS])
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            r = _subprocess("env_replay", rec, rep)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            with open(rec, "rb") as f:
                recorded = pickle.load(f)
            with open(rep, "rb") as f:
                replayed = pickle.load(f)
            for key in ALL_ENVS:
                with self.subTest(env=key):
                    self.assertGreater(len(recorded[key]["state"]["env"]["actions"]), 0, "capture was not mid-episode")
                    ends = sum(bool(o[2][0] or o[3][0]) for o in recorded[key]["outputs"])
                    self.assertGreater(ends, 0, "continuation never crossed an episode reset")
                    for i, (a, b) in enumerate(zip(recorded[key]["outputs"], replayed[key])):
                        for x, y in zip(a, b):
                            np.testing.assert_array_equal(x, y, err_msg=f"{key} step {i}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class EnvRestoreBreakTest(unittest.TestCase):
    """Break-and-restore evidence: each half of the restore mechanism is necessary."""

    def _capture(self, env_type, env_name):
        env = _env(env_type, env_name)
        env.reset()
        rng = np.random.default_rng(0)
        shape = (1,) + env.single_action_space.shape
        for a in rng.uniform(-1, 1, (1230,) + shape):  # third DMC episode
            env.step(a)
        return env, exp12_envs.env_restore_state(env), rng.uniform(-1, 1, (400,) + shape)

    def test_restore_is_exact_and_each_component_is_necessary(self):
        for env_type, env_name in [("dmc", "hopper-hop"), ("myosuite", "myo-pose-hard")]:
            with self.subTest(env=env_name, variant="intact"):
                env, state, actions = self._capture(env_type, env_name)
                fresh = _env(env_type, env_name)
                fresh.reset()
                exp12_envs.restore_env(fresh, state)
                self.assertEqual(_mismatches(env, fresh, actions), 0)
            with self.subTest(env=env_name, variant="rng not restored"):
                env, state, actions = self._capture(env_type, env_name)
                fresh = _env(env_type, env_name)
                fresh.reset()
                with mock.patch.object(exp12_envs, "set_env_rng_state", lambda *a: None):
                    exp12_envs.restore_env(fresh, state)
                self.assertGreater(_mismatches(env, fresh, actions), 0)
            with self.subTest(env=env_name, variant="actions not replayed"):
                env, state, actions = self._capture(env_type, env_name)
                fresh = _env(env_type, env_name)
                fresh.reset()
                state["env"]["actions"] = []
                exp12_envs.restore_env(fresh, state)
                self.assertGreater(_mismatches(env, fresh, actions), 0)


class MyoSuiteSeedingTest(unittest.TestCase):
    """G3: Exp 1/2 MyoSuite envs are reproducible from their seed; the repo factory is not."""

    def _first_obs(self, factory):
        env = factory()
        obs, _ = env.reset()
        env.close()
        return obs

    def test_seeded_factory_is_reproducible_and_repo_factory_is_not(self):
        from scale_rl.envs.myosuite import make_myosuite_env as repo_factory

        a = self._first_obs(lambda: exp12_envs.make_myosuite_env("myo-pose-hard", 1))
        b = self._first_obs(lambda: exp12_envs.make_myosuite_env("myo-pose-hard", 1))
        c = self._first_obs(lambda: exp12_envs.make_myosuite_env("myo-pose-hard", 2))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        r1 = self._first_obs(lambda: repo_factory("myo-pose-hard", 1))
        r2 = self._first_obs(lambda: repo_factory("myo-pose-hard", 1))
        self.assertFalse(np.array_equal(r1, r2), "repo factory unexpectedly reproducible")


class ResumeExactnessTest(unittest.TestCase):
    """A save/restore at any step continues bit-identically to an uninterrupted run."""

    @classmethod
    def setUpClass(cls):
        cls.patchers = patch_wandb()

    @classmethod
    def tearDownClass(cls):
        for p in cls.patchers:
            p.stop()

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _trainer(self, overrides):
        cfg = compose(overrides)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        return Exp12Trainer(cfg, self.tmp)

    def _reference(self, overrides, last):
        t = self._trainer(overrides)
        t.start()
        t.train(last)
        return t.save(os.path.join(self.tmp, "ref"))

    def _resumed(self, overrides, split, last, tag):
        t = self._trainer(overrides)
        t.start()
        t.train(split)
        mid = t.save(os.path.join(self.tmp, f"mid_{tag}"))
        t2 = self._trainer(overrides)
        t2.restore(mid)
        t2.train(last)
        return t2.save(os.path.join(self.tmp, f"res_{tag}"))

    def test_resume_is_bit_exact_for_each_suite(self):
        for env_name, group in SUITES:
            overrides = tiny_overrides(env_name, group, steps=300)
            ref = self._reference(overrides, 300)
            for split in (5, 137):  # before the first update; mid-episode mid-window after updates
                with self.subTest(env=env_name, split=split):
                    self.assertEqual(state_differences(ref, self._resumed(overrides, split, 300, f"{env_name}{split}")), [])

    def test_breaking_each_restored_component_is_detected(self):
        overrides = tiny_overrides("hopper-hop", "dmc_medium", steps=300)
        ref = self._reference(overrides, 300)
        breaks = {
            "numpy reseeded instead of restored (the old resume bug)": mock.patch.object(
                trainer_module.np.random, "set_state", lambda state: np.random.seed(1)
            ),
            "train env reset instead of restored": mock.patch.object(
                trainer_module, "restore_env",
                lambda env, state: env.reset() if env is self._current.train_env else exp12_envs.restore_env(env, state),
            ),
        }
        for name, patcher in breaks.items():
            with self.subTest(break_=name):
                t = self._trainer(overrides)
                t.start()
                t.train(137)
                mid = t.save(os.path.join(self.tmp, "mid_break"))
                self._current = self._trainer(overrides)
                with patcher:
                    self._current.restore(mid)
                self._current.train(300)
                res = self._current.save(os.path.join(self.tmp, "res_break"))
                diffs = state_differences(ref, res)
                self.assertTrue(any(d.startswith("agent:") for d in diffs), diffs)
        with self.subTest(break_="agent window buffers dropped"):
            t = self._trainer(overrides)
            t.start()
            t.train(137)
            mid = t.save(os.path.join(self.tmp, "mid_win"))
            meta_path = Path(mid) / "meta.pkl"
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            meta["agent_window_buffers"] = {k: [] for k in meta["agent_window_buffers"]}
            with open(meta_path, "wb") as f:
                pickle.dump(meta, f)
            t2 = self._trainer(overrides)
            t2.restore(mid)
            t2.train(300)
            diffs = state_differences(ref, t2.save(os.path.join(self.tmp, "res_win")))
            self.assertIn("meta:metrics_rows", diffs)


class KillAndResumeEntryPointTest(unittest.TestCase):
    """exp1.run() killed at different points and relaunched ends bit-identical to an uninterrupted run."""

    def _run(self, run_dir, crash_step=None, interval=60):
        spec = {"overrides": tiny_overrides("hopper-hop", "dmc_medium", steps=300), "checkpoint_dir": run_dir,
                "checkpoint_interval": interval, "crash_step": crash_step}
        spec_path = os.path.join(self.tmp, "spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)
        return _subprocess("exp1", spec_path)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_kill_points(self):
        ref_dir = os.path.join(self.tmp, "ref")
        r = self._run(ref_dir)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        ref = latest_state_dir(Path(ref_dir) / "state")
        # 5: before any update or save; 60: right after a save; 95: mid-interval, mid-episode.
        for crash_step in (5, 60, 95):
            with self.subTest(crash_step=crash_step):
                run_dir = os.path.join(self.tmp, f"crash{crash_step}")
                r = self._run(run_dir, crash_step=crash_step)
                self.assertEqual(r.returncode, 3, r.stderr[-3000:])
                self.assertFalse((Path(run_dir) / "DONE").exists())
                r = self._run(run_dir)
                self.assertEqual(r.returncode, 0, r.stderr[-3000:])
                self.assertTrue((Path(run_dir) / "DONE").exists())
                res = latest_state_dir(Path(run_dir) / "state")
                self.assertEqual(state_differences(ref, res), [])
        r = self._run(ref_dir)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("DONE", r.stderr)


class Angle1ParityTest(unittest.TestCase):
    """With probes off, exp1's loop reproduces experiments/angle_1.py's training bit-for-bit."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patchers = patch_wandb()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_same_params_optimizer_state_rng_and_metrics(self):
        from experiments import angle_1, exp1

        steps = 290  # not an evaluation step, so angle_1's mid-step checkpoint equals end-of-step
        overrides = tiny_overrides("cheetah-run", "dmc_medium", steps=steps)
        da, de = os.path.join(self.tmp, "a"), os.path.join(self.tmp, "e")
        angle1_overrides = [o for o in overrides if not o.startswith("probe.")]
        angle_1.run({"experiment": "angle_1", "config_path": CONFIG_PATH, "config_name": "base_sac",
                     "overrides": angle1_overrides + ["updates_per_interaction_step=2"], "checkpoint_dir": da,
                     "checkpoint_interval": steps, "checkpoint_start_frac": 0.0})
        exp1.run({"experiment": "exp1", "config_path": CONFIG_PATH, "config_name": "base_exp12",
                  "overrides": overrides + ["probe.enabled=false"], "checkpoint_dir": de, "checkpoint_interval": 10**9,
                  "checkpoint_start_frac": 0.0})
        state = latest_state_dir(Path(de) / "state")
        leaves_a = jax.tree_util.tree_leaves_with_path(load_agent_tree(da))
        leaves_e = jax.tree_util.tree_leaves_with_path(load_agent_tree(state))
        self.assertEqual(len(leaves_a), len(leaves_e))
        for (path, a), (_, e) in zip(leaves_a, leaves_e):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(e), err_msg=str(path))
        with open(Path(da) / "obs_rms.pkl", "rb") as f:
            rms_a = pickle.load(f)
        with open(state / "obs_rms.pkl", "rb") as f:
            rms_e = pickle.load(f)
        for k in rms_a:
            np.testing.assert_array_equal(rms_a[k], rms_e[k])
        csv_a = pd.read_csv(next((Path(da) / "logs").glob("*.csv")))
        csv_e = pd.read_csv(next(p for p in (Path(de) / "logs").glob("*.csv") if "eval_episodes" not in p.name))
        self.assertEqual(set(csv_a.columns), set(csv_e.columns))
        pd.testing.assert_frame_equal(csv_a, csv_e[csv_a.columns], check_exact=True)


if __name__ == "__main__":
    unittest.main()
