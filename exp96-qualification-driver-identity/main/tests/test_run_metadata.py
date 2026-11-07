"""Run metadata and version guards (A7): every run persists what it was produced
under; resume and analysis refuse to mix runs produced under different settings."""

import json
import os
import pickle
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import wandb
from hydra.core.global_hydra import GlobalHydra

from analysis.metrics_store import RunIdentity, metrics_path, run_metadata_path
from utils.run_metadata import (
    SNAPSHOT_RUN_METADATA_FILENAME,
    RunMetadataMismatch,
    build_run_metadata,
    check_resume_matches,
    check_runs_comparable,
    save_run_metadata,
)

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs")


class _FakeRun:
    name, id, summary = "fake-run", "fake-run-id", {}

    def log(self, *a, **k):
        pass

    def finish(self):
        pass


def _fake_init(*a, **k):
    wandb.run = _FakeRun()
    return wandb.run


def _overrides(results_root, extra=()):
    return [
        "env_name=cheetah-run", "seed=2", "actor_num_blocks=1", "actor_hidden_dim=8",
        "critic_num_blocks=1", "critic_hidden_dim=8", "env.num_env_steps=40",
        "buffer.min_length=5", "buffer.max_length=200", "buffer.sample_batch_size=4",
        "evaluation_per_interaction_step=10", "logging_per_interaction_step=5",
        "num_eval_episodes=1", "critic_degradation=true", f"results_root={results_root}", *extra,
    ]


class Angle1RunMetadataTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.results = os.path.join(self.tmpdir, "results")
        self.ckpt = os.path.join(self.tmpdir, "ckpt")
        for patcher in (mock.patch("wandb.init", side_effect=_fake_init), mock.patch("wandb.log")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run(self, overrides, kill_at=None):
        from experiments import angle_1

        def killing(iterable, **kwargs):
            for step in iterable:
                if step == kill_at:
                    raise KeyboardInterrupt
                yield step

        GlobalHydra.instance().clear()
        args = {
            "experiment": "angle_1", "config_path": CONFIG_PATH, "config_name": "base_sac",
            "overrides": overrides, "checkpoint_dir": self.ckpt, "checkpoint_interval": 10,
            "checkpoint_start_frac": 0.0,
        }
        with mock.patch.object(angle_1.tqdm, "tqdm", killing if kill_at else angle_1.tqdm.tqdm):
            angle_1.run(args)

    def test_metadata_is_persisted_next_to_checkpoint_and_metrics(self):
        self._run(_overrides(self.results, ["+save_probe_capture_snapshot=true"]))
        meta = json.loads((Path(self.ckpt) / "run_metadata.json").read_text())
        settings = meta["settings"]
        self.assertEqual(settings["architecture"], "D1W8")
        self.assertEqual(settings["env_id"], "cheetah-run")
        self.assertIn("dm_control", settings["simulator_versions"])
        for key, expected in (("seed", 2), ("num_env_steps", 40), ("num_interaction_steps", 20),
                              ("updates_per_interaction_step", 5), ("action_repeat", 2),
                              ("logging_per_interaction_step", 5), ("evaluation_per_interaction_step", 10),
                              ("num_eval_episodes", 1)):
            self.assertEqual(settings[key], expected, key)
        self.assertEqual(settings["discount"], meta["resolved_config"]["gamma"])
        self.assertEqual(len(meta["code"]["commit"]), 40)
        self.assertEqual(len(meta["config_hash"]), 64)
        self.assertEqual(meta["resolved_config"]["env"]["env_name"], "cheetah-run")

        identity = RunIdentity("angle_1", "D1W8", "cheetah-run", 2)
        sidecar = json.loads(run_metadata_path(identity, root=f"{self.results}/metrics").read_text())
        self.assertEqual(sidecar["config_hash"], meta["config_hash"])
        snapshot = (Path(self.results) / "baseline_calibration_pool" / "cheetah-run" / "D1W8" / "seed2"
                    / "baseline_pool" / SNAPSHOT_RUN_METADATA_FILENAME)
        self.assertEqual(json.loads(snapshot.read_text())["protocol_hash"], meta["protocol_hash"])

    def test_resume_under_a_different_config_is_refused(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run(_overrides(self.results), kill_at=15)
        with self.assertRaisesRegex(RunMetadataMismatch, r"actor_grad_cosine_every"):
            self._run(_overrides(self.results, ["actor_grad_cosine_every=7"]))

    def test_resume_with_only_a_moved_storage_root_is_allowed_and_recorded(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run(_overrides(self.results), kill_at=15)
        self._run(_overrides(os.path.join(self.tmpdir, "moved_results")))
        meta = json.loads((Path(self.ckpt) / "run_metadata.json").read_text())
        self.assertEqual([launch["resumed"] for launch in meta["launches"]], [False, True])

    def test_resuming_a_checkpoint_without_metadata_marks_it_unrecorded(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run(_overrides(self.results), kill_at=15)
        (Path(self.ckpt) / "run_metadata.json").unlink()
        self._run(_overrides(self.results))
        meta = json.loads((Path(self.ckpt) / "run_metadata.json").read_text())
        self.assertTrue(meta["first_launch_unrecorded"])


def _metadata(overrides=None, commit="a" * 40):
    cfg = {
        "env_name": "cheetah-run", "seed": 1, "num_env_steps": 500_000, "num_interaction_steps": 250_000,
        "action_repeat": 2, "gamma": 0.99, "updates_per_interaction_step": 5,
        "logging_per_interaction_step": 2000, "results_root": None,
        "env": {"env_type": "dmc", "env_name": "cheetah-run", "seed": 1},
        "agent": {"seed": 1, "critic_num_blocks": 2, "critic_hidden_dim": 512, "actor_learning_rate": 1e-4},
        "buffer": {"max_length": 1_000_000},
    }
    for key, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = key.split(".")
        for parent in parents:
            node = node[parent]
        node[leaf] = value
    with mock.patch("utils.run_metadata.code_version", return_value={"commit": commit, "dirty": False}):
        return build_run_metadata(cfg, identity={"architecture": "D2W512"}, launch={})


class ComparabilityRulesTest(unittest.TestCase):
    def test_seed_architecture_and_locations_do_not_break_comparability(self):
        runs = {
            "a": _metadata(),
            "b": _metadata({"seed": 2, "env.seed": 2, "agent.seed": 2, "agent.critic_num_blocks": 7,
                            "agent.critic_hidden_dim": 1024, "results_root": "/elsewhere"}),
        }
        self.assertEqual(check_runs_comparable(runs, "test"), [])

    def test_any_protocol_difference_is_refused(self):
        for key, value in (("logging_per_interaction_step", 3000), ("num_env_steps", 1_000_000),
                           ("action_repeat", 1), ("gamma", 0.995), ("updates_per_interaction_step", 2),
                           ("agent.actor_learning_rate", 3e-4)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(RunMetadataMismatch, key.split(".")[-1]):
                    check_runs_comparable({"a": _metadata(), "b": _metadata({key: value})}, "test")

    def test_missing_metadata_is_reported_not_refused(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(check_runs_comparable({"a": _metadata(), "legacy": None}, "test"), ["legacy"])
        self.assertTrue(any("not machine-verified" in str(w.message) for w in caught))

    def test_resume_check_ignores_locations_and_warns_on_a_code_change(self):
        stored = _metadata()
        check_resume_matches(stored, _metadata({"results_root": "/moved"}), "ckpt")
        with self.assertWarnsRegex(UserWarning, "code commit"):
            check_resume_matches(stored, _metadata(commit="b" * 40), "ckpt")
        moved = _metadata()
        moved["protocol"]["simulator_versions"] = {"dm_control": "0.0.0"}
        with self.assertWarnsRegex(UserWarning, "simulator versions"):
            check_resume_matches(stored, moved, "ckpt")
        with self.assertRaisesRegex(RunMetadataMismatch, "num_env_steps"):
            check_resume_matches(stored, _metadata({"num_env_steps": 1}), "ckpt")


class AnalysisRefusesMismatchedRunsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.metrics = os.path.join(self.tmpdir, "metrics")

    def _write_run(self, identity, metadata):
        path = metrics_path(identity, root=self.metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        steps = np.arange(0, 200_001, 2000)
        pd.DataFrame({"interaction_step": steps, "env_step": steps * 2,
                      "td_error_variance": np.linspace(1, 2, len(steps)),
                      "actor_grad_cosine": np.linspace(0, 1, len(steps))}).to_csv(path, index=False)
        if metadata is not None:
            save_run_metadata(run_metadata_path(identity, root=self.metrics), metadata)

    def _analyze(self):
        from analysis.pipeline import run_post_hoc_onset_analysis

        run = RunIdentity("angle_1", "D7W1024", "cheetah-run", 1)
        baselines = [RunIdentity("angle_1", "D2W512", "cheetah-run", s) for s in range(1, 6)]
        onset_cfg = {"baseline_percentile": 95, "acf_ccf_burn_in_fraction": 0.25, "ccf_lag_fraction_cap": 0.2,
                     "critic_degradation_method_version": "v", "propagation_method_version": "v"}
        return run, run_post_hoc_onset_analysis(
            run, True, False, baselines, onset_cfg, 2000, metrics_root=self.metrics,
            baseline_root=os.path.join(self.tmpdir, "baselines"), ledger_root=os.path.join(self.tmpdir, "ledgers"),
        )

    def test_a_baseline_with_a_different_cadence_is_refused_and_logged(self):
        self._write_run(RunIdentity("angle_1", "D7W1024", "cheetah-run", 1), _metadata())
        for seed in range(1, 6):
            overrides = {"logging_per_interaction_step": 3000} if seed == 4 else {}
            self._write_run(RunIdentity("angle_1", "D2W512", "cheetah-run", seed), _metadata(overrides))
        with self.assertRaisesRegex(RunMetadataMismatch, "logging_per_interaction_step"):
            self._analyze()
        ledger = pd.read_csv(Path(self.tmpdir, "ledgers", "angle_1", "architectures", "D7W1024", "onset_events.csv"))
        self.assertEqual(ledger.iloc[0]["status"], "needs_manual_review")
        self.assertIn("refused to combine runs", ledger.iloc[0]["detection_notes"])

    def test_legacy_runs_without_metadata_are_analyzed_and_flagged(self):
        self._write_run(RunIdentity("angle_1", "D7W1024", "cheetah-run", 1), _metadata())
        for seed in range(1, 6):
            self._write_run(RunIdentity("angle_1", "D2W512", "cheetah-run", seed), None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, ledger_path = self._analyze()
        notes = pd.read_csv(ledger_path).iloc[0]["detection_notes"]
        self.assertIn("not machine-verified", notes)
        self.assertIn("angle_1_D2W512_cheetah-run_seed1", notes)


class CachedBaselineCadenceTest(unittest.TestCase):
    def test_cached_thresholds_are_rejected_for_a_different_logging_cadence(self):
        from analysis.baseline_calibration import load_or_calibrate_baseline

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        identities = [RunIdentity("angle_1", "D2W512", "cheetah-run", s) for s in range(1, 6)]
        for ident in identities:
            path = metrics_path(ident, root=f"{tmp}/metrics")
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"interaction_step": range(0, 100, 2), "td_error_variance": np.arange(50.0)}).to_csv(path)
        kwargs = dict(metrics_root=f"{tmp}/metrics", baseline_root=f"{tmp}/baselines")
        load_or_calibrate_baseline(identities, "td_error_variance", expected_logging_interval=2, **kwargs)
        with self.assertRaisesRegex(ValueError, "logging interval"):
            load_or_calibrate_baseline(identities, "td_error_variance", expected_logging_interval=4, **kwargs)


class PoolComparabilityTest(unittest.TestCase):
    def test_pool_agents_trained_under_different_settings_are_refused(self):
        from analysis.baseline_calibration_pool import check_pool_comparable
        from experiments.angle_2a.storage import matchup_dir

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        for seed, overrides in ((1, {}), (2, {}), (6, {"num_env_steps": 1_250_000})):
            out = matchup_dir("cheetah-run", seed, "baseline_pool", root=tmp, architecture="D2W512")
            out.mkdir(parents=True)
            save_run_metadata(out / SNAPSHOT_RUN_METADATA_FILENAME, _metadata(overrides))
        self.assertEqual(check_pool_comparable("cheetah-run", [1, 2], tmp, "test"), [])
        with self.assertRaisesRegex(RunMetadataMismatch, "num_env_steps"):
            check_pool_comparable("cheetah-run", [1, 2, 6], tmp, "test")


if __name__ == "__main__":
    unittest.main()
