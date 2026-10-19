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


def _metadata(overrides=None, commit="a" * 40):
    """Minimal current Exp1 metadata for protocol-comparability tests."""
    cfg = {
        "env_name": "cheetah-run", "seed": 1, "num_env_steps": 500_000,
        "num_interaction_steps": 250_000, "action_repeat": 2, "gamma": 0.99,
        "updates_per_interaction_step": 5, "logging_per_interaction_step": 2000,
        "results_root": None,
        "env": {"env_type": "dmc", "env_name": "cheetah-run", "seed": 1},
        "agent": {"seed": 1, "critic_num_blocks": 2, "critic_hidden_dim": 512,
                  "actor_learning_rate": 1e-4},
        "buffer": {"max_length": 1_000_000},
    }
    for key, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = key.split(".")
        for parent in parents:
            node = node[parent]
        node[leaf] = value
    with mock.patch("utils.run_metadata.code_version",
                    return_value={"commit": commit, "dirty": False}):
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



if __name__ == "__main__":
    unittest.main()
