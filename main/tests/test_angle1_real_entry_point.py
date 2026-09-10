"""Exercises experiments.angle_1.run() itself, end to end, on tiny/synthetic
config overrides - not the building-block tests that have substituted for
this so far (no test_angle_1*.py existed before this file). This is the
literal entry point run.py invokes, including config composition via real
Hydra YAML files, the real training loop, real checkpoint saving, and
(separately) the real onset-detection/ledger post-hoc analysis path and the
save_probe_capture_snapshot instrumentation added 2026-09-08.

wandb.init() is unconditionally called inside run() (os.environ["WANDB_MODE"]
= "online" is set right before constructing WandbTrainerLogger, so setting
WANDB_MODE beforehand would just be overwritten) - real network credentials
are not available in this environment, so wandb.init/log/run are mocked at
the module-attribute level (wandb.init, wandb.log), not by disabling wandb
via an env var. This exercises every other real line of run() unmodified.

GlobalHydra is explicitly cleared before each hydra.initialize_config_dir()
call: run() calls it directly (not as a context manager), and multiple real
run() invocations (across these test methods, and across
test_angle2a_real_entry_point.py / test_angle2b_real_entry_point.py sharing
one `unittest discover` process) would otherwise hit "GlobalHydra is already
initialized".
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import wandb
from hydra.core.global_hydra import GlobalHydra

from experiments.angle_2a.agent_runner import ProbeCapture
from experiments.angle_2b.checkpoint_io import load_frozen_agent_snapshot

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs")


class _FakeWandbRun:
    def __init__(self):
        self.name = "fake-run"
        self.id = "fake-run-id"
        self.summary = {}

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def _fake_wandb_init(*args, **kwargs):
    run = _FakeWandbRun()
    wandb.run = run
    return run


def _fast_overrides(env_name="cheetah-run", seed=1, extra=()):
    return [
        f"env_name={env_name}",
        f"seed={seed}",
        "actor_num_blocks=1", "actor_hidden_dim=8",
        "critic_num_blocks=1", "critic_hidden_dim=8",
        "env.num_env_steps=40",
        "buffer.min_length=5", "buffer.max_length=200", "buffer.sample_batch_size=4",
        "evaluation_per_interaction_step=10", "logging_per_interaction_step=5",
        "num_eval_episodes=1",
        *extra,
    ]


class Angle1RealEntryPointTest(unittest.TestCase):
    def setUp(self):
        GlobalHydra.instance().clear()
        self.tmpdir = tempfile.mkdtemp()
        self.wandb_init_patcher = mock.patch("wandb.init", side_effect=_fake_wandb_init)
        self.wandb_log_patcher = mock.patch("wandb.log")
        self.wandb_init_patcher.start()
        self.wandb_log_patcher.start()

    def tearDown(self):
        self.wandb_init_patcher.stop()
        self.wandb_log_patcher.stop()
        GlobalHydra.instance().clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_args(self, overrides, checkpoint_interval=1000):
        return {
            "experiment": "angle_1",
            "config_path": CONFIG_PATH,
            "config_name": "base_sac",
            "overrides": overrides,
            "checkpoint_dir": self.tmpdir,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_start_frac": 0.0,
        }

    def test_plain_run_completes_and_logs_csv(self):
        from experiments.angle_1 import run

        run(self._run_args(_fast_overrides(seed=1)))

        csv_files = list(Path(self.tmpdir, "logs").glob("*.csv"))
        self.assertEqual(len(csv_files), 1, f"expected exactly one CSV log, found {csv_files}")

    def test_run_checkpoints_and_resumes(self):
        from experiments.angle_1 import run

        # checkpoint_interval=10 with num_interaction_steps=20 guarantees at
        # least one real mid-run checkpoint save.
        run(self._run_args(_fast_overrides(seed=2), checkpoint_interval=10))
        self.assertTrue((Path(self.tmpdir) / "meta.pkl").exists())

        # Resuming the identical invocation must not crash (is_resumed=True
        # path) - GlobalHydra must be cleared again before this second real
        # hydra.initialize_config_dir() call within the same test.
        GlobalHydra.instance().clear()
        run(self._run_args(_fast_overrides(seed=2), checkpoint_interval=10))

    def test_save_probe_capture_snapshot_produces_a_loadable_pool_snapshot(self):
        from experiments.angle_1 import run

        environment = "cheetah-run"
        seed = 3
        pool_root = str(Path(self.tmpdir) / "pool")
        # Note: save_probe_capture_snapshot/probe_capture_snapshot_root are
        # new keys not present in any YAML schema, so Hydra's struct mode
        # requires the "+" add-key prefix - a plain
        # save_probe_capture_snapshot=true override fails with
        # "Could not override... To append to your config use
        # +save_probe_capture_snapshot=true" (confirmed directly). Anyone
        # launching this for real needs to remember the "+".
        run(self._run_args(_fast_overrides(
            env_name=environment, seed=seed,
            extra=[
                "+save_probe_capture_snapshot=true",
                f"+probe_capture_snapshot_root={pool_root}",
            ],
        )))

        loaded = load_frozen_agent_snapshot(environment, seed, "baseline_pool", "pool", root=pool_root)
        self.assertGreater(loaded.states.shape[0], 0)
        self.assertEqual(loaded.states.shape[0], loaded.actions.shape[0])

        checkpoint_dir = Path(pool_root) / environment / f"seed{seed}" / "baseline_pool" / "checkpoints" / "pool"
        reloaded_capture = ProbeCapture.load_fresh(str(checkpoint_dir))
        self.assertGreater(len(reloaded_capture), 0)

    def test_critic_degradation_onset_detection_writes_a_real_ledger_row(self):
        """The one remaining piece of Angle 1's real entry point never
        exercised via run() itself: critic_degradation=true's post-hoc
        onset-detection/ledger-write path (analysis/pipeline.py). Only
        tested previously by calling that function directly with synthetic
        inputs, never through run()'s own config-driven call site.

        analysis/pipeline.run_post_hoc_onset_analysis's metrics_root/
        baseline_root/ledger_root default to relative "results/..." paths,
        and run()'s own call site doesn't expose a way to override them -
        so this test temporarily chdirs into a tmpdir (restored in
        tearDown) rather than writing into this repo's real results/
        directory, which must stay untouched."""
        from experiments.angle_1 import run

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.addCleanup(os.chdir, original_cwd)

        environment = "cheetah-run"
        # Larger than this file's other tests' overrides: the ACF/CCF window
        # calibration needs enough post-burn-in recorded points for a real,
        # successful calibration - too few (this file's usual 4 logged
        # points) hits pipeline.py's graceful "needs review" degradation
        # path instead, which is also real behavior but not the happy path.
        onset_overrides = [
            "env.num_env_steps=400", "logging_per_interaction_step=2",
            "buffer.min_length=5", "buffer.max_length=500", "buffer.sample_batch_size=4",
            "evaluation_per_interaction_step=100", "num_eval_episodes=1",
            "actor_num_blocks=1", "actor_hidden_dim=8",
            "critic_degradation=true",
            "onset_detection.default_architecture=D1W8",
        ]
        for seed in range(1, 6):
            GlobalHydra.instance().clear()
            run(self._run_args([
                f"env_name={environment}", f"seed={seed}",
                "critic_num_blocks=1", "critic_hidden_dim=8",
                *onset_overrides,
            ]))

        GlobalHydra.instance().clear()
        run(self._run_args([
            f"env_name={environment}", "seed=99",
            "critic_num_blocks=3", "critic_hidden_dim=16",  # scaled: D3W16, != baseline D1W8
            *onset_overrides,
        ]))

        ledger_path = Path("results/ledgers/angle_1/architectures/D3W16/onset_events.csv")
        self.assertTrue(ledger_path.exists(), f"expected a real onset ledger at {ledger_path}")
        import pandas as pd
        ledger_df = pd.read_csv(ledger_path)
        self.assertEqual(len(ledger_df), 1)
        self.assertEqual(ledger_df.iloc[0]["environment"], environment)
        self.assertEqual(int(ledger_df.iloc[0]["seed"]), 99)
        # "success" (a real onset found) and "no_onset_detected" (calibration
        # ran cleanly, legitimately found no sustained exceedance - expected
        # and fine for tiny random-weight synthetic data) are both healthy
        # outcomes; only "needs_manual_review" indicates the pipeline itself
        # hit a problem.
        self.assertIn(
            ledger_df.iloc[0]["status"], ("success", "no_onset_detected"),
            f"expected a clean pipeline run, got notes: {ledger_df.iloc[0].get('detection_notes')}",
        )


if __name__ == "__main__":
    unittest.main()
