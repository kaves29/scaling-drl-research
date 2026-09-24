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

import gymnasium as gym
import numpy as np
import wandb
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import ProbeCapture
from experiments.angle_2b.checkpoint_io import load_frozen_agent_snapshot
from scale_rl.agents import create_agent
from scale_rl.common.logger import WandbTrainerLogger

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


OBS_DIM = 4
ACT_DIM = 2
COSINE_KEY = "train/actor_grad_cosine"


def _make_agent(seed=0):
    cfg = OmegaConf.create({
        "agent_type": "sac", "seed": seed, "num_train_envs": 1, "max_episode_steps": 100,
        "normalize_observation": True, "actor_block_type": "residual", "actor_num_blocks": 1,
        "actor_hidden_dim": 8, "actor_learning_rate": 1e-4, "actor_weight_decay": 1e-2,
        "critic_block_type": "residual", "critic_num_blocks": 1, "critic_hidden_dim": 8,
        "critic_learning_rate": 1e-4, "critic_weight_decay": 1e-2, "critic_use_cdq": False,
        "temp_target_entropy": None, "temp_target_entropy_coef": -0.5, "temp_initial_value": 0.01,
        "temp_learning_rate": 1e-4, "temp_weight_decay": 0.0, "target_tau": 0.005, "gamma": 0.99,
        "n_step": 1, "mixed_precision": False, "actor_sparsity": 0.0, "critic_sparsity": 0.0,
    })
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
    return create_agent(observation_space=observation_space, action_space=action_space, cfg=cfg)


def _make_batches(num_updates, batch_size=8, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "observation": rng.standard_normal((batch_size, OBS_DIM)).astype(np.float32),
            "action": rng.uniform(-1, 1, size=(batch_size, ACT_DIM)).astype(np.float32),
            "reward": rng.standard_normal(batch_size).astype(np.float32),
            "terminated": np.zeros(batch_size, dtype=np.float32),
            "truncated": np.zeros(batch_size, dtype=np.float32),
            "next_observation": rng.standard_normal((batch_size, OBS_DIM)).astype(np.float32),
        }
        for _ in range(num_updates)
    ]


def _bare_logger():
    """The real WandbTrainerLogger averaging path, without wandb.init()."""
    logger = object.__new__(WandbTrainerLogger)
    logger.reset()
    return logger


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
            # ~10 updates per logging window here, so the default cadence
            # (30) would leave most windows with no actor_grad_cosine sample.
            "actor_grad_cosine_every=1",
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


class ActorGradCosineCadenceTest(unittest.TestCase):
    """actor_grad_cosine_every: computed only on update steps divisible by it,
    and the logged window average is a mean over computed samples only."""

    NUM_UPDATES = 90

    def _run(self, every):
        agent = _make_agent()
        logger = _bare_logger()
        infos = []
        for update_step, batch in enumerate(_make_batches(self.NUM_UPDATES)):
            info = agent.update(
                update_step, batch, compute_actor_grad_cosine=update_step % every == 0
            )
            logger.update_metric(**info)
            infos.append(info)
        return infos, logger.average_meter_dict

    def test_cadence_and_window_average(self):
        infos_1, meters_1 = self._run(every=1)
        infos_30, meters_30 = self._run(every=30)

        self.assertTrue(all(COSINE_KEY in info for info in infos_1))
        computed_30 = [step for step, info in enumerate(infos_30) if COSINE_KEY in info]
        self.assertEqual(computed_30, [0, 30, 60])

        # Diagnostic only: skipping it must not perturb training.
        for info_1, info_30 in zip(infos_1, infos_30):
            for key in ("train/actor_loss", "train/critic_loss", "train/td_error_var"):
                self.assertEqual(info_1[key], info_30[key])
        for step in computed_30:
            self.assertEqual(infos_30[step][COSINE_KEY], infos_1[step][COSINE_KEY])

        samples_1 = [info[COSINE_KEY] for info in infos_1]
        samples_30 = [infos_30[step][COSINE_KEY] for step in computed_30]
        self.assertEqual(meters_1[COSINE_KEY].count, self.NUM_UPDATES)
        self.assertEqual(meters_30[COSINE_KEY].count, len(samples_30))
        self.assertAlmostEqual(meters_1[COSINE_KEY].avg, float(np.mean(samples_1)), places=6)
        self.assertAlmostEqual(meters_30[COSINE_KEY].avg, float(np.mean(samples_30)), places=6)
        self.assertNotAlmostEqual(
            meters_30[COSINE_KEY].avg, sum(samples_30) / self.NUM_UPDATES, places=3
        )
        self.assertEqual(meters_30["train/actor_loss"].count, self.NUM_UPDATES)


if __name__ == "__main__":
    unittest.main()
