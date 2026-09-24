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
import jax
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

    def _logged_windows(self, overrides):
        from experiments.angle_1 import run

        GlobalHydra.instance().clear()
        wandb.log.reset_mock()
        run(self._run_args(overrides))
        return [(c.kwargs["step"], list(c.args[0].items())) for c in wandb.log.call_args_list]

    def test_deferred_update_metrics_are_bit_identical_to_eager(self):
        """Materializing update metrics only at logging time must be a pure
        timing change: identical logged averages, key order, and types."""
        from experiments import angle_1

        class EagerUpdateMetrics(angle_1.PendingUpdateMetrics):
            def add(self, first_update_step, update_info):
                super().add(first_update_step, update_info)
                self.flush()

        overrides = _fast_overrides(seed=4, extra=[
            "env.num_env_steps=60", "logging_per_interaction_step=10",
            "evaluation_per_interaction_step=15", "actor_grad_cosine_every=4",
        ])
        deferred = self._logged_windows(overrides)
        with mock.patch.object(angle_1, "PendingUpdateMetrics", EagerUpdateMetrics):
            eager = self._logged_windows(overrides)

        self.assertGreaterEqual(len([w for w in deferred if w[0] > 0]), 2)
        self.assertTrue(any("train/actor_loss" in dict(items) for _, items in deferred))
        self.assertEqual(len(deferred), len(eager))
        for (step_d, items_d), (step_e, items_e) in zip(deferred, eager):
            self.assertEqual(step_d, step_e)
            self.assertEqual([k for k, _ in items_d], [k for k, _ in items_e])
            for (key, v_d), (_, v_e) in zip(items_d, items_e):
                self.assertIs(type(v_d), type(v_e), key)
                if isinstance(v_d, float):
                    self.assertEqual(v_d.hex(), v_e.hex(), f"{key} at step {step_d}")
                else:
                    self.assertEqual(v_d, v_e, key)

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

    def _compose(self, overrides):
        from hydra import compose, initialize_config_dir

        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
            cfg = compose(config_name="base_sac", overrides=overrides)
        return OmegaConf.to_container(cfg)

    def test_multienv_variant_preserves_utd_and_leaves_defaults_alone(self):
        variant = self._compose(["env=dmc_hard_multienv"])
        self.assertEqual(variant["env"]["num_train_envs"], 4)
        self.assertEqual(variant["updates_per_interaction_step"], 20)

        for env in ("dmc_hard", "dmc_medium", "myosuite_hard", "myosuite_medium"):
            default = self._compose([f"env={env}"])
            self.assertEqual(default["env"]["num_train_envs"], 1, env)
            self.assertEqual(default["updates_per_interaction_step"], 5, env)
            self.assertEqual(
                variant["updates_per_interaction_step"] / variant["env"]["num_train_envs"],
                default["updates_per_interaction_step"] / default["env"]["num_train_envs"],
            )
        hard = self._compose(["env=dmc_hard"])["env"]
        self.assertEqual(
            {k: v for k, v in variant["env"].items() if k != "num_train_envs"},
            {k: v for k, v in hard.items() if k != "num_train_envs"},
        )

    def test_multienv_variant_rejects_save_probe_capture_snapshot(self):
        from experiments.angle_1 import run

        with self.assertRaisesRegex(ValueError, r"num_train_envs\s+== 1 \(got 4\)"):
            run(self._run_args(_fast_overrides(extra=[
                "env=dmc_hard_multienv",
                "+save_probe_capture_snapshot=true",
                f"+probe_capture_snapshot_root={Path(self.tmpdir) / 'pool'}",
            ])))

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

    UPDATES_PER_CALL = 5
    NUM_UPDATES = 90

    def _run(self, every):
        from experiments.angle_1 import PendingUpdateMetrics

        agent = _make_agent()
        logger = _bare_logger()
        pending = PendingUpdateMetrics(logger, every)
        batches = _make_batches(self.NUM_UPDATES)
        infos = []
        for first in range(0, self.NUM_UPDATES, self.UPDATES_PER_CALL):
            chunk = batches[first:first + self.UPDATES_PER_CALL]
            stacked = {key: np.stack([b[key] for b in chunk]) for key in chunk[0]}
            info = agent.update_many(first, stacked, every)
            pending.add(first, info)
            infos.append(jax.device_get(info))
        pending.flush()
        per_update = {key: np.concatenate([info[key] for info in infos]) for key in infos[0]}
        return per_update, logger.average_meter_dict

    def test_cadence_and_window_average(self):
        per_update_1, meters_1 = self._run(every=1)
        per_update_30, meters_30 = self._run(every=30)

        self.assertTrue(np.all(np.isfinite(per_update_1[COSINE_KEY])))
        computed_30 = np.flatnonzero(~np.isnan(per_update_30[COSINE_KEY])).tolist()
        self.assertEqual(computed_30, [0, 30, 60])

        # Diagnostic only: skipping it must not perturb training.
        for key in ("train/actor_loss", "train/critic_loss", "train/td_error_var"):
            np.testing.assert_allclose(per_update_30[key], per_update_1[key], rtol=1e-4, err_msg=key)
        np.testing.assert_allclose(
            per_update_30[COSINE_KEY][computed_30], per_update_1[COSINE_KEY][computed_30], rtol=1e-4
        )

        samples_1 = [float(v) for v in per_update_1[COSINE_KEY]]
        samples_30 = [float(per_update_30[COSINE_KEY][step]) for step in computed_30]
        self.assertEqual(meters_1[COSINE_KEY].count, self.NUM_UPDATES)
        self.assertEqual(meters_30[COSINE_KEY].count, len(samples_30))
        self.assertEqual(meters_1[COSINE_KEY].avg, sum(samples_1) / len(samples_1))
        self.assertEqual(meters_30[COSINE_KEY].avg, sum(samples_30) / len(samples_30))
        self.assertNotAlmostEqual(
            meters_30[COSINE_KEY].avg, sum(samples_30) / self.NUM_UPDATES, places=3
        )
        self.assertEqual(meters_30["train/actor_loss"].count, self.NUM_UPDATES)

    def test_cosine_branch_executes_only_on_cadence_steps(self):
        """Counts real executions of the gated branch (debug callbacks are
        unsupported on METAL, so this runs on CPU)."""
        from scale_rl.agents.sac import sac_agent

        executed = []
        real_fn = sac_agent.compute_actor_gradient_cosine

        def counted(**kwargs):
            jax.debug.callback(lambda key: executed.append(1), kwargs["key"])
            return real_fn(**kwargs)

        self.addCleanup(jax.clear_caches)
        for every, first_update_step, expected in ((30, 0, 3), (1, 0, 90), (4, 1, 22)):
            executed.clear()
            jax.clear_caches()
            with self.subTest(every=every), jax.default_device(jax.devices("cpu")[0]), \
                    mock.patch.object(sac_agent, "compute_actor_gradient_cosine", counted):
                agent = _make_agent()
                batches = _make_batches(self.NUM_UPDATES)
                stacked = {key: np.stack([b[key] for b in batches]) for key in batches[0]}
                info = agent.update_many(first_update_step, stacked, every)
                cosine = np.asarray(info[COSINE_KEY])
                self.assertEqual(len(executed), expected)
                steps = first_update_step + np.arange(self.NUM_UPDATES)
                np.testing.assert_array_equal(~np.isnan(cosine), steps % every == 0)


class ScanFusionEquivalenceTest(unittest.TestCase):
    """update_many (one lax.scan call) vs. sequential update() calls.

    Compared per call from an identical starting state, so differences can
    only come from float reassociation within one fused call. Over a long
    free-running trajectory those differences compound into slow, persistent
    drift between the two runs, which a sign test cannot tell apart from a
    real behavioral change - so that comparison is not asserted here.
    """

    NUM_CALLS = 600
    UPDATES_PER_CALL = 5
    REL_TOL = 1e-5
    KEYS = (
        "train/actor_loss", "train/critic_loss", "train/td_error_var",
        "train/q1_mean", "train/entropy", "train/actor_grad_cosine",
    )

    def _max_rel_and_sign_balance(self, device):
        with jax.default_device(device):
            sequential, fused = _make_agent(), _make_agent()
            rng = np.random.default_rng(0)
            diffs = {key: [] for key in self.KEYS}
            refs = {key: [] for key in self.KEYS}
            for call in range(self.NUM_CALLS):
                for attr in ("_rng", "_actor", "_critic", "_target_critic", "_temperature"):
                    setattr(fused.agent, attr, getattr(sequential.agent, attr))
                fused.agent.churn_ref_batch = sequential.agent.churn_ref_batch
                chunk = _make_batches(self.UPDATES_PER_CALL, batch_size=32, seed=int(rng.integers(1 << 31)))
                first = call * self.UPDATES_PER_CALL
                stacked = {key: np.stack([b[key] for b in chunk]) for key in chunk[0]}
                ref = [
                    sequential.update(first + i, {k: v.copy() for k, v in b.items()})
                    for i, b in enumerate(chunk)
                ]
                out = jax.device_get(fused.update_many(first, stacked, 1))
                for key in self.KEYS:
                    for i in range(self.UPDATES_PER_CALL):
                        diffs[key].append(float(out[key][i]) - ref[i][key])
                        refs[key].append(ref[i][key])
        return {key: (np.array(diffs[key]), np.array(refs[key])) for key in self.KEYS}

    def test_fused_update_matches_sequential_per_call(self):
        devices = {jax.devices()[0].platform: jax.devices()[0], "cpu": jax.devices("cpu")[0]}
        for platform, device in devices.items():
            for key, (diff, ref) in self._max_rel_and_sign_balance(device).items():
                with self.subTest(platform=platform, key=key):
                    self.assertTrue(np.all(np.isfinite(diff)))
                    rel = np.max(np.abs(diff)) / np.max(np.abs(ref))
                    self.assertLessEqual(rel, self.REL_TOL, f"max relative deviation {rel:.3e}")
                    nonzero = diff[diff != 0]
                    if len(nonzero) >= 100:
                        positive_frac = float(np.mean(nonzero > 0))
                        self.assertTrue(
                            0.35 <= positive_frac <= 0.65,
                            f"systematic deviation: {positive_frac:.2f} of {len(nonzero)} nonzero "
                            f"differences are positive",
                        )


if __name__ == "__main__":
    unittest.main()
