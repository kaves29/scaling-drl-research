"""Exercises experiments.angle_2_a.run() itself, end to end, on tiny/synthetic
config overrides - not the building-block tests that have substituted for
this so far (matchup.py/agent_runner.py/probes.py are all individually
tested, but no test previously called angle_2_a.run() itself). Specifically
targets the refactored Phase 3 (pool_null_baseline.py), since that's the
newest structural change to this angle's real entry point.

wandb.init() is called unconditionally by matchup.py's _log_to_wandb - mocked
at the module-attribute level, same approach as test_angle1_real_entry_point.py.

Every relevant path this module touches (onset ledger root, shared pool
storage root, pool-null cache root, and run_matchup's own hardcoded
output_root="results/angle_2a") defaults to a relative "results/..." path
with no CLI/config override - so, same as test_angle1_real_entry_point.py's
critic_degradation test, this test chdirs into a tmpdir for its duration
(restored in tearDown) rather than writing into this repo's real results/
directory.

r_calibration and prereq_check are disabled via overrides: both are
substantial, already-separately-tested phases (r_calibration.py,
prereq_check.py each have their own dedicated tests) unrelated to what
changed recently; enabling them here would only add runtime and setup
complexity without additional coverage of the actual change being tested.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import wandb
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import ProbeCapture
from experiments.angle_2a.storage import matchup_dir, save_frozen_agent_snapshot
from utils.onset_ledger import WandbIdentity, log_onset_event

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs")
ENVIRONMENT = "cheetah-run"


class _FakeWandbRun:
    def __init__(self):
        self.name = "fake-run"
        self.id = "fake-run-id"
        self.summary = {}

    def update(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def _fake_wandb_init(*args, **kwargs):
    run = _FakeWandbRun()
    wandb.run = run
    return run


class _FakeWandbRunIdentityObj:
    def __init__(self, architecture, seed):
        self.name = f"fake-run-{architecture}-{seed}"
        self.id = "fake-wandb-id"


def _write_onset_ledger_entry(ledger_root, environment, seed, architecture, onset_step):
    row = {
        "run_key": f"angle1_{architecture}_{environment}_seed{seed}",
        "exact_run_name": f"fake-run-{architecture}-{seed}",
        "wandb_run_id": "fake-wandb-id",
        "architecture": architecture,
        "environment": environment,
        "seed": seed,
        "critic_degradation_onset_step": onset_step,
        "critic_degradation_method": "td_variance_p95_sustained_v1",
        "propagation_onset_step": None,
        "propagation_method": None,
        "propagation_lag": None,
        "status": "success",
        "detection_notes": "synthetic entry for Angle 2A real-entry-point test",
    }
    log_onset_event(
        exp_name="angle_1", architecture=architecture, row=row,
        identity=WandbIdentity(run_obj=_FakeWandbRunIdentityObj(architecture, seed)),
        root=ledger_root, mirror_to_wandb=False,
    )


def _make_pool_agent_cfg(seed):
    return {
        "agent_type": "sac", "seed": seed, "num_train_envs": 1, "max_episode_steps": 20,
        "normalize_observation": False, "actor_block_type": "residual", "actor_num_blocks": 1,
        "actor_hidden_dim": 8, "actor_learning_rate": 1e-4, "actor_weight_decay": 1e-2,
        "critic_block_type": "residual", "critic_num_blocks": 1, "critic_hidden_dim": 8,
        "critic_learning_rate": 1e-4, "critic_weight_decay": 1e-2, "critic_use_cdq": False,
        "temp_target_entropy": None, "temp_target_entropy_coef": -0.5, "temp_initial_value": 0.01,
        "temp_learning_rate": 1e-4, "temp_weight_decay": 0.0, "target_tau": 0.005, "gamma": 0.99,
        "n_step": 1, "mixed_precision": False, "actor_sparsity": 0.0, "critic_sparsity": 0.0,
    }


class Angle2ARealEntryPointTest(unittest.TestCase):
    """NOTE on a real bug found while writing this test (see the End-of-Task
    Summary): SACAgent.save_checkpoint()/load_checkpoint()
    (scale_rl/agents/sac/sac_agent.py) pass checkpoint_dir straight to
    orbax with no os.path.abspath() conversion, and orbax's checkpointer
    REJECTS any relative path outright ("Checkpoint path should be
    absolute"), regardless of cwd - confirmed directly, not assumed. This
    is pre-existing (not introduced by the pool refactor) and affects
    Angle 1's own --checkpoint_dir too if it were ever passed relative; it
    has simply never been hit because operators have always passed absolute
    paths by convention, never enforced. It WOULD immediately crash a real
    Angle 2A run, though: angle_2_a.py's Phase 2 hardcodes
    output_root="results/angle_2a" (relative, no override mechanism at
    all), so agent.save_checkpoint() inside save_frozen_agent_snapshot
    would always fail on an unmodified real invocation. Patched here
    (absolute-ifying checkpoint_dir before delegating to the real method)
    so this test can still validate everything else about the real pipeline
    - this is a test-only workaround, NOT a fix to the underlying bug."""

    def setUp(self):
        GlobalHydra.instance().clear()
        self.tmpdir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.wandb_init_patcher = mock.patch("wandb.init", side_effect=_fake_wandb_init)
        self.wandb_init_patcher.start()

        from scale_rl.agents.sac.sac_agent import SACAgent

        self._real_save_checkpoint = SACAgent.save_checkpoint
        self._real_load_checkpoint = SACAgent.load_checkpoint

        def _abs_save_checkpoint(agent_self, checkpoint_dir):
            return self._real_save_checkpoint(agent_self, os.path.abspath(checkpoint_dir))

        def _abs_load_checkpoint(agent_self, checkpoint_dir):
            return self._real_load_checkpoint(agent_self, os.path.abspath(checkpoint_dir))

        self.checkpoint_patcher = mock.patch.multiple(
            SACAgent, save_checkpoint=_abs_save_checkpoint, load_checkpoint=_abs_load_checkpoint,
        )
        self.checkpoint_patcher.start()

    def tearDown(self):
        self.checkpoint_patcher.stop()
        self.wandb_init_patcher.stop()
        os.chdir(self.original_cwd)
        GlobalHydra.instance().clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _seed_pool(self, pool_root, num_agents=4, num_steps=20):
        """Real, tiny standalone pool agents trained on a real (tiny) live
        cheetah-run environment - env_state must be real (not None), since
        Angle 2A's own refactored null-baseline runs real MC rollouts
        against these agents (unlike Angle 2B, which never needs env_state).
        Mirrors tests/test_angle2a_pool_null_baseline.py's own helper."""
        from scale_rl.agents import create_agent
        from scale_rl.envs import create_envs
        from experiments.angle_2a.env_state import capture_env_state

        for seed in range(1, num_agents + 1):
            train_env, eval_env = create_envs(
                env_type="dmc", seed=seed, env_name=ENVIRONMENT, num_train_envs=1,
                num_eval_envs=1, rescale_action=True, no_termination=False,
                action_repeat=1, reward_scale=1.0, max_episode_steps=20,
            )
            try:
                single_env = train_env.envs[0]
                observation_space = train_env.observation_space
                action_space = train_env.action_space
                agent_cfg = _make_pool_agent_cfg(seed)
                agent = create_agent(observation_space, action_space, OmegaConf.create(agent_cfg))
                probe_capture = ProbeCapture(
                    capacity=num_steps, observation_shape=observation_space.shape[-1:],
                    action_shape=action_space.shape[-1:],
                )
                observations, _ = train_env.reset()
                timestep = None
                for interaction_step in range(1, num_steps + 1):
                    env_state = capture_env_state(single_env, "dmc")
                    if timestep is not None:
                        actions = agent.sample_actions(interaction_step, prev_timestep=timestep, training=True)
                    else:
                        actions = train_env.action_space.sample()
                    probe_capture.add(interaction_step - 1, observations[0], actions[0], env_state)
                    next_observations, _r, _t, _tr, _i = train_env.step(actions)
                    timestep = {"next_observation": next_observations}
                    observations = next_observations

                snapshot_paths = save_frozen_agent_snapshot(
                    ENVIRONMENT, seed, "baseline_pool", "pool", agent, probe_capture,
                    agent_cfg=agent_cfg, root=pool_root,
                )
                probe_capture.save(str(snapshot_paths["checkpoint_dir"]))
            finally:
                train_env.close()
                eval_env.close()

    def test_run_completes_end_to_end_with_pool_based_null_baseline(self):
        from analysis.baseline_calibration_pool import all_unique_pairs, get_baseline_calibration_pool
        from experiments.angle_2_a import run
        from experiments.angle_2a.pool_null_baseline import load_pool_null_distribution

        ledger_root = "results/ledgers"
        pool_root = "results/baseline_calibration_pool"
        onset_step = 15

        _write_onset_ledger_entry(ledger_root, ENVIRONMENT, seed=1, architecture="D2W8", onset_step=onset_step)
        _write_onset_ledger_entry(ledger_root, ENVIRONMENT, seed=1, architecture="D3W8", onset_step=onset_step)

        # Only 4 pool agents (not the real 10) - patch the pool identity
        # lookup to a matching 4-seed subset, exactly like
        # tests/test_angle2a_pool_null_baseline.py already does.
        self._seed_pool(pool_root, num_agents=4)
        fake_identities = [type("Ident", (), {"seed": s})() for s in range(1, 5)]

        with mock.patch(
            "experiments.angle_2a.pool_null_baseline.get_baseline_calibration_pool",
            return_value=fake_identities,
        ):
            run({
                "config_path": CONFIG_PATH,
                "config_name": "base_angle2a",
                "overrides": [
                    f"env_name={ENVIRONMENT}", "seed=1",
                    "actor_num_blocks=1", "actor_hidden_dim=8",
                    "angle_2_a.scaled_a.critic_num_blocks=2", "angle_2_a.scaled_a.critic_hidden_dim=8",
                    "angle_2_a.scaled_b.critic_num_blocks=3", "angle_2_a.scaled_b.critic_hidden_dim=8",
                    "angle_2_a.reference.critic_num_blocks=1", "angle_2_a.reference.critic_hidden_dim=8",
                    "buffer.min_length=3", "buffer.max_length=50", "buffer.sample_batch_size=4",
                    "angle_2_a.num_probes_per_source=2",
                    "angle_2_a.r_calibration.enabled=false", "angle_2_a.num_mc_rollouts=2",
                    "angle_2_a.prereq_check.enabled=false",
                    "angle_2_a.run_null_baseline=true",
                    "angle_2_a.onset_ledger_root=results/ledgers",
                ],
                "checkpoint_dir": None,
                "checkpoint_interval": None,
            })

        # --- real matchup outputs exist ---
        for matchup_name in ("matchup_1", "matchup_2"):
            out_dir = matchup_dir(ENVIRONMENT, 1, matchup_name, root="results/angle_2a")
            self.assertTrue((out_dir / "run_metadata.json").exists(), f"missing {matchup_name} run_metadata.json")
            self.assertTrue((out_dir / "probes.csv").exists())

        # --- pool null distribution genuinely computed and cached (not
        # skipped/empty) - confirms Phase 3's refactor actually ran ---
        cached = load_pool_null_distribution(ENVIRONMENT, root="results/angle_2a_pool_null")
        self.assertIsNotNone(cached, "expected a cached pool null distribution after a real run")
        expected_pairs = len(all_unique_pairs(list(range(1, 5))))
        self.assertEqual(len(cached.pairs), expected_pairs)
        self.assertEqual(cached.null_n, expected_pairs * 2)


if __name__ == "__main__":
    unittest.main()
