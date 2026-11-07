"""Covers Angle 2A's resumability (audit item 3, 2026-09-07): training-phase
checkpoint/resume (agent_runner.py) and MC-validation-phase per-rollout
resumability (probes.py's run_monte_carlo_rollouts).

Live-tested throughout: real dm_control training, real SACAgent checkpoints,
real ProbeCapture persistence, real MC rollouts. Not tested: bit-exact
reproducibility of a resumed run against an uninterrupted control - that is
NOT the guarantee this provides (see agent_runner.train_agent_to_step's
docstring: global RNG is unconditionally reseeded on resume, matching
experiments/angle_1.py's own precedent, so the post-resume buffer-sampling
sequence is a fresh continuation, not a bit-identical replay). What's
actually tested: state genuinely carries over (not restarted from scratch),
training reaches the correct final step, and MC-validation genuinely skips
already-completed rollouts and resumes a partial probe from the exact
interrupted rollout rather than redoing it.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import (
    ProbeCapture,
    TrainedAgentHandle,
    build_agent_and_env,
    load_training_checkpoint_meta,
    train_agent_to_step,
)
from experiments.angle_2a.config import RoleArchitecture
from experiments.angle_2a.env_state import restore_env_state
from experiments.angle_2a.probes import (
    Probe,
    SOURCE_D,
    _load_mc_progress,
    _save_mc_progress,
    run_monte_carlo_rollouts,
)

ARCHITECTURE = RoleArchitecture(role="scaled_a", critic_num_blocks=1, critic_hidden_dim=8)
ARCHITECTURE_LABEL = "D1W8"


def _make_cfg(seed: int = 202):
    return OmegaConf.create(
        {
            "seed": seed, "gamma": 0.99, "num_interaction_steps": 60,
            "env": {
                "env_type": "dmc", "env_name": "cheetah-run", "seed": seed,
                "num_train_envs": 1, "num_eval_envs": 1, "rescale_action": True,
                "no_termination": False, "action_repeat": 1, "reward_scale": 1.0, "max_episode_steps": 20,
            },
            "buffer": {
                "buffer_class_type": "numpy", "buffer_type": "uniform", "n_step": 1, "gamma": 0.99,
                "max_length": 1000, "min_length": 3, "add_batch_size": 1, "sample_batch_size": 4,
            },
            "updates_per_interaction_step": 1,
            "agent": {
                "agent_type": "sac", "seed": seed, "num_train_envs": 1, "max_episode_steps": 20,
                "normalize_observation": False, "actor_block_type": "residual", "actor_num_blocks": 1,
                "actor_hidden_dim": 8, "actor_learning_rate": 1e-4, "actor_weight_decay": 1e-2,
                "critic_block_type": "residual", "critic_num_blocks": 1, "critic_hidden_dim": 8,
                "critic_learning_rate": 1e-4, "critic_weight_decay": 1e-2, "critic_use_cdq": False,
                "temp_target_entropy": None, "temp_target_entropy_coef": -0.5, "temp_initial_value": 0.01,
                "temp_learning_rate": 1e-4, "temp_weight_decay": 0.0, "target_tau": 0.005, "gamma": 0.99,
                "n_step": 1, "mixed_precision": False, "actor_sparsity": 0.0, "critic_sparsity": 0.0,
            },
        }
    )


class TestTrainingResumability(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_partial_run_then_resume_reaches_correct_final_step_and_preserves_earlier_state(self):
        checkpoint_dir = os.path.join(self.tmpdir, "d_training")
        seed_context = "resumability_test:D:D1W8"
        stop_step = 30  # fixed throughout - matches real usage, where
        # onset_step never changes between an interrupted attempt and its
        # resume (run_matchup() always passes the same looked-up onset_step)

        # "Attempt 1": a process interrupted partway through - driven
        # directly via build_agent_and_env/run_training_loop (rather than
        # train_agent_to_step, which would run this to completion) so it can
        # be abandoned after only 15 of the full 30 steps, exactly like a
        # real kill mid-run would leave things (last checkpoint at step 15,
        # since checkpoint_interval=5).
        from experiments.angle_2a.agent_runner import (
            build_agent_and_env,
            run_training_loop,
            seed_global_rng_for_agent,
        )

        cfg = _make_cfg()
        check_env, check_eval_env, check_single_env, check_buffer, check_agent, obs_space, act_space = (
            build_agent_and_env(ARCHITECTURE, cfg)
        )
        probe_capacity = min(int(cfg.buffer.max_length), stop_step)
        check_probe_capture = ProbeCapture(capacity=probe_capacity, observation_shape=obs_space.shape[-1:], action_shape=act_space.shape[-1:])
        seed_global_rng_for_agent(int(cfg.seed), seed_context)
        loop = run_training_loop(
            check_agent, check_buffer, check_env, check_single_env, check_probe_capture, cfg, stop_step,
            checkpoint_dir=checkpoint_dir, checkpoint_interval=5,
        )
        for step in loop:
            if step == 15:
                break  # abandon here - simulates the process dying right after this checkpoint
        early_observations = check_probe_capture._observations[:15].copy()
        early_actions = check_probe_capture._actions[:15].copy()
        check_env.close()
        check_eval_env.close()

        meta_after_attempt_1 = load_training_checkpoint_meta(checkpoint_dir)
        self.assertIsNotNone(meta_after_attempt_1)
        self.assertEqual(meta_after_attempt_1["interaction_step"], 15)

        # "Attempt 2": the same command re-invoked (same checkpoint_dir, same
        # seed_context, same stop_step) - must detect the existing checkpoint
        # and resume from 16, not restart from 1.
        handle_2 = train_agent_to_step(
            role="D", architecture=ARCHITECTURE, architecture_label=ARCHITECTURE_LABEL,
            base_cfg=_make_cfg(), stop_step=30, seed_context=seed_context,
            checkpoint_dir=checkpoint_dir, checkpoint_interval=5,
        )
        try:
            self.assertEqual(len(handle_2.probe_capture), 30)
            # The pre-interruption transitions (steps 1-15) must be exactly
            # preserved, not overwritten by a restart - proves this was a
            # genuine resume, not a fresh run that happened to also reach 30.
            np.testing.assert_array_equal(handle_2.probe_capture._observations[:15], early_observations)
            np.testing.assert_array_equal(handle_2.probe_capture._actions[:15], early_actions)
        finally:
            handle_2.close()

        meta_after_attempt_2 = load_training_checkpoint_meta(checkpoint_dir)
        self.assertEqual(meta_after_attempt_2["interaction_step"], 30)

    def test_resuming_an_already_fully_trained_checkpoint_is_a_no_op(self):
        checkpoint_dir = os.path.join(self.tmpdir, "d_training_full")
        seed_context = "resumability_test:D:full"
        handle_1 = train_agent_to_step(
            role="D", architecture=ARCHITECTURE, architecture_label=ARCHITECTURE_LABEL,
            base_cfg=_make_cfg(), stop_step=20, seed_context=seed_context,
            checkpoint_dir=checkpoint_dir, checkpoint_interval=5,
        )
        handle_1.close()

        # Re-invoking with the SAME stop_step=20 on an already-complete
        # checkpoint must not error and must not redo any training - the
        # resumed range(21, 21) is empty by construction.
        handle_2 = train_agent_to_step(
            role="D", architecture=ARCHITECTURE, architecture_label=ARCHITECTURE_LABEL,
            base_cfg=_make_cfg(), stop_step=20, seed_context=seed_context,
            checkpoint_dir=checkpoint_dir, checkpoint_interval=5,
        )
        try:
            self.assertEqual(len(handle_2.probe_capture), 20)
        finally:
            handle_2.close()

    def test_no_checkpoint_dir_means_no_checkpoint_written(self):
        checkpoint_dir = os.path.join(self.tmpdir, "never_written")
        handle = train_agent_to_step(
            role="D", architecture=ARCHITECTURE, architecture_label=ARCHITECTURE_LABEL,
            base_cfg=_make_cfg(), stop_step=10, seed_context="resumability_test:D:none",
        )
        handle.close()
        self.assertFalse(os.path.exists(checkpoint_dir))
        self.assertIsNone(load_training_checkpoint_meta(None))
        self.assertIsNone(load_training_checkpoint_meta(checkpoint_dir))


class TestMonteCarloValidationResumability(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        cfg = _make_cfg()
        self.train_env, self.eval_env, self.single_env, self.buffer, self.agent, obs_space, act_space = (
            build_agent_and_env(ARCHITECTURE, cfg)
        )
        probe_capture = ProbeCapture(capacity=30, observation_shape=obs_space.shape[-1:], action_shape=act_space.shape[-1:])
        from experiments.angle_2a.agent_runner import run_training_loop, seed_global_rng_for_agent

        seed_global_rng_for_agent(202, "resumability_test:mc")
        for _ in run_training_loop(self.agent, self.buffer, self.train_env, self.single_env, probe_capture, cfg, 20):
            pass
        self.handle = TrainedAgentHandle(
            role="D", architecture_label=ARCHITECTURE_LABEL, architecture=ARCHITECTURE,
            agent=self.agent, buffer=self.buffer, train_env=self.train_env, eval_env=self.eval_env,
            single_env=self.single_env, stop_step=20, probe_capture=probe_capture,
        )
        idxs, states, actions, env_states = probe_capture.sample(3, np.random.default_rng(0))
        self.probes = [
            Probe(probe_id=f"mc_resume_probe_{i}", source=SOURCE_D, state=states[i], action=actions[i], env_state=env_states[i])
            for i in range(3)
        ]

    def tearDown(self):
        self.handle.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_already_fully_done_probe_is_not_recomputed_on_resume(self):
        checkpoint_path = os.path.join(self.tmpdir, "mc_progress.pkl")
        run_monte_carlo_rollouts(self.probes, self.handle, self.handle, num_rollouts=4, gamma=0.99, max_rollout_steps=60, checkpoint_path=checkpoint_path)
        full_returns = {p.probe_id: list(p.mc_rollout_returns) for p in self.probes}

        # Reset the Probe objects to blank, as a resumed process would have
        # (Probe objects themselves aren't persisted - only checkpoint_path's
        # raw per-probe-id progress survives a resume).
        for p in self.probes:
            p.mc_rollout_returns = []
            p.mc_return = None
            p.mc_return_se = None

        call_count = {"n": 0}
        real_restore = restore_env_state

        def counting_restore(env, captured):
            call_count["n"] += 1
            return real_restore(env, captured)

        import experiments.angle_2a.probes as probes_module
        probes_module.restore_env_state = counting_restore
        try:
            run_monte_carlo_rollouts(self.probes, self.handle, self.handle, num_rollouts=4, gamma=0.99, max_rollout_steps=60, checkpoint_path=checkpoint_path)
        finally:
            probes_module.restore_env_state = real_restore

        self.assertEqual(call_count["n"], 0, "already-fully-done probes must not trigger any new rollouts")
        for p in self.probes:
            self.assertEqual(list(p.mc_rollout_returns), full_returns[p.probe_id])

    def test_partially_done_probe_resumes_from_the_exact_interrupted_rollout(self):
        checkpoint_path = os.path.join(self.tmpdir, "mc_progress_partial.pkl")
        num_rollouts = 4
        target_probe = self.probes[0]

        # Simulate "2 of 4 rollouts completed, then interrupted" by seeding
        # the progress file directly with 2 fabricated (but distinguishable)
        # returns for this probe, and nothing for the other two probes.
        fabricated_partial = [111.0, 222.0]
        _save_mc_progress(checkpoint_path, {target_probe.probe_id: fabricated_partial})

        run_monte_carlo_rollouts(
            self.probes, self.handle, self.handle, num_rollouts=num_rollouts,
            gamma=0.99, max_rollout_steps=60, checkpoint_path=checkpoint_path,
        )

        self.assertEqual(len(target_probe.mc_rollout_returns), num_rollouts)
        # The first 2 are exactly preserved, not recomputed/discarded.
        self.assertEqual(target_probe.mc_rollout_returns[:2], fabricated_partial)
        # The remaining 2 are genuinely new (not also 111.0/222.0 by coincidence).
        self.assertNotEqual(target_probe.mc_rollout_returns[2:], [111.0, 222.0])

        # The other two probes had no prior progress - fully computed, 4 each.
        for p in self.probes[1:]:
            self.assertEqual(len(p.mc_rollout_returns), num_rollouts)

        # Progress file reflects the final, complete state for all 3 probes.
        final_progress = _load_mc_progress(checkpoint_path)
        for p in self.probes:
            self.assertEqual(len(final_progress[p.probe_id]), num_rollouts)


if __name__ == "__main__":
    unittest.main()
