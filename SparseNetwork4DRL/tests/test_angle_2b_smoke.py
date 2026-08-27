"""Smoke test for Angle 2B: Actor-Facing Signal Falsification.

Exercises the full pipeline end to end on tiny synthetic agents (no real
Angle 1/Angle 2A run exists in this checkout yet - see the End-of-Task
Summary for why), matching testing-and-verification.md's requirement that
training/metric changes get a real execution-path smoke test, not just a
clean exit code:

  1. builds tiny synthetic SAC agents (playing D/R for a primary matchup,
     and A/B pairs for two null-baseline seeds) and trains each briefly so
     their critics actually diverge from initialization
  2. persists them via experiments.angle_2a.storage.save_frozen_agent_snapshot
     (the exact function added to Angle 2A for this project)
  3. runs experiments.angle_2b.matchup_2b.run_angle_2b_analysis end to end,
     which internally performs all four gradient computations
     (g_{D|D}, g_{D|R}, g_{R|R}, g_{R|D}) under jax.jit, builds the null
     distribution from the two persisted null pairs, and compares primary
     against it using the (mean + 2*std) rule
  4. asserts every distortion metric is finite (non-NaN/non-Inf), the null
     distribution is non-degenerate (more than one distinct value, not all
     seeds collapsing to the same number), and persisted output files are
     present, loadable, and internally consistent
  5. separately verifies ObservationNormalizer.save_checkpoint/load_checkpoint
     round-trips obs_rms exactly (the correctness fix this project made -
     without it, Angle 2B would silently evaluate frozen agents on the wrong
     input distribution)
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import ProbeCapture
from experiments.angle_2a.storage import save_frozen_agent_snapshot
from experiments.angle_2b.matchup_2b import run_angle_2b_analysis
from experiments.angle_2b.storage import analysis_dir
from scale_rl.agents import create_agent
from scale_rl.agents.wrappers.normalization import ObservationNormalizer

OBS_DIM = 4
ACT_DIM = 2
NUM_STATES_PER_SOURCE = 5  # small on purpose: this is a smoke test, not a full run


def make_agent_cfg(seed: int, critic_num_blocks: int, critic_hidden_dim: int, normalize_observation: bool = True) -> dict:
    return {
        "agent_type": "sac",
        "seed": seed,
        "num_train_envs": 1,
        "max_episode_steps": 100,
        "normalize_observation": normalize_observation,
        "actor_block_type": "residual",
        "actor_num_blocks": 1,
        "actor_hidden_dim": 8,
        "actor_learning_rate": 1e-4,
        "actor_weight_decay": 1e-2,
        "critic_block_type": "residual",
        "critic_num_blocks": critic_num_blocks,
        "critic_hidden_dim": critic_hidden_dim,
        "critic_learning_rate": 1e-4,
        "critic_weight_decay": 1e-2,
        # critic_use_cdq=False (not True) deliberately: SACClippedDoubleCritic
        # (the True/CDQ path) fails to even initialize under this
        # environment's installed jax==0.4.34/flax==0.8.4 (nn.vmap with
        # in_axes=None rejects the observations/actions kwargs with "Expected
        # None, got (Array...)") - confirmed by reproducing it in total
        # isolation from any Angle 2B code (see the End-of-Task Summary).
        # This is a pre-existing environment/dependency bug in
        # scale_rl/agents/sac/sac_network.py, unrelated to and out of scope
        # for Angle 2B; gradients.py's _actor_loss still mirrors
        # update_actor's CDQ branch verbatim for when this is fixed upstream.
        "critic_use_cdq": False,
        "temp_target_entropy": None,
        "temp_target_entropy_coef": -0.5,
        "temp_initial_value": 0.01,
        "temp_learning_rate": 1e-4,
        "temp_weight_decay": 0.0,
        "target_tau": 0.005,
        "gamma": 0.99,
        "n_step": 1,
        "mixed_precision": False,
        "actor_sparsity": 0.0,
        "critic_sparsity": 0.0,
    }


def make_spaces():
    import gymnasium as gym

    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
    return observation_space, action_space


def make_random_batch(rng: np.random.Generator, batch_size: int = 16) -> dict:
    return {
        "observation": rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(batch_size, ACT_DIM)).astype(np.float32),
        "reward": rng.normal(size=(batch_size,)).astype(np.float32),
        "terminated": np.zeros((batch_size,), dtype=np.float32),
        "next_observation": rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
    }


def make_trained_agent(seed: int, critic_num_blocks: int, critic_hidden_dim: int, num_updates: int = 3):
    """Builds a tiny SAC agent and runs a few real .update() calls on random
    batches, so its critic actually diverges from initialization rather than
    every agent trivially agreeing at t=0."""
    observation_space, action_space = make_spaces()
    agent_cfg_dict = make_agent_cfg(seed, critic_num_blocks, critic_hidden_dim)
    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        cfg=OmegaConf.create(agent_cfg_dict),
    )
    rng = np.random.default_rng(seed)
    for step in range(num_updates):
        agent.update(step, make_random_batch(rng))
    return agent, agent_cfg_dict


def make_probe_capture(seed: int, n: int = 40) -> ProbeCapture:
    rng = np.random.default_rng(seed + 1000)
    pc = ProbeCapture(capacity=n, observation_shape=(OBS_DIM,), action_shape=(ACT_DIM,))
    for i in range(n):
        pc.add(
            i,
            rng.normal(size=(OBS_DIM,)).astype(np.float32),
            rng.uniform(-1, 1, size=(ACT_DIM,)).astype(np.float32),
            env_state=None,
        )
    return pc


class Angle2BSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path(tempfile.mkdtemp(prefix="angle2b_smoke_"))
        self.angle_2a_root = self.tmp_root / "angle_2a"
        self.angle_2b_root = self.tmp_root / "angle_2b"
        self.environment = "smoketest_env"

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _persist(self, environment, seed, matchup_name, role, seed_for_agent, critic_num_blocks, critic_hidden_dim):
        agent, agent_cfg_dict = make_trained_agent(seed_for_agent, critic_num_blocks, critic_hidden_dim)
        probe_capture = make_probe_capture(seed_for_agent)
        save_frozen_agent_snapshot(
            environment, seed, matchup_name, role, agent, probe_capture,
            agent_cfg=agent_cfg_dict, root=str(self.angle_2a_root),
        )
        return agent

    def test_full_pipeline_smoke(self):
        seed = 1

        # Primary matchup: D (a "scaled" 3x16 critic) vs R (a "reference" 1x8 critic).
        self._persist(self.environment, seed, "matchup_1", "D", seed_for_agent=101, critic_num_blocks=3, critic_hidden_dim=16)
        self._persist(self.environment, seed, "matchup_1", "R", seed_for_agent=102, critic_num_blocks=1, critic_hidden_dim=8)

        # Two independent null-baseline pairs (two seeds), both default (1x8) architecture.
        for null_seed, (a_init, b_init) in {1: (201, 202), 2: (203, 204)}.items():
            self._persist(self.environment, null_seed, "null_matchup_1", "D", seed_for_agent=a_init, critic_num_blocks=1, critic_hidden_dim=8)
            self._persist(self.environment, null_seed, "null_matchup_1", "R", seed_for_agent=b_init, critic_num_blocks=1, critic_hidden_dim=8)

        result = run_angle_2b_analysis(
            environment=self.environment,
            seed=seed,
            matchup_name="matchup_1",
            null_seeds=[1, 2],
            analysis_seed=42,
            num_states_per_source=NUM_STATES_PER_SOURCE,
            angle_2a_root=str(self.angle_2a_root),
            output_root=str(self.angle_2b_root),
        )

        # --- primary / secondary: finite, sane ---
        for label, metrics in (("primary", result.primary), ("secondary", result.secondary)):
            for metric_name in ("d_dir", "d_mag", "d_grad"):
                value = metrics[metric_name]
                self.assertTrue(np.isfinite(value), f"{label}.{metric_name} is not finite: {value}")
            self.assertGreaterEqual(metrics["d_dir"], -1e-6, "D_dir should be >= 0 (1 - cosine, cosine <= 1)")
            self.assertGreaterEqual(metrics["d_grad"], 0.0, "D_grad is a norm, must be >= 0")

        # --- null distribution: non-degenerate ---
        self.assertEqual(len(result.null_pairs), 2, "expected both null seeds to be found")
        null_d_dirs = [p.d_dir for p in result.null_pairs]
        for p in result.null_pairs:
            self.assertTrue(np.isfinite(p.d_dir) and np.isfinite(p.d_mag) and np.isfinite(p.d_grad))
        self.assertNotEqual(
            null_d_dirs[0], null_d_dirs[1],
            "the two null pairs collapsed to an identical D_dir - null distribution is degenerate",
        )

        # --- null comparison: well-formed for all three metrics ---
        for metric_name in ("d_dir", "d_mag", "d_grad"):
            comparison = result.null_comparison[metric_name]
            self.assertTrue(np.isfinite(comparison.null_mean))
            self.assertTrue(np.isfinite(comparison.threshold))
            self.assertEqual(comparison.null_n, 2)
            self.assertIsInstance(comparison.exceeds_null, bool)

        # --- persisted outputs exist and round-trip ---
        out_dir = analysis_dir(self.environment, seed, "matchup_1", root=str(self.angle_2b_root))
        self.assertTrue((out_dir / "run_metadata.json").exists())
        self.assertTrue((out_dir / "null_distribution.csv").exists())
        self.assertTrue((out_dir / "gradients.npz").exists())

        import json

        with open(out_dir / "run_metadata.json") as f:
            metadata = json.load(f)
        self.assertEqual(metadata["matchup_name"], "matchup_1")
        self.assertEqual(metadata["null_matchup_name"], "null_matchup_1")
        self.assertAlmostEqual(metadata["primary"]["d_dir"], result.primary["d_dir"], places=5)

        with np.load(out_dir / "gradients.npz") as npz:
            for key in ("g_D_given_D", "g_D_given_R", "g_R_given_R", "g_R_given_D"):
                self.assertIn(key, npz.files)
                self.assertTrue(np.all(np.isfinite(npz[key])))
                self.assertGreater(npz[key].shape[0], 0)

    def test_observation_normalizer_checkpoint_round_trip(self):
        """Verifies the ObservationNormalizer.save_checkpoint/load_checkpoint
        fix this project made: obs_rms must survive a save/load round trip
        exactly, or Angle 2B would evaluate frozen agents on the wrong input
        distribution (they were trained on normalized observations)."""
        observation_space, action_space = make_spaces()
        agent_cfg_dict = make_agent_cfg(seed=1, critic_num_blocks=1, critic_hidden_dim=8)
        agent = create_agent(observation_space, action_space, OmegaConf.create(dict(agent_cfg_dict)))
        self.assertIsInstance(agent, ObservationNormalizer)

        rng = np.random.default_rng(0)
        # Drive obs_rms away from its (mean=0, var=1, count=eps) initial state.
        for _ in range(5):
            batch = make_random_batch(rng)
            agent.sample_actions(
                interaction_step=0,
                prev_timestep={"next_observation": batch["observation"]},
                training=True,
            )

        mean_before = agent.obs_rms.mean.copy()
        var_before = agent.obs_rms.var.copy()
        count_before = agent.obs_rms.count

        ckpt_dir = self.tmp_root / "obs_rms_ckpt"
        agent.save_checkpoint(str(ckpt_dir))

        # Fresh agent shell with default-initialized obs_rms.
        fresh_agent = create_agent(observation_space, action_space, OmegaConf.create(dict(agent_cfg_dict)))
        fresh_agent.load_checkpoint(str(ckpt_dir))

        np.testing.assert_array_equal(fresh_agent.obs_rms.mean, mean_before)
        np.testing.assert_array_equal(fresh_agent.obs_rms.var, var_before)
        self.assertEqual(fresh_agent.obs_rms.count, count_before)


if __name__ == "__main__":
    unittest.main()
