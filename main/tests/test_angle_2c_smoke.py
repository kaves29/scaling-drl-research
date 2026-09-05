"""Smoke test for Angle 2C: Distortion Decomposition.

Builds a full synthetic Angle 2A -> Angle 2B -> Angle 2C pipeline on tiny
agents (no real Angle 1/2A/2B run exists in this checkout - matching
tests/test_angle_2b_smoke.py's convention for the same reason), matching
testing-and-verification.md's requirement that changes get a real
execution-path smoke test, not just a clean exit code:

  1. builds tiny synthetic SAC agents (D/R for matchup_1, two null pairs for
     null_matchup_1) and trains each briefly so critics actually diverge
  2. persists them via experiments.angle_2a.storage.save_frozen_agent_snapshot
     (exactly as Angle 2A would) PLUS a synthetic Angle 2A run_metadata.json
     (scaled_architecture/scaled_onset_step - the two fields Angle 2C's
     onset_lookup.py actually reads) and a synthetic Angle 1 onset-ledger
     entry via the real utils.onset_ledger.log_onset_event writer
  3. runs experiments.angle_2b.matchup_2b.run_angle_2b_analysis to produce
     real Angle 2B output (including the (s,a)/nabla_a Q/Q arrays added for
     Angle 2C)
  4. runs experiments.angle_2c.matchup_2c.run_angle_2c_analysis on top,
     asserting: all three properties are non-NaN for primary and secondary,
     the null distribution is well-formed and non-degenerate, the
     onset-timing lookup correctly finds the synthetic Angle 1 ledger entry,
     and persisted output files are present/loadable/consistent
  5. separately, directly unit-tests the reconstruction test's math
     (experiments.angle_2c.reconstruction) with hand-built inputs designed
     to force a co-occurrence-like scenario, since a full pipeline built on
     tiny random synthetic agents cannot reliably be relied on to trigger
     real co-occurrence on its own - this guarantees that code path is
     actually exercised, not just reachable in principle
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import ProbeCapture
from experiments.angle_2a.storage import save_frozen_agent_snapshot
from experiments.angle_2b.matchup_2b import run_angle_2b_analysis
from experiments.angle_2c.matchup_2c import run_angle_2c_analysis
from experiments.angle_2c.reconstruction import run_reconstruction_test
from experiments.angle_2c.storage import analysis_dir as angle_2c_analysis_dir
from scale_rl.agents import create_agent
from utils.atomic_io import atomic_write_text
from utils.onset_ledger import WandbIdentity, log_onset_event

OBS_DIM = 4
ACT_DIM = 2
NUM_PERTURBATIONS = 5  # small on purpose: this is a smoke test, not a full run
NUM_STATES_PER_SOURCE = 5


def make_agent_cfg(seed: int, critic_num_blocks: int, critic_hidden_dim: int) -> dict:
    return {
        "agent_type": "sac",
        "seed": seed,
        "num_train_envs": 1,
        "max_episode_steps": 100,
        "normalize_observation": True,
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
        # critic_use_cdq=False: see tests/test_angle_2b_smoke.py for why
        # (CDQ path fails to initialize under the installed jax/flax combo).
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
    """A few real .update() calls before returning - an immediately-saved,
    freshly-initialized agent was observed (see the Angle 2B sourcing
    End-of-Task Summary) to reliably trigger a pre-existing orbax/jax-metal
    checkpoint-restore quirk on this machine, unrelated to Angle 2C."""
    observation_space, action_space = make_spaces()
    agent_cfg_dict = make_agent_cfg(seed, critic_num_blocks, critic_hidden_dim)
    agent = create_agent(observation_space, action_space, OmegaConf.create(agent_cfg_dict))
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


class Angle2CSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path(tempfile.mkdtemp(prefix="angle2c_smoke_"))
        self.angle_2a_root = self.tmp_root / "angle_2a"
        self.angle_2b_root = self.tmp_root / "angle_2b"
        self.angle_2c_root = self.tmp_root / "angle_2c"
        self.ledger_root = self.tmp_root / "ledgers"
        self.environment = "smoketest_env"
        self.addCleanup(shutil.rmtree, self.tmp_root, ignore_errors=True)

    def _persist(self, environment, seed, matchup_name, role, seed_for_agent, critic_num_blocks, critic_hidden_dim):
        agent, agent_cfg_dict = make_trained_agent(seed_for_agent, critic_num_blocks, critic_hidden_dim)
        probe_capture = make_probe_capture(seed_for_agent)
        save_frozen_agent_snapshot(
            environment, seed, matchup_name, role, agent, probe_capture,
            agent_cfg=agent_cfg_dict, root=str(self.angle_2a_root),
        )

    def _write_angle_2a_run_metadata(self, environment, seed, matchup_name, scaled_architecture, scaled_onset_step):
        """Angle 2C's onset_lookup.py reads scaled_architecture/
        scaled_onset_step from Angle 2A's run_metadata.json - written here
        by experiments.angle_2a.matchup.run_matchup() in real usage, but
        this smoke test only calls save_frozen_agent_snapshot() (matching
        test_angle_2b_smoke.py's convention), which does not produce this
        file - so it is constructed directly here, matching the real
        schema's field names exactly."""
        import json

        out_dir = self.angle_2a_root / environment / f"seed{seed}" / matchup_name
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            out_dir / "run_metadata.json",
            json.dumps({"scaled_architecture": scaled_architecture, "scaled_onset_step": scaled_onset_step}),
        )

    def _write_angle_1_ledger_entry(self, environment, seed, architecture, onset_step):
        # log_onset_event force-downgrades status to needs_manual_review
        # whenever the run identity can't be verified against a live/API
        # WandB run (identity=None) - see utils/onset_ledger.py's
        # _validate_identity. A minimal duck-typed run_obj (just .name/.id)
        # is sufficient to satisfy that check without a real WandB run.
        class _FakeWandbRun:
            name = f"fake-run-{architecture}-{seed}"
            id = "fake-wandb-id"

        row = {
            "run_key": f"angle1_{architecture}_{environment}_seed{seed}",
            "exact_run_name": _FakeWandbRun.name,
            "wandb_run_id": _FakeWandbRun.id,
            "architecture": architecture,
            "environment": environment,
            "seed": seed,
            "critic_degradation_onset_step": onset_step,
            "critic_degradation_method": "td_variance_p95_sustained_v1",
            "propagation_onset_step": None,
            "propagation_method": None,
            "propagation_lag": None,
            "status": "success",
            "detection_notes": "synthetic entry for Angle 2C smoke test",
        }
        log_onset_event(
            exp_name="angle_1", architecture=architecture, row=row,
            identity=WandbIdentity(run_obj=_FakeWandbRun()),
            root=str(self.ledger_root), mirror_to_wandb=False,
        )

    def test_full_pipeline_smoke(self):
        seed = 1
        onset_step = 120_000

        self._persist(self.environment, seed, "matchup_1", "D", seed_for_agent=101, critic_num_blocks=3, critic_hidden_dim=16)
        self._persist(self.environment, seed, "matchup_1", "R", seed_for_agent=102, critic_num_blocks=1, critic_hidden_dim=8)
        for null_seed, (a_init, b_init) in {1: (201, 202), 2: (203, 204)}.items():
            self._persist(self.environment, null_seed, "null_matchup_1", "D", seed_for_agent=a_init, critic_num_blocks=1, critic_hidden_dim=8)
            self._persist(self.environment, null_seed, "null_matchup_1", "R", seed_for_agent=b_init, critic_num_blocks=1, critic_hidden_dim=8)

        self._write_angle_2a_run_metadata(self.environment, seed, "matchup_1", scaled_architecture="D3W16", scaled_onset_step=onset_step)
        self._write_angle_1_ledger_entry(self.environment, seed, architecture="D3W16", onset_step=onset_step)

        run_angle_2b_analysis(
            environment=self.environment, seed=seed, matchup_name="matchup_1",
            null_seeds=[1, 2], analysis_seed=42, num_states_per_source=NUM_STATES_PER_SOURCE,
            angle_2a_root=str(self.angle_2a_root), output_root=str(self.angle_2b_root),
        )

        result = run_angle_2c_analysis(
            environment=self.environment, seed=seed, matchup_name="matchup_1",
            num_perturbations=NUM_PERTURBATIONS, perturbation_sigma=0.01, analysis_seed=42,
            onset_source_experiment="angle_1", onset_ledger_root=str(self.ledger_root),
            angle_2a_root=str(self.angle_2a_root), angle_2b_root=str(self.angle_2b_root),
            output_root=str(self.angle_2c_root),
        )

        # --- primary / secondary: finite, sane ---
        for label, props in (("primary", result.primary_properties), ("secondary", result.secondary_properties)):
            for key, value in props.items():
                self.assertTrue(np.isfinite(value), f"{label}.{key} is not finite: {value}")

        # --- null comparison: well-formed for all three properties ---
        for name in ("direction", "magnitude", "instability"):
            comparison = result.null_comparison[name]
            self.assertTrue(np.isfinite(comparison.null_mean))
            self.assertTrue(np.isfinite(comparison.threshold))
            self.assertEqual(comparison.null_n, 2)
            self.assertIsInstance(comparison.exceeds_null, bool)

        # --- non-result handling: exactly one of these two states holds ---
        if result.non_result:
            self.assertIsNone(result.dominant_property)
        else:
            self.assertIn(result.dominant_property, ("direction", "magnitude", "instability"))
            # --- onset-timing lookup: correctly finds the synthetic Angle 1 entry ---
            self.assertIsNotNone(result.onset_timing)
            self.assertEqual(result.onset_timing.angle1_degradation_onset_step, onset_step)
            self.assertEqual(result.onset_timing.t_star, onset_step)

        # --- persisted outputs exist and round-trip ---
        out_dir = angle_2c_analysis_dir(self.environment, seed, "matchup_1", root=str(self.angle_2c_root))
        self.assertTrue((out_dir / "run_metadata.json").exists())
        self.assertTrue((out_dir / "null_distribution.csv").exists())
        self.assertTrue((out_dir / "properties.npz").exists())

        import json

        with open(out_dir / "run_metadata.json") as f:
            metadata = json.load(f)
        self.assertEqual(metadata["matchup_name"], "matchup_1")
        self.assertAlmostEqual(
            metadata["primary"]["direction_raw"], result.primary_properties["direction_raw"], places=5,
        )

        with np.load(out_dir / "properties.npz") as npz:
            for key in ("primary_direction_raw", "primary_magnitude_raw", "primary_raw_offset", "primary_instability_ratio"):
                self.assertIn(key, npz.files)
                self.assertTrue(np.all(np.isfinite(npz[key])))
                self.assertGreater(npz[key].shape[0], 0)


class ReconstructionTestMathTest(unittest.TestCase):
    """Directly exercises the reconstruction test's math (see module
    docstring: a full pipeline built on tiny random synthetic agents cannot
    reliably be relied on to trigger real co-occurrence on its own, so this
    guarantees the code path is actually run, with a controlled, designed
    scenario where the answer is knowable in advance."""

    def test_reconstruction_correctly_identifies_the_dominant_property(self):
        observation_space, action_space = make_spaces()
        agent_cfg = make_agent_cfg(seed=7, critic_num_blocks=1, critic_hidden_dim=8)
        agent, _ = make_trained_agent(seed=7, critic_num_blocks=1, critic_hidden_dim=8)

        n = 6
        rng = np.random.default_rng(0)
        observations = jnp.asarray(rng.normal(size=(n, OBS_DIM)).astype(np.float32))
        key = jax.random.PRNGKey(0)

        # Designed scenario: R's nabla_a Q is D's, rotated ~90 degrees (large
        # direction change) but with matched magnitude. If direction is
        # truly what's being tested, "R-direction + D-magnitude" should
        # reproduce g_{D|R} almost exactly (since that IS g_{D|R}'s actual
        # nabla_a Q, up to floating point), while "D-direction + R-magnitude"
        # (~unchanged direction) should reproduce it poorly.
        grad_aq_d_at_d = rng.normal(size=(n, ACT_DIM)).astype(np.float32)
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)  # 90-degree rotation
        grad_aq_r_at_d = grad_aq_d_at_d @ rotation.T  # same magnitude, orthogonal direction
        grad_aq_d_at_d = jnp.asarray(grad_aq_d_at_d)
        grad_aq_r_at_d = jnp.asarray(grad_aq_r_at_d)

        # target_g_d_given_r = the REAL actor-parameter gradient using
        # nabla_a Q_R at D's point - constructed via the same synthesis
        # primitive (mathematically identical to what
        # experiments/angle_2b/gradients.py's real critic call would have
        # produced, since synthesize_actor_gradient's chain-rule derivation
        # is exact, not an approximation).
        from experiments.angle_2c.reconstruction import _flatten, synthesize_actor_gradient

        target_pytree = synthesize_actor_gradient(agent.agent.actor, agent.agent.temperature, observations, key, grad_aq_r_at_d)
        target_g_d_given_r = _flatten(target_pytree)

        result = run_reconstruction_test(
            actor=agent.agent.actor,
            temperature=agent.agent.temperature,
            observations=observations,
            key=key,
            grad_aq_d_at_d=grad_aq_d_at_d,
            grad_aq_r_at_d=grad_aq_r_at_d,
            target_g_d_given_r=target_g_d_given_r,
        )

        for label in ("d_direction_r_magnitude", "r_direction_d_magnitude"):
            for metric_name, value in result[label].items():
                self.assertTrue(np.isfinite(value), f"{label}.{metric_name} is not finite: {value}")

        # R-direction+D-magnitude uses the REAL nabla_a Q_R's direction,
        # exactly matching how target_g_d_given_r was constructed - it must
        # reproduce the target far more closely than the D-direction variant.
        r_dir_score = result["r_direction_d_magnitude"]["cosine_similarity_to_target"]
        d_dir_score = result["d_direction_r_magnitude"]["cosine_similarity_to_target"]
        self.assertGreater(r_dir_score, d_dir_score)
        self.assertGreater(r_dir_score, 0.99)


if __name__ == "__main__":
    unittest.main()
