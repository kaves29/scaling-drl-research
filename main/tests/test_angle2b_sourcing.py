"""Verifies Angle 2B's own-buffer-only sourcing requirement: each analysis's
state batch must come exclusively from its held-fixed actor's own
probe-capture buffer, never the other agent's - see
experiments/angle_2b/sampling.py's module docstring for why (avoiding an
out-of-distribution confound).

Two independent checks:
  1. sample_state_batch() itself only ever draws from the single array it's
     given - proven directly, not inferred.
  2. End-to-end through run_angle_2b_analysis(), with D's and R's persisted
     probe-capture buffers populated from disjoint, mutually-exclusive
     numeric ranges, so it's possible to tell by inspection alone which
     buffer any given sampled state actually came from. Confirms the
     primary batch (pi_D held fixed) is 100% D-range and the secondary
     batch (pi_R held fixed) is 100% R-range - and that a null pair's batch
     is 100% A-range, never B-range.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import ProbeCapture
from experiments.angle_2a.storage import save_frozen_agent_snapshot
from experiments.angle_2b import matchup_2b, null_baseline, sampling
from scale_rl.agents import create_agent

OBS_DIM = 4
ACT_DIM = 2

# Disjoint ranges so a sampled state's provenance is identifiable by value alone.
D_RANGE = (1000.0, 1001.0)
R_RANGE = (-1001.0, -1000.0)
A_RANGE = (2000.0, 2001.0)
B_RANGE = (-3001.0, -3000.0)


def _agent_cfg(seed, critic_num_blocks, critic_hidden_dim):
    return {
        "agent_type": "sac",
        "seed": seed,
        "num_train_envs": 1,
        "max_episode_steps": 100,
        # True (matching test_angle_2b_smoke.py's proven-working config):
        # normalize_observation=False was tried first here and reliably hit
        # a pre-existing orbax/jax-metal checkpoint-restore quirk on this
        # machine, unrelated to sourcing - see End-of-Task Summary. Since
        # ranges below are separated by 1000+ units, a freshly-initialized
        # obs_rms (~identity transform) doesn't threaten range separation.
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
        "critic_use_cdq": False,  # see test_angle_2b_smoke.py for why
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


def _make_spaces():
    import gymnasium as gym

    return (
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32),
    )


def _make_probe_capture(value_range, seed, n=60):
    """All states drawn uniformly from `value_range`, actions irrelevant."""
    lo, hi = value_range
    rng = np.random.default_rng(seed)
    pc = ProbeCapture(capacity=n, observation_shape=(OBS_DIM,), action_shape=(ACT_DIM,))
    for i in range(n):
        pc.add(
            i,
            rng.uniform(lo, hi, size=(OBS_DIM,)).astype(np.float32),
            rng.uniform(-1, 1, size=(ACT_DIM,)).astype(np.float32),
            env_state=None,
        )
    return pc


def _random_update_batch(rng: np.random.Generator, batch_size: int = 16) -> dict:
    return {
        "observation": rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(batch_size, ACT_DIM)).astype(np.float32),
        "reward": rng.normal(size=(batch_size,)).astype(np.float32),
        "terminated": np.zeros((batch_size,), dtype=np.float32),
        "next_observation": rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
    }


def _in_range(states: np.ndarray, value_range) -> bool:
    lo, hi = value_range
    return bool(np.all(states >= lo) and np.all(states <= hi))


class SampleStateBatchUnitTest(unittest.TestCase):
    """Direct check on the primitive: only ever draws from the one array given."""

    def test_returned_states_are_a_subset_of_the_single_source_array(self):
        rng = np.random.default_rng(0)
        source = rng.uniform(500.0, 501.0, size=(50, OBS_DIM)).astype(np.float32)

        batch = sampling.sample_state_batch(source, seed=1, context="unit-test", num_states_per_source=10)

        self.assertEqual(batch.shape, (10, OBS_DIM))
        for row in batch:
            self.assertTrue(
                np.any(np.all(np.isclose(source, row), axis=1)),
                "sampled row is not present in the source array",
            )

    def test_signature_takes_exactly_one_source_array(self):
        # Regression guard: the old two-source (states_a, states_b) signature
        # must not silently come back - see sampling.py's module docstring.
        import inspect

        params = list(inspect.signature(sampling.sample_state_batch).parameters)
        self.assertEqual(params[0], "states")
        self.assertNotIn("states_a", params)
        self.assertNotIn("states_b", params)


class EndToEndSourcingTest(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path(tempfile.mkdtemp(prefix="angle2b_sourcing_"))
        self.angle_2a_root = self.tmp_root / "angle_2a"
        self.environment = "sourcing_test_env"
        self.addCleanup(shutil.rmtree, self.tmp_root, ignore_errors=True)

    def _persist(self, seed, matchup_name, role, value_range, agent_seed, num_blocks=1, hidden_dim=8):
        observation_space, action_space = _make_spaces()
        cfg = _agent_cfg(agent_seed, num_blocks, hidden_dim)
        agent = create_agent(observation_space, action_space, OmegaConf.create(cfg))
        # A few real .update() calls before saving (mirroring
        # test_angle_2b_smoke.py's make_trained_agent) - an
        # immediately-saved, freshly-initialized agent's params were
        # observed to reliably trigger a pre-existing orbax/jax-metal
        # checkpoint-restore quirk on this machine, unrelated to sourcing
        # (see End-of-Task Summary); at least one jitted update forces real
        # device materialization first.
        rng = np.random.default_rng(agent_seed)
        for step in range(3):
            agent.update(step, _random_update_batch(rng))
        probe_capture = _make_probe_capture(value_range, seed=agent_seed)
        save_frozen_agent_snapshot(
            self.environment, seed, matchup_name, role, agent, probe_capture,
            agent_cfg=cfg, root=str(self.angle_2a_root),
        )

    def test_primary_and_secondary_each_use_only_their_own_actors_buffer(self):
        seed = 7
        # Deliberately varied architectures per role (mirroring
        # test_angle_2b_smoke.py's pattern) rather than four identical
        # shapes - four checkpoints of identical pytree structure saved/
        # loaded in one process was observed to trigger a pre-existing
        # orbax/jax-metal restore quirk on this machine unrelated to this
        # test's actual subject (state sourcing) - see End-of-Task Summary.
        self._persist(seed, "matchup_1", "D", D_RANGE, agent_seed=11, num_blocks=3, hidden_dim=16)
        self._persist(seed, "matchup_1", "R", R_RANGE, agent_seed=12, num_blocks=1, hidden_dim=8)
        # One null pair so run_angle_2b_analysis doesn't need to fail on a
        # missing null distribution - not the focus of this test, but must
        # exist for the call to succeed.
        self._persist(1, "null_matchup_1", "D", A_RANGE, agent_seed=21, num_blocks=2, hidden_dim=8)
        self._persist(1, "null_matchup_1", "R", B_RANGE, agent_seed=22, num_blocks=2, hidden_dim=8)

        captured = []
        real_sample = sampling.sample_state_batch

        def spy(states, seed, context, num_states_per_source):
            batch = real_sample(states, seed, context, num_states_per_source)
            captured.append((context, batch.copy()))
            return batch

        with mock.patch.object(matchup_2b, "sample_state_batch", side_effect=spy), \
             mock.patch.object(null_baseline, "sample_state_batch", side_effect=spy):
            result = matchup_2b.run_angle_2b_analysis(
                environment=self.environment,
                seed=seed,
                matchup_name="matchup_1",
                null_seeds=[1],
                analysis_seed=42,
                num_states_per_source=10,
                angle_2a_root=str(self.angle_2a_root),
                output_root=str(self.tmp_root / "angle_2b"),
            )

        by_context = {ctx: batch for ctx, batch in captured}
        primary_batch = next(b for ctx, b in captured if ctx.startswith("primary:"))
        secondary_batch = next(b for ctx, b in captured if ctx.startswith("secondary:"))
        null_batch = next(b for ctx, b in captured if ctx.startswith("null:"))

        self.assertTrue(_in_range(primary_batch, D_RANGE), "primary batch contains non-D-range states")
        self.assertFalse(_in_range(primary_batch, R_RANGE), "primary batch leaked R-range states")

        self.assertTrue(_in_range(secondary_batch, R_RANGE), "secondary batch contains non-R-range states")
        self.assertFalse(_in_range(secondary_batch, D_RANGE), "secondary batch leaked D-range states")

        self.assertTrue(_in_range(null_batch, A_RANGE), "null batch contains non-A-range states")
        self.assertFalse(_in_range(null_batch, B_RANGE), "null batch leaked B-range states")

        # Sanity: the analysis itself still produced finite results.
        for metrics in (result.primary, result.secondary):
            for v in metrics.values():
                self.assertTrue(np.isfinite(v))


if __name__ == "__main__":
    unittest.main()
