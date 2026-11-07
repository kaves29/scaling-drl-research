"""Covers Angle 2A's Monte Carlo rollout-count (R) calibration
(experiments/angle_2a/r_calibration.py).

Live-tested: the full real pipeline (real dm_control training, real
SACAgent critics/rollouts, a real dedicated null pair, real onset-independent
sigma measurement) via calibrate_r/load_or_calibrate_r against real
environments. The pooling-vs-not-pooled distinction and the R-solving
formula's genuine sensitivity to its inputs are tested directly at the unit
level with synthetic data instead, since that's the only way to construct a
scenario where the two approaches (pooled vs within-pair) provably disagree
- real training's actual between-state/within-state variance split isn't
something a short test can dictate exactly.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.config import RoleArchitecture
from experiments.angle_2a.errors import Angle2AConfigError
from experiments.angle_2a.r_calibration import (
    calibrate_r,
    load_or_calibrate_r,
    load_r_calibration,
    save_r_calibration,
    sigma_rollout_from_per_pair_returns,
    solve_calibrated_r,
)


def _make_cfg(env_name: str, seed: int = 108, num_interaction_steps: int = 40, temp_initial_value: float = 0.01):
    return OmegaConf.create(
        {
            "seed": seed,
            "gamma": 0.99,
            "num_interaction_steps": num_interaction_steps,
            "env": {
                "env_type": "dmc", "env_name": env_name, "seed": seed,
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
                "temp_target_entropy": None, "temp_target_entropy_coef": -0.5,
                "temp_initial_value": temp_initial_value, "temp_learning_rate": 1e-4, "temp_weight_decay": 0.0,
                "target_tau": 0.005, "gamma": 0.99, "n_step": 1, "mixed_precision": False,
                "actor_sparsity": 0.0, "critic_sparsity": 0.0,
            },
        }
    )


ARCHITECTURE = RoleArchitecture(role="reference", critic_num_blocks=1, critic_hidden_dim=8)
ARCHITECTURE_LABEL = "D1W8"


class TestSigmaRolloutNotPooled(unittest.TestCase):
    def test_within_pair_std_used_not_pooled_across_pairs(self):
        # Pair A: tightly clustered around 10 (small within-pair spread).
        # Pair B: tightly clustered around 100 (small within-pair spread).
        # If pooled together first, the combined array's std is dominated by
        # the ~90 gap BETWEEN the pairs' means, not genuine rollout noise -
        # this test would pass with a much larger "sigma" if pooling were
        # used instead of the correct within-pair-then-combine approach.
        pair_a = [9.9, 10.0, 10.1, 9.95, 10.05]
        pair_b = [99.9, 100.0, 100.1, 99.95, 100.05]

        sigma, per_pair_sigmas = sigma_rollout_from_per_pair_returns([pair_a, pair_b])

        pooled_sigma_if_bug = float(np.std(pair_a + pair_b, ddof=1))
        self.assertGreater(pooled_sigma_if_bug, 40.0, "test setup sanity check: pooling should look huge here")

        self.assertLess(sigma, 1.0, "within-pair sigma should be small, not dominated by the between-pair gap")
        self.assertAlmostEqual(per_pair_sigmas[0], float(np.std(pair_a, ddof=1)))
        self.assertAlmostEqual(per_pair_sigmas[1], float(np.std(pair_b, ddof=1)))
        # conservative combination: max, not average, of the per-pair sigmas
        self.assertEqual(sigma, max(per_pair_sigmas))

    def test_single_pair_still_works(self):
        sigma, per_pair_sigmas = sigma_rollout_from_per_pair_returns([[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.assertEqual(len(per_pair_sigmas), 1)
        self.assertAlmostEqual(sigma, float(np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)))


class TestSolveCalibratedR(unittest.TestCase):
    def test_larger_sigma_requires_larger_r(self):
        r_small_sigma, _, _ = solve_calibrated_r(sigma_rollout=1.0, threshold_95=10.0, se_margin_divisor=2.0, r_cap=1000)
        r_large_sigma, _, _ = solve_calibrated_r(sigma_rollout=10.0, threshold_95=10.0, se_margin_divisor=2.0, r_cap=1000)
        self.assertGreater(r_large_sigma, r_small_sigma, "R must respond to sigma, not be a hardcoded constant")

    def test_smaller_threshold_requires_larger_r(self):
        r_loose, _, _ = solve_calibrated_r(sigma_rollout=1.0, threshold_95=10.0, se_margin_divisor=2.0, r_cap=1000)
        r_tight, _, _ = solve_calibrated_r(sigma_rollout=1.0, threshold_95=1.0, se_margin_divisor=2.0, r_cap=1000)
        self.assertGreater(r_tight, r_loose, "R must respond to threshold_95, not be a hardcoded constant")

    def test_matches_closed_form(self):
        # sigma=4, threshold_95=8, margin=2 -> target_se=4 -> ideal_r = (4/4)^2 = 1
        r, ideal_r, underpowered = solve_calibrated_r(sigma_rollout=4.0, threshold_95=8.0, se_margin_divisor=2.0, r_cap=1000)
        self.assertEqual(ideal_r, 1)
        self.assertEqual(r, 1)
        self.assertFalse(underpowered)

        # sigma=12, threshold_95=8, margin=2 -> target_se=4 -> ideal_r = (12/4)^2 = 9
        r, ideal_r, underpowered = solve_calibrated_r(sigma_rollout=12.0, threshold_95=8.0, se_margin_divisor=2.0, r_cap=1000)
        self.assertEqual(ideal_r, 9)
        self.assertEqual(r, 9)
        self.assertFalse(underpowered)

    def test_cap_binds_and_flags_underpowered(self):
        # Deliberately huge sigma relative to target -> ideal_r far exceeds a small cap.
        r, ideal_r, underpowered = solve_calibrated_r(sigma_rollout=1000.0, threshold_95=1.0, se_margin_divisor=2.0, r_cap=35)
        self.assertEqual(r, 35)
        self.assertGreater(ideal_r, 35)
        self.assertTrue(underpowered)

    def test_cap_does_not_bind_when_unnecessary(self):
        r, ideal_r, underpowered = solve_calibrated_r(sigma_rollout=0.5, threshold_95=10.0, se_margin_divisor=2.0, r_cap=35)
        self.assertEqual(r, ideal_r)
        self.assertFalse(underpowered)

    def test_nonpositive_threshold_raises(self):
        with self.assertRaises(Angle2AConfigError):
            solve_calibrated_r(sigma_rollout=1.0, threshold_95=0.0, se_margin_divisor=2.0, r_cap=35)


class TestCalibrateRRealEnvironment(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_real_calibration_produces_sensible_structure(self):
        cfg = _make_cfg("cheetah-run")
        result = calibrate_r(
            ARCHITECTURE, ARCHITECTURE_LABEL, cfg, "cheetah-run", calibration_seed=555,
            burn_in_fraction=0.25, num_representative_pairs=3, num_rollouts_for_sigma=10,
        )
        self.assertEqual(result.environment, "cheetah-run")
        self.assertGreaterEqual(result.calibrated_r, 1)
        self.assertGreaterEqual(result.sigma_rollout, 0.0)
        self.assertTrue(np.isfinite(result.threshold_95))
        self.assertEqual(len(result.per_pair_sigmas), 3)
        self.assertEqual(result.stop_step, 10)  # floor(0.25 * 40)

    def test_load_or_calibrate_caches_and_is_keyed_per_environment(self):
        cfg_a = _make_cfg("cheetah-run")
        cfg_b = _make_cfg("walker-walk")
        root = os.path.join(self.tmpdir, "r_calibration")

        result_a = load_or_calibrate_r(
            ARCHITECTURE, ARCHITECTURE_LABEL, cfg_a, "cheetah-run", calibration_seed=555,
            burn_in_fraction=0.25, root=root, num_representative_pairs=3, num_rollouts_for_sigma=10,
        )
        result_b = load_or_calibrate_r(
            ARCHITECTURE, ARCHITECTURE_LABEL, cfg_b, "walker-walk", calibration_seed=555,
            burn_in_fraction=0.25, root=root, num_representative_pairs=3, num_rollouts_for_sigma=10,
        )

        # Each environment gets its own cache entry - a bug that shared one
        # global cache path (ignoring `environment`) would make these
        # identical regardless of the actual underlying computation.
        self.assertIsNotNone(load_r_calibration("cheetah-run", root=root))
        self.assertIsNotNone(load_r_calibration("walker-walk", root=root))
        self.assertEqual(load_r_calibration("cheetah-run", root=root).environment, "cheetah-run")
        self.assertEqual(load_r_calibration("walker-walk", root=root).environment, "walker-walk")

        # Genuinely computed independently, not a shared hardcoded value:
        # not forcing a specific direction (real short-training noise could
        # go either way), just that at least one measured quantity differs -
        # both being exactly equal across two different real environments,
        # seeds unchanged otherwise, would be a near-impossible coincidence
        # if these were actually computed independently.
        different = (
            result_a.sigma_rollout != result_b.sigma_rollout
            or result_a.threshold_95 != result_b.threshold_95
            or result_a.calibrated_r != result_b.calibrated_r
        )
        self.assertTrue(different, "cheetah-run and walker-walk calibrated to identical values - suspicious")

        # Cache hit: recomputing for the same environment returns the cached
        # object without re-running training (verified by checking it's
        # exactly the same values, and that a second call is fast/no-op-safe
        # by construction of load_or_calibrate_r's cache-hit branch).
        cached_a = load_or_calibrate_r(
            ARCHITECTURE, ARCHITECTURE_LABEL, cfg_a, "cheetah-run", calibration_seed=555,
            burn_in_fraction=0.25, root=root,
        )
        self.assertEqual(cached_a.sigma_rollout, result_a.sigma_rollout)
        self.assertEqual(cached_a.calibrated_r, result_a.calibrated_r)

    def test_save_and_load_round_trip(self):
        cfg = _make_cfg("cheetah-run")
        root = os.path.join(self.tmpdir, "r_calibration")
        result = calibrate_r(
            ARCHITECTURE, ARCHITECTURE_LABEL, cfg, "cheetah-run", calibration_seed=555,
            burn_in_fraction=0.25, num_representative_pairs=3, num_rollouts_for_sigma=10,
        )
        save_r_calibration(result, root=root)
        reloaded = load_r_calibration("cheetah-run", root=root)
        self.assertEqual(reloaded, result)


if __name__ == "__main__":
    unittest.main()
