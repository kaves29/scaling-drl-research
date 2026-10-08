"""Regressions for methodology-preserving input and reporting corrections."""

import copy
import json
import os
import pickle
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import exp12_helpers
import exp12_reports
import positive_control
import probe_fresh_checks
from analysis import exp2_analysis
from experiments.exp12 import m_selection, state
import generate_manifest as gm
from experiments.exp12.probe import summarize
from experiments.exp12.twin import combine
from experiments.exp1 import compose_config


class TwinLossOrderTest(unittest.TestCase):
    def test_mean_of_paired_losses(self):
        fresh = np.tile(np.array([[0.49], [0.48]], np.float32), (1, 5))
        current = np.tile(np.array([[0.48], [0.47]], np.float32), (1, 5))
        result = {}
        for name, scores in (("fresh", fresh), ("current", current)):
            for q in range(2):
                result[f"{name}_q{q + 1}"] = {
                    "score": scores[q],
                    "final_loss": np.float32(0.5) - scores[q],
                    "b": np.full(5, 0.5, np.float32),
                }
        actual = summarize(combine(result, ["fresh", "current"]))["loss_rounds"]
        reference = np.array(
            [
                np.float32(
                    np.float32(fresh[0, r] - current[0, r])
                    + np.float32(fresh[1, r] - current[1, r])
                )
                / np.float32(2)
                for r in range(5)
            ],
            np.float32,
        )
        np.testing.assert_array_equal(actual, reference)


class PositiveControlInputsTest(unittest.TestCase):
    def loss(self):
        return {
            "degraded": np.full(5, 0.8),
            "injected_last": np.full(5, 0.7),
            "injected_half": np.full(5, 0.1),
            "injected_all": np.full(5, 0.05),
        }

    def test_nonfinite_candidate_stops_without_selecting(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                loss = self.loss()
                loss["injected_half"][0] = value
                result = m_selection.evaluate(loss, 0.1, 0.1)
                self.assertIsNone(result["chosen_m"])
                self.assertIn("finite", result["stop"])

    def test_missing_extra_or_wrong_round_count_cannot_select(self):
        missing = self.loss()
        missing.pop("injected_half")
        extra = {**self.loss(), "shared_offset": np.ones(5)}
        short = self.loss()
        short["injected_half"] = np.ones(4)
        for loss in (missing, extra, short):
            with self.subTest(keys=list(loss)):
                result = m_selection.evaluate(loss, 0.1, 0.1)
                self.assertIsNone(result["chosen_m"])
                self.assertTrue(result["stop"])

    def test_valid_result_matches_independent_pooled_variance(self):
        loss = self.loss()
        loss["injected_all"][-1] = 0.06
        result = m_selection.evaluate(loss, 0.1, 0.1)
        variance = (
            sum(
                sum((float(x) - sum(map(float, values)) / 5) ** 2 for x in values)
                for values in loss.values()
            )
            / 16
        )
        self.assertEqual(result["chosen_m"], "half")
        self.assertAlmostEqual(result["noise_sd"], variance**0.5)
        self.assertAlmostEqual(result["noise"], variance**0.5 / 0.8)

    def test_approved_pc_settings_are_accepted_but_probe_or_trigger_edits_are_not(self):
        cfg = compose_config(
            exp12_helpers.CONFIG_PATH,
            "base_exp12",
            [
                "env_name=dog-run",
                "env=dmc_hard",
                "run_role=dev",
                "seed=101",
                "critic_num_blocks=4",
                "critic_hidden_dim=1536",
            ],
        )
        self.assertEqual(positive_control._setting_errors(cfg, False), [])
        for key in (
            "probe.steps",
            "trigger.resamples",
            "positive_control.noise_threshold",
        ):
            edited = copy.deepcopy(cfg)
            parent, field = key.split(".")
            edited[parent][field] = cfg[parent][field] * 2
            self.assertTrue(positive_control._setting_errors(edited, False), key)

    def test_forced_run_cannot_qualify(self):
        cfg = compose_config(
            exp12_helpers.CONFIG_PATH,
            "base_exp12",
            [
                "env_name=dog-run",
                "env=dmc_hard",
                "run_role=dev",
                "seed=101",
                "critic_num_blocks=4",
                "critic_hidden_dim=1536",
                "testing.force_trigger_check=2",
            ],
        )
        self.assertTrue(
            any("forced" in e for e in positive_control._setting_errors(cfg, False))
        )

    def test_noncanonical_budget_cannot_qualify(self):
        cfg = compose_config(
            exp12_helpers.CONFIG_PATH,
            "base_exp12",
            [
                "env_name=dog-run",
                "env=dmc_hard",
                "run_role=dev",
                "seed=101",
                "critic_num_blocks=4",
                "critic_hidden_dim=1536",
                "num_env_steps=240000",
            ],
        )
        self.assertTrue(positive_control._setting_errors(cfg, False))


class CudaComparisonInputsTest(unittest.TestCase):
    def test_structure_and_dtype_mismatch_raise(self):
        cases = [
            ([np.ones(1), np.ones(1)], [np.ones(1)]),
            ([np.ones((1, 2))], [np.ones((2, 1))]),
            ([np.ones(1, np.float32)], [np.ones(1, np.float64)]),
        ]
        for actual, reference in cases:
            with self.subTest(shapes=[a.shape for a in actual]):
                with self.assertRaises(ValueError):
                    exp12_helpers.max_relative_deviation(actual, reference)

    def test_nonfinite_model_leaves_raise(self):
        for actual, reference in (
            ([2, np.nan], [1, np.nan]),
            ([np.nan], [1]),
            ([np.inf], [np.inf]),
        ):
            with self.subTest(actual=actual):
                with self.assertRaises(ValueError):
                    exp12_helpers.max_relative_deviation(
                        [np.array(actual, float)], [np.array(reference, float)]
                    )

    def test_integer_state_difference_is_exact(self):
        actual = np.array([2**31, 4], np.uint32)
        reference = np.array([2**31, 3], np.uint32)
        self.assertEqual(
            exp12_helpers.max_relative_deviation([actual], [reference]), float("inf")
        )

    def test_explicit_diagnostic_nan_marker_checks_masks_and_finite_values(self):
        actual, reference = np.array([2.0, np.nan]), np.array([1.0, np.nan])
        self.assertEqual(
            exp12_helpers.max_relative_deviation(
                [actual], [reference], allow_nan_indices=(0,)
            ),
            1.0,
        )
        with self.assertRaises(ValueError):
            exp12_helpers.max_relative_deviation(
                [actual], [np.array([1.0, 0.0])], allow_nan_indices=(0,)
            )

    def test_valid_float_measurement_unchanged_and_thresholds_unset(self):
        self.assertEqual(
            exp12_helpers.max_relative_deviation(
                [np.array([3.0, 1.0])], [np.array([2.0, 1.0])]
            ),
            0.5,
        )
        self.assertEqual(
            exp12_helpers.max_relative_deviation([np.array([0.5])], [np.zeros(1)]), 0.5
        )
        self.assertTrue(all(t is None for t in exp12_helpers.GPU_TOLERANCES.values()))


class RangeCriterionTest(unittest.TestCase):
    def rows(self, ratio):
        return [
            {
                "arch": "D4W1536",
                "is_configured_pool": True,
                "score_over_b": ratio,
                "within_10_90_pct_of_b": 0.1 <= ratio <= 0.9,
            }
        ]

    def cli(self, root, ratio):
        rows = self.rows(ratio)

        def produce(*args):
            (Path(root) / "range_D4W1536.json").write_text(json.dumps(rows))
            return rows

        args = [
            "probe_fresh_checks",
            "--mode",
            "range",
            "--archs",
            "D4W1536",
            "--out_dir",
            root,
        ]
        with mock.patch.object(sys, "argv", args), mock.patch.object(
            probe_fresh_checks, "_stub_wandb"
        ), mock.patch.object(probe_fresh_checks, "range_mode", side_effect=produce):
            return probe_fresh_checks.main()

    def test_cli_uses_new_rule_and_exit_status(self):
        for ratio, passed in (
            (0.602, False),
            (0.9, True),
            (0.987, True),
            (float("inf"), False),
            (float("nan"), False),
        ):
            with self.subTest(ratio=ratio), tempfile.TemporaryDirectory() as root:
                code = self.cli(root, ratio)
                verdict = json.loads((Path(root) / "range_verdict.json").read_text())
                self.assertEqual(verdict["PASS"], passed)
                self.assertEqual(code == 0, passed)
                self.assertEqual(
                    verdict["PASS"],
                    exp12_reports.range_criterion(root, ["D4W1536"])["pass"],
                )

    def test_duplicate_missing_or_nonfinite_range_cannot_pass(self):
        for rows in ([], self.rows(0.99) * 2, self.rows(float("inf"))):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as root:
                (Path(root) / "range_D4W1536.json").write_text(json.dumps(rows))
                self.assertFalse(
                    exp12_reports.range_criterion(root, ["D4W1536"])["pass"]
                )


class ForkMeasurementLabelsTest(unittest.TestCase):
    def test_inherited_fork_record_is_labeled_without_changing_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            key = "exp1_D4W1024_dog-run_seed1"
            directory = Path(root) / "exp12" / "exp2" / key
            directory.mkdir(parents=True)
            (directory / "fork.json").write_text(
                json.dumps(
                    {
                        "fork_step": 10,
                        "fork_check_index": 2,
                        "horizon_steps": 25,
                        "eval_every_steps": 1,
                    }
                )
            )
            rows = pd.DataFrame(
                [
                    {"check_index": 2, "steps_since_fork": 0, "loss_iqm": 0.8},
                    {"check_index": 3, "steps_since_fork": 5, "loss_iqm": 0.2},
                ]
            )
            for arm in ("control", "injected"):
                path = directory / f"arm_{arm}"
                path.mkdir()
                rows.to_csv(path / "checks.csv", index=False)
            source = (directory / "arm_injected/checks.csv").read_bytes()
            runs = pd.DataFrame(
                [
                    dict(
                        run_key=key,
                        run_role="confirmatory",
                        architecture="D4W1024",
                        environment="dog-run",
                        seed=1,
                    )
                ]
            )
            with mock.patch.object(
                exp2_analysis.ledger, "load", return_value=(runs, pd.DataFrame())
            ):
                checks = exp2_analysis.load(root)["checks"]
            injected = checks[checks.arm == "injected"]
            self.assertEqual(
                list(injected.measurement_phase), ["pre_injection", "post_injection"]
            )
            self.assertEqual(list(injected.loss_iqm), [0.8, 0.2])
            self.assertEqual(
                (directory / "arm_injected/checks.csv").read_bytes(), source
            )


class ManifestPersistenceTest(unittest.TestCase):
    def test_manifest_generation_refuses_stale_files_without_overwriting(self):
        with tempfile.TemporaryDirectory() as root:
            old = Path(root) / "exp2_arms_OLD_GPU.txt"
            old.write_text("old approved commands\n")
            cwd = os.getcwd()
            try:
                os.chdir(root)
                with self.assertRaises(ValueError):
                    gm.main(
                        [
                            "--grid",
                            "exp12",
                            "--ckpt-root",
                            root + "/ckpt",
                            "--results-root",
                            root + "/results",
                        ]
                    )
                self.assertFalse((Path(root) / "exp12_exp1_jobs.txt").exists())
                self.assertEqual(old.read_text(), "old approved commands\n")
            finally:
                os.chdir(cwd)


class PersistenceInputsTest(unittest.TestCase):
    def buffer(self):
        return SimpleNamespace(
            **{
                name: np.arange(6, dtype=np.float32).reshape(3, 2)
                for name in state.BUFFER_ARRAYS
            },
            _num_in_buffer=2,
            _current_idx=2,
            _n_step_transitions=deque(),
        )

    def test_latest_rejects_path_escape_and_dangling_pointer(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("../outside", "missing", ".", ""):
                with self.subTest(name=name):
                    (Path(root) / "LATEST").write_text(name)
                    with self.assertRaises(ValueError):
                        state.latest_state_dir(root)

        # Offline certification must also reject a parent-directory pointer.
        from analysis import exp12_validation

        with tempfile.TemporaryDirectory() as root:
            parent = Path(root)
            state_root = parent / "state"
            state_root.mkdir()
            (state_root / "LATEST").write_text("..")
            for name in ("meta.pkl", "buffer.npz", "buffer_meta.pkl", "obs_rms.pkl"):
                (parent / name).write_bytes(b"not checkpoint evidence")
            (parent / "agent_ckpt").mkdir()
            (parent / "agent_ckpt/dummy").write_bytes(b"not checkpoint evidence")
            with self.assertRaises(ValueError):
                exp12_validation._state(exp12_validation.Snapshot(), state_root)

    def test_replay_rejects_shape_or_dtype_before_mutation(self):
        for kind in ("shape", "dtype"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                path = Path(root)
                buffer = self.buffer()
                state.save_buffer(buffer, path)
                with np.load(path / "buffer.npz") as data:
                    arrays = {k: data[k] for k in data.files}
                arrays["actions"] = (
                    arrays["actions"][:1]
                    if kind == "shape"
                    else arrays["actions"].astype(np.float64)
                )
                np.savez(path / "buffer.npz", **arrays)
                target = self.buffer()
                target._observations[:] = -1
                before = copy.deepcopy(target)
                with self.assertRaises(ValueError):
                    state.load_buffer(target, path)
                np.testing.assert_array_equal(
                    target._observations, before._observations
                )

    def test_valid_replay_restore_and_rng_are_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            path, buffer, restored = Path(root), self.buffer(), self.buffer()
            state.save_buffer(buffer, path)
            restored._observations[:] = -1
            rng = pickle.dumps(np.random.get_state())
            state.load_buffer(restored, path)
            np.testing.assert_array_equal(
                restored._observations[:2], buffer._observations[:2]
            )
            self.assertEqual(pickle.dumps(np.random.get_state()), rng)

    def test_null_resume_rejects_wrong_identity_before_reusing_pair(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(env="dog-run", seed=990, null_pairs=1)
            row = {
                "pair": 0,
                "arch": "D4W1024",
                "env": "hopper-hop",
                "seed": 990,
                "loss_iqm": 0.01,
                "fired": False,
            }
            (Path(root) / "null_pairs_D4W1024.jsonl").write_text(json.dumps(row) + "\n")
            cfg = exp12_helpers.compose(exp12_helpers.tiny_overrides())
            trainer = SimpleNamespace(
                cfg=cfg,
                _sac_agent=SimpleNamespace(critic=SimpleNamespace(network_def=None)),
                buffer=SimpleNamespace(
                    _observations=np.zeros((10, 3)), _actions=np.zeros((10, 2))
                ),
                close=mock.Mock(),
            )
            with mock.patch.object(
                probe_fresh_checks, "_fresh_trainer", return_value=trainer
            ):
                with self.assertRaises(ValueError):
                    probe_fresh_checks.null_mode(args, "D4W1024", 4, 1024, Path(root))

    def test_null_resume_requires_matching_provenance_and_check_index(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "pairs.jsonl"
            provenance = {
                "arch": "D4W1024",
                "env": "dog-run",
                "seed": 990,
                "code": {"commit": "fixture", "dirty": False},
            }
            row = {
                **{k: provenance[k] for k in ("arch", "env", "seed")},
                "pair": 0,
                "check_index": 1,
                "provenance": provenance,
            }
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(probe_fresh_checks._done_pairs(path, provenance), {0: row})
            for bad in (
                {**row, "env": "hopper-hop"},
                {**row, "check_index": 2},
                {**row, "provenance": {}},
            ):
                path.write_text(json.dumps(bad) + "\n")
                with self.assertRaises(ValueError):
                    probe_fresh_checks._done_pairs(path, provenance)

    def test_null_resume_rejects_duplicate_pairs(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "pairs.jsonl"
            path.write_text('{"pair":0}\n{"pair":0}\n')
            with self.assertRaises(ValueError):
                probe_fresh_checks._done_pairs(path)


if __name__ == "__main__":
    unittest.main()
