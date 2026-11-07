"""Adversarial saved-evidence checks; no GPU qualification or alternative criteria."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.exp1 import compose_config
from experiments.exp12 import fork, probe, trigger
from scripts import positive_control, probe_fresh_checks
from scripts import exp12_reports


class Check1EvidenceTest(unittest.TestCase):
    def test_broadcastable_shape_corruption_is_rejected(self):
        pre = {"q": np.ones(4, np.float32), "dq_da": np.ones((4, 2), np.float32)}
        for field in pre:
            after = copy.deepcopy(pre)
            after[field] = after[field][..., None]
            with self.assertRaisesRegex(ValueError, "matching"):
                fork.check1(pre, after, pre, 64, True)

    def test_nonfinite_empty_and_non_fp32_panels_are_rejected(self):
        pre = {"q": np.ones(4, np.float32), "dq_da": np.ones((4, 2), np.float32)}
        for bad in (
            np.full(4, np.nan, np.float32),
            np.full(4, np.inf, np.float32),
            np.ones(4, np.float64),
            np.ones(4, np.int32),
            np.empty(0, np.float32),
        ):
            with self.subTest(dtype=bad.dtype, shape=bad.shape):
                after = {**pre, "q": bad}
                with self.assertRaises(ValueError):
                    fork.check1(pre, after, pre, 64, True)

    def test_valid_fp32_report_preserves_64_eps_and_exact_identity(self):
        pre = {"q": np.ones(4, np.float32), "dq_da": np.ones((4, 2), np.float32)}
        after = {**pre, "q": np.nextafter(pre["q"], np.float32(2))}
        self.assertTrue(fork.check1(pre, after, pre, 64, True)["pass"])
        self.assertFalse(fork.check1(pre, after, pre, 64, False)["pass"])


class NullEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.cfg = probe.ProbeConfig(5, 1000, 20, 25600, 256, 1e5, 2560)
        self.tc = trigger.TriggerConfig(10000, 0.95, 2, 0, 0.95)

    def row(self, loss=0.0):
        fresh = np.full(5, 0.2, np.float32)
        current = fresh - np.float32(loss)
        values = fresh - current
        low, high = trigger.bootstrap_interval(values, 990, 1, 10000, 0.95)
        return dict(
            pair=0,
            seed=990,
            check_index=1,
            score_fresh_rounds=fresh.tolist(),
            score_current_rounds=current.tolist(),
            loss_rounds=values.tolist(),
            b_rounds=[0.5] * 5,
            valid=True,
            loss_iqm=probe.iqm(values),
            ci_low=low,
            ci_high=high,
            resamples=10000,
            confidence=0.95,
            null_threshold=0.0,
            fired=low > 0,
        )

    def test_resume_recomputes_contradictory_firing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            row = self.row()
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(
                probe_fresh_checks._done_pairs(path, cfg=self.cfg, tc=self.tc), {0: row}
            )
            row["fired"] = True
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "fired"):
                probe_fresh_checks._done_pairs(path, cfg=self.cfg, tc=self.tc)

    def test_pairing_statistics_and_protocol_corruption_are_rejected(self):
        row = self.row(0.1)
        for field, bad in (
            ("loss_rounds", [0.2] * 5),
            ("ci_low", -1),
            ("loss_iqm", 0.9),
            ("valid", False),
            ("resamples", 9999),
            ("confidence", 0.9),
            ("null_threshold", 0.1),
            ("score_fresh_rounds", [0.2] * 4),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe_fresh_checks._validate_null_row(
                    {**row, field: bad}, self.cfg, self.tc
                )

    def test_nonfinite_original_evidence_stays_invalid_and_never_fires(self):
        row = self.row()
        row.update(
            score_current_rounds=[float("nan")] * 5,
            loss_rounds=[float("nan")] * 5,
            valid=False,
            loss_iqm=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
        )
        probe_fresh_checks._validate_null_row(row, self.cfg, self.tc)
        with self.assertRaises(ValueError):
            probe_fresh_checks._validate_null_row(
                {**row, "fired": True}, self.cfg, self.tc
            )


class CalibrationPopulationTest(unittest.TestCase):
    ARCHS = ("D2W512", "D4W1024", "D4W1536")

    def summaries(self):
        return [
            dict(
                arch=arch,
                null_pairs=100,
                per_check_fire_rate=0.05,
                fire_rate_exceeds_5pct=False,
                invalid_pairs=0,
            )
            for arch in self.ARCHS
        ]

    def test_all_sizes_100_pairs_and_5pct_boundary_are_preserved(self):
        rows = self.summaries()
        self.assertTrue(exp12_reports.null_criterion_rows(rows, self.ARCHS)["pass"])
        rows[1].update(per_check_fire_rate=0.13, fire_rate_exceeds_5pct=True)
        self.assertFalse(exp12_reports.null_criterion_rows(rows, self.ARCHS)["pass"])

    def test_missing_duplicate_small_and_contradictory_summaries_cannot_pass(self):
        rows = self.summaries()
        variants = [
            rows[:-1],
            rows + [rows[0]],
            [{**r, "null_pairs": 99} for r in rows],
            [{**r, "per_check_fire_rate": 0.13} for r in rows],
            [{**r, "per_check_fire_rate": float("nan")} for r in rows],
        ]
        for variant in variants:
            self.assertFalse(
                exp12_reports.null_criterion_rows(variant, self.ARCHS)["pass"]
            )

    def test_block_b_range_cannot_infer_its_population_from_present_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "gpu2" / "range_hopper_hop"
            root.mkdir(parents=True)
            row = dict(
                arch="D2W512",
                pool_size=25600,
                is_configured_pool=True,
                score_over_b=0.95,
                score_iqm=0.475,
                b_iqm=0.5,
                score_std=0.0,
                score_range=0.0,
                final_loss_rounds=[0.025] * 5,
                within_10_90_pct_of_b=False,
            )
            (root / "range_D2W512.json").write_text(json.dumps([row]))
            report = exp12_reports.blockB_report(directory)
            self.assertIn("FAIL           hopper-hop range, amendment (w)", report)
            self.assertIn("missing ['D4W1024', 'D4W1536']", report)

    def test_invalid_pair_acceptance_rule_is_not_silently_selected(self):
        rows = self.summaries()
        rows[0].pop("invalid_pairs")
        rows[1]["invalid_pairs"] = 1
        result = exp12_reports.null_criterion_rows(rows, self.ARCHS)
        self.assertTrue(result["pass"])
        self.assertEqual(result["invalid_pair_counts"][self.ARCHS[0]], None)
        self.assertEqual(result["invalid_pair_counts"][self.ARCHS[1]], 1)
        self.assertIn("no new", result["validity_note"])


class PositiveControlEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.cfg = compose_config(
            str(Path(__file__).resolve().parents[1] / "configs"),
            "base_exp12",
            [
                "env_name=dog-run",
                "env=dmc_hard",
                "critic_num_blocks=4",
                "critic_hidden_dim=1536",
                "run_role=dev",
                "seed=990",
            ],
        )

    def evidence(self, firing=(1, 2), end=2):
        records = [dict(check_index=0, interaction_step=5000)]
        schedule = probe.check_steps(int(self.cfg.num_interaction_steps), 20)
        for k in range(1, end + 1):
            current = np.full(5, 0.1, np.float32)
            fresh = np.full(5, 0.2 if k in firing else 0.1, np.float32)
            loss = fresh - current
            low, high = trigger.bootstrap_interval(loss, 990, k, 10000, 0.95)
            records.append(
                dict(
                    check_index=k,
                    interaction_step=schedule[k - 1],
                    valid=True,
                    triggered=low > 0,
                    ci_low=low,
                    ci_high=high,
                    loss_iqm=probe.iqm(loss),
                    **{f"score_current_r{r}": float(current[r]) for r in range(5)},
                    **{f"score_fresh_r{r}": float(fresh[r]) for r in range(5)},
                    **{f"loss_r{r}": float(loss[r]) for r in range(5)},
                )
            )
        natural = trigger.f_star(records, 20, 2, 0.95)
        plan = {
            **fork.fork_plan(self.cfg, schedule[min(end, 19) - 1], min(end, 19)),
            "fork_step": schedule[end - 1],
            "fork_check_index": end,
            "run_key": "fixture",
        }
        meta = dict(
            interaction_step=plan["fork_step"],
            extra_state=dict(probe_records=records, f_star=natural),
        )
        return plan, meta

    def test_full_settings_natural_first_trigger_is_cpu_verifiable(self):
        self.assertEqual(positive_control._setting_errors(self.cfg, False), [])
        positive_control._validate_natural_fork(self.cfg, *self.evidence())

    def test_false_fork_at_a_nonfiring_check_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "first eligible natural"):
            positive_control._validate_natural_fork(self.cfg, *self.evidence(firing=()))

    def test_bad_fork_is_rejected_before_agent_creation_or_loading(self):
        from experiments.exp12 import state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = fork.fork_dir(root) / "state"
            path = state.new_state_dir(checkpoint_root, 50000)
            plan, meta = self.evidence(firing=())
            state.save_meta(path, meta)
            state.commit_state_dir(checkpoint_root, path)
            (fork.fork_dir(root) / "fork.json").write_text(json.dumps(plan))
            with mock.patch("experiments.exp12.trainer.Exp12Trainer") as trainer:
                with self.assertRaisesRegex(ValueError, "first eligible natural"):
                    positive_control.load_fork(self.cfg, root, validate_natural=True)
                trainer.assert_not_called()

    def test_late_and_nonfirst_forks_are_rejected(self):
        for firing, end in (((1, 2, 3), 3), ((19, 20), 20), ((1, 3), 3)):
            with self.subTest(firing=firing), self.assertRaisesRegex(
                ValueError, "first eligible natural"
            ):
                positive_control._validate_natural_fork(
                    self.cfg, *self.evidence(firing, end)
                )

    def test_corrupt_records_state_and_horizons_are_rejected(self):
        plan, meta = self.evidence()
        mutations = [
            lambda p, m: m["extra_state"]["probe_records"].pop(1),
            lambda p, m: m["extra_state"]["probe_records"].append(
                m["extra_state"]["probe_records"][1]
            ),
            lambda p, m: m["extra_state"]["probe_records"][1].update(forced=True),
            lambda p, m: m["extra_state"]["probe_records"][1].update(ci_low=-1),
            lambda p, m: m["extra_state"]["probe_records"][1].update(
                interaction_step=1
            ),
            lambda p, m: m.update(interaction_step=1),
            lambda p, m: m["extra_state"].update(f_star=None),
            lambda p, m: p.update(horizon_steps=1),
        ]
        for mutate in mutations:
            p, m = copy.deepcopy(plan), copy.deepcopy(meta)
            mutate(p, m)
            with self.assertRaises(ValueError):
                positive_control._validate_natural_fork(self.cfg, p, m)

    def test_existing_evidence_is_never_overwritten(self):
        for filename in ("positive_control.json", "positive_control_curves.npz"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / filename
                path.write_bytes(b"historical evidence")
                args = SimpleNamespace(
                    run_dir="/unused", out_dir=directory, allow_any_setting=False
                )
                with mock.patch("utils.run_metadata.load_run_metadata") as load:
                    with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                        positive_control.run(args)
                    load.assert_not_called()
                self.assertEqual(path.read_bytes(), b"historical evidence")


if __name__ == "__main__":
    unittest.main()
