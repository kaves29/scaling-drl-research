"""Confirmatory certificates require independent population and source evidence."""

import copy
import io
import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from analysis import exp12_validation as v
from experiments.exp12 import exp2_ledger, ledger
from tests import exp12_complete_fixture as f


class CompletenessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.study = f.study(cls.base)
        f.populate(cls.base, cls.study)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def item(self, a="D4W1024", e="dog-run", s=1):
        directory = ledger.ledger_root(str(self.base / "results")) / v.run_key(a, e, s)
        return v._parent(v.Snapshot(), directory, a, e, s, self.study)[2]

    def mutate_csv(self, path, change, validate):
        original = path.read_bytes()
        try:
            frame = pd.read_csv(path)
            change(frame).to_csv(path, index=False)
            with self.assertRaises((ValueError, AssertionError, KeyError)):
                validate()
        finally:
            path.write_bytes(original)

    def test_full_population_and_b5_invalids_pass(self):
        _, cert = v.validate(
            str(self.base / "results"), self.base / "study.json", "exp1"
        )
        self.assertEqual(len(cert["census"]), 195)
        self.assertTrue(cert["pass"])
        self.assertEqual(sum(bool(x["invalid_checks"]) for x in cert["census"]), 15)
        _, cert2 = v.validate(
            str(self.base / "results"), self.base / "study.json", "exp2"
        )
        self.assertEqual(len(cert2["census"]), 130)
        self.assertEqual(
            sum(x["status"] == "eligible_trigger" for x in cert2["census"]), 2
        )
        self.assertEqual(
            sum(x["status"] == "valid_no_trigger" for x in cert2["census"]), 128
        )

    def test_independent_population_cannot_shrink_to_observed(self):
        path = self.base / "study.json"
        original = path.read_bytes()
        try:
            study = json.loads(original)
            study["configs"].pop("D4W1536/h1-reach-v0")
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(v.ValidationError, "config population"):
                v.validate(str(self.base / "results"), path, "exp1")
        finally:
            path.write_bytes(original)

    def test_complete_keys_with_shortened_methodology_are_rejected(self):
        path = self.base / "study.json"
        original = path.read_bytes()
        try:
            study = json.loads(original)
            study["configs"]["D4W1536/h1-reach-v0"]["probe"]["steps"] = 10
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(
                v.ValidationError, "approved study configurations"
            ):
                v.validate(str(self.base / "results"), path, "exp1")
        finally:
            path.write_bytes(original)

    def test_missing_seed_environment_and_architecture_rejected(self):
        root = ledger.ledger_root(str(self.base / "results"))
        for label, predicate in [
            ("seed", lambda p: p.name.endswith("seed5")),
            ("environment", lambda p: "h1-reach-v0" in p.name),
            ("architecture", lambda p: "D4W1536" in p.name),
        ]:
            moved = []
            try:
                for p in root.iterdir():
                    if predicate(p):
                        target = self.base / p.name
                        p.rename(target)
                        moved.append((target, p))
                with self.subTest(label=label), self.assertRaises(
                    v.ValidationError
                ) as cm:
                    v.validate(
                        str(self.base / "results"), self.base / "study.json", "exp1"
                    )
                self.assertIn("population mismatch", str(cm.exception))
                self.assertTrue(
                    any(
                        x["status"] == "invalid_incomplete" for x in cm.exception.census
                    )
                )
            finally:
                for source, target in moved:
                    source.rename(target)

    def test_missing_endpoints_duplicates_times_and_provenance(self):
        a, e, s = "D4W1024", "dog-run", 1
        directory = ledger.ledger_root(str(self.base / "results")) / v.run_key(a, e, s)
        parent = lambda: self.item(a, e, s)
        mutations = [
            ("run.csv", lambda df: pd.concat([df, df], ignore_index=True)),
            ("run.csv", lambda df: df.assign(final_loss_iqm=99.0)),
            ("run.csv", lambda df: df.assign(status="running")),
            ("run.csv", lambda df: df.assign(code_commit="2" * 40)),
            ("checks.csv", lambda df: df.iloc[:-1]),
            ("checks.csv", lambda df: pd.concat([df, df.iloc[-1:]], ignore_index=True)),
            (
                "checks.csv",
                lambda df: df.assign(interaction_step=df.interaction_step + 1),
            ),
            ("checks.csv", lambda df: df.assign(seed=99)),
            ("metrics.csv", lambda df: df.iloc[:-1]),
            ("metrics.csv", lambda df: df.drop(columns="train/churn")),
        ]
        for filename, change in mutations:
            with self.subTest(filename=filename, change=change):
                self.mutate_csv(directory / filename, change, parent)
        source = self.base / "checkpoints/exp1" / a / e / "seed_1"
        for path in [
            source / "DONE",
            source / "fresh_critic/dummy",
            source / "state/LATEST",
        ]:
            original = path.read_bytes()
            try:
                path.unlink()
                with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                    parent()
            finally:
                path.write_bytes(original)
        path = source / "run_metadata.json"
        original = path.read_bytes()
        try:
            data = json.loads(original)
            data["launches"][0]["runtime"]["jax"] = "wrong"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "runtime mismatch"):
                parent()
        finally:
            path.write_bytes(original)

    def test_episode_grid_breaks_and_early_termination(self):
        item = self.item()
        v._fork(v.Snapshot(), str(self.base / "results"), item, self.study)
        root = exp2_ledger.run_root(item["run_key"], str(self.base / "results"))
        path = root / "arm_injected/eval_episodes.csv"
        for change in [
            lambda df: df.iloc[:-1],
            lambda df: pd.concat([df, df.iloc[-1:]], ignore_index=True),
            lambda df: df.assign(eval_index=df.eval_index + 1),
            lambda df: df.assign(steps_since_fork=df.steps_since_fork + 1),
            lambda df: df.assign(**{"return": np.nan}),
            lambda df: df.assign(length=0),
            lambda df: df.assign(arm="identity"),
        ]:
            with self.subTest(change=change):
                self.mutate_csv(
                    path,
                    change,
                    lambda: v._fork(
                        v.Snapshot(), str(self.base / "results"), item, self.study
                    ),
                )
        self.assertTrue((pd.read_csv(path).length == 1).all())

    def test_missing_fork_arm_check1_and_check2_evidence(self):
        item = self.item()
        root = exp2_ledger.run_root(item["run_key"], str(self.base / "results"))
        for path in [
            root / "fork.json",
            root / "check1_injected.json",
            root / "check2.json",
            root / "arm_control/checks.csv",
            root / "arm_injected/eval_episodes.csv",
            self.base
            / "checkpoints/exp2_arm/D4W1024/dog-run/seed_1/injected/probes/check2_curves.npz",
        ]:
            original = path.read_bytes()
            try:
                path.unlink()
                with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                    v._fork(v.Snapshot(), str(self.base / "results"), item, self.study)
            finally:
                path.write_bytes(original)
        path = root / "check1_injected.json"
        original = path.read_bytes()
        try:
            data = json.loads(original)
            data["pass"] = False
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "unresolved"):
                v._fork(v.Snapshot(), str(self.base / "results"), item, self.study)
        finally:
            path.write_bytes(original)
        path = root / "check2.json"
        original = path.read_bytes()
        try:
            data = json.loads(original)
            self.assertFalse(data["pass"])
            data["paired_difference_iqm"] = 99.0
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "stale Check 2"):
                v._fork(v.Snapshot(), str(self.base / "results"), item, self.study)
        finally:
            path.write_bytes(original)

    def test_routine_pruned_checkpoints_not_required(self):
        source = self.item()["source"]
        self.assertEqual(len(list((source / "state").iterdir())), 2)
        self.item()

    def test_snapshot_change_and_exploratory_atomic_markers(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "input"
            p.write_text("before")
            snap = v.Snapshot()
            snap.read(p)
            p.write_text("after")
            with self.assertRaisesRegex(ValueError, "input changed"):
                snap.unchanged()
            out = Path(d) / "explore"
            v.publish(
                out,
                lambda path, cert: (path / "data.csv").write_text("x"),
                self.base / "results",
                "exp1",
                exploratory=True,
            )
            self.assertFalse((out / "validation.json").exists())
            self.assertFalse(
                json.loads((out / "EXPLORATORY.json").read_text())[
                    "validated_confirmatory"
                ]
            )
            failed = Path(d) / "refused"
            with self.assertRaises(v.ValidationError):
                v.publish(
                    failed,
                    lambda *args: self.fail("writer called"),
                    self.base / "results",
                    "exp1",
                )
            self.assertFalse(failed.exists())
            self.assertTrue(Path(str(failed) + ".validation_failed.json").exists())
            with self.assertRaisesRegex(ValueError, "overwrite"):
                v.publish(
                    out,
                    lambda *args: None,
                    self.base / "results",
                    "exp1",
                    exploratory=True,
                )

    def test_control_original_panel_change_is_rejected_in_exp1(self):
        source = self.item()["source"]
        path = source / "fork/check1_pre.npz"
        original = path.read_bytes()
        try:
            with np.load(path) as data:
                values = dict(data)
            values["q"][0] += 1
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "original/control Check 1"):
                self.item()
        finally:
            path.write_bytes(original)

    def test_terminal_episode_loss_and_nonfinite_return_rejected(self):
        source = self.item()["source"]
        root = source / "state"
        path = root / (root / "LATEST").read_text() / "meta.pkl"
        original = path.read_bytes()
        try:
            for change in ("missing", "nonfinite"):
                data = pickle.loads(original)
                if change == "missing":
                    data["eval_rows"].pop()
                else:
                    data["eval_rows"][0]["return"] = float("nan")
                path.write_bytes(pickle.dumps(data))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.item()
        finally:
            path.write_bytes(original)

    def test_missing_and_wrong_per_q_curve_rejected(self):
        directory = ledger.ledger_root(str(self.base / "results")) / v.run_key(
            "D4W1536", "h1-reach-v0", 1
        )
        path = directory / "probe_curves.npz"
        original = path.read_bytes()
        try:
            for change in ("missing", "shape", "score"):
                with np.load(io.BytesIO(original)) as data:
                    values = dict(data)
                field = "check_20/current_q2_losses"
                if change == "missing":
                    values.pop(field)
                elif change == "shape":
                    values[field] = values[field][:4]
                else:
                    values["check_20/current_q2_score"] = np.zeros(5)
                np.savez_compressed(path, **values)
                with self.subTest(change=change), self.assertRaises(
                    (ValueError, KeyError)
                ):
                    self.item("D4W1536", "h1-reach-v0", 1)
        finally:
            path.write_bytes(original)

    def test_directory_addition_prevents_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            snap = v.Snapshot()
            snap.directory(root)
            (root / "late").mkdir()
            with self.assertRaisesRegex(ValueError, "directory changed"):
                snap.unchanged()

    def test_primary_contains_check2_failure(self):
        from analysis.exp2_analysis import load, paired_returns

        data = load(str(self.base / "results"))
        paired = paired_returns(data["evals"], data["forks"])
        self.assertEqual(len(data["forks"]), 2)
        self.assertFalse(data["forks"].check2_pass.any())
        self.assertEqual(len(paired), 52)
        self.assertEqual(set(paired.seed), {1, 2})

    def test_confirmatory_output_contains_full_population_certificate(self):
        from analysis import exp1_analysis as a1, exp2_analysis as a2

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "exp1"
            with mock.patch.object(a1, "plot_trajectories"), mock.patch.object(
                a1, "plot_every_seed"
            ):
                a1.run_analysis(
                    str(out),
                    str(self.base / "results"),
                    study_manifest=self.base / "study.json",
                )
            cert = json.loads((out / "validation.json").read_text())
            self.assertTrue(cert["validated_confirmatory"])
            self.assertEqual(len(cert["census"]), 195)
            endpoints = pd.read_csv(out / "primary_endpoint.csv")
            self.assertTrue((endpoints.runs_scaled == 65).all())
            self.assertTrue((endpoints.runs_default == 65).all())
            out2 = Path(d) / "exp2"
            with mock.patch.object(a2, "plot_paired"), mock.patch.object(
                a2, "plot_both_arms"
            ), mock.patch.object(a2, "shared_time_axis"):
                a2.run_analysis(
                    str(out2),
                    results_root=str(self.base / "results"),
                    study_manifest=self.base / "study.json",
                )
            cert2 = json.loads((out2 / "validation.json").read_text())
            self.assertEqual(len(cert2["census"]), 130)
            summary = pd.read_csv(out2 / "eligibility_by_environment.csv")
            self.assertEqual(len(summary), 26)
            self.assertEqual(summary.zero_eligible.sum(), 25)
            self.assertEqual(len(pd.read_csv(out2 / "paired_returns.csv")), 52)
            self.assertTrue(
                pd.read_csv(out2 / "paired_returns_check2_success_only.csv").empty
            )

    def test_one_seed_reporting_is_blocked_pending_decision(self):
        from analysis.exp2_analysis import run_analysis

        census = [
            dict(
                architecture="D4W1024",
                environment="dog-run",
                status="eligible_trigger",
                check2_pass=False,
            )
        ]
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            v, "validate", return_value=(v.Snapshot(), {"census": census})
        ):
            with self.assertRaises(v.ReportingDecisionRequired):
                run_analysis(
                    str(Path(d) / "out"),
                    results_root=str(self.base / "results"),
                    study_manifest=self.base / "study.json",
                )
            self.assertFalse((Path(d) / "out/validation.json").exists())

    def test_last_eligible_check_keeps_full_horizon_and_nominal_endpoint(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            study = f.study(base)
            directory, source = f.parent(
                base, study, "D4W1536", "dog-run", 1, eligible=True, fork_check=19
            )
            row, checks, item = v._parent(
                v.Snapshot(), directory, "D4W1536", "dog-run", 1, study
            )
            self.assertEqual(item["fork_step"], 475000)
            self.assertEqual(checks.check_index.max(), 24)
            self.assertEqual(
                row.final_loss_iqm, checks[checks.check_index == 20].loss_iqm.iloc[0]
            )
            v._fork(v.Snapshot(), str(base / "results"), item, study)

    def test_twin_aggregate_cannot_disagree_with_individual_measurements(self):
        a, e, s = "D4W1536", "h1-run-v0", 1
        directory = ledger.ledger_root(str(self.base / "results")) / v.run_key(a, e, s)
        source = self.base / "checkpoints/exp1" / a / e / "seed_1"
        state = source / "state" / (source / "state/LATEST").read_text()
        meta = pickle.loads((state / "meta.pkl").read_bytes())
        row = meta["extra_state"]["probe_records"][1]
        for r in range(5):
            row[f"score_fresh_r{r}"] = 2.0
            row[f"loss_r{r}"] = 0.0
        row.update(score_fresh_iqm=2.0, loss_iqm=0.0, ci_low=0.0, ci_high=0.0)
        path = directory / "checks.csv"
        original = path.read_bytes()
        try:
            frame = pd.read_csv(path)
            for key, value in row.items():
                frame.loc[1, key] = value
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "per-Q mean"):
                v._checks(
                    v.Snapshot(),
                    path,
                    meta,
                    1000000,
                    1000000,
                    s,
                    self.study["configs"][f"{a}/{e}"],
                    a,
                    e,
                )
        finally:
            path.write_bytes(original)

    def test_zero_eligible_is_explicit(self):
        from analysis.exp2_analysis import _write_analysis

        cert = {
            "census": [
                dict(
                    run_key="null",
                    architecture="D4W1024",
                    environment="dog-run",
                    status="valid_no_trigger",
                )
            ]
        }
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d) / "empty"
            empty.mkdir()
            result = _write_analysis(
                str(Path(d) / "out"), results_root=str(empty), certification=cert
            )
            self.assertTrue(result["forks"].empty)
            self.assertEqual(len(pd.read_csv(Path(d) / "out/candidate_census.csv")), 1)


if __name__ == "__main__":
    unittest.main()
