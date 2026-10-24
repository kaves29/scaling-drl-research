import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from analysis.campaign.ingest import ingest, local_artifacts
from analysis.campaign.statistics import summary, paired, trajectory
from analysis.campaign.provenance import critic_snapshot, write_sidecar
from analysis.campaign.diagnostic_plots import validate_pair
from analysis.campaign.pipeline import run


def fixture(path, n=103):
    with zipfile.ZipFile(path, "w") as z:
        for i in range(n):
            m = {
                "id": str(i),
                "name": "same display name",
                "campaign": "exp96" if i < 96 else "pilot",
                "state": "finished",
                "code": {"commit": "a" * 40, "dirty": False},
                "config": {
                    "seed": (i // 12 + 1) if i < 96 else 1,
                    "env_name": (
                        "dog-run",
                        "humanoid-run",
                        "myo-key-turn",
                        "myo-pen-twirl",
                    )[(i % 12) // 3],
                    "architecture": ("D2W512", "D4W1024", "D4W1536")[i % 3],
                    "num_interaction_steps": 10,
                },
            }
            z.writestr(f"{i}/metadata.json", json.dumps(m))
            z.writestr(
                f"{i}/history.jsonl",
                "\n".join(
                    json.dumps({"interaction_step": t, "eval/avg_return": v})
                    for t, v in [(0, 0), (5, i), (10, i * 2)]
                ),
            )


class CampaignTests(unittest.TestCase):
    def test_known_statistics(self):
        r = summary([0, 1, 2, 3, 100], reps=100)
        self.assertEqual(r["iqm"], 2)
        self.assertEqual(r["median"], 2)
        self.assertEqual(r["mean"], 21.2)
        self.assertEqual(r, summary([0, 1, 2, 3, 100], reps=100))
        self.assertIsNone(summary([1])["mean_ci95"])

    def test_pairing(self):
        r = paired({1: 3, 2: 7}, {1: 1, 3: 4}, reps=10)
        self.assertEqual(r["difference"]["mean"], 2)
        self.assertEqual(r["left_only"], [2])

    def test_integral(self):
        self.assertEqual(trajectory([(0, 0), (5, 2), (10, 4)], 10)["observed_auc"], 20)
        self.assertIsNone(trajectory([(0, 0), (5, 2)], 10)["final_window"])
        with self.assertRaises(ValueError):
            trajectory([(1, 2), (1, 3)], 10)

    def test_103_record_names(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.zip"
            fixture(p)
            a = ingest(p)
            self.assertEqual(len(a["records"]), 103)
            self.assertEqual(a["coverage"]["exp96_verified_records"], 96)
            self.assertEqual(a["coverage"]["pilot_verified_records"], 7)
            self.assertFalse(a["relationships"])

    def test_quality_gates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.zip"
            with zipfile.ZipFile(p, "w") as z:
                for i in range(2):
                    z.writestr(
                        f"{i}/metadata.json",
                        json.dumps(
                            {
                                "id": "same",
                                "state": "crashed",
                                "config": {
                                    "seed": 1,
                                    "architecture": "A",
                                    "env_name": "e",
                                },
                            }
                        ),
                    )
                    z.writestr(
                        f"{i}/history.jsonl",
                        '{"interaction_step":2,"q":NaN}\n{"interaction_step":1}',
                    )
            codes = {x["code"] for x in ingest(p)["issues"]}
            self.assertTrue(
                {
                    "duplicate_run_id",
                    "nonincreasing_steps",
                    "nonfinite_metric",
                    "nonfinished_status",
                    "unverified_source",
                }
                <= codes
            )

    def test_missing_history(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("metadata.json", '{"id":"r"}')
            self.assertIn("missing_history", {x["code"] for x in ingest(p)["issues"]})

    def test_unsafe_zip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("../metadata.json", "{}")
            with self.assertRaises(ValueError):
                ingest(p)

    def test_distribution(self):
        r = critic_snapshot([1, 3], [0, 0], [0, 1])
        self.assertEqual(r["td_mse"], 5)
        self.assertEqual(r["q"]["variance"], 1)
        with self.assertRaises(ValueError):
            critic_snapshot([1], [0, 1])

    def test_scores(self):
        a = dict(
            run_id="r",
            training_seed=1,
            round=0,
            checkpoint="x",
            task_id="t",
            target_hash="h",
            optimizer_hash="o",
            completed_updates=2000,
            status="complete",
            baseline=0.5,
            final_pool_mse=0.1,
            score=0.4,
        )
        b = {**a, "final_pool_mse": 0.3, "score": 0.2}
        self.assertAlmostEqual(validate_pair(a, b), 0.2)
        for bad in ({}, {**b, "task_id": "wrong"}, {**b, "score": 0.9}):
            with self.assertRaises(ValueError):
                validate_pair(a, bad)

    def test_adapters(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.npz"
            np.savez(p, loss=np.arange(3))
            self.assertEqual(local_artifacts(p)["loss"], [0, 1, 2])
            side = Path(d) / "receipt.json"
            write_sidecar(side, {"status": "partial"})
            with self.assertRaises(FileExistsError):
                write_sidecar(side, {})

    def test_deterministic_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.zip"
            fixture(p, 3)
            run(p, Path(d) / "a", reps=50)
            run(p, Path(d) / "b", reps=50)
            self.assertEqual(
                (Path(d) / "a/statistics.json").read_bytes(),
                (Path(d) / "b/statistics.json").read_bytes(),
            )


class IndependenceTests(unittest.TestCase):
    def test_repeated_checkpoints_rejected(self):
        from analysis.campaign.statistics import independent_seed_summary

        with self.assertRaises(ValueError):
            independent_seed_summary([{"seed": 1, "x": 2}, {"seed": 1, "x": 3}], "x")

    def test_longitudinal_missing_steps(self):
        from analysis.campaign.statistics import longitudinal

        a = longitudinal(
            [
                {"seed": 1, "step": 0, "value": 1},
                {"seed": 2, "step": 0, "value": 3},
                {"seed": 1, "step": 1, "value": 5},
            ],
            reps=10,
        )
        self.assertEqual([x["n_seeds"] for x in a], [2, 1])
        self.assertIsNone(a[1]["mean_ci95"])

    def test_provenance_corruption(self):
        from analysis.campaign.provenance import validate

        self.assertIn("config_hash_mismatch", validate({"config": {"lr": 1}}))
        self.assertIn("incomplete_optimizer", validate({}))

    def test_partial_crashed_and_mixed_configs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.zip"
            with zipfile.ZipFile(p, "w") as z:
                for i in range(4):
                    m = {
                        "id": str(i),
                        "campaign": "exp96",
                        "state": "crashed" if i == 0 else "finished",
                        "code": {"commit": "a" * 40, "dirty": False},
                        "config": {
                            "seed": i,
                            "env_name": "e",
                            "architecture": "A",
                            "num_interaction_steps": 10,
                            "lr": 1 if i < 3 else 2,
                        },
                    }
                    z.writestr(f"{i}/metadata.json", json.dumps(m))
                    end = 5 if i == 1 else 10
                    z.writestr(
                        f"{i}/history.jsonl",
                        "\n".join(
                            json.dumps({"interaction_step": t, "eval/avg_return": 1})
                            for t in [0, end]
                        ),
                    )
            run(p, Path(d) / "out", reps=10)
            a = json.loads((Path(d) / "out/statistics.json").read_text())
            self.assertEqual(a["missing_data_sensitivity"]["excluded_n"], 2)
            self.assertEqual(len(a["groups"]), 2)

    def test_nan_statistic_rejected(self):
        with self.assertRaises(ValueError):
            summary([1, float("nan")])

    def test_duplicate_zip_members(self):
        import warnings

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with zipfile.ZipFile(p, "w") as z:
                    z.writestr("metadata.json", "{}")
                    z.writestr("metadata.json", "{}")
            with self.assertRaises(ValueError):
                ingest(p)


class SnapshotTests(unittest.TestCase):
    def test_initial_fixed_and_varying_without_update(self):
        from analysis.campaign.provenance import fitting_snapshot
        from types import SimpleNamespace

        class Net:
            def apply(self, p, obs, act):
                return np.zeros(len(obs))

        state = SimpleNamespace(params={"x": np.array([3.0])})
        r = fitting_snapshot(
            Net(),
            state,
            np.zeros((2, 1)),
            np.zeros((2, 1)),
            np.array([1.0, 10.0]),
            [0],
            [1],
        )
        self.assertEqual(r["initial_full_pool_mse"], 50.5)
        self.assertEqual(r["fixed_batch_mse"], 1)
        self.assertEqual(r["varying_minibatch_mse"], 100)
        self.assertEqual(state.params["x"][0], 3)
