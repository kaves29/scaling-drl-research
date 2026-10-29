"""Array-only certification regressions; no training or scientific probe fits."""

import io
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import numpy as np

from analysis import exp12_validation as v
from experiments.exp12 import ledger
from tests import exp12_complete_fixture as f


class MethodCCertificateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.study = f.study(cls.base)
        for config in cls.study["configs"].values():
            config["probe"]["steps"] = 2000
        cls.parents = {}
        for env in ("dog-run", "h1-run-v0"):
            directory, source = f.parent(
                cls.base, cls.study, "D4W1024", env, 1, eligible=True
            )
            cls.parents[env] = directory, source

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def item(self, env):
        return v._parent(
            v.Snapshot(), self.parents[env][0], "D4W1024", env, 1, self.study
        )[2]

    def check_arm(self, env):
        return v._fork(
            v.Snapshot(), str(self.base / "results"), self.item(env), self.study
        )

    def change_archive(self, path, change, check):
        original = path.read_bytes()
        try:
            with np.load(io.BytesIO(original)) as data:
                arrays = dict(data)
            change(arrays)
            np.savez_compressed(path, **arrays)
            with self.assertRaises((ValueError, KeyError)):
                check()
        finally:
            path.write_bytes(original)

    def test_complete_2000_step_single_and_twin_forks_are_accepted(self):
        for env in self.parents:
            with self.subTest(env=env):
                # A failed Check 2 remains valid primary evidence.
                self.assertFalse(self.check_arm(env)["pass"])

    def test_parent_rejects_all_curves_shortened_to_1000(self):
        for env in self.parents:
            self.change_archive(
                self.parents[env][0] / "probe_curves.npz",
                lambda a: a.update(
                    {k: x[:, :1000] for k, x in a.items() if k.endswith("_losses")}
                ),
                lambda: self.item(env),
            )

    def test_check2_rejects_all_curves_shortened_to_1000(self):
        for env in self.parents:
            self.change_archive(
                self.arm_curves(env),
                lambda a: a.update(
                    {k: x[:, :1000] for k, x in a.items() if k.endswith("_losses")}
                ),
                lambda: self.check_arm(env),
            )

    def arm_curves(self, env):
        return (
            self.base
            / "checkpoints/exp2_arm/D4W1024"
            / env
            / "seed_1/injected/probes/check2_curves.npz"
        )

    def test_incomplete_oversized_transposed_or_wrong_round_count_rejected(self):
        for path, check in (
            (
                self.parents["dog-run"][0] / "probe_curves.npz",
                lambda: self.item("dog-run"),
            ),
            (self.arm_curves("dog-run"), lambda: self.check_arm("dog-run")),
        ):
            for shape in (
                (5, 1999),
                (5, 2001),
                (4, 2000),
                (6, 2000),
                (2000, 5),
                (5, 2000, 1),
            ):

                def mutate(arrays):
                    key = next(k for k in arrays if k.endswith("_losses"))
                    arrays[key] = np.zeros(shape, np.float32)

                with self.subTest(path=path.name, shape=shape):
                    self.change_archive(path, mutate, check)

    def test_missing_loss_array_rejected(self):
        for env in self.parents:
            for path, check in (
                (self.parents[env][0] / "probe_curves.npz", lambda: self.item(env)),
                (self.arm_curves(env), lambda: self.check_arm(env)),
            ):

                def mutate(arrays):
                    arrays.pop(next(k for k in arrays if k.endswith("_losses")))

                self.change_archive(path, mutate, check)

    def test_duplicate_archive_keys_rejected(self):
        for path, check in (
            (
                self.parents["dog-run"][0] / "probe_curves.npz",
                lambda: self.item("dog-run"),
            ),
            (self.arm_curves("dog-run"), lambda: self.check_arm("dog-run")),
        ):
            original = path.read_bytes()
            try:
                with zipfile.ZipFile(path, "a") as archive, warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    key = next(
                        k for k in archive.namelist() if k.endswith("_losses.npy")
                    )
                    archive.writestr(key, archive.read(key))
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    check()
            finally:
                path.write_bytes(original)

    def test_step_contract_is_explicit_and_not_inferred_from_array_length(self):
        for steps in (1000, 2000):
            self.assertEqual(
                v._probe_curve_shape({"probe": {"rounds": 5, "steps": steps}}),
                (5, steps),
            )
        for probe in (
            {},
            {"rounds": 5},
            {"rounds": 10, "steps": 2000},
            {"rounds": 5, "steps": 1999},
            {"rounds": 5, "steps": 2000.0},
            {"rounds": 5, "steps": "2000"},
            {"rounds": True, "steps": 2000},
        ):
            with self.subTest(probe=probe), self.assertRaises(ValueError):
                v._probe_curve_shape({"probe": probe})

    def test_certification_does_not_rewrite_or_truncate_arrays(self):
        before = {
            path: path.read_bytes()
            for env in self.parents
            for path in (
                self.parents[env][0] / "probe_curves.npz",
                self.arm_curves(env),
            )
        }
        for env in self.parents:
            self.check_arm(env)
        for path, value in before.items():
            self.assertEqual(path.read_bytes(), value)


class CurveRetentionTest(unittest.TestCase):
    def test_raw_to_ledger_round_trip_retains_every_update(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probe = root / "run/probes"
            probe.mkdir(parents=True)
            values = np.arange(10000, dtype=np.float32).reshape(5, 2000)
            np.savez_compressed(probe / "check_01.npz", fresh_losses=values)
            identity = dict(
                run_key="retention",
                run_role="dev",
                architecture="D1W8",
                environment="dog-run",
                seed=101,
                budget_env_steps=1000000,
                num_interaction_steps=500000,
                num_checks=20,
            )
            directory = ledger.write_run(
                identity,
                [{"check_index": 1, "interaction_step": 25000}],
                None,
                "running",
                probe,
                str(root),
            )
            with np.load(directory / "probe_curves.npz") as arrays:
                actual = arrays["check_01/fresh_losses"]
                np.testing.assert_array_equal(actual, values)
                self.assertEqual(actual[4, 1999], 9999)
