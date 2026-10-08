"""Resumable CPU test receipts cannot convert failure, skipping or stale source to success."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_exp12_cpu_validation import check_provenance, reusable

DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "run_exp12_cpu_validation.py"


class CPUValidationDriverTest(unittest.TestCase):
    def run_worker(self, failing):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fakecase.py").write_text(
                "import unittest\nclass T(unittest.TestCase):\n"
                " def test_pass(self): self.assertTrue(True)\n"
                " @unittest.skip('fixture dependency')\n def test_skip(self): pass\n"
                + (
                    " def test_failure(self): self.fail('intentional fixture failure')\n"
                    if failing
                    else ""
                )
            )
            receipt = root / "receipt.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(DRIVER),
                    "--worker",
                    "fakecase",
                    "--receipt",
                    str(receipt),
                    "--source-hash",
                    "f" * 64,
                ],
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(root)},
                capture_output=True,
                text=True,
            )
            return completed, json.loads(receipt.read_text())

    def test_failure_is_explicit_even_when_other_tests_pass_or_skip(self):
        completed, receipt = self.run_worker(True)
        self.assertEqual(completed.returncode, 1)
        self.assertFalse(receipt["success"])
        self.assertEqual(
            (receipt["tests_run"], receipt["failures"], receipt["errors"]), (3, 1, 0)
        )
        self.assertEqual(len(receipt["skips"]), 1)

    def test_success_retains_skips_instead_of_counting_them_as_passes(self):
        completed, receipt = self.run_worker(False)
        self.assertEqual(completed.returncode, 0)
        self.assertTrue(receipt["success"])
        self.assertEqual(receipt["tests_run"], 2)
        self.assertEqual(
            receipt["skips"],
            [{"test": "fakecase.T.test_skip", "reason": "fixture dependency"}],
        )

    def test_stale_resume_provenance_is_rejected_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            original = {"commit": "a" * 40, "source_hash": "f" * 64}
            check_provenance(path, original, False)
            check_provenance(path, original, True)
            before = path.read_bytes()
            for changed in (
                {**original, "commit": "b" * 40},
                {**original, "source_hash": "e" * 64},
            ):
                with self.assertRaises(ValueError):
                    check_provenance(path, changed, True)
                self.assertEqual(path.read_bytes(), before)

    def test_resume_requires_matching_successful_receipt_and_intact_owned_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "unittest.log"
            log.write_text("completed module log")
            data = dict(
                module="fixture",
                success=True,
                exit_code=0,
                source_hash="f" * 64,
                failures=0,
                errors=0,
                tests_run=1,
                test_ids=["fixture.T.test_pass"],
                log=str(log),
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            )
            self.assertTrue(reusable("fixture", data, "f" * 64, root))
            for field, bad in (
                ("success", False),
                ("exit_code", 1),
                ("source_hash", "e" * 64),
                ("tests_run", 2),
                ("errors", 1),
                ("log_sha256", "a" * 64),
                ("module", "other"),
            ):
                self.assertFalse(
                    reusable("fixture", {**data, field: bad}, "f" * 64, root)
                )
            log.write_text("tampered log")
            self.assertFalse(reusable("fixture", data, "f" * 64, root))


if __name__ == "__main__":
    unittest.main()
