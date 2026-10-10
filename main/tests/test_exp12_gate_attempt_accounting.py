"""CPU-only adversarial checks of immutable full-width attempt receipts."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.exp12_fullwidth_gate import (
    STAGES,
    attempt_accounting,
    compare,
    strict_json,
)
from tests import test_exp12_launch_preparation as fixtures


class AttemptAccountingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "evidence"
        self.root.mkdir()
        self.spec = fixtures.FullwidthContractTest().spec()
        self.records = fixtures.FullwidthContractTest().records()
        (self.root / "spec.json").write_text(json.dumps(self.spec))
        for i, stage in enumerate(STAGES):
            r = self.records[stage]
            step = (
                self.spec["crash_step"] if stage == "crash" else self.spec["end_step"]
            )
            r.update(
                stage=stage,
                expected_commit=self.spec["expected_commit"],
                spec_sha256=hashlib.sha256(
                    json.dumps(self.spec, sort_keys=True).encode()
                ).hexdigest(),
                interaction_step=step,
                update_step=2 * (step - 4999),
                restored_step=6000,
                restored_update=2002,
                roundtrip_differences=[],
            )
            directory = self.root / stage
            directory.mkdir()
            (directory / "receipt.json").write_text(json.dumps(r))
            (directory / "payload").write_text("checkpoint fixture")
            self.attempt(stage, str(100 + i), 3 if stage == "crash" else 0)

    def attempt(self, stage, job, code):
        p = self.root / "attempts" / (stage + "_" + job)
        p.mkdir(parents=True)
        verdict = (
            "EXPECTED_CRASH"
            if stage == "crash" and code == 3
            else "PROCESS_COMPLETE" if code == 0 else "INCOMPLETE"
        )
        (p / "process.json").write_text(
            json.dumps(dict(exit=code, stage=stage, verdict=verdict, timeout=False))
        )
        (p / "exit_status.txt").write_text(str(code) + "\n")
        (p / "commit.txt").write_text(self.spec["expected_commit"] + "\n")
        (p / "submitted-spec.json").write_text(json.dumps(self.spec))
        for name in ("dependencies.txt", "nvidia-smi.csv", "workload.log"):
            (p / name).write_text("fixture " + name)
        self.manifest(p)
        return p

    def manifest(self, p):
        files = sorted(
            q for q in self.root.rglob("*") if q.is_file() and q.name != "SHA256SUMS"
        )
        (p / "SHA256SUMS").write_text(
            "".join(
                hashlib.sha256(q.read_bytes()).hexdigest()
                + "  ./"
                + q.relative_to(self.root).as_posix()
                + "\n"
                for q in files
            )
        )

    def test_failed_then_success_keeps_failed_evidence(self):
        failed = self.attempt("reference_warm", "90", 1)
        # Historical failed manifests may legitimately contain an older global receipt.
        text = (failed / "SHA256SUMS").read_text()
        lines = [
            (
                line
                if not line.endswith("./reference_warm/receipt.json")
                else "0" * 64 + "  ./reference_warm/receipt.json"
            )
            for line in text.splitlines()
        ]
        (failed / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        before = {q: q.read_bytes() for q in self.root.rglob("*") if q.is_file()}
        audit = attempt_accounting(self.root, self.spec, self.records)
        self.assertEqual(audit["reference_warm"]["accepted"]["job_id"], "101")
        self.assertEqual(len(audit["reference_warm"]["attempts"]), 2)
        self.assertEqual(before, {q: q.read_bytes() for q in before})

    def test_two_successes_rejected(self):
        self.attempt("reference_warm", "200", 0)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            attempt_accounting(self.root, self.spec, self.records)

    def test_same_job_cannot_bind_two_stages(self):
        self.attempt("reference_warm", "100", 1)
        with self.assertRaisesRegex(ValueError, "job ID reused"):
            attempt_accounting(self.root, self.spec, self.records)

    def test_missing_attempt_rejected(self):
        import shutil

        shutil.rmtree(self.root / "attempts/resume_103")
        with self.assertRaisesRegex(ValueError, "missing accepted"):
            attempt_accounting(self.root, self.spec, self.records)

    def test_wrong_spec_and_commit_rejected_even_for_failed_attempt(self):
        p = self.attempt("reference_warm", "90", 1)
        for name, content in [
            ("submitted-spec.json", json.dumps(dict(self.spec, seed=103))),
            ("commit.txt", "b" * 40),
        ]:
            original = (p / name).read_text()
            (p / name).write_text(content)
            self.manifest(p)
            with self.assertRaisesRegex(ValueError, "provenance"):
                attempt_accounting(self.root, self.spec, self.records)
            (p / name).write_text(original)
            self.manifest(p)

    def test_forged_stage_receipt_rejected(self):
        self.records["reference_warm"]["stage"] = "reference_cold"
        (self.root / "reference_warm/receipt.json").write_text(
            json.dumps(self.records["reference_warm"])
        )
        self.manifest(self.root / "attempts/reference_warm_101")
        with self.assertRaisesRegex(ValueError, "receipt provenance"):
            attempt_accounting(self.root, self.spec, self.records)

    def test_receipt_replacement_and_corrupt_payload_rejected(self):
        for path in (
            "reference_warm/receipt.json",
            "reference_warm/payload",
            "attempts/reference_warm_101/workload.log",
        ):
            q = self.root / path
            original = q.read_bytes()
            q.write_bytes(original + b"corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                attempt_accounting(self.root, self.spec, self.records)
            q.write_bytes(original)

    def test_malformed_and_duplicate_json_rejected(self):
        p = self.root / "attempts/reference_warm_101/process.json"
        for text in ("{}", '{"exit":0,"exit":1}', '{"exit":NaN}'):
            p.write_text(text)
            with self.assertRaises((ValueError, TypeError)):
                attempt_accounting(self.root, self.spec, self.records)

    def test_forged_exit_or_job_stage_rejected(self):
        p = self.root / "attempts/reference_warm_101"
        (p / "exit_status.txt").write_text("1")
        with self.assertRaisesRegex(ValueError, "exit status"):
            attempt_accounting(self.root, self.spec, self.records)
        (p / "exit_status.txt").write_text("0")
        p.rename(p.with_name("resume_101"))
        with self.assertRaisesRegex(ValueError, "stage"):
            attempt_accounting(self.root, self.spec, self.records)

    def test_missing_failed_record_is_not_ignored(self):
        p = self.attempt("reference_warm", "90", 1)
        (p / "commit.txt").unlink()
        with self.assertRaises(OSError):
            attempt_accounting(self.root, self.spec, self.records)

    def test_unsafe_checksum_path_rejected(self):
        p = self.root / "attempts/reference_warm_101/SHA256SUMS"
        p.write_text(p.read_text() + "0" * 64 + "  ./../outside\n")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            attempt_accounting(self.root, self.spec, self.records)

    def test_deleted_checksummed_payload_rejected(self):
        (self.root / "reference_warm/payload").unlink()
        with self.assertRaises(OSError):
            attempt_accounting(self.root, self.spec, self.records)

    def test_missing_manifest_and_stage_receipt_rejected(self):
        for name in (
            "attempts/reference_warm_101/SHA256SUMS",
            "reference_warm/receipt.json",
        ):
            path = self.root / name
            original = path.read_bytes()
            path.unlink()
            with self.assertRaises((ValueError, OSError)):
                attempt_accounting(self.root, self.spec, self.records)
            path.write_bytes(original)

    def test_cpu_cli_separates_reviewer_revision_and_preserves_evidence(self):
        import sys
        from scripts.exp12_fullwidth_gate import main

        self.attempt("reference_warm", "90", 1)
        report = self.root.parent / "cpu-report.json"
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        argv = [
            "gate",
            "--stage",
            "compare",
            "--spec",
            str(self.root / "spec.json"),
            "--out",
            str(self.root),
            "--comparison-commit",
            "c" * 40,
            "--report",
            str(report),
        ]
        with patch.object(sys, "argv", argv), patch(
            "utils.run_metadata.code_version",
            return_value={"commit": "c" * 40, "dirty": False},
        ), patch("subprocess.check_output", return_value=b""), patch(
            "experiments.exp12.state.latest_state_dir", side_effect=lambda p: p
        ), patch(
            "experiments.exp12.state.load_meta",
            return_value=dict(interaction_step=6000, update_step=2002),
        ), patch(
            "experiments.exp12.state.state_differences", return_value=[]
        ):
            self.assertEqual(main(), 0)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertEqual(strict_json(report)["comparator_commit"], "c" * 40)
            with self.assertRaisesRegex(ValueError, "fresh absolute"):
                main()  # Never overwrite the CPU re-analysis receipt either.
            report.unlink()
            argv[-1] = str(self.root / "forbidden.json")
            with self.assertRaisesRegex(ValueError, "fresh absolute"):
                main()

    def test_comparator_keeps_existing_checks_and_writes_external_report(self):
        self.attempt("reference_warm", "90", 1)
        report = self.root.parent / "comparison.json"
        with patch(
            "experiments.exp12.state.latest_state_dir", side_effect=lambda p: p
        ), patch(
            "experiments.exp12.state.load_meta",
            return_value=dict(interaction_step=6000, update_step=2002),
        ), patch(
            "experiments.exp12.state.state_differences", return_value=[]
        ):
            self.assertEqual(compare(self.root, self.spec, report, "c" * 40), 0)
            result = strict_json(report)
            self.assertEqual(result["expected_commit"], "a" * 40)
            self.assertEqual(result["comparator_commit"], "c" * 40)
            self.assertEqual(result["verdict"], "PASS")
            self.records["reference_warm"]["gpu_cache_hits"] = 0
            (self.root / "reference_warm/receipt.json").write_text(
                json.dumps(self.records["reference_warm"])
            )
            self.manifest(self.root / "attempts/reference_warm_101")
            self.assertEqual(compare(self.root, self.spec, report, "c" * 40), 1)
            self.assertEqual(strict_json(report)["verdict"], "INCOMPLETE")
        self.assertFalse((self.root / "validation.json").exists())
