"""Verify complete byte preservation, frozen-source provenance and refusal paths."""

import hashlib
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.package_exp12_diagnostic import REQUIRED, SOURCE_PATHS, package


class ArtifactPackageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        for name in (*SOURCE_PATHS, "main/configs/base_exp12.yaml"):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frozen source\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        )
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.run = self.root / "run"
        self.run.mkdir()
        for name in REQUIRED:
            (self.run / name).touch()
        (self.run / "commit.txt").write_text(self.commit + "\n")
        (self.run / "backend.json").write_text(json.dumps({"commit": self.commit}))
        (self.run / "trace.jsonl").write_text(
            json.dumps(
                dict(
                    event="wrapper_start",
                    command=["scripts/profile_exp12.py", "--archs", "D4W1536"],
                    slurm_job_id="123",
                )
            )
            + "\n"
        )
        (self.run / "profile.log").write_bytes(b"complete log\x00\n")
        (self.run / "exit_status.txt").write_text("124\n")
        for suffix in ("out", "err"):
            Path(f"{self.run}.slurm-123.{suffix}").touch()
        self.out = self.root / "package.zip"

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, text=True, stderr=subprocess.DEVNULL
        )

    def package(self, **kwargs):
        return package(self.run, "123", self.commit, self.repo, self.out, **kwargs)

    def test_full_bytes_checksums_and_frozen_source(self):
        (self.repo / "main/configs/base_exp12.yaml").write_text("uncommitted change\n")
        (self.run / "cache_single_writer").mkdir()
        (self.run / "cache_single_writer/large.cache").write_bytes(b"excluded")
        report = self.package()
        self.assertEqual(report["missing_required"], [])
        self.assertTrue(report["configuration"]["runtime_command_available"])
        self.assertFalse(report["configuration"]["resolved_config_available"])
        self.assertEqual(report["qualification_status"], "not_assessed")
        with zipfile.ZipFile(self.out) as archive:
            self.assertIsNone(archive.testzip())
            for name in REQUIRED:
                self.assertEqual(
                    archive.read("artifacts/" + name), (self.run / name).read_bytes()
                )
            self.assertEqual(
                archive.read("source/main/configs/base_exp12.yaml"), b"frozen source\n"
            )
            self.assertEqual(archive.read("artifacts/slurm.stderr"), b"")
            self.assertFalse(any("cache" in p for p in archive.namelist()))
            for line in archive.read("SHA256SUMS").decode().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
            stored = json.loads(archive.read("MANIFEST.json"))
            self.assertEqual(stored, report)
        self.assertEqual(
            Path(str(self.out) + ".sha256").read_text().split()[0],
            hashlib.sha256(self.out.read_bytes()).hexdigest(),
        )

    def test_partial_trace_is_preserved_and_flagged(self):
        with (self.run / "trace.jsonl").open("ab") as handle:
            handle.write(b'{"event":"begin"}\n{"partial":')
        report = self.package()
        trace = report["traces"]["artifacts/trace.jsonl"]
        self.assertEqual(trace["valid_records"], 2)
        self.assertEqual(trace["invalid_records"][0]["line"], 3)
        with zipfile.ZipFile(self.out) as archive:
            self.assertEqual(
                archive.read("artifacts/trace.jsonl"),
                (self.run / "trace.jsonl").read_bytes(),
            )

    def test_exit_zero_is_not_a_qualification_and_bad_status_is_preserved(self):
        (self.run / "exit_status.txt").write_text("0\n")
        report = self.package()
        self.assertEqual(report["exit_status"], 0)
        self.assertEqual(report["qualification_status"], "not_assessed")
        self.out = self.root / "malformed-status.zip"
        (self.run / "exit_status.txt").write_bytes(b"partial-status")
        report = self.package()
        self.assertIn("malformed", report["exit_status"])
        with zipfile.ZipFile(self.out) as archive:
            self.assertEqual(
                archive.read("artifacts/exit_status.txt"), b"partial-status"
            )

    def test_missing_artifacts_are_explicit_not_invented(self):
        (self.run / "commit.txt").unlink()
        (self.run / "backend.json").unlink()
        (self.run / "trace.jsonl").unlink()
        Path(f"{self.run}.slurm-123.err").unlink()
        report = self.package()
        self.assertEqual(
            set(report["missing_required"]),
            {
                "artifacts/commit.txt",
                "artifacts/backend.json",
                "artifacts/trace.jsonl",
                "artifacts/slurm.stderr",
            },
        )
        self.assertFalse(report["source_revision_recorded"])
        self.assertFalse(report["configuration"]["runtime_command_available"])

    def test_recorded_commit_mismatch_refused_without_output(self):
        (self.run / "commit.txt").write_text("0" * 40)
        with self.assertRaisesRegex(ValueError, "recorded commit"):
            self.package()
        self.assertFalse(self.out.exists())

    def test_backend_commit_mismatch_refused(self):
        (self.run / "backend.json").write_text(json.dumps({"commit": "0" * 40}))
        with self.assertRaisesRegex(ValueError, "backend.json: recorded commit"):
            self.package()

    def test_trace_job_id_mismatch_refused(self):
        (self.run / "trace.jsonl").write_text(
            '{"event":"wrapper_start","slurm_job_id":"456"}\n'
        )
        with self.assertRaisesRegex(ValueError, "recorded job ID mismatch"):
            self.package()

    def test_slurm_filename_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, "Slurm log job ID mismatch"):
            self.package(stderr=str(self.root / "run.slurm-456.err"))

    def test_symlink_artifact_refused(self):
        (self.run / "profile.log").unlink()
        (self.run / "profile.log").symlink_to(
            self.repo / "main/configs/base_exp12.yaml"
        )
        with self.assertRaisesRegex(ValueError, "symlink artifact"):
            self.package()

    def test_invalid_identifiers_refused(self):
        for job, commit in (("../123", self.commit), ("123", self.commit[:8])):
            with self.subTest(job=job, commit=commit), self.assertRaises(ValueError):
                package(self.run, job, commit, self.repo, self.out)

    def test_output_inside_input_refused(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            package(self.run, "123", self.commit, self.repo, self.run / "package.zip")

    def test_existing_archive_or_sidecar_never_overwritten(self):
        self.package()
        before = self.out.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.package()
        self.assertEqual(self.out.read_bytes(), before)
        self.out.unlink()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.package()


if __name__ == "__main__":
    unittest.main()
