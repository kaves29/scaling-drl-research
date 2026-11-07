"""Exercise validation checkout guards before any cluster or accelerator setup."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
WRAPPERS = ("sbatch_exp12_blockA.sh", "sbatch_exp12_blockB.sh")


class ValidationCheckoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.main = self.repo / "main"
        scripts = self.main / "scripts"
        scripts.mkdir(parents=True)
        for name in (*WRAPPERS, "check_exp12_validation_checkout.sh"):
            shutil.copyfile(SCRIPTS / name, scripts / name)
        for name in (
            "exp12_blockA.sh",
            "exp12_blockB.sh",
            "collect_report.sh",
            "probe_fresh_checks.py",
            "profile_exp12.py",
            "positive_control.py",
            "preflight_checkpoint_check.py",
        ):
            (scripts / name).touch()
        (self.main / "run.py").touch()
        (self.main / "configs").mkdir()
        (self.main / "configs/base_exp12.yaml").touch()
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
        self.git("checkout", "--detach", "-q", self.commit)
        self.env = dict(
            os.environ,
            SLURM_JOB_ID="123",
            SLURM_SUBMIT_DIR=str(self.main),
            EXPECTED_COMMIT=self.commit,
        )
        self.env.pop("BLOCKB_TEST_HOOKS", None)

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, text=True, stderr=subprocess.DEVNULL
        )

    def run_wrapper(self, name):
        # Slurm runs a spool copy; the executable's own location is not a checkout.
        spool = self.root / name
        shutil.copyfile(SCRIPTS / name, spool)
        command = 'module() { printf "SETUP:%s\\n" "$PWD"; exit 73; }; export -f module; bash "$1"'
        return subprocess.run(
            ["bash", "-c", command, "test", str(spool)],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def assert_failure(self, reason):
        for name in WRAPPERS:
            with self.subTest(name=name):
                result = self.run_wrapper(name)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(reason, result.stderr)
                self.assertNotIn("SETUP:", result.stdout)

    def test_detached_spool_uses_submission_checkout_before_setup(self):
        for name in WRAPPERS:
            result = self.run_wrapper(name)
            self.assertEqual(result.returncode, 73, result.stderr)
            self.assertIn(f"SETUP:{self.main}", result.stdout)
            self.assertIn(self.commit, result.stdout)

    def test_requires_submission_directory(self):
        self.env.pop("SLURM_SUBMIT_DIR")
        self.assert_failure("SLURM_SUBMIT_DIR")

    def test_requires_full_expected_commit(self):
        self.env.pop("EXPECTED_COMMIT")
        self.assert_failure("EXPECTED_COMMIT is required")
        self.env["EXPECTED_COMMIT"] = self.commit[:8]
        self.assert_failure("40-character")

    def test_wrong_commit_rejected(self):
        self.env["EXPECTED_COMMIT"] = "0" * 40
        self.assert_failure("HEAD mismatch")

    def test_tracked_and_staged_changes_rejected(self):
        (self.main / "run.py").write_text("# changed\n")
        self.assert_failure("tracked or staged")
        self.git("add", ".")
        self.assert_failure("tracked or staged")

    def test_root_untracked_source_rejected(self):
        (self.repo / "unreviewed.py").touch()
        self.assert_failure("untracked source")

    def test_opened_slurm_logs_allowed_but_tracked_logs_still_checked(self):
        logs = self.main / "logs"
        logs.mkdir()
        (logs / "exp12_blockB_123.err").touch()
        self.test_detached_spool_uses_submission_checkout_before_setup()
        self.git("add", "main/logs")
        self.assert_failure("tracked or staged")

    def test_missing_required_file_rejected(self):
        (self.main / "configs/base_exp12.yaml").unlink()
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "missing config",
        )
        self.env["EXPECTED_COMMIT"] = self.git("rev-parse", "HEAD").strip()
        self.assert_failure("missing required file: configs/base_exp12.yaml")

    def test_test_hooks_rejected_before_setup(self):
        self.env["BLOCKB_TEST_HOOKS"] = "/unreviewed/hooks.sh"
        self.assert_failure("BLOCKB_TEST_HOOKS")


if __name__ == "__main__":
    unittest.main()
