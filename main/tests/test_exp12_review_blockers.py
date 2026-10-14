"""Regression tests for the independent review of claude/exp12-min-qualification.

1. Ledger.finish/claim must not certify evidence that changed after it was validated.
2. The qualification command script must reserve each submission atomically, record job ids
   durably, never resubmit, and refuse to continue past a partial submission.
3. The shared-filesystem verifier must be repeatable and must not modify its evidence.
4. Numerical-bound proposals must be source-bound and reject invalid or mixed measurements.
5. (Idle GPUs) An outcome that cannot be recorded must not keep an allocation waiting.
"""

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))

from scripts import exp12_dynamic_scheduler as scheduler  # noqa: E402
from scripts import exp12_numeric_bounds as bounds  # noqa: E402
from scripts import exp12_qualification_commands as commands  # noqa: E402
from scripts import exp12_shared_fs_claim_check as claim_check  # noqa: E402
from tests.test_exp12_dynamic_scheduler import evidence, item, owner  # noqa: E402

COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"


def _row(root, seed=1):
    row = item(root, seed=seed)
    evidence(Path(row["row"]["checkpoint_dir"]))
    receipt = root / f"receipt_{seed}.json"
    receipt.write_text('{"engineering_verdict":"COMPLETE"}')
    return row, receipt


# --------------------------------------------------------------------------------------- 1
class EvidenceBindingTest(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.row, self.receipt = _row(self.root)
        self.owner = owner(101, "first")
        self.ledger = scheduler.Ledger(self.root, query=lambda _: True, finish_timeout=0.2)
        self.assertTrue(self.ledger.claim(self.row, self.owner, lambda _: None))
        self.directory = self.ledger._row_dir(self.row)
        self.done = Path(self.row["row"]["checkpoint_dir"]) / "DONE"

    def tearDown(self):
        self._directory.cleanup()

    def _job(self, attempt_name="attempt"):
        attempt = self.root / attempt_name
        attempt.mkdir()
        process = mock.Mock()
        process.poll.return_value = 0
        return {"process": process, "item": self.row, "attempt": attempt, "owner": self.owner}

    def test_evidence_changed_while_a_finish_waited_is_blocked_not_certified(self):
        job = self._job()
        orphan = self.directory / "transition_guard"
        orphan.mkdir()
        with mock.patch.object(scheduler, "validate_done", return_value=self.receipt):
            with self.assertRaises(scheduler.LedgerContention):
                scheduler._finish(job, self.ledger, COMMIT, self.root)
            self.done.write_text('{"interaction_step": 1}')  # changed after validation
            orphan.rmdir()
            scheduler._finish(job, self.ledger, COMMIT, self.root)
        self.assertFalse((self.directory / "completed.json").exists())
        blocked = scheduler._read(self.directory / "blocked.json")
        self.assertIn("changed after validation", blocked["error"])
        self.assertEqual(self.ledger.state(self.row), "blocked")

    def test_receipt_changed_after_validation_is_blocked(self):
        job = self._job()
        with mock.patch.object(scheduler, "validate_done", return_value=self.receipt):
            original = scheduler._evidence_digests

            def evidence_then_tamper(item_):
                digests = original(item_)
                if job.get("outcome"):
                    self.receipt.write_text('{"engineering_verdict":"EDITED"}')
                return digests

            with mock.patch.object(scheduler, "_evidence_digests", side_effect=evidence_then_tamper):
                scheduler._finish(job, self.ledger, COMMIT, self.root)
        self.assertFalse((self.directory / "completed.json").exists())
        self.assertIn("changed after validation", scheduler._read(self.directory / "blocked.json")["error"])

    def test_evidence_changing_during_validation_is_blocked(self):
        job = self._job()

        def validate(*_):
            self.done.write_text('{"interaction_step": 2}')  # the run's files move under the validator
            return self.receipt

        with mock.patch.object(scheduler, "validate_done", side_effect=validate):
            scheduler._finish(job, self.ledger, COMMIT, self.root)
        self.assertFalse((self.directory / "completed.json").exists())
        self.assertIn("changed during validation", scheduler._read(self.directory / "blocked.json")["error"])

    def test_a_receipt_without_validated_evidence_is_refused(self):
        with self.assertRaisesRegex(ValueError, "validated evidence"):
            self.ledger.finish(self.row, self.owner, receipt=self.receipt)
        self.assertFalse((self.directory / "completed.json").exists())
        self.assertTrue((self.directory / "claim" / "ready").is_file())

    def test_unchanged_evidence_is_completed_with_the_validated_digests(self):
        job = self._job()
        with mock.patch.object(scheduler, "validate_done", return_value=self.receipt):
            scheduler._finish(job, self.ledger, COMMIT, self.root)
        completed = scheduler._read(self.directory / "completed.json")
        self.assertEqual(completed["evidence_sha256"], job["outcome"]["evidence"])
        self.assertEqual(completed["receipt_sha256"], job["outcome"]["receipt_sha256"])
        self.assertEqual(self.ledger.state(self.row), "complete")

    def test_reconciliation_does_not_certify_evidence_that_changed_during_it(self):
        row, receipt = _row(self.root, seed=2)
        done = Path(row["row"]["checkpoint_dir"]) / "DONE"

        def reconcile(_):
            done.write_text('{"interaction_step": 3}')
            return receipt

        ledger = scheduler.Ledger(self.root, query=lambda _: False)
        self.assertFalse(ledger.claim(row, owner(202, "later"), reconcile))
        directory = ledger._row_dir(row)
        self.assertFalse((directory / "completed.json").exists())
        self.assertIn("changed during validation", scheduler._read(directory / "blocked.json")["error"])


# --------------------------------------------------------------------------------------- 2
SBATCH_STUB = """import fcntl
import os
import sys
from pathlib import Path

log = Path(os.environ["FAKE_SBATCH_LOG"])
with Path(str(log) + ".lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    count = len(log.read_text().splitlines()) if log.exists() else 0
    fail_at = os.environ.get("FAKE_SBATCH_FAIL_AT")
    if fail_at and count >= int(fail_at):
        print("sbatch: error: simulated failure", file=sys.stderr)
        sys.exit(1)
    number = count + 1
    with log.open("a") as stream:
        stream.write(f"{number} {' '.join(sys.argv[1:])}\\n")
    print(5000 + number)
"""


class SubmissionReservationTest(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        base = Path(self._directory.name)
        self.checkout = base / "checkout"
        (self.checkout / "main").mkdir(parents=True)
        (self.checkout / "main" / "README").write_text("x")
        git = ["git", "-C", str(self.checkout), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(["git", "init", "-q", str(self.checkout)], check=True)
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(git + ["commit", "-q", "-m", "x"], check=True)
        self.sha = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
        self.qual = base / "qual"
        self.hb = base / "hb"
        (self.hb / "bin").mkdir(parents=True)
        python = self.hb / "bin" / "python"
        python.write_text("#!/bin/sh\n")
        python.chmod(0o755)
        self.bin = base / "bin"
        self.bin.mkdir()
        stub = self.bin / "sbatch"
        stub.write_text(f"#!{sys.executable}\n" + SBATCH_STUB)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        self.log = base / "sbatch.log"
        self.script = base / "submit.sh"
        self.script.write_text(commands.render(self.sha, str(self.checkout), str(self.qual), str(self.hb)))

    def tearDown(self):
        self._directory.cleanup()

    def _run(self, **env):
        environment = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                           FAKE_SBATCH_LOG=str(self.log), **env)
        return subprocess.run(["bash", str(self.script)], capture_output=True, text=True, env=environment)

    def _calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_a_repeated_run_never_resubmits(self):
        first = self._run()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(len(self._calls()), 7)
        second = self._run()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(len(self._calls()), 7)
        status = commands.reconcile_status(str(self.qual))
        self.assertEqual({s["state"] for s in status.values()}, {"SUBMITTED"})
        self.assertEqual(sorted(int(s["job_id"]) for s in status.values()), list(range(5001, 5008)))

    def test_concurrent_runs_submit_each_job_once(self):
        environment = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", FAKE_SBATCH_LOG=str(self.log))
        runs = [subprocess.Popen(["bash", str(self.script)], env=environment, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True) for _ in range(3)]
        for run in runs:
            run.communicate(timeout=120)
        self.assertEqual(len(self._calls()), 7)
        names = [line.split("--job-name=")[1].split()[0] for line in self._calls()]
        self.assertEqual(len(set(names)), 7)

    def test_a_partial_submission_requires_explicit_reconciliation(self):
        failed = self._run(FAKE_SBATCH_FAIL_AT="2")  # the third sbatch fails
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(len(self._calls()), 2)
        status = commands.reconcile_status(str(self.qual))
        states = [s["state"] for s in status.values()]
        self.assertEqual(states.count("SUBMITTED"), 2)
        self.assertEqual(states.count("PARTIAL"), 1)
        again = self._run()
        self.assertEqual(again.returncode, 3, again.stderr)
        self.assertIn("reconcile", again.stderr)
        self.assertEqual(len(self._calls()), 2)  # nothing submitted past a partial reservation
        partial = next(n for n, s in status.items() if s["state"] == "PARTIAL")
        commands.reconcile(str(self.qual), partial, release=True, reason="sacct shows no job")
        self.assertEqual(commands.reconcile_status(str(self.qual))[partial]["state"], "NOT_SUBMITTED")
        done = self._run()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(len(self._calls()), 7)
        released = list((self.qual / "submissions" / ".released").iterdir())
        self.assertEqual(len(released), 1)  # the partial reservation is kept as evidence

    def test_reconcile_can_record_a_job_found_in_slurm_and_refuses_bad_ids(self):
        self._run(FAKE_SBATCH_FAIL_AT="0")
        status = commands.reconcile_status(str(self.qual))
        partial = next(n for n, s in status.items() if s["state"] == "PARTIAL")
        with self.assertRaises(ValueError):
            commands.reconcile(str(self.qual), partial, job_id="12x", reason="r")
        commands.reconcile(str(self.qual), partial, job_id="777", reason="found in sacct")
        self.assertEqual(commands.reconcile_status(str(self.qual))[partial],
                         {"state": "SUBMITTED", "job_id": "777"})
        with self.assertRaises(ValueError):  # a submitted entry is never re-reconciled
            commands.reconcile(str(self.qual), partial, release=True, reason="r")


# --------------------------------------------------------------------------------------- 3
def _tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        digest.update(str(path.relative_to(root)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


class RepeatableVerifyTest(unittest.TestCase):
    def test_verify_twice_gives_the_same_report_and_leaves_the_evidence_untouched(self):
        no_jax = ("import sys, runpy; sys.modules['jax'] = None; sys.argv = ['x'] + sys.argv[1:]; "
                  f"runpy.run_path({str(MAIN / 'scripts/exp12_shared_fs_claim_check.py')!r}, run_name='__main__')")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "race"
            root.mkdir()
            procs = [subprocess.Popen([sys.executable, "-c", no_jax, "worker", "--root", str(root),
                                       "--rank", str(rank), "--items", "60"], stderr=subprocess.PIPE, text=True)
                     for rank in (0, 1)]
            for proc in procs:
                self.assertEqual(proc.wait(timeout=300), 0, proc.stderr.read())
            before = _tree_digest(root)
            first = claim_check.verify(root, require_distinct_hosts=False)
            second = claim_check.verify(root, require_distinct_hosts=False)
            self.assertEqual(_tree_digest(root), before)
            self.assertEqual(first, second)
            self.assertEqual(first["verdict"], "PASS", first["failures"])
            self.assertTrue(first["recovery"]["preserved"])
            out = Path(directory) / "verdict.json"
            self.assertEqual(claim_check.main_verify(root, out, require_distinct_hosts=False), 0)
            self.assertEqual(json.loads(out.read_text()), first)
            with self.assertRaises(ValueError):  # a verdict is never written inside the evidence
                claim_check.main_verify(root, root / "verdict.json", require_distinct_hosts=False)


# --------------------------------------------------------------------------------------- 4
def _record(name, mode, deviation=1e-7, commit=COMMIT, backend="gpu", **extra):
    return {"name": name, "deviation": deviation, "tolerance": None, "commit": commit, "backend": backend,
            "device_kind": "NVIDIA A100-SXM4-40GB", "mode": mode, **extra}


def _numerics(root, commit=COMMIT, edit=None):
    root.mkdir()
    (root / "submission.json").write_text(json.dumps({"commit": commit, "job_id": "1"}))
    for mode, filename in (("tf32", "numerics_tf32.jsonl"), ("highest_deterministic", "numerics_deterministic.jsonl")):
        rows = [_record(f"diagnostics_update/{mode}", mode, updates=4),
                _record(f"diagnostics_update/{mode}", mode, updates=4, twin=True),
                _record(f"diagnostics_training/{mode}", mode, interaction_steps=400)]
        if edit:
            rows = edit(mode, rows)
        (root / filename).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


class SourceBoundNumericsTest(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.base = Path(self._directory.name)

    def tearDown(self):
        self._directory.cleanup()

    def _propose(self, name, **kw):
        return bounds.propose(_numerics(self.base / name, **kw), COMMIT)

    def test_valid_measurements_are_proposed_and_bound_to_the_source(self):
        report = self._propose("ok")
        self.assertEqual(report["status"], "PROPOSED_NOT_APPLIED", report["errors"])
        self.assertEqual(report["source_commit"], COMMIT)
        self.assertTrue(report["complete"])
        self.assertEqual(report["bounds"]["diagnostics_update/tf32"]["records"], 2)

    def _invalid(self, name, pattern, **kw):
        report = self._propose(name, **kw)
        self.assertEqual(report["status"], "INVALID")
        self.assertFalse(report["complete"])
        self.assertTrue(any(pattern in e for e in report["errors"]), report["errors"])
        self.assertNotIn("proposed_bound", json.dumps(report["bounds"]))

    def test_a_job_from_another_commit_is_rejected(self):
        self._invalid("job", "submission commit", commit=OTHER)

    def test_mixed_record_commits_are_rejected(self):
        self._invalid("mixed", "commit", edit=lambda m, rows: rows[:1] + [{**rows[1], "commit": OTHER}] + rows[2:])

    def test_cpu_backend_or_wrong_device_is_rejected(self):
        self._invalid("cpu", "backend", edit=lambda m, rows: [{**r, "backend": "cpu"} for r in rows])
        self._invalid("device", "device", edit=lambda m, rows: [{**r, "device_kind": "Tesla V100"} for r in rows])

    def test_a_record_in_the_wrong_mode_file_is_rejected(self):
        swapped = {"tf32": "highest_deterministic", "highest_deterministic": "tf32"}
        self._invalid("mode", "mode", edit=lambda m, rows: [{**rows[0], "mode": swapped[m]}] + rows[1:])

    def test_missing_or_extra_records_are_rejected(self):
        self._invalid("missing", "expected 2", edit=lambda m, rows: rows[1:])
        self._invalid("extra", "expected 1", edit=lambda m, rows: rows + [rows[2]])

    def test_negative_nonfinite_and_nonnumeric_deviations_are_rejected(self):
        for name, value in (("negative", -1e-9), ("nan", float("nan")), ("inf", float("inf")), ("bool", True),
                            ("text", "1e-7")):
            with self.subTest(name):
                self._invalid(name, "deviation", edit=lambda m, rows, v=value: [{**rows[0], "deviation": v}] + rows[1:])

    def test_records_carry_their_provenance(self):
        sys.path.insert(0, str(MAIN / "tests"))
        import exp12_helpers

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "n.jsonl"
            with mock.patch.dict(os.environ, {"EXP12_NUMERICS_OUT": str(out), "EXP12_NUMERICS_COMMIT": COMMIT}):
                exp12_helpers.record_numerics("diagnostics_update/tf32", deviation=1e-7)
            record = json.loads(out.read_text())
        self.assertEqual(record["commit"], COMMIT)
        self.assertEqual(record["backend"], "cpu")  # this test process
        self.assertIn("device_kind", record)
        self.assertIn(record["mode"], ("cpu", "tf32", "highest_deterministic"))
        job = (MAIN / "scripts/sbatch_exp12_numeric_bounds.sh").read_text()
        self.assertIn('EXP12_NUMERICS_COMMIT="$EXPECTED_COMMIT"', job)
        self.assertIn('--expected-commit "$EXPECTED_COMMIT"', job)


# --------------------------------------------------------------------------------------- 5
class UnrecordedFinishDoesNotIdleTheAllocationTest(unittest.TestCase):
    def test_a_job_is_abandoned_after_its_bounded_finish_attempts(self):
        from concurrent.futures import Future

        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            job = {"item": {"id": "a"}, "attempt": attempt, "outcome": {"error": "exit 1"}}
            unrecorded = []
            for n in range(scheduler.FINISH_MAX_ATTEMPTS):
                future = Future()
                future.set_exception(scheduler.LedgerContention("busy"))
                scheduler._collect_finished({future: job}, unrecorded, 100.0 * n)
                if n < scheduler.FINISH_MAX_ATTEMPTS - 1:
                    self.assertEqual(unrecorded, [job])
                    unrecorded.clear()
            self.assertEqual(unrecorded, [])
            self.assertTrue((attempt / "finish_unrecorded.json").is_file())
        bound = scheduler.FINISH_MAX_ATTEMPTS * (scheduler.FINISH_GUARD_TIMEOUT_SECONDS + scheduler.FINISH_RETRY_SECONDS)
        self.assertLessEqual(bound, 30 * 60)  # at most half an hour per job, not until walltime

    def test_an_idle_allocation_does_not_wait_for_unrecorded_outcomes(self):
        self.assertTrue(scheduler._idle(active=[], pending=[], validating={}, unrecorded=[{"item": {"id": "a"}}]))
        self.assertFalse(scheduler._idle(active=[1], pending=[], validating={}, unrecorded=[]))
        self.assertFalse(scheduler._idle(active=[], pending=[], validating={1: 1}, unrecorded=[]))


if __name__ == "__main__":
    unittest.main()


# ------------------------------------------------------------- follow-up review (f880f1c)
class CrossCommitReservationTest(SubmissionReservationTest):
    """Reservations made for one commit must never count as submissions for another."""

    # Only the cross-commit case runs here; the inherited tests run in the parent class.
    test_a_repeated_run_never_resubmits = None
    test_concurrent_runs_submit_each_job_once = None
    test_a_partial_submission_requires_explicit_reconciliation = None
    test_reconcile_can_record_a_job_found_in_slurm_and_refuses_bad_ids = None

    def test_reservations_from_another_commit_fail_closed(self):
        first = self._run()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(len(self._calls()), 7)
        (self.checkout / "main" / "README").write_text("y")
        git = ["git", "-C", str(self.checkout), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(git + ["commit", "-q", "-am", "y"], check=True)
        other = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
        self.script.write_text(commands.render(other, str(self.checkout), str(self.qual), str(self.hb)))
        again = self._run()
        self.assertEqual(again.returncode, 3, again.stdout + again.stderr)
        self.assertIn("another commit", again.stderr)
        self.assertNotIn("already submitted", again.stdout)
        self.assertEqual(len(self._calls()), 7)  # nothing reused, nothing submitted
        status = commands.reconcile_status(str(self.qual), commit=other)
        self.assertEqual({s["state"] for s in status.values()}, {"FOREIGN"})
        self.assertEqual({s["state"] for s in commands.reconcile_status(str(self.qual), commit=self.sha).values()},
                         {"SUBMITTED"})
        with self.assertRaises(ValueError):
            commands.reconcile(str(self.qual), next(iter(status)), release=True, reason="r", commit=other)


class UnrecordedFailureRecoveryTest(unittest.TestCase):
    """A failed run whose outcome was never recorded must not be reclaimed as pending."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.row = item(self.root)
        self.first = owner(101, "first")
        live = scheduler.Ledger(self.root, query=lambda _: True)
        self.assertTrue(live.claim(self.row, self.first, lambda _: None))
        self.attempt = self.root / "attempts" / self.row["kind"] / self.row["id"] / "first"
        self.attempt.mkdir(parents=True)
        self.later = scheduler.Ledger(self.root, query=lambda _: False)  # allocation 101 has ended

    def tearDown(self):
        self._directory.cleanup()

    def _record(self, returncode, outcome):
        scheduler._write_new(self.attempt / "exit.json", {"returncode": returncode, "ended_at": 0.0})
        job = {"item": self.row, "attempt": self.attempt, "outcome": outcome}
        scheduler._abandon_unrecorded([job])

    def test_an_unrecorded_failure_is_blocked_for_review_retry(self):
        self._record(1, {"error": f"exit 1; inspect {self.attempt}"})
        reconcile = mock.Mock(return_value=None)
        self.assertFalse(self.later.claim(self.row, owner(202, "later"), reconcile))
        reconcile.assert_not_called()
        directory = self.later._row_dir(self.row)
        blocked = scheduler._read(directory / "blocked.json")
        self.assertIn("unrecorded failure", blocked["error"])
        self.assertIn("exit 1", blocked["error"])
        self.assertEqual(self.later.state(self.row), "blocked")
        self.assertFalse((directory / "claim").exists())
        self.assertEqual(len(list((directory / "retired_claims").iterdir())), 1)  # old claim kept
        self.assertFalse(self.later.claim(self.row, owner(303, "again"), reconcile))  # stays blocked

    def test_a_nonzero_exit_without_a_finish_record_is_also_blocked(self):
        scheduler._write_new(self.attempt / "exit.json", {"returncode": 9, "ended_at": 0.0})
        self.assertFalse(self.later.claim(self.row, owner(202, "later"), lambda _: None))
        self.assertIn("exit 9", scheduler._read(self.later._row_dir(self.row) / "blocked.json")["error"])

    def test_an_unrecorded_success_is_still_reconciled_from_its_evidence(self):
        evidence(Path(self.row["row"]["checkpoint_dir"]))
        receipt = self.root / "receipt.json"
        receipt.write_text('{"engineering_verdict":"COMPLETE"}')
        self._record(0, scheduler.bound_receipt(self.row, receipt))
        self.assertFalse(self.later.claim(self.row, owner(202, "later"), lambda _: receipt))
        self.assertEqual(self.later.state(self.row), "complete")


# ------------------------------------------------------------- follow-up review (27561ee)
class ReservationMetadataTest(SubmissionReservationTest):
    """Only reserved.json's own top-level commit binds a reservation; anything else fails closed."""

    test_a_repeated_run_never_resubmits = None
    test_concurrent_runs_submit_each_job_once = None
    test_a_partial_submission_requires_explicit_reconciliation = None
    test_reconcile_can_record_a_job_found_in_slurm_and_refuses_bad_ids = None

    def _plant(self, text, job_id="4242"):
        entry = self.qual / "submissions" / "identity_D4W1536_dog-run"
        entry.mkdir(parents=True)
        (entry / "reserved.json").write_text(text)
        if job_id is not None:
            (entry / "job_id").write_text(job_id + "\n")
        return entry

    def _refused(self, text, job_id="4242"):
        self._plant(text, job_id)
        result = self._run()
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertNotIn("already submitted", result.stdout)
        self.assertIn("FOREIGN", result.stderr)
        self.assertEqual(self._calls(), [])
        status = commands.reconcile_status(str(self.qual), commit=self.sha)["identity_D4W1536_dog-run"]
        self.assertEqual(status["state"], "FOREIGN")

    def test_a_nested_occurrence_of_the_commit_does_not_count(self):
        self._refused(json.dumps({"name": "identity_D4W1536_dog-run", "commit": OTHER,
                                  "note": {"commit": self.sha}}, separators=(",", ":")))

    def test_a_json_array_does_not_count(self):
        self._refused(json.dumps([{"name": "identity_D4W1536_dog-run", "commit": self.sha}],
                                 separators=(",", ":")))

    def test_malformed_or_missing_metadata_fails_closed(self):
        self._refused('{"commit":"%s"' % self.sha)  # truncated JSON containing the commit

    def test_missing_reservation_metadata_fails_closed(self):
        entry = self.qual / "submissions" / "identity_D4W1536_dog-run"
        entry.mkdir(parents=True)
        (entry / "job_id").write_text("4242\n")
        result = self._run()
        self.assertEqual(result.returncode, 3)
        self.assertEqual(self._calls(), [])

    def test_a_reservation_for_another_name_does_not_count(self):
        self._refused(json.dumps({"name": "numerics_01", "commit": self.sha}, separators=(",", ":")))

    def test_a_valid_reservation_is_still_skipped(self):
        self._plant(json.dumps({"name": "identity_D4W1536_dog-run", "commit": self.sha}))
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already submitted: identity_D4W1536_dog-run (job 4242)", result.stdout)
        self.assertEqual(len(self._calls()), 6)


class MalformedUnrecordedFailureTest(UnrecordedFailureRecoveryTest):
    """Malformed attempt records always block the row for review; never an exception or retry."""

    test_an_unrecorded_failure_is_blocked_for_review_retry = None
    test_a_nonzero_exit_without_a_finish_record_is_also_blocked = None
    test_an_unrecorded_success_is_still_reconciled_from_its_evidence = None

    def _blocked(self, files):
        for name, content in files.items():
            path = self.attempt / name
            if content is None:
                path.mkdir()  # unreadable as a file
            elif isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content)
        reconcile = mock.Mock(return_value=None)
        self.assertFalse(self.later.claim(self.row, owner(202, "later"), reconcile))
        reconcile.assert_not_called()
        directory = self.later._row_dir(self.row)
        self.assertEqual(self.later.state(self.row), "blocked")
        self.assertFalse((directory / "claim").exists())
        return scheduler._read(directory / "blocked.json")["error"]

    def test_malformed_records_block_the_row(self):
        cases = {
            "unrecorded array": {"finish_unrecorded.json": "[1, 2]"},
            "outcome array": {"finish_unrecorded.json": '{"outcome": ["error"]}'},
            "outcome string": {"finish_unrecorded.json": '{"outcome": "exit 1"}'},
            "outcome missing": {"finish_unrecorded.json": '{"id": "x"}'},
            "receipt not a path": {"finish_unrecorded.json": '{"outcome": {"receipt": 7}}'},
            "unrecorded not JSON": {"finish_unrecorded.json": "{oops"},
            "unrecorded not UTF-8": {"finish_unrecorded.json": b"\xff\xfe\x00"},
            "unrecorded a directory": {"finish_unrecorded.json": None},
            "exit array": {"exit.json": "[0]"},
            "exit code string": {"exit.json": '{"returncode": "0"}'},
            "exit code bool": {"exit.json": '{"returncode": false}'},
            "exit code float": {"exit.json": '{"returncode": 0.0}'},
            "exit code missing": {"exit.json": '{"ended_at": 1}'},
            "exit a directory": {"exit.json": None},
        }
        for label, files in cases.items():
            with self.subTest(label):
                self.tearDown()
                self.setUp()
                error = self._blocked(files)
                self.assertIn("unrecorded failure", error)
