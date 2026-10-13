"""Ledger.finish under transition-guard contention: bounded retry, exactly-once, fail closed."""

import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts import exp12_dynamic_scheduler as scheduler
from tests.test_exp12_dynamic_scheduler import evidence, item, owner


def _completed_row(root, seed=1):
    row = item(root, seed=seed)
    evidence(Path(row["row"]["checkpoint_dir"]))
    receipt = root / f"receipt_{seed}.json"
    receipt.write_text('{"engineering_verdict":"COMPLETE"}')
    return row, receipt


def _hold_guard(ledger, row, seconds):
    """Another worker's claim attempt: holds the row's guard, then releases it."""
    guard = ledger._row_dir(row) / "transition_guard"
    guard.mkdir()

    def release():
        time.sleep(seconds)
        guard.rmdir()

    thread = threading.Thread(target=release)
    thread.start()
    return thread


class FinishGuardContentionTest(unittest.TestCase):
    def setUp(self):
        self._directory = TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self):
        self._directory.cleanup()

    def _files(self, ledger, row):
        return sorted(str(p.relative_to(ledger._row_dir(row))) for p in ledger._row_dir(row).rglob("*"))

    def test_transient_contention_is_waited_out_and_recorded_once(self):
        ledger = scheduler.Ledger(self.root, query=lambda _: True, finish_timeout=10)
        row, receipt = _completed_row(self.root)
        first = owner(101, "first")
        self.assertTrue(ledger.claim(row, first, lambda _: None))
        holder = _hold_guard(ledger, row, 0.4)
        started = time.monotonic()
        ledger.finish(row, first, **scheduler.bound_receipt(row, receipt))
        waited = time.monotonic() - started
        holder.join()
        self.assertGreaterEqual(waited, 0.35)
        self.assertEqual(ledger.state(row), "complete")
        directory = ledger._row_dir(row)
        self.assertFalse((directory / "claim").exists())
        self.assertFalse((directory / "transition_guard").exists())
        self.assertEqual(len(list((directory / "retired_claims").iterdir())), 1)
        self.assertEqual(scheduler._read(directory / "completed.json")["attempt_token"], "first")
        with self.assertRaisesRegex(ValueError, "claim owner changed"):  # a repeat cannot record twice
            ledger.finish(row, first, **scheduler.bound_receipt(row, receipt))

    def test_retry_exhaustion_fails_closed_without_writing_anything(self):
        ledger = scheduler.Ledger(self.root, query=lambda _: True, finish_timeout=0.3)
        row, receipt = _completed_row(self.root)
        first = owner(101, "first")
        self.assertTrue(ledger.claim(row, first, lambda _: None))
        orphan = ledger._row_dir(row) / "transition_guard"
        orphan.mkdir()
        before = self._files(ledger, row)
        started = time.monotonic()
        with self.assertRaises(scheduler.LedgerContention) as raised:
            ledger.finish(row, first, **scheduler.bound_receipt(row, receipt))
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsInstance(raised.exception, ValueError)  # existing ValueError handlers still apply
        self.assertIn("nothing recorded, claim kept", str(raised.exception))
        self.assertEqual(self._files(ledger, row), before)  # no completed/blocked, claim intact
        self.assertTrue(orphan.is_dir())  # a foreign guard is never removed
        self.assertTrue((ledger._row_dir(row) / "claim" / "ready").is_file())
        self.assertFalse(ledger.claim(row, owner(102, "second"), lambda _: None))  # still held

    def test_recovery_after_exhaustion_records_exactly_once(self):
        ledger = scheduler.Ledger(self.root, query=lambda _: True, finish_timeout=0.2)
        row, receipt = _completed_row(self.root)
        first = owner(101, "first")
        self.assertTrue(ledger.claim(row, first, lambda _: None))
        orphan = ledger._row_dir(row) / "transition_guard"
        orphan.mkdir()
        with self.assertRaises(scheduler.LedgerContention):
            ledger.finish(row, first, **scheduler.bound_receipt(row, receipt))
        orphan.rmdir()  # reviewed and cleared
        ledger.finish(row, first, **scheduler.bound_receipt(row, receipt))  # the worker's later retry
        self.assertEqual(ledger.state(row), "complete")
        self.assertEqual(len(list((ledger._row_dir(row) / "retired_claims").iterdir())), 1)

    def test_owner_gone_before_retry_is_reconciled_once_and_late_finish_is_refused(self):
        row, receipt = _completed_row(self.root)
        first = owner(101, "first")
        live = scheduler.Ledger(self.root, query=lambda _: True, finish_timeout=0.2)
        self.assertTrue(live.claim(row, first, lambda _: None))
        orphan = live._row_dir(row) / "transition_guard"
        orphan.mkdir()
        with self.assertRaises(scheduler.LedgerContention):
            live.finish(row, first, **scheduler.bound_receipt(row, receipt))
        orphan.rmdir()
        # The allocation ended without recording; a later allocation reconciles the row.
        later = scheduler.Ledger(self.root, query=lambda _: False)
        self.assertFalse(later.claim(row, owner(202, "later"), lambda _: receipt))
        completed = scheduler._read(later._row_dir(row) / "completed.json")
        self.assertTrue(completed["reconciled"])
        with self.assertRaisesRegex(ValueError, "claim owner changed"):
            live.finish(row, first, **scheduler.bound_receipt(row, receipt))
        self.assertEqual(scheduler._read(later._row_dir(row) / "completed.json"), completed)

    def test_concurrent_completion_while_other_workers_keep_claiming(self):
        rows = [_completed_row(self.root, seed) for seed in range(1, 13)]
        owners = [owner(100 + i, f"owner{i}") for i in range(len(rows))]
        ledger = scheduler.Ledger(self.root, query=lambda _: True, finish_timeout=30)
        for (row, _), who in zip(rows, owners):
            self.assertTrue(ledger.claim(row, who, lambda _: None))
        stop = threading.Event()
        wrongly_claimed, errors = [], []

        def claimer(number):
            other = scheduler.Ledger(self.root, query=lambda _: True)
            while not stop.is_set():
                for row, _ in rows:
                    try:
                        if other.claim(row, owner(900 + number, f"intruder{number}"), lambda _: None):
                            wrongly_claimed.append(row["id"])
                    except Exception as exc:
                        errors.append(repr(exc))

        def finisher(index):
            row, receipt = rows[index]
            try:
                ledger.finish(row, owners[index], **scheduler.bound_receipt(row, receipt))
            except Exception as exc:
                errors.append(repr(exc))

        claimers = [threading.Thread(target=claimer, args=(n,)) for n in range(3)]
        for thread in claimers:
            thread.start()
        time.sleep(0.05)
        finishers = [threading.Thread(target=finisher, args=(i,)) for i in range(len(rows))]
        for thread in finishers:
            thread.start()
        for thread in finishers:
            thread.join()
        stop.set()
        for thread in claimers:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(wrongly_claimed, [])
        for (row, _), who in zip(rows, owners):
            directory = ledger._row_dir(row)
            self.assertEqual(ledger.state(row), "complete")
            self.assertEqual(scheduler._read(directory / "completed.json")["attempt_token"], who["token"])
            self.assertEqual(len(list((directory / "retired_claims").iterdir())), 1)
            self.assertFalse((directory / "claim").exists() or (directory / "transition_guard").exists())
            self.assertFalse(list(directory.rglob("*.tmp")))


class WorkerFinishRetryTest(unittest.TestCase):
    """The allocation loop keeps a contended job and re-records the same outcome later."""

    def setUp(self):
        self._directory = TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.row, self.receipt = _completed_row(self.root)
        self.owner = owner(101, "first")
        self.attempt = self.root / "attempt"
        self.attempt.mkdir()
        process = mock.Mock()
        process.poll.return_value = 0
        self.job = {"process": process, "item": self.row, "attempt": self.attempt, "owner": self.owner}

    def tearDown(self):
        self._directory.cleanup()

    def test_finish_retry_records_the_first_outcome_once(self):
        ledger = scheduler.Ledger(self.root, query=lambda _: True, finish_timeout=0.1)
        self.assertTrue(ledger.claim(self.row, self.owner, lambda _: None))
        orphan = ledger._row_dir(self.row) / "transition_guard"
        orphan.mkdir()
        validate = mock.Mock(return_value=self.receipt)
        with mock.patch.object(scheduler, "validate_done", validate):
            with self.assertRaises(scheduler.LedgerContention):
                scheduler._finish(self.job, ledger, "c" * 40, self.root)
            self.assertEqual(self.job["outcome"], scheduler.bound_receipt(self.row, self.receipt))
            orphan.rmdir()
            self.job["process"].poll.return_value = 9  # must not be re-read
            self.assertTrue(scheduler._finish(self.job, ledger, "c" * 40, self.root))
        validate.assert_called_once()
        self.assertEqual(scheduler._read(self.attempt / "exit.json")["returncode"], 0)
        self.assertEqual(ledger.state(self.row), "complete")

    def test_collect_keeps_contended_jobs_and_reraises_other_errors(self):
        contended, broken, fine = Future(), Future(), Future()
        contended.set_exception(scheduler.LedgerContention("busy"))
        broken.set_exception(ValueError("claim owner changed before finish"))
        fine.set_result(True)
        jobs = [{"item": {"id": n}} for n in ("a", "b", "c")]
        validating = {contended: jobs[0], fine: jobs[2]}
        unrecorded = []
        scheduler._collect_finished(validating, unrecorded, 1000.0)
        self.assertEqual(validating, {})
        self.assertEqual(unrecorded, [jobs[0]])
        self.assertEqual(jobs[0]["retry_finish_at"], 1000.0 + scheduler.FINISH_RETRY_SECONDS)
        with self.assertRaisesRegex(ValueError, "claim owner changed"):
            scheduler._collect_finished({broken: jobs[1]}, [], 0.0)

    def test_unrecorded_outcomes_are_kept_for_review_at_the_allocation_end(self):
        self.job["outcome"] = {"receipt": self.receipt}
        scheduler._abandon_unrecorded([self.job])
        record = scheduler._read(self.attempt / "finish_unrecorded.json")
        self.assertEqual(record["outcome"], {"receipt": str(self.receipt)})

    def test_startup_failure_under_contention_does_not_stop_the_worker(self):
        ledger = scheduler.Ledger(self.root, query=lambda _: True, finish_timeout=0.1)
        approval = {"expected_commit": "c" * 40}

        def contended_resume(row, commit):
            (ledger._row_dir(row) / "transition_guard").mkdir()
            raise RuntimeError("startup failed")

        with mock.patch.object(scheduler, "validate_resume", contended_resume), \
                mock.patch.object(scheduler, "validate_done", return_value=None):
            result = scheduler._start(self.row, "0", approval, self.root, ledger, "101")
        self.assertIsNone(result)
        directory = ledger._row_dir(self.row)
        self.assertTrue((directory / "claim" / "ready").is_file())  # kept: fail closed
        self.assertFalse((directory / "blocked.json").exists())


if __name__ == "__main__":
    unittest.main()
