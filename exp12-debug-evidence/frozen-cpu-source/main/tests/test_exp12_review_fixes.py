"""Regressions for the 2026-10-13 adversarial-review engineering fixes (scheduler, state, fork IO)."""

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import exp12_dynamic_scheduler as scheduler


def item(root, arch="D4W1536", kind="exp1", env="hopper-hop", group="dmc_medium"):
    return {"id": f"{kind}_{arch}_{env}", "kind": kind,
            "row": {"architecture": arch, "environment": env, "group": group, "seed": 1,
                    "config_hash": "cfg", "checkpoint_dir": str(Path(root) / "run"),
                    "run_key": f"exp1_{arch}_{env}_seed1"}}


def completed(stdout="", stderr="", code=0):
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr=stderr)


class SlurmLivenessTest(unittest.TestCase):
    def check(self, squeue, sacct=None):
        calls = [squeue] + ([sacct] if sacct is not None else [])
        with mock.patch.object(scheduler.subprocess, "run", side_effect=calls) as run:
            result = scheduler.slurm_alive("123")
        return result, run.call_count

    def test_running_and_pending_jobs_are_alive(self):
        for state in ("RUNNING", "PENDING", "COMPLETING", "SUSPENDED", "CONFIGURING"):
            self.assertEqual(self.check(completed(state + "\n")), (True, 1))

    def test_listed_terminal_jobs_are_ended(self):
        for state in ("COMPLETED", "TIMEOUT", "FAILED", "CANCELLED", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"):
            self.assertEqual(self.check(completed(state + "\n")), (False, 1))
        self.assertEqual(self.check(completed("")), (False, 1))

    def test_purged_job_id_is_ended_not_a_crash(self):
        purged = completed(stderr="slurm_load_jobs error: Invalid job id specified\n", code=1)
        self.assertEqual(self.check(purged), (False, 1))

    def test_squeue_failure_falls_back_to_sacct_and_refuses_when_unknown(self):
        down = completed(stderr="slurm_load_jobs error: Socket timed out\n", code=1)
        self.assertEqual(self.check(down, completed("RUNNING\n")), (True, 2))
        self.assertEqual(self.check(down, completed("CANCELLED by 1\n")), (False, 2))
        with self.assertRaises(RuntimeError):
            self.check(down, completed(code=1))

    def test_unrecognised_state_is_treated_as_alive(self):
        self.assertEqual(self.check(completed("SOMETHING_NEW\n")), (True, 1))


class ClaimContainmentTest(unittest.TestCase):
    def test_unknown_liveness_skips_the_row_without_killing_the_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = item(root)
            first = scheduler.Ledger(root, query=lambda job: True)
            self.assertTrue(first.claim(row, {"job_id": "1", "token": "a", "host": "h", "pid": 1}, lambda _: None))

            def unknown(job):
                raise RuntimeError("squeue and sacct failed; claim recovery refused")

            ledger = scheduler.Ledger(root, query=unknown)
            approval = {"expected_commit": "c" * 40}
            scheduler._CLAIM_BACKOFF.clear()
            with mock.patch.object(scheduler, "validate_resume") as resume:
                self.assertIsNone(scheduler._start(row, "0", approval, root, ledger, "2"))
                self.assertIsNone(scheduler._start(row, "0", approval, root, ledger, "2"))  # backoff
                resume.assert_not_called()
            self.assertGreater(scheduler._CLAIM_BACKOFF[row["id"]], time.time())
            self.assertEqual(ledger.state(row), "pending")  # never blocked or reassigned

    def test_ended_owner_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = item(root)
            scheduler.Ledger(root, query=lambda job: True).claim(
                row, {"job_id": "1", "token": "a", "host": "h", "pid": 1}, lambda _: None)
            ledger = scheduler.Ledger(root, query=lambda job: False)
            self.assertTrue(ledger.claim(row, {"job_id": "2", "token": "b", "host": "h", "pid": 2}, lambda _: None))
            self.assertEqual(len(list((ledger._row_dir(row) / "retired_claims").iterdir())), 1)

    def test_orphaned_transition_guard_is_reported_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = item(root)
            ledger = scheduler.Ledger(root, query=lambda job: False)
            guard = ledger._row_dir(row) / "transition_guard"
            guard.mkdir(parents=True)
            old = time.time() - 2 * scheduler.STALE_GUARD_SECONDS
            os.utime(guard, (old, old))
            scheduler._WARNED_GUARDS.clear()
            with mock.patch("sys.stderr") as err:
                self.assertFalse(ledger.claim(row, {"job_id": "3", "token": "c"}, lambda _: None))
                self.assertFalse(ledger.claim(row, {"job_id": "3", "token": "c"}, lambda _: None))
            warnings = [c for c in err.write.call_args_list if "transition guard" in str(c)]
            self.assertEqual(len(warnings), 1)
            self.assertTrue(guard.exists())  # never removed automatically


class EnvironmentGroupsTest(unittest.TestCase):
    def test_every_manifest_group_has_an_estimate_slot_and_is_admissible(self):
        from generate_manifest import EXP12_BUDGETS

        self.assertEqual(set(scheduler.GROUPS), set(EXP12_BUDGETS))
        profile = {"cpus": 1, "host_ram_gb": 1, "gpu_memory_gb": 1, "min_interactions_per_second": 1,
                   "estimated_minutes_by_group": {g: 10 for g in scheduler.GROUPS}}
        admission = {"stop_margin_minutes": 1, "host_headroom_gb": 1, "gpu_headroom_gb": 1,
                     "architectures": {a: profile for a in scheduler.WEIGHTS}}
        with tempfile.TemporaryDirectory() as directory:
            for group in scheduler.GROUPS:
                self.assertTrue(scheduler.can_admit(
                    item(directory, group=group), "0", [], admission,
                    {"cpus_per_node": 64, "memory_gb_per_node": 256}, time.time() + 3600, (40960, 0), 100))
        template = json.loads((Path(scheduler.MAIN) / "configs/launch_readiness/"
                               "dynamic_allocation_approval.template.json").read_text())
        for arch, prof in template["admission"]["architectures"].items():
            self.assertEqual(set(prof["estimated_minutes_by_group"]), set(scheduler.GROUPS))
            if "estimated_arm_minutes_by_group" in prof:
                self.assertEqual(set(prof["estimated_arm_minutes_by_group"]), set(scheduler.GROUPS))

    def test_prepare_creates_submission_and_log_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text("{}")
            rows = [{"environment": "dog-run", "architecture": "D2W512", "seed": n, "config_hash": f"c{n}"}
                    for n in range(195)]
            plan = {"stage": "parents", "commit": "a" * 40, "root": str(root / "campaign"), "entries": rows}
            with mock.patch.object(scheduler, "_check_plan", return_value=plan):
                scheduler.prepare(plan_path, root / "queue")
            self.assertTrue((root / "queue" / "slurm").is_dir())
            for name in scheduler.ALLOCATIONS:
                self.assertTrue((root / "queue" / "submissions" / name).is_dir())


class DurableStateAndAtomicNpzTest(unittest.TestCase):
    def test_state_is_flushed_before_latest_names_it(self):
        from experiments.exp12 import state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "step_000000010_aaaaaaaa"
            old.mkdir()
            new = root / "step_000000020_bbbbbbbb"
            (new / "agent_ckpt").mkdir(parents=True)
            (new / "buffer.npz").write_bytes(b"x")
            (new / "agent_ckpt" / "d").write_bytes(b"y")
            events = []
            real_fsync = os.fsync

            def fsync(fd):
                events.append(("fsync", os.readlink(f"/proc/self/fd/{fd}")))
                real_fsync(fd)

            real_write = state.atomic_write_text

            def write(path, text):
                events.append(("latest", str(path)))
                real_write(path, text)

            with mock.patch.object(state.os, "fsync", side_effect=fsync), \
                    mock.patch.object(state, "atomic_write_text", side_effect=write):
                state.commit_state_dir(root, new)
            latest_at = next(i for i, e in enumerate(events) if e[0] == "latest")
            flushed_before = {e[1] for e in events[:latest_at]}
            for path in (new / "buffer.npz", new / "agent_ckpt" / "d", new, new / "agent_ckpt", root):
                self.assertIn(str(path), flushed_before)
            self.assertIn(("fsync", str(root)), events[latest_at:])
            self.assertEqual((root / "LATEST").read_text(), new.name)
            self.assertFalse(old.exists())

    def test_save_npz_replaces_atomically(self):
        from experiments.exp12 import fork

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "check1_control.npz"
            fork.save_npz(path, {"q": np.arange(3.0)})
            fork.save_npz(path, {"q": np.arange(4.0)})
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["check1_control.npz"])
            self.assertTrue(np.array_equal(fork.load_npz(path)["q"], np.arange(4.0)))
            with mock.patch.object(np, "savez", side_effect=RuntimeError("disk full")):
                with self.assertRaises(RuntimeError):
                    fork.save_npz(path, {"q": np.zeros(1)})
            self.assertTrue(np.array_equal(fork.load_npz(path)["q"], np.arange(4.0)))  # old file intact
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["check1_control.npz"])


if __name__ == "__main__":
    unittest.main()
