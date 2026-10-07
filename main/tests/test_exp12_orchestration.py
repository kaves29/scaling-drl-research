"""Orchestration failures must not duplicate work or imply scientific nulls."""

import json
import multiprocessing
import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from scripts import claim_launcher as claims

REPO = Path(__file__).resolve().parents[1]


def process_claim(root, barrier, release, queue):
    barrier.wait(10)
    queue.put(claims._try_claim(root, "none", claims.socket.gethostname()))
    release.wait(10)


class ClaimOwnershipTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.claim = self.root / claims.CLAIM_DIR_NAME

    def old_claim(self, job="old", pid=111, host="old-host"):
        self.claim.mkdir()
        (self.claim / claims.OWNER_FILE_NAME).write_text(
            json.dumps(dict(slurm_job_id=job, hostname=host, pid=pid))
        )

    def test_stale_contenders_cannot_both_own_the_claim(self):
        self.old_claim()
        read, resume = threading.Event(), threading.Event()
        original = claims._read_owner
        results = {}

        def paused(path):
            owner = original(path)
            if (
                threading.current_thread().name == "stale-B"
                and owner["slurm_job_id"] == "old"
            ):
                read.set()
                if not resume.wait(10):
                    raise RuntimeError("test release timed out")
            return owner

        def contender():
            results["B"] = claims._try_claim(str(self.root), "B", "host-B")

        with mock.patch.object(
            claims, "_read_owner", side_effect=paused
        ), mock.patch.object(
            claims,
            "_claim_owner_alive",
            side_effect=lambda owner: owner["slurm_job_id"] != "old",
        ):
            thread = threading.Thread(name="stale-B", target=contender)
            thread.start()
            try:
                self.assertTrue(read.wait(10))
                results["A"] = claims._try_claim(str(self.root), "A", "host-A")
            finally:
                resume.set()
                thread.join(10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(sum(results.values()), 1, results)
        winner = next(name for name, owns in results.items() if owns)
        self.assertEqual(original(self.claim)["slurm_job_id"], winner)

    def test_single_stale_claim_is_still_recoverable(self):
        self.old_claim()
        with mock.patch.object(claims, "_claim_owner_alive", return_value=False):
            self.assertTrue(claims._try_claim(str(self.root), "new", "host-new"))
        self.assertEqual(claims._read_owner(self.claim)["slurm_job_id"], "new")

    def test_live_or_unpublished_owner_is_not_stolen(self):
        self.old_claim()
        with mock.patch.object(claims, "_claim_owner_alive", return_value=True):
            self.assertFalse(claims._try_claim(str(self.root), "new", "host-new"))
        (self.claim / claims.OWNER_FILE_NAME).unlink()
        self.assertFalse(claims._try_claim(str(self.root), "new", "host-new"))

    def test_foreign_release_preserves_new_owner(self):
        self.old_claim(job="new", pid=222, host="new-host")
        original = (self.claim / claims.OWNER_FILE_NAME).read_bytes()
        with mock.patch.object(
            claims.os, "getpid", return_value=111
        ), mock.patch.object(claims.socket, "gethostname", return_value="old-host"):
            claims._release_claim(str(self.root))
        self.assertEqual((self.claim / claims.OWNER_FILE_NAME).read_bytes(), original)

    def test_own_release_removes_only_owned_claim(self):
        self.old_claim(
            job=os.environ.get("SLURM_JOB_ID", "none"),
            pid=os.getpid(),
            host=claims.socket.gethostname(),
        )
        claims._release_claim(str(self.root))
        self.assertFalse(self.claim.exists())

    def test_orphan_transition_guard_blocks_without_deleting_evidence(self):
        guard = self.root / "claim_guard"
        guard.mkdir()
        self.assertFalse(claims._try_claim(str(self.root), "new", "host-new"))
        self.assertTrue(guard.is_dir())
        self.assertFalse(self.claim.exists())

    def test_stale_recovery_is_exclusive_across_processes(self):
        self.old_claim(job="none")
        ctx = multiprocessing.get_context("spawn")
        barrier, release, queue = ctx.Barrier(3), ctx.Event(), ctx.Queue()
        children = [
            ctx.Process(
                target=process_claim, args=(str(self.root), barrier, release, queue)
            )
            for _ in range(2)
        ]
        for child in children:
            child.start()
        try:
            barrier.wait(10)
            outcomes = [queue.get(timeout=10) for _ in children]
        finally:
            release.set()
            for child in children:
                child.join(10)
                if child.is_alive():
                    child.terminate()
                    child.join()
        self.assertEqual(sum(outcomes), 1)
        self.assertEqual([child.exitcode for child in children], [0, 0])

    def test_failed_owner_inspection_releases_transition_guard(self):
        self.old_claim()
        with mock.patch.object(
            claims, "_claim_owner_alive", side_effect=RuntimeError("inspection")
        ):
            with self.assertRaisesRegex(RuntimeError, "inspection"):
                claims._try_claim(str(self.root), "new", "host-new")
        self.assertFalse((self.root / "claim_guard").exists())
        self.assertEqual(claims._read_owner(self.claim)["slurm_job_id"], "old")


class BlockBParentStatusTest(unittest.TestCase):
    def function(self, name):
        source = (REPO / "scripts/exp12_blockB.sh").read_text()
        marker = name + "() {"
        if marker not in source:
            return ""
        start = source.index(marker)
        end = source.index("\n}\n", start) + 3
        return source[start:end]

    def run_lanes(self, root, code, done=False, fork=False):
        env = dict(
            os.environ,
            OUT=str(root),
            PARENT_CODE=str(code),
            MAKE_DONE=str(int(done)),
            MAKE_FORK=str(int(fork)),
        )
        for lane in ("gpu0", "gpu1"):
            (root / lane).mkdir(parents=True)
        body = """
set -u
PREFLIGHT_ARCHS=""; ARM_M=last; WATCH_POLL=0
record() { printf '%s\\t%s\\t%s\\n' "$1" "$2" "$3" >> "$OUT/status.tsv"; }
remaining() { echo 1000; }
run_step() {
  echo "$2" >> "$OUT/calls.txt"
  if [ "$2" = dev_run ]; then
    mkdir -p "$OUT/gpu0/dev_run"
    [ "$MAKE_DONE" = 1 ] && touch "$OUT/gpu0/dev_run/DONE"
    if [ "$MAKE_FORK" = 1 ]; then
      mkdir -p "$OUT/gpu0/dev_run/fork"
      touch "$OUT/gpu0/dev_run/fork/FORK_READY"
    fi
    return "$PARENT_CODE"
  fi
  return 0
}
"""
        body += "\n".join(
            self.function(name)
            for name in ("parent_no_fork_status", "lane_gpu0", "lane_gpu1")
        )
        body += "\nlane_gpu0\nlane_gpu1\n"
        subprocess.run(
            ["bash", "-c", body],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        path = root / "status.tsv"
        return path.read_text().splitlines() if path.exists() else []

    def test_incomplete_parents_are_not_no_trigger_results(self):
        for code in (124, 137, 1, 0):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                rows = self.run_lanes(Path(tmp), code)
                self.assertEqual(
                    rows,
                    [
                        "gpu0\tpositive_control\tBLOCKED_PARENT_INCOMPLETE",
                        "gpu1\tarm_injected\tBLOCKED_PARENT_INCOMPLETE",
                    ],
                )

    def test_completed_parent_without_fork_retains_operational_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.run_lanes(Path(tmp), 0, done=True)
            self.assertEqual(
                rows,
                [
                    "gpu0\tpositive_control\tSKIPPED_NO_TRIGGER",
                    "gpu1\tarm_injected\tSKIPPED_NO_TRIGGER",
                ],
            )

    def test_existing_ready_fork_preserves_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(self.run_lanes(root, 124, fork=True), [])
            self.assertEqual(
                (root / "calls.txt").read_text().splitlines(),
                ["dev_run", "positive_control", "arm_injected"],
            )


if __name__ == "__main__":
    unittest.main()
