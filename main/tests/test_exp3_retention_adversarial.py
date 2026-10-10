"""Independent safety assertions against Claude's logical-retention correction.

Known failures are left as ordinary failures, not converted into passing checks.
Run from main/: python -m unittest -v tests.test_exp3_retention_adversarial
"""

import concurrent.futures
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.exp12.state import state_differences
from experiments.exp12.trainer import Exp12Trainer
from experiments.exp3 import retention
from experiments.exp3.artifacts import state_fingerprint
from experiments.exp3.recording import finalize_capture, record_exp2_scope
from experiments.exp3.streams import StreamReader, StreamWriter
from tests import test_exp3_capture_review as fixtures
from tests.exp12_helpers import compose, tiny_overrides


class RetentionSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ResaveRetentionTest.setUpClass()
        cls.first = fixtures.ResaveRetentionTest.first
        cls.resaved = fixtures.ResaveRetentionTest.resaved

    @classmethod
    def tearDownClass(cls):
        fixtures.ResaveRetentionTest.tearDownClass()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def clone(self, name):
        return Path(shutil.copytree(self.first, self.root / name))

    def rewrite_meta(self, state, change):
        path = state / "meta.pkl"
        value = pickle.loads(path.read_bytes())
        change(value)
        path.write_bytes(pickle.dumps(value, protocol=4))

    def assert_conflict(self, first, second, mode):
        keep = self.root / f"keep_{mode}"
        target = retention.retain_state(first, keep, 20, mode)
        retention.retain_state(second, keep, 20, mode)
        self.assertTrue(list(keep.glob("CONFLICT_*.json")))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            retention.read_snapshot(target)

    def test_equivalent_pickle_and_replay_reencoding_preserves_original(self):
        second = self.clone("reencoded")
        meta = second / "meta.pkl"
        meta.write_bytes(pickle.dumps(pickle.loads(meta.read_bytes()), protocol=3))
        with np.load(second / "buffer.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        np.savez_compressed(second / "buffer.npz", **arrays)
        self.assertEqual(state_differences(self.first, second), [])
        for mode in ("omit", "retain"):
            keep = self.root / mode
            target = retention.retain_state(self.first, keep, 20, mode)
            before = (target / retention.SNAPSHOT).read_bytes()
            retention.retain_state(second, keep, 20, mode)
            self.assertFalse(list(keep.glob("CONFLICT_*.json")))
            self.assertEqual(len(list(keep.glob("EQUIVALENT_*.json"))), 1)
            self.assertEqual((target / retention.SNAPSHOT).read_bytes(), before)
            retention.read_snapshot(target)

    def test_same_step_counter_difference_is_a_conflict(self):
        second = self.clone("counter")
        self.rewrite_meta(
            second, lambda m: m.update(update_counter=m["update_counter"] + 1)
        )
        for mode in ("omit", "retain"):
            self.assert_conflict(self.first, second, mode)

    def test_incompatible_injection_provenance_is_a_conflict(self):
        second = self.clone("provenance")
        self.rewrite_meta(
            second,
            lambda m: m["extra_state"].update(injection={"m": "last", "seed": 999}),
        )
        for mode in ("omit", "retain"):
            self.assert_conflict(self.first, second, mode)

    def test_normalization_dtype_difference_is_a_conflict(self):
        second = self.clone("dtype")
        path = second / "obs_rms.pkl"
        value = pickle.loads(path.read_bytes())
        original = np.asarray(value["mean"])
        value["mean"] = original.astype(
            np.float32 if original.dtype != np.float32 else np.float64
        )
        path.write_bytes(pickle.dumps(value))
        for mode in ("omit", "retain"):
            self.assert_conflict(self.first, second, mode)

    def test_metadata_signed_zero_difference_is_a_conflict(self):
        first, second = self.clone("positive_zero"), self.clone("negative_zero")
        self.rewrite_meta(first, lambda m: m["extra_state"].update(signed_zero=0.0))
        self.rewrite_meta(second, lambda m: m["extra_state"].update(signed_zero=-0.0))
        for mode in ("omit", "retain"):
            self.assert_conflict(first, second, mode)

    def test_replay_difference_is_rejected_only_when_replay_is_retained(self):
        second = self.clone("replay")
        with np.load(second / "buffer.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        arrays["rewards"].flat[0] = np.nextafter(
            arrays["rewards"].flat[0], np.float32(np.inf)
        )
        np.savez(second / "buffer.npz", **arrays)
        self.assert_conflict(self.first, second, "retain")
        target = retention.retain_state(self.first, self.root / "omit", 20, "omit")
        retention.retain_state(second, self.root / "omit", 20, "omit")
        retention.read_snapshot(target)

    def test_truncated_source_agent_is_rejected(self):
        second = self.clone("truncated")
        (second / "agent_ckpt/checkpoint").write_bytes(b"broken")
        for mode in ("omit", "retain"):
            keep = self.root / mode
            retention.retain_state(self.first, keep, 20, mode)
            with self.assertRaises(Exception):
                retention.retain_state(second, keep, 20, mode)
            self.assertFalse(list(keep.glob("EQUIVALENT_*.json")))

    def test_unrestorable_sharding_is_not_logically_equivalent(self):
        second = self.clone("broken_sharding")
        (second / "agent_ckpt/_sharding").write_text("not valid JSON")
        # Establish that comparison's NumPy restoration is weaker than actual SAC restore.
        self.assertEqual(state_differences(self.first, second), [])
        cfg = compose(
            tiny_overrides(
                steps=40,
                extra=[
                    "run_role=dev",
                    "diagnostics.kl_reference_size=4",
                    "actor_grad_cosine_every=10000",
                ],
            )
        )
        trainer = Exp12Trainer(cfg, str(self.root / "restore_attempt"))
        try:
            with self.assertRaises(Exception):
                trainer.restore(second)
        finally:
            trainer.close()
        for mode in ("omit", "retain"):
            keep = self.root / mode
            retention.retain_state(self.first, keep, 20, mode)
            try:
                retention.retain_state(second, keep, 20, mode)
            except Exception:
                continue
            self.assertTrue(
                list(keep.glob("CONFLICT_*.json")),
                "unrestorable source was labelled equivalent",
            )
            self.assertFalse(list(keep.glob("EQUIVALENT_*.json")))

    def test_corrupt_retained_bytes_are_not_accepted_by_the_fast_path(self):
        for mode in ("omit", "retain"):
            keep = self.root / mode
            source = self.clone("source_" + mode)
            target = retention.retain_state(source, keep, 20, mode)
            (target / "agent_ckpt/_sharding").write_text("corrupt retained bytes")
            with self.assertRaisesRegex(ValueError, "checksum"):
                retention.read_snapshot(target)
            with self.assertRaises(
                Exception, msg="fast path returned a corrupted snapshot"
            ):
                retention.retain_state(self.first, keep, 20, mode)

    def test_corrupt_retained_bytes_are_not_accepted_by_logical_fallback(self):
        source = self.clone("snapshot_source")
        target = retention.retain_state(source, self.root / "keep", 20, "omit")
        (target / "agent_ckpt/_sharding").write_text("corrupt retained bytes")
        with self.assertRaises(Exception, msg="fallback returned a corrupted snapshot"):
            retention.retain_state(self.resaved, self.root / "keep", 20, "omit")

    def test_concurrent_different_states_never_both_succeed_on_initial_publication(
        self,
    ):
        first, second = self.clone("writer_a"), self.clone("writer_b")
        self.rewrite_meta(
            second, lambda m: m.update(update_counter=m["update_counter"] + 1)
        )
        barrier = threading.Barrier(2)
        dimensions = retention.replay_dimensions

        def pause_after_absence_check(path):
            value = dimensions(path)
            barrier.wait(timeout=20)
            return value

        keep = self.root / "race"
        with mock.patch.object(
            retention, "replay_dimensions", pause_after_absence_check
        ):
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures = [
                    pool.submit(retention.retain_state, s, keep, 20, "omit")
                    for s in (first, second)
                ]
                results = []
                for f in futures:
                    try:
                        results.append((True, f.result(timeout=30)))
                    except OSError:
                        results.append((False, None))
        self.assertEqual(sum(ok for ok, _ in results), 1)
        target = next(path for ok, path in results if ok)
        retention.read_snapshot(target)

    def test_concurrent_conflicting_resave_blocks_snapshot_use(self):
        second = self.clone("conflicting_resave")
        self.rewrite_meta(
            second, lambda m: m.update(update_counter=m["update_counter"] + 1)
        )
        keep = self.root / "concurrent_existing"
        target = retention.retain_state(self.first, keep, 20, "omit")
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(retention.retain_state, s, keep, 20, "omit")
                for s in (self.first, second)
            ]
            for f in futures:
                f.result(timeout=30)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            retention.read_snapshot(target)

    def test_independent_processes_cannot_both_publish_different_states(self):
        first, second = self.clone("process_a"), self.clone("process_b")
        self.rewrite_meta(
            second, lambda m: m.update(update_counter=m["update_counter"] + 1)
        )
        keep = self.root / "process_race"
        script = textwrap.dedent("""
            import json, sys, time
            from pathlib import Path
            from experiments.exp3 import retention
            source, keep, coordination, number = sys.argv[1:]
            coordination = Path(coordination)
            dimensions = retention.replay_dimensions
            def ready(path):
                result = dimensions(path)
                (coordination / ('ready_' + number)).write_text('ready')
                deadline = time.monotonic() + 20
                while not (coordination / 'go').exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError('process barrier expired')
                    time.sleep(.02)
                return result
            retention.replay_dimensions = ready
            try:
                target = retention.retain_state(source, keep, 20, 'omit')
                result = {'accepted': True, 'target': str(target)}
            except OSError as error:
                result = {'accepted': False, 'error': type(error).__name__}
            (coordination / ('result_' + number + '.json')).write_text(json.dumps(result))
        """)
        children = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    script,
                    str(source),
                    str(keep),
                    str(self.root),
                    str(i),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for i, source in enumerate((first, second))
        ]
        try:
            deadline = time.monotonic() + 20
            while not all((self.root / f"ready_{i}").is_file() for i in (0, 1)):
                if time.monotonic() > deadline:
                    self.fail("independent processes did not reach the controlled race")
                time.sleep(0.02)
            (self.root / "go").write_text("publish")
            for child in children:
                _, stderr = child.communicate(timeout=30)
                self.assertEqual(child.returncode, 0, stderr.decode()[-2000:])
            results = [
                json.loads((self.root / f"result_{i}.json").read_text()) for i in (0, 1)
            ]
            self.assertEqual(sum(r["accepted"] for r in results), 1)
            winner = next(r["target"] for r in results if r["accepted"])
            retention.read_snapshot(winner)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.communicate()


class RealFinalizationSafetyTest(unittest.TestCase):
    """Actual Exp1 fork and Exp2 arm; death after DONE, before stream sealing."""

    @classmethod
    def setUpClass(cls):
        fixtures.FinalSaveDeathTest.setUpClass()
        cls.base = fixtures.FinalSaveDeathTest.tmp
        cls.parent = fixtures.FinalSaveDeathTest.parent
        cls.arm = cls.base / "completed_arm"
        cls.stream = cls.base / "unfinished_stream"
        cls.keep = cls.base / "retained"
        retain = {"root": str(cls.keep), "until_step": "arm_end", "replay": "omit"}
        with mock.patch.object(
            StreamWriter, "seal", side_effect=RuntimeError("death after DONE")
        ):
            with cls().assertRaisesRegex(RuntimeError, "death after DONE"):
                with record_exp2_scope(cls.stream, 16, retain=retain):
                    fixtures.run_arm(
                        str(cls.arm),
                        fixtures.FinalSaveDeathTest.overrides,
                        str(cls.parent),
                        "injected",
                        interval=20,
                    )
        assert (cls.arm / "DONE").is_file()
        assert not StreamReader(cls.stream, require_complete=False).manifest["complete"]

    @classmethod
    def tearDownClass(cls):
        fixtures.FinalSaveDeathTest.tearDownClass()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stream_copy = Path(shutil.copytree(self.stream, self.root / "stream"))
        self.keep_copy = Path(shutil.copytree(self.keep, self.root / "retained"))
        manifest_path = self.stream_copy / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["provenance"]["retention"]["root"] = str(self.keep_copy)
        manifest_path.write_text(json.dumps(manifest))
        from experiments.exp12.state import latest_state_dir, load_meta

        self.state = latest_state_dir(self.arm / "state")
        step = load_meta(self.state)["interaction_step"]
        self.snapshot = self.keep_copy / f"step_{step:09d}"
        retention.read_snapshot(self.snapshot)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_done_finalization_is_idempotent_and_never_trains_or_saves(self):
        before = state_fingerprint(self.state)
        with mock.patch.object(
            Exp12Trainer, "train", side_effect=AssertionError("training forbidden")
        ):
            with mock.patch.object(
                Exp12Trainer,
                "save",
                side_effect=AssertionError("source save forbidden"),
            ):
                first = finalize_capture(self.stream_copy, self.arm)
                self.assertEqual(first, finalize_capture(self.stream_copy, self.arm))
        self.assertEqual(first["capture_verdict"], "COMPLETE")
        self.assertFalse(first["scientific_qualification"])
        self.assertEqual(state_fingerprint(self.state), before)
        self.assertTrue(StreamReader(self.stream_copy).manifest["complete"])
        retention.read_snapshot(self.snapshot)

    def test_real_done_finalization_rejects_changed_source_provenance(self):
        arm_copy = Path(shutil.copytree(self.arm, self.root / "wrong_source"))
        metadata = arm_copy / "run_metadata.json"
        value = json.loads(metadata.read_text())
        value["identity"]["seed"] += 1
        metadata.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "provenance"):
            finalize_capture(self.stream_copy, arm_copy)

    def test_real_done_finalization_refuses_an_active_writer(self):
        import fcntl

        lock_path = self.stream_copy.parent / (self.stream_copy.name + ".capture.lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "active writer"):
                finalize_capture(self.stream_copy, self.arm)

    def test_real_done_finalization_does_not_claim_complete_with_corrupt_retention(
        self,
    ):
        (self.snapshot / "agent_ckpt/_sharding").write_text("corrupt retained metadata")
        with self.assertRaisesRegex(ValueError, "checksum"):
            retention.read_snapshot(self.snapshot)
        with self.assertRaises(
            Exception,
            msg="COMPLETE reported although mandatory retained endpoint is corrupt",
        ):
            finalize_capture(self.stream_copy, self.arm)

    def test_real_done_finalization_does_not_claim_complete_with_conflicted_retention(
        self,
    ):
        (self.keep_copy / f"CONFLICT_{self.snapshot.name}_review.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            retention.read_snapshot(self.snapshot)
        with self.assertRaises(
            Exception, msg="COMPLETE reported although retained endpoint is conflicted"
        ):
            finalize_capture(self.stream_copy, self.arm)


if __name__ == "__main__":
    unittest.main()
