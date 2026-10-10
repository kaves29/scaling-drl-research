"""Crash-boundary regressions independent of successful training fixtures."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from experiments.exp3.streams import StreamWriter, StreamReader
from experiments.exp3.retention import retain_state, read_snapshot
from experiments.exp3 import gpu_checks
from experiments.exp3.recording import record_exp2_scope
from experiments.exp12.trainer import Exp12Trainer


class BoundaryTest(unittest.TestCase):
    def row(self):
        return dict(
            observation=np.zeros((1, 2), np.float32),
            next_observation=np.ones((1, 2), np.float32),
            action=np.zeros((1, 1), np.float32),
            reward=np.zeros(1, np.float32),
            terminated=np.zeros(1, np.float32),
            truncated=np.zeros(1, np.float32),
        )

    def norm(self):
        return dict(mean=np.zeros((1, 2)), var=np.ones((1, 2)), count=1.0)

    def test_identical_retention_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            state = p / "state"
            state.mkdir()
            np.savez(
                state / "buffer.npz",
                observations=np.zeros((2, 2)),
                actions=np.zeros((2, 1)),
            )
            (state / "buffer_meta.pkl").write_bytes(b"meta")
            (state / "payload").write_bytes(b"same")
            for mode in ("retain", "omit"):
                snap = retain_state(state, p / mode, 10, mode)
                retain_state(state, p / mode, 10, mode)
                self.assertEqual(list((p / mode).glob("CONFLICT_*")), [])
                self.assertIsNotNone(read_snapshot(snap))

    def test_multichunk_commit_failure_never_publishes_partial_interval(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s"
            w = StreamWriter(p, {"start_step": 10}, 1, autoflush=False)
            w.append(11, self.row(), 2, self.norm())
            w.append(12, self.row(), 2, self.norm())
            import experiments.exp3.streams as streams

            original = streams.digest
            calls = 0

            def fail_second(path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("second chunk write failed")
                return original(path)

            with patch.object(streams, "digest", fail_second):
                with self.assertRaises(OSError):
                    w.flush()
            self.assertEqual(
                StreamReader(p, require_complete=False).manifest["count"], 0
            )

    def test_failed_source_save_never_advances_stream(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / "run"
            run.mkdir()
            fork = run / "fork/state"
            fork.mkdir(parents=True)
            (fork / "saved").mkdir()
            (fork / "saved/payload").write_bytes(b"fork")
            (fork / "LATEST").write_text("saved")
            (run / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "identity": {"environment": "fixture", "seed": 1},
                        "config_hash": "fixture",
                    }
                )
            )
            t = Exp12Trainer.__new__(Exp12Trainer)
            t.run_dir = run
            t.cfg = SimpleNamespace(fork=SimpleNamespace(source=None))
            t.extra_state = {"fork": {"fork_step": 10}}
            t.interaction_step = 10
            t.update_step = 20
            t.buffer = SimpleNamespace(add=lambda _: None)
            t.agent = SimpleNamespace(
                agent=SimpleNamespace(_rng=np.array([0, 1], np.uint32)),
                obs_rms=SimpleNamespace(**self.norm()),
            )

            def train(trainer, last_step, after_step=None, before_first_update=None):
                trainer.interaction_step = 11
                trainer.buffer.add(self.row())
                trainer.update_step = 22
                after_step(trainer)
                trainer.save(run / "state")

            with patch.object(Exp12Trainer, "train", train), patch.object(
                Exp12Trainer, "save", side_effect=OSError("source checkpoint failed")
            ):
                with self.assertRaisesRegex(OSError, "source checkpoint failed"):
                    with record_exp2_scope(root / "stream", 1):
                        t.train(11)
            self.assertEqual(
                StreamReader(root / "stream", require_complete=False).manifest["count"],
                0,
            )

    def test_bitwise_reporting_preserves_dtype_and_signed_zero(self):
        self.assertFalse(
            gpu_checks.difference(
                np.array([1.0], np.float32), np.array([1.0], np.float64)
            )["bitwise_equal"]
        )
        self.assertFalse(
            gpu_checks.difference(
                np.array([0.0], np.float32), np.array([-0.0], np.float32)
            )["bitwise_equal"]
        )

    def test_signed_zero_reporting_is_bitwise(self):
        self.assertFalse(
            gpu_checks.difference(
                np.array([0.0], np.float32), np.array([-0.0], np.float32)
            )["bitwise_equal"]
        )

    def test_checkpoint_publication_recovers_both_sides_and_postcommit_crash(self):
        from experiments.exp3.streams import recover_checkpoint_publication

        for source_step in (10, 12):
            with self.subTest(
                source_step=source_step
            ), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "stream"
                w = StreamWriter(root, {"start_step": 10}, 1, autoflush=False)
                w.append(11, self.row(), 2, self.norm())
                w.append(12, self.row(), 2, self.norm())
                w.prepare_checkpoint(12)
                self.assertEqual(
                    StreamReader(root, require_complete=False).manifest["count"], 0
                )
                recover_checkpoint_publication(root, source_step)
                self.assertEqual(
                    StreamReader(root, require_complete=False).manifest["count"],
                    source_step - 10,
                )
                self.assertFalse((root / "checkpoint_publication.json").exists())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "s"
            w = StreamWriter(root, {"start_step": 10}, 1, autoflush=False)
            w.append(11, self.row(), 2, self.norm())
            w.prepare_checkpoint(11)
            candidate = json.loads((root / "checkpoint_publication.json").read_text())[
                "manifest"
            ]
            (root / "manifest.json").write_text(json.dumps(candidate))
            recover_checkpoint_publication(root, 11)
            self.assertEqual(
                StreamReader(root, require_complete=False).manifest["count"], 1
            )

    def test_publication_rejects_wrong_cursor_and_corrupted_chunks(self):
        from experiments.exp3.streams import recover_checkpoint_publication

        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "s"
            w = StreamWriter(root, {"start_step": 10}, 1, autoflush=False)
            w.append(11, self.row(), 2, self.norm())
            w.prepare_checkpoint(11)
            with self.assertRaisesRegex(ValueError, "neither publication"):
                recover_checkpoint_publication(root, 9)
            next(root.glob("chunk_*.npz")).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                recover_checkpoint_publication(root, 11)
            self.assertEqual(
                StreamReader(root, require_complete=False).manifest["count"], 0
            )

    def test_launch_append_does_not_change_immutable_metadata_binding(self):
        from experiments.exp3.artifacts import metadata_fingerprint

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "metadata.json"
            m = {
                "config_hash": "same",
                "code": {"commit": "a" * 40},
                "launches": [{"job_id": 1}],
            }
            p.write_text(json.dumps(m))
            original = metadata_fingerprint(p)
            m["launches"].append({"job_id": 2})
            p.write_text(json.dumps(m))
            self.assertEqual(metadata_fingerprint(p), original)
            m["config_hash"] = "different"
            p.write_text(json.dumps(m))
            self.assertNotEqual(metadata_fingerprint(p), original)

    def test_concurrent_source_writer_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "did not execute"):
                with record_exp2_scope(Path(d) / "stream", 1):
                    with self.assertRaisesRegex(ValueError, "active recording writer"):
                        with record_exp2_scope(Path(d) / "stream", 1):
                            self.fail("concurrent writer entered")

    def test_completed_source_capture_can_be_finalized_without_training(self):
        import pickle
        from experiments.exp3.recording import finalize_capture
        from experiments.exp3.artifacts import metadata_fingerprint

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            run = base / "run"
            run.mkdir()
            metadata = run / "run_metadata.json"
            metadata.write_text(
                json.dumps({"identity": {"seed": 1}, "launches": [{"job": 1}]})
            )
            provenance = {
                "start_step": 10,
                "source_identity": {"seed": 1},
                "source_metadata_fingerprint": metadata_fingerprint(metadata),
                "arm": "control",
                "injection": None,
                "retention": None,
            }
            root = base / "stream"
            w = StreamWriter(root, provenance, 1, autoflush=False)
            w.append(11, self.row(), 2, self.norm())
            w.prepare_checkpoint(11)
            state = run / "state/saved"
            state.mkdir(parents=True)
            (run / "state/LATEST").write_text("saved")
            (state / "meta.pkl").write_bytes(
                pickle.dumps({"interaction_step": 11, "extra_state": {}})
            )
            (run / "DONE").write_text(json.dumps({"interaction_step": 11}))
            result = finalize_capture(root, run)
            self.assertEqual(result["capture_verdict"], "COMPLETE")
            self.assertFalse(result["scientific_qualification"])
            self.assertEqual([r["step"] for r in StreamReader(root)], [11])
            self.assertEqual(finalize_capture(root, run), result)
            (run / "DONE").write_text(json.dumps({"interaction_step": 12}))
            with self.assertRaisesRegex(ValueError, "endpoint differs"):
                finalize_capture(root, run)

    def test_repeat_detector_rejects_signed_zero_and_dtype_changes(self):
        with patch.object(
            gpu_checks,
            "measurement_fields",
            side_effect=[
                {"v": np.array([0.0], np.float32)},
                {"v": np.array([-0.0], np.float32)},
            ],
        ):
            self.assertFalse(gpu_checks.repeat_bitwise(None, None, None, None, False))
        with patch.object(
            gpu_checks,
            "measurement_fields",
            side_effect=[
                {"v": np.array([1.0], np.float32)},
                {"v": np.array([1.0], np.float64)},
            ],
        ):
            self.assertFalse(gpu_checks.repeat_bitwise(None, None, None, None, False))

    def test_natural_fork_storage_window_requires_explicit_policy(self):
        from experiments.exp3.retention import validate_retention, retention_end

        policy = validate_retention(
            {
                "root": "/tmp/unused-storage-policy",
                "until_step": "arm_end",
                "replay": "omit",
            }
        )
        self.assertEqual(retention_end(policy, {"arm_end_step": 150000}), 150000)
        policy["until_step"] = 175000
        self.assertEqual(retention_end(policy, {"arm_end_step": 150000}), 175000)
        policy["until_step"] = "choose_for_me"
        with self.assertRaises(ValueError):
            validate_retention(policy)

    def test_retention_cannot_be_deleted_by_routine_checkpoint_pruning(self):
        with tempfile.TemporaryDirectory() as d:
            routine = Path(d) / "state"
            state = routine / "step_000000010_fixture"
            state.mkdir(parents=True)
            (routine / "LATEST").write_text(state.name)
            with self.assertRaisesRegex(ValueError, "mutable routine"):
                retain_state(state, routine / "snapshots", 10, "omit")
            with self.assertRaisesRegex(ValueError, "outside the source state"):
                retain_state(state, state / "snapshots", 10, "omit")

    def test_captured_key_must_not_be_silently_cast(self):
        with tempfile.TemporaryDirectory() as d:
            w = StreamWriter(
                Path(d) / "s", {"start_step": 0, "captures": ["agent_key"]}, 1
            )
            with self.assertRaises(ValueError):
                w.append(
                    1,
                    self.row(),
                    2,
                    self.norm(),
                    {"agent_key": np.array([0, 2**32], np.uint64)},
                )


if __name__ == "__main__":
    unittest.main()
