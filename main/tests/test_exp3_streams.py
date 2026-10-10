"""Storage-only negative tests, independent of environments and SAC training."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp3.streams import StreamReader, StreamWriter


class StreamTest(unittest.TestCase):
    def row(self):
        return {
            "observation": np.zeros((1, 2), np.float32),
            "next_observation": np.ones((1, 2), np.float32),
            "action": np.array([[-0.0]], np.float32),
            "reward": np.array([1.0], np.float32),
            "terminated": np.zeros(1, np.float32),
            "truncated": np.ones(1, np.float32),
        }

    def norm(self):
        return {
            "mean": np.zeros((1, 2), np.float64),
            "var": np.ones((1, 2), np.float64),
            "count": 1.0,
        }

    def test_prefix_resume_and_sealing_preserve_all_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "stream"
            w = StreamWriter(p, {"start_step": 10}, 1)
            w.append(11, self.row(), 2, self.norm())
            w = StreamWriter(p, {"start_step": 10}, 1, resume=True)
            w.append(12, self.row(), 2, self.norm())
            w.seal()
            rows = list(StreamReader(p))
            self.assertEqual([r["step"] for r in rows], [11, 12])
            for r in rows:
                for k, v in self.row().items():
                    self.assertEqual(r["transition"][k].dtype, v.dtype)
                    self.assertEqual(r["transition"][k].tobytes(), v.tobytes())
            with self.assertRaisesRegex(ValueError, "sealed"):
                StreamWriter(p, {"start_step": 10}, 1, resume=True)

    def test_resume_cannot_change_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "stream"
            StreamWriter(p, {"start_step": 10}, 1)
            with self.assertRaisesRegex(ValueError, "provenance"):
                StreamWriter(p, {"start_step": 11}, 1, resume=True)

    def test_duplicate_manifest_chunks_and_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s"
            w = StreamWriter(p, {"start_step": 0}, 1)
            w.append(1, self.row(), 2, self.norm())
            w.seal()
            m = json.loads((p / "manifest.json").read_text())
            m["chunks"] *= 2
            m["count"] = 2
            (p / "manifest.json").write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "duplicated"):
                StreamReader(p)
            m["chunks"] = m["chunks"][:1]
            m["count"] = 1
            m["chunks"][0]["name"] = "../escaped.npz"
            (p / "manifest.json").write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "path"):
                list(StreamReader(p))

    def test_bad_normalization_fields_and_arrival_shapes_refused(self):
        with tempfile.TemporaryDirectory() as d:
            w = StreamWriter(Path(d) / "s", {"start_step": 0}, 2)
            norm = self.norm()
            norm["var"][0, 0] = -1
            with self.assertRaisesRegex(ValueError, "normalization"):
                w.append(1, self.row(), 2, norm)
            row = self.row()
            row["reward"] = np.zeros((1, 1), np.float32)
            with self.assertRaisesRegex(ValueError, "scalar shape"):
                w.append(1, row, 2, self.norm())

    def test_stream_writes_do_not_consume_global_rng(self):
        np.random.seed(77)
        before = np.random.get_state()
        with tempfile.TemporaryDirectory() as d:
            w = StreamWriter(Path(d) / "s", {"start_step": 0}, 2)
            w.append(1, self.row(), 2, self.norm())
            w.seal()
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_partial_diagnostic_array_publication_is_never_visible(self):
        from unittest.mock import patch
        from experiments.exp3.runner import array_file

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "measurement.npz"

            def fail(stream, **values):
                stream.write(b"partial")
                raise OSError("simulated disk write failure")

            with patch("experiments.exp3.runner.np.savez", side_effect=fail):
                with self.assertRaisesRegex(OSError, "simulated"):
                    array_file(path, {"a": np.ones(3, np.float32)})
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(d).iterdir()), [])
            array_file(path, {"a": np.ones(3, np.float32)})
            with np.load(path) as actual:
                np.testing.assert_array_equal(actual["a"], np.ones(3, np.float32))
