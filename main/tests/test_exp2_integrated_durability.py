"""Exact reviewed durability class; scheduler tests intentionally excluded.

Source: 087feab341b9be1f377697668f2d1c750ec0467f
main/tests/test_exp12_review_fixes.py, DurableStateAndAtomicNpzTest.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


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
