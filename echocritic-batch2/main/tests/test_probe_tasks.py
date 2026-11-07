import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp12.probe import iqm
from experiments.exp12.probe_tasks import FrozenTask, fixed_task_paired_summary


class ProbeTaskTests(unittest.TestCase):
    def task(self):
        return FrozenTask(
            np.arange(24, dtype=np.float32).reshape(8, 3),
            np.zeros((8, 2), np.float32),
            np.arange(8, dtype=np.float32),
            np.array([[0, 1, 2, 3], [4, 5, 6, 7]], np.int32),
            {"source": "f597ee9", "normalizer_id": "fixed-example"},
        )

    def test_roundtrip_immutable_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task"
            task = self.task()
            task.save(path)
            restored = FrozenTask.load(path)
            self.assertEqual(task.task_id, restored.task_id)
            with self.assertRaises(ValueError):
                restored.targets[0] = 10
            with self.assertRaises(ValueError):
                restored.targets.flags.writeable = True
            with self.assertRaises(FileExistsError):
                task.save(path)

    def test_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task"
            self.task().save(path)
            m = json.loads((path / "manifest.json").read_text())
            m["provenance"]["normalizer_id"] = "changed"
            (path / "manifest.json").write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                FrozenTask.load(path)

    def test_paired_iqm_is_not_difference_of_iqms(self):
        f = np.array([0, 0, 0, 0, 0, 0, 1, 2, 4, 8], np.float32)
        c = np.array([8, 0, 0, 0, 0, 0, 0, 1, 2, 4], np.float32)
        b = np.ones(10, np.float32)
        ids = [str(i) for i in range(10)]
        s = fixed_task_paired_summary(f, c, b, ids, ids, 2000, 2000)
        self.assertEqual(s["paired_loss_iqm10"], iqm((b - f) - (b - c)))
        self.assertNotEqual(s["paired_loss_iqm10"], iqm(b - f) - iqm(b - c))

    def test_mismatched_tasks_horizons_or_target_count_are_rejected(self):
        x = np.zeros(10, np.float32)
        ids = [str(i) for i in range(10)]
        for fresh, current, fs, cs in [
            (ids, ids[::-1], 2000, 2000),
            (ids, ids, 2000, 8000),
        ]:
            with self.assertRaises(ValueError):
                fixed_task_paired_summary(x, x, x, fresh, current, fs, cs)
        with self.assertRaises(ValueError):
            fixed_task_paired_summary(x[:3], x[:3], x[:3], ids[:3], ids[:3], 2000, 2000)


if __name__ == "__main__":
    unittest.main()
