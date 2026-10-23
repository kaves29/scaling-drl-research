"""Exact task persistence through the runner and real JAX scan."""

from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest

import jax.numpy as jnp
import numpy as np

from experiments.exp12 import fixed_task as ft
from experiments.exp12.probe_tasks import FrozenTask
from tests import test_fixed_task as fixtures


class TaskBankIntegrationTests(unittest.TestCase):
    def test_runner_bank_preserves_fitting_and_pair_identity(self):
        root = Path(__file__).resolve().parents[1]
        runner_type = runpy.run_path(
            str(root / "scripts/exp96_fixed_task_diagnostics.py")
        )["Investigation"]
        runner = object.__new__(runner_type)
        runner.args = SimpleNamespace(commit="cpu-test-source")
        runner.optimizer_sha256 = "cpu-test-optimizer"
        runner.inputs = []
        f = fixtures.FixedTaskTests()
        f.setUp()
        targets = f.z + jnp.float32(10)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "first_learner"
            task_id = runner.bank_task(
                directory, f.o, f.a, targets, f.idx[:1000], f.z, "shared"
            )
            other_id = runner.bank_task(
                Path(tmp) / "second_learner",
                f.o,
                f.a,
                targets,
                f.idx[:1000],
                f.z,
                "shared",
            )
            restored = FrozenTask.load(directory)
            self.assertEqual(task_id, other_id)
            self.assertEqual(task_id, restored.task_id)
            start = ft.FitState(f.p, f.tx.init(f.p))
            direct, direct_curve = ft.fit_segment(
                f.net, f.tx, start, f.o, f.a, targets, f.idx[:1000]
            )
            replayed, replayed_curve = ft.fit_segment(
                f.net,
                f.tx,
                start,
                jnp.asarray(restored.observations),
                jnp.asarray(restored.actions),
                jnp.asarray(restored.targets),
                jnp.asarray(restored.minibatch_indices),
            )
            np.testing.assert_array_equal(direct_curve, replayed_curve)
            self.assertEqual(ft.tree_sha256(direct), ft.tree_sha256(replayed))

    def test_numeric_payload_corruption_is_rejected(self):
        from tests.test_probe_tasks import ProbeTaskTests

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "task"
            ProbeTaskTests().task().save(directory)
            with np.load(directory / "arrays.npz", allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in data.files}
            arrays["targets"][0] += np.float32(1)
            np.savez(directory / "arrays.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                FrozenTask.load(directory)


if __name__ == "__main__":
    unittest.main()
