"""Budget, execution-plan integration, and failure-preserving evidence export."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.exp12.diagnostic_budget import admission, reconcile
from scripts.check_exp96_diagnostic_budget import check_live, prepare, walltime_seconds
from scripts.export_exp96_diagnostics import export

RAW = (
    "23120674|CANCELLED by 97950|0:0|0||0|gpuA100x4\n"
    "23120792|FAILED|1:0|181|billing=3000,cpu=16,gres/gpu=1,mem=60G,node=1|0|gpuA100x8-interactive\n"
    "23120792.batch|FAILED|1:0|181|gres/gpu=1||\n"
)
JOBS = ["23120674", "23120792"]


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = (
            Path(__file__).resolve().parents[1]
            / "configs/diagnostics/reconstruction.json"
        )
        self.commit = "a" * 40

    def test_physical_usage_is_parent_elapsed_not_steps_or_billing(self):
        result = reconcile(RAW, JOBS)
        self.assertEqual(result["physical_gpu_seconds"], 181)
        self.assertEqual(
            admission(result, 3600)["remaining_after_request_seconds"], 10619
        )
        with self.assertRaises(ValueError):
            admission(result, 7200, 7200)
        with self.assertRaises(ValueError):
            reconcile(RAW + RAW, JOBS)

    def test_unsettled_missing_unknown_and_restarted_allocations_fail(self):
        for raw in (
            RAW.replace("FAILED", "RUNNING"),
            RAW.replace("|0|gpuA100x8", "|1|gpuA100x8"),
            RAW.splitlines()[0],
            RAW.replace("23120792|", "23120999|"),
            RAW.replace("gres/gpu=1", "gres/gpu=2"),
        ):
            with self.assertRaises(ValueError):
                reconcile(raw, JOBS)

    def receipt(self):
        return prepare(
            RAW, JOBS, self.plan, self.commit, 3600, 0, "2026-10-22T22:00:00"
        )

    def test_live_preflight_rechecks_accounting_and_actual_slurm_limit(self):
        job = "JobId=42 Requeue=0 Restarts=0 TimeLimit=01:00:00 AllocTRES=gres/gpu=1,node=1 Partition=gpuA100x8-interactive"
        with mock.patch(
            "scripts.check_exp96_diagnostic_budget.subprocess.check_output",
            side_effect=[job, "42\n", RAW],
        ):
            result = check_live(self.receipt(), self.plan, self.commit, "42", 3000)
        self.assertEqual(result["accounting"]["physical_gpu_seconds"], 181)
        for invalid in (
            job.replace("01:00:00", "02:00:00"),
            job.replace("Requeue=0", "Requeue=1"),
            job.replace("gpu=1", "gpu=2"),
        ):
            with mock.patch(
                "scripts.check_exp96_diagnostic_budget.subprocess.check_output",
                return_value=invalid,
            ):
                with self.assertRaises(ValueError):
                    check_live(self.receipt(), self.plan, self.commit, "42", 3000)
        with mock.patch(
            "scripts.check_exp96_diagnostic_budget.subprocess.check_output",
            side_effect=[job, "42\n43\n"],
        ):
            with self.assertRaises(ValueError):
                check_live(self.receipt(), self.plan, self.commit, "42", 3000)
        with self.assertRaises(ValueError):
            check_live(self.receipt(), self.plan, "b" * 40, "42", 3000)
        self.assertEqual(walltime_seconds("1-01:02:03"), 90123)

    def test_atomic_numeric_publication_preserves_prior_file_on_failed_write(self):
        import numpy as np
        from experiments.exp12.diagnostic_artifacts import atomic_npz

        path = self.root / "curve.npz"
        atomic_npz(path, losses=np.ones(2, dtype=np.float32))
        before = path.read_bytes()
        with mock.patch(
            "experiments.exp12.diagnostic_artifacts.np.savez",
            side_effect=RuntimeError("interrupted write"),
        ):
            with self.assertRaises(RuntimeError):
                atomic_npz(path, losses=np.zeros(2))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob(".curve.npz*")), [])

    def test_failed_preflight_can_be_exported_without_results(self):
        source = self.root / "failed"
        source.mkdir()
        (source / "runtime.json").write_text(
            json.dumps({"expected_commit": self.commit})
        )
        (source / "status.json").write_text('{"status":"FAILED_PREFLIGHT"}')
        result = export(source, self.root / "failed-export")
        self.assertEqual(result["status"]["status"], "FAILED_PREFLIGHT")
        self.assertEqual(result["source"], self.commit)

    def test_export_includes_partial_results_and_verifiable_checksums_not_replay(self):
        import numpy as np

        source = self.root / "source"
        source.mkdir()
        out = self.root / "export"
        fit = source / "fit"
        fit.mkdir()
        results = {
            "source": self.commit,
            "frozen_source": "f" * 40,
            "plan_sha256": "c" * 64,
            "conditions": [{"id": "fit", "status": "PARTIAL_BUDGET"}],
            "rows": [
                {
                    "name": "fit",
                    "status": "PARTIAL",
                    "task_id": "task",
                    "steps": 1000,
                    "first_update": {"prediction_change_rms": 0.1},
                    "measurements": [
                        {"step": 0, "mse": 2.0},
                        {"step": 1000, "mse": 1.0},
                    ],
                }
            ],
        }
        (source / "results.json").write_text(json.dumps(results))
        (source / "status.json").write_text('{"status":"PARTIAL_BUDGET"}')
        np.savez(fit / "curve_through_1000.npz", losses=np.ones(1000, dtype=np.float32))
        np.savez(source / "buffer.npz", observations=np.ones((32, 3)))
        manifest = export(source, out)
        self.assertEqual(manifest["status"]["status"], "PARTIAL_BUDGET")
        self.assertNotIn("buffer.npz", manifest["files"])
        self.assertIn(
            "first_update_prediction_change_rms", (out / "measurements.csv").read_text()
        )
        for line in (out / "SHA256SUMS.txt").read_text().splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(
                hashlib.sha256((out / name).read_bytes()).hexdigest(), digest
            )
        with self.assertRaises(FileExistsError):
            export(source, out)
        (source / "alias").symlink_to("/tmp")
        with self.assertRaises(ValueError):
            export(source, self.root / "unsafe")


class EntryPointTest(unittest.TestCase):
    def exercise(self, *, partial=False, save_failure=False, unsafe_output=False):
        from contextlib import ExitStack
        from types import SimpleNamespace
        import time
        from scripts import exp96_fixed_task_diagnostics as module

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            output = root / "output"
            campaign = root / "campaign"
            if unsafe_output:
                output = campaign / "nested-output"
            args = SimpleNamespace(
                root=str(output),
                campaign=str(campaign),
                commit="a" * 40,
                cap_seconds=3000,
                plan=str(
                    Path(__file__).resolve().parents[1]
                    / "configs/diagnostics/reconstruction.json"
                ),
            )
            runner = mock.Mock(start=time.monotonic())
            runner.run.return_value = not partial
            if save_failure:
                runner.run.side_effect = RuntimeError("original fit error")
                runner.save.side_effect = ValueError("nonfinite results")
            stack.enter_context(
                mock.patch.object(
                    module.argparse.ArgumentParser, "parse_args", return_value=args
                )
            )
            stack.enter_context(
                mock.patch.dict(module.os.environ, {"SLURM_JOB_ID": "42"})
            )
            stack.enter_context(
                mock.patch.object(
                    module.subprocess, "check_output", side_effect=[args.commit, ""]
                )
            )
            backend = stack.enter_context(
                mock.patch.object(module.jax, "default_backend", return_value="gpu")
            )
            stack.enter_context(
                mock.patch.object(
                    module.jax,
                    "devices",
                    return_value=[SimpleNamespace(device_kind="NVIDIA A100-SXM4-40GB")],
                )
            )
            stack.enter_context(mock.patch.object(module, "set_matmul_precision"))
            stack.enter_context(
                mock.patch.object(
                    module,
                    "runtime_info",
                    return_value={
                        "jax_cuda12_plugin": "0.4.34",
                        "jax_cuda12_pjrt": "0.4.34",
                    },
                )
            )
            stack.enter_context(
                mock.patch.object(module, "Investigation", return_value=runner)
            )
            stack.enter_context(mock.patch.object(module.threading, "Thread"))
            if unsafe_output:
                with self.assertRaisesRegex(
                    ValueError, "outside source and frozen campaign"
                ):
                    module.main()
                backend.assert_not_called()
                self.assertFalse(output.exists())
            elif save_failure:
                with self.assertRaisesRegex(RuntimeError, "original fit error"):
                    module.main()
                status = json.loads((output / "status.json").read_text())
                self.assertEqual(status["status"], "FAILED")
                self.assertIn("nonfinite results", status["result_save_error"])
            else:
                self.assertEqual(module.main(), 3)
                self.assertEqual(
                    json.loads((output / "status.json").read_text())["status"],
                    "PARTIAL_BUDGET",
                )

    def test_partial_execution_has_nonzero_exit_and_preserved_status(self):
        self.exercise(partial=True)

    def test_primary_failure_survives_result_serialization_failure(self):
        self.exercise(save_failure=True)

    def test_output_cannot_be_created_in_frozen_campaign(self):
        self.exercise(unsafe_output=True)


if __name__ == "__main__":
    unittest.main()
