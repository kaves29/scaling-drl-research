import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import exp12_dynamic_scheduler as scheduler
from scripts.exp12_dynamic_scheduler import identity
from scripts.exp12_launch_plan import cells
from scripts.exp12_pilot_plasticity_wandb import records
from scripts.exp12_seven_parent_pilot import FROZEN, PLACEMENT, select, validate_resources


class SevenParentPilotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = cells(Path("/tmp/exp12-seven-parent-test-root"), "parents")
        cls.queue = {"commit": FROZEN, "parents": [
            {"kind": "exp1", "id": identity("exp1", row, FROZEN), "row": row}
            for row in rows]}
        cls.selected = select(cls.queue)

    def test_exact_frozen_cells_and_placement(self):
        self.assertEqual(len(self.queue["parents"]), 195)
        self.assertEqual([(gpu, item["row"]["architecture"], item["row"]["environment"],
                           item["row"]["seed"]) for gpu, item in self.selected], list(PLACEMENT))
        self.assertEqual(len({item["id"] for _, item in self.selected}), 7)
        self.assertEqual([item["row"]["budget"] for _, item in self.selected],
                         [1000000, 1000000, 1000000, 2000000, 1000000, 1000000, 1000000])

    def test_resource_fail_closed(self):
        estimates = {item["id"]: {"cpus": 4, "host_ram_gb": 8,
                                  "gpu_memory_gb": 5 if gpu == 0 else 6}
                     for gpu, item in self.selected}
        spec = {"resources": {"gpus": 2, "cpus": 32, "memory_gb": 96,
                              "wall_minutes": 600}, "per_parent_resource_estimates": estimates}
        validate_resources(spec, self.selected)
        spec["per_parent_resource_estimates"][self.selected[-1][1]["id"]]["gpu_memory_gb"] = 20
        with self.assertRaisesRegex(ValueError, "GPU 1"):
            validate_resources(spec, self.selected)
        del spec["per_parent_resource_estimates"][self.selected[-1][1]["id"]]
        with self.assertRaisesRegex(ValueError, "every selected parent"):
            validate_resources(spec, self.selected)

    def test_wrong_source_or_parent_id_rejected(self):
        wrong_source = {**self.queue, "commit": "0" * 40}
        with self.assertRaisesRegex(ValueError, "frozen scientific commit"):
            select(wrong_source)
        rows = [dict(item) for item in self.queue["parents"]]
        rows[10] = {**rows[10], "id": "0" * 64}
        with self.assertRaisesRegex(ValueError, "logical identity"):
            select({**self.queue, "parents": rows})

    def test_plasticity_export_uses_training_env_steps_and_undefined_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "probe_checks.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["check_index", "interaction_step",
                                                           "score_fresh_iqm", "score_current_iqm"])
                writer.writeheader()
                writer.writerow({"check_index": 0, "interaction_step": 1000,
                                 "score_fresh_iqm": 2, "score_current_iqm": ""})
                for k in range(1, 21):
                    writer.writerow({"check_index": k, "interaction_step": k * 25000,
                                     "score_fresh_iqm": 2, "score_current_iqm": 1})
            output, baseline, undefined = records(path, 2)
            self.assertEqual((baseline, undefined), (2, False))
            self.assertEqual(output[0]["training_env_steps"], 50000)
            self.assertEqual(output[0]["plasticity/degradation_pct"], 50)
            text = path.read_text().replace("0,1000,2,", "0,1000,0,")
            path.write_text(text)
            output, _, undefined = records(path, 2)
            self.assertTrue(undefined)
            self.assertNotIn("plasticity/degradation_pct", output[0])

    def test_training_process_uses_separate_frozen_checkout(self):
        item = self.selected[0][1]
        approval = {"expected_commit": FROZEN, "resources": {}, "admission": {
            "architectures": {item["row"]["architecture"]: {"cpus": 4}}}}
        with tempfile.TemporaryDirectory() as td:
            ledger = mock.Mock()
            ledger.claim.return_value = True
            with (mock.patch.object(scheduler, "validate_resume", return_value=False),
                  mock.patch.object(scheduler, "checkpoint_progress", return_value=None),
                  mock.patch.object(scheduler.subprocess, "Popen") as popen):
                job = scheduler._start(item, "0", approval, td, ledger, "12345",
                                       training_checkout="/separate/frozen/main")
            self.assertIsNotNone(job)
            self.assertEqual(popen.call_args.kwargs["cwd"], "/separate/frozen/main")
            self.assertEqual(popen.call_args.kwargs["env"]["PYTHONPATH"], "/separate/frozen/main")
            self.assertEqual(popen.call_args.kwargs["env"]["WANDB_RUN_ID"], item["id"][:16])
            self.assertEqual(popen.call_args.args[0][3:5], ["--experiment", "exp1"])
            ledger.claim.assert_called_once()


if __name__ == "__main__":
    unittest.main()
