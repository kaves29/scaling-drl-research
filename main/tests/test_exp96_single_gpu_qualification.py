"""Required new checks for sequential qualification on a partial x4 node."""

import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import exp96_multigpu as m
from scripts import exp96_multigpu_qualification as q


def resources(minutes=30):
    return dict(
        partition="gpuA100x4",
        nodes=1,
        gpus=1,
        cpus=16,
        memory_gb=60,
        wall_minutes=minutes,
    )


class SingleGpu(unittest.TestCase):
    def test_partial_node_uses_slurm_physical_identity(self):
        inventory = "\n".join(
            f"{i}, GPU-{i}, NVIDIA A100-SXM4-40GB, 40960" for i in range(4)
        )
        with patch.dict(
            os.environ, SLURM_JOB_GPUS="2", CUDA_VISIBLE_DEVICES="0"
        ), patch.object(m, "checked", return_value=inventory):
            self.assertEqual(q.qualification_device()["uuid"], "GPU-2")
            with patch.dict(os.environ, SLURM_JOB_GPUS="1,2"):
                with self.assertRaises(ValueError):
                    q.qualification_device()
        m.resources_valid(resources(), qualification=True)
        self.assertEqual(m.rate(resources()), 1)
        for composition in ((7, 0, 0), (0, 5, 0), (0, 0, 4)):
            self.assertTrue(m.composition_allowed(composition))
        self.assertFalse(m.composition_allowed((1, 1, 0)))
        with self.assertRaises(ValueError):
            m.resources_valid(dict(resources(), gpus=8), qualification=True)

    def test_cancelled_before_allocation_releases_only_zero_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = m.Budget(Path(tmp) / "budget")
            b.initialize("synthetic owner fixture")
            b.reserve("cancelled", resources(), "qualification")
            b.submitted("cancelled", "123")
            with patch.object(m, "checked", return_value="123|CANCELLED by 97950|0||"):
                result = b.settle("cancelled")
            self.assertEqual(result["actual_su"], 0)
            self.assertEqual(result["actual_gpu_hours"], 0)
            b.reserve("missing", resources(), "qualification")
            b.submitted("missing", "124")
            with patch.object(m, "checked", return_value="124|CANCELLED|1||"):
                with self.assertRaises(ValueError):
                    b.settle("missing")

    def test_three_phases_never_overlap_and_all_required_children_finish(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp).resolve()
            smoke = root / "smoke"
            smoke.mkdir()
            m.write(smoke / "smoke.json", dict(synthetic=True))
            previous = root / "prior.json"
            m.write(previous, dict(environment_input_bounds={}))
            processes = []
            phase_counts = []

            class Process:
                def __init__(self, cmd, **kwargs):
                    path = Path(cmd[cmd.index("--root") + 1])
                    arch = cmd[cmd.index("--arch") + 1]
                    if not any(p.path.parent == path.parent for p in processes):
                        self.assert_prior_finished()
                        phase_counts.append(0)
                    phase_counts[-1] += 1
                    self.pid = 100 + len(processes)
                    self.path = path
                    path.mkdir()
                    curve = path / "curves"
                    curve.write_bytes(b"synthetic curves")
                    ids = list(
                        map(int, kwargs["env"]["EXP96_ASSIGNED_CPU_IDS"].split(","))
                    )
                    m.write(path / "ready.json", dict(pid=self.pid, gpu="GPU-2"))
                    m.write(
                        path / "child_receipt.json",
                        dict(
                            source="test-source",
                            gpu_uuid="GPU-2",
                            cpu_ids=ids,
                            status="PASS",
                            curve_path=str(curve),
                            curve_sha256=m.digest(curve),
                            architecture=arch,
                            device_memory_stats=dict(peak_bytes_in_use=100 * 2**20),
                            warm_update_seconds=0.001,
                            probe_seconds=1,
                            elapsed=2,
                        ),
                    )
                    processes.append(self)

                def assert_prior_finished(self):
                    if any(p.poll() is None for p in processes):
                        raise AssertionError("phases overlapped")

                def poll(self):
                    return 0 if (self.path.parent / "EXIT_GO.json").exists() else None

            def sample(records):
                return (
                    {"GPU-2": (40960, 1024)},
                    {p["process"].pid: ("GPU-2", 128) for p in records},
                    {p["process"].pid: 2**30 for p in records},
                )

            stack.enter_context(
                patch.dict(
                    os.environ,
                    SLURM_JOB_ID="123",
                    EXP96_BUDGET="fixture",
                    EXP96_RESERVATION="fixture",
                    EXP96_REUSE_CHECKS=str(previous),
                    EXP96_QUALIFICATION_MINUTES="30",
                )
            )
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "scripts.exp12_launch_plan": NS(
                            require_pinned_stack=lambda: None
                        )
                    },
                )
            )
            for name in ("source", "verify_actual", "reserved_job"):
                stack.enter_context(patch.object(m, name, return_value=None))
            stack.enter_context(
                patch.object(m, "training_fingerprint", return_value="unchanged")
            )
            stack.enter_context(
                patch.object(m, "BASE_SMOKE_DIGEST", m.digest(smoke / "smoke.json"))
            )
            stack.enter_context(
                patch.object(m, "checked", return_value="synthetic lscpu")
            )
            stack.enter_context(patch.object(q, "SMOKE", smoke))
            stack.enter_context(
                patch.object(
                    q, "reusable_checks", return_value=dict(evidence_sha256={})
                )
            )
            stack.enter_context(
                patch.object(
                    q,
                    "qualification_device",
                    return_value=dict(uuid="GPU-2", memory_mb=40960),
                )
            )
            stack.enter_context(
                patch.object(
                    os, "sched_getaffinity", return_value=set(range(16)), create=True
                )
            )
            stack.enter_context(patch.object(q.subprocess, "Popen", Process))
            stack.enter_context(patch.object(q, "sample", side_effect=sample))
            stack.enter_context(patch.object(q.time, "sleep", return_value=None))
            stack.enter_context(contextlib.redirect_stdout(__import__("io").StringIO()))
            (root / "run").mkdir()
            self.assertEqual(q.run(root / "run", "test-source"), 0)
            self.assertEqual(phase_counts, [7, 5, 4])
            report = m.read(root / "run/qualification.json")
            self.assertEqual(report["resources"]["gpus"], 1)
            self.assertEqual(len(report["attempts"]), 16)
            self.assertEqual(len(report["fitting_barrier_sha256"]), 3)


if __name__ == "__main__":
    unittest.main()
