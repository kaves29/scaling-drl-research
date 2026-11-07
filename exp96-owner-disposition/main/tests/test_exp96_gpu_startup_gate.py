"""New startup admission/cleanup regressions, with explicit synthetic processes."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import exp96_multigpu as m
from scripts import exp96_multigpu_qualification as q


class Startup(unittest.TestCase):
    def exercise(self, wrong_gpu=False, timeout=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "startup"
            events = []

            class Process:
                pid = 1234

                def __init__(self, command, **kwargs):
                    self.code = None
                    events.append(("launch", command, kwargs["env"]))
                    if not timeout:
                        root.mkdir()
                        m.write(
                            root / "ready.json",
                            dict(
                                source="fixture",
                                pid=self.pid,
                                gpu="GPU-2",
                                cpu_ids=[16, 17],
                                kernel_finite_fp32=True,
                                runtime=dict(
                                    platform="gpu",
                                    jax="0.4.34",
                                    matmul_precision="tensorfloat32",
                                ),
                            ),
                        )

                def poll(self):
                    return 0 if (root / "EXIT.json").exists() else self.code

                def wait(self, timeout):
                    if (root / "EXIT.json").exists():
                        return 0
                    self.code = -15
                    return -15

            def killed(*args):
                events.append(("kill", args))

            telemetry = (
                {"GPU-2": (40960, 512)},
                {1234: ("GPU-wrong" if wrong_gpu else "GPU-2", 400)},
                {1234: 2**20},
            )
            with patch.object(q.subprocess, "Popen", Process), patch.object(
                q, "sample", return_value=telemetry
            ), patch.object(q.os, "killpg", side_effect=killed):
                if timeout:
                    with patch.object(q.time, "monotonic", side_effect=[0, 100]):
                        with self.assertRaises(TimeoutError):
                            q.gpu_startup_gate(
                                root, "fixture", dict(uuid="GPU-2"), [16, 17, 18]
                            )
                elif wrong_gpu:
                    with self.assertRaisesRegex(ValueError, "verification failed"):
                        q.gpu_startup_gate(
                            root, "fixture", dict(uuid="GPU-2"), [16, 17, 18]
                        )
                else:
                    proof = q.gpu_startup_gate(
                        root, "fixture", dict(uuid="GPU-2"), [16, 17, 18]
                    )
                    self.assertTrue(proof["passed"])
                    self.assertEqual(
                        m.digest(root / "ready.json"), proof["ready_sha256"]
                    )
            launch = events[0]
            self.assertEqual(launch[2]["CUDA_VISIBLE_DEVICES"], "GPU-2")
            self.assertEqual(launch[2]["EXP96_ASSIGNED_CPU_IDS"], "16,17")
            self.assertEqual(launch[2]["JAX_PLATFORMS"], "cuda,cpu")
            self.assertNotIn("JAX_PLATFORM_NAME", launch[2])
            self.assertEqual(any(e[0] == "kill" for e in events), wrong_gpu or timeout)

    def test_success_uses_real_telemetry_before_admission(self):
        self.exercise()

    def test_incorrect_physical_gpu_is_not_admitted(self):
        self.exercise(wrong_gpu=True)

    def test_startup_timeout_cleans_up_without_fitting(self):
        self.exercise(timeout=True)


if __name__ == "__main__":
    unittest.main()
