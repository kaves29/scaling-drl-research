"""Driver ABI and identity mismatch regressions; no GPU is used by these tests."""

import ctypes
import os
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import exp96_multigpu as m
from scripts import exp96_multigpu_qualification as q

TOKEN = "GPU-00112233-4455-6677-8899-aabbccddeeff"
IDENTITY = dict(
    uuid=TOKEN,
    pci_bus_id="0000:87:00.0",
    visible_count=1,
    driver_api_version=12040,
    controller_context=None,
)
INVENTORY = f"0, {TOKEN}, NVIDIA A100-SXM4-40GB, 40960, 00000000:87:00.0, 3"
ENV = dict(
    SLURM_JOB_GPUS="3",
    SLURM_STEP_GPUS="3",
    SLURM_GPUS_ON_NODE="1",
    CUDA_VISIBLE_DEVICES="0",
)


class Identity(unittest.TestCase):
    def test_driver_abi_and_no_context_created(self):
        def integer(pointer, value):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = value
            return 0

        def token(pointer, device):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ubyte * 16))[0][:] = uuid.UUID(
                TOKEN[4:]
            ).bytes
            return 0

        def bus(buffer, length, device):
            buffer.value = b"0000:87:00.0"
            return 0

        driver = NS(
            cuInit=Mock(return_value=0),
            cuDeviceGetCount=Mock(side_effect=lambda p: integer(p, 1)),
            cuDeviceGet=Mock(side_effect=lambda p, n: integer(p, 3)),
            cuDeviceGetUuid=Mock(side_effect=token),
            cuDeviceGetPCIBusId=Mock(side_effect=bus),
            cuCtxGetCurrent=Mock(return_value=0),
            cuDriverGetVersion=Mock(side_effect=lambda p: integer(p, 12040)),
        )
        with patch.object(q.ctypes, "CDLL", return_value=driver):
            self.assertEqual(q.cuda_visible_identity(), IDENTITY)
            driver.cuInit.assert_called_once_with(0)
            driver.cuDeviceGetCount.side_effect = lambda p: integer(p, 2)
            with self.assertRaisesRegex(ValueError, "exactly one GPU"):
                q.cuda_visible_identity()
            driver.cuInit.return_value = 100
            with self.assertRaisesRegex(RuntimeError, "CUDA error 100"):
                q.cuda_visible_identity()

    def test_numeric_namespaces_can_differ_and_optional_ids_can_be_absent(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, ENV
        ), patch.object(m, "checked", return_value=INVENTORY), patch.object(
            q, "cuda_visible_identity", return_value=IDENTITY
        ):
            file = Path(tmp) / "discovery.json"
            self.assertEqual(q.qualification_device(file)["uuid"], TOKEN)
            self.assertEqual(m.read(file)["selected"]["minor_number"], "3")
            with patch.dict(os.environ, SLURM_JOB_GPUS="", SLURM_STEP_GPUS=""):
                self.assertEqual(q.qualification_device()["uuid"], TOKEN)
            # Multiple NVML rows are harmless; only CUDA's actual UUID matches.
            other = "3, GPU-ffffffff-ffff-ffff-ffff-ffffffffffff, NVIDIA A100-SXM4-40GB, 40960, 00000000:ca:00.0, 0"
            with patch.object(m, "checked", return_value=other + "\n" + INVENTORY):
                self.assertEqual(q.qualification_device()["index"], "0")

    def test_conflicting_identity_and_oversized_visibility_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, ENV
        ), patch.object(m, "checked", return_value=INVENTORY), patch.object(
            q, "cuda_visible_identity", return_value=IDENTITY
        ):
            with patch.object(
                m, "checked", return_value=INVENTORY.replace("87:00.0", "88:00.0")
            ):
                with self.assertRaisesRegex(ValueError, "PCI identity"):
                    q.qualification_device()
            with patch.object(m, "checked", return_value=INVENTORY + "\n" + INVENTORY):
                with self.assertRaisesRegex(ValueError, "uniquely matched"):
                    q.qualification_device()
            for key, value in [
                ("SLURM_GPUS_ON_NODE", "2"),
                ("SLURM_JOB_GPUS", "0,1"),
                ("SLURM_JOB_GPUS", "GPU-wrong"),
                ("CUDA_VISIBLE_DEVICES", ""),
            ]:
                file = Path(tmp) / (key + value.replace(",", "_"))
                with patch.dict(os.environ, {key: value}):
                    with self.assertRaises(ValueError):
                        q.qualification_device(file)
                self.assertIn("error", m.read(file))


if __name__ == "__main__":
    unittest.main()
