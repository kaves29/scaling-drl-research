"""Verifies utils/hardware.py's OS-level GPU vendor detection.

No real NVIDIA or AMD hardware is available in CI/dev environments, so these
tests use stub executables on a temporary PATH to exercise the real
shutil.which + subprocess.run code path (not just mocked-out logic), plus a
few pure-mock cases for the "tool present but errors" branch.
"""

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.hardware import _VENDOR_SENTINEL_ENV_VAR, configure_hardware_env, detect_gpu_vendor


def _make_stub(dir_path: Path, name: str, exit_code: int) -> None:
    script = dir_path / name
    script.write_text(f"#!/bin/sh\nexit {exit_code}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


class DetectGpuVendorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.bin_dir = Path(self._tmp.name)
        self._path_patch = mock.patch.dict(os.environ, {"PATH": str(self.bin_dir)}, clear=False)
        self._path_patch.start()
        self.addCleanup(self._path_patch.stop)

    def test_no_tools_on_path_returns_none(self):
        self.assertIsNone(detect_gpu_vendor())

    def test_nvidia_smi_success_returns_nvidia(self):
        _make_stub(self.bin_dir, "nvidia-smi", exit_code=0)
        self.assertEqual(detect_gpu_vendor(), "nvidia")

    def test_rocm_smi_success_returns_amd(self):
        _make_stub(self.bin_dir, "rocm-smi", exit_code=0)
        self.assertEqual(detect_gpu_vendor(), "amd")

    def test_nvidia_smi_present_but_fails_falls_through_to_rocm(self):
        # e.g. nvidia-smi installed but "couldn't communicate with the
        # NVIDIA driver" (real-world exit code 9 on a non-NVIDIA box).
        _make_stub(self.bin_dir, "nvidia-smi", exit_code=9)
        _make_stub(self.bin_dir, "rocm-smi", exit_code=0)
        self.assertEqual(detect_gpu_vendor(), "amd")

    def test_both_tools_succeed_nvidia_takes_precedence(self):
        _make_stub(self.bin_dir, "nvidia-smi", exit_code=0)
        _make_stub(self.bin_dir, "rocm-smi", exit_code=0)
        self.assertEqual(detect_gpu_vendor(), "nvidia")

    def test_both_tools_present_but_fail_returns_none(self):
        _make_stub(self.bin_dir, "nvidia-smi", exit_code=1)
        _make_stub(self.bin_dir, "rocm-smi", exit_code=1)
        self.assertIsNone(detect_gpu_vendor())


class ConfigureHardwareEnvTest(unittest.TestCase):
    """Covers the env-var-setting logic that used to live as untested,
    inline module-top code in experiments/angle_1.py before it was
    extracted into utils.hardware.configure_hardware_env() so run.py could
    also call it directly (see run.py's entry point and the End-of-Task
    Summary for why). Each test fully controls os.environ (clear=True) so
    the real test-runner's own PATH/CUDA_VISIBLE_DEVICES/etc never leak in.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.bin_dir = Path(self._tmp.name)

    def _patched_env(self, extra_env=None, platform="linux"):
        env = {"PATH": str(self.bin_dir)}
        if extra_env:
            env.update(extra_env)
        patchers = [
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("sys.platform", platform),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def test_amd_sets_rocm_platforms_egl_and_mirrors_hip_from_cuda(self):
        _make_stub(self.bin_dir, "rocm-smi", exit_code=0)
        self._patched_env(extra_env={"CUDA_VISIBLE_DEVICES": "3"}, platform="linux")

        vendor = configure_hardware_env()

        self.assertEqual(vendor, "amd")
        self.assertEqual(os.environ["JAX_PLATFORMS"], "rocm,cpu")
        self.assertEqual(os.environ["PYOPENGL_PLATFORM"], "egl")
        self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "3")

    def test_amd_without_cuda_visible_devices_set_does_not_invent_hip(self):
        _make_stub(self.bin_dir, "rocm-smi", exit_code=0)
        self._patched_env(platform="linux")

        configure_hardware_env()

        self.assertNotIn("HIP_VISIBLE_DEVICES", os.environ)

    def test_amd_never_overrides_operator_set_values(self):
        _make_stub(self.bin_dir, "rocm-smi", exit_code=0)
        self._patched_env(
            extra_env={
                "JAX_PLATFORMS": "operator_chosen",
                "PYOPENGL_PLATFORM": "operator_chosen",
                "HIP_VISIBLE_DEVICES": "operator_chosen",
                "CUDA_VISIBLE_DEVICES": "5",
            },
            platform="linux",
        )

        configure_hardware_env()

        self.assertEqual(os.environ["JAX_PLATFORMS"], "operator_chosen")
        self.assertEqual(os.environ["PYOPENGL_PLATFORM"], "operator_chosen")
        self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "operator_chosen")

    def test_nvidia_on_linux_leaves_jax_platforms_untouched(self):
        # This is the "zero behavior change to the NVIDIA path" guarantee:
        # JAX_PLATFORMS must stay unset so JAX auto-detects CUDA/CPU, exactly
        # as it did before AMD support existed.
        _make_stub(self.bin_dir, "nvidia-smi", exit_code=0)
        self._patched_env(platform="linux")

        vendor = configure_hardware_env()

        self.assertEqual(vendor, "nvidia")
        self.assertNotIn("JAX_PLATFORMS", os.environ)
        self.assertEqual(os.environ["PYOPENGL_PLATFORM"], "glfw")

    def test_no_vendor_on_linux_matches_nvidia_behavior_exactly(self):
        # No tools on PATH at all: must be indistinguishable from the
        # NVIDIA branch's env-var outcome (same "leave auto-detect alone"
        # fallback that existed before AMD support was added).
        self._patched_env(platform="linux")

        vendor = configure_hardware_env()

        self.assertIsNone(vendor)
        self.assertNotIn("JAX_PLATFORMS", os.environ)
        self.assertEqual(os.environ["PYOPENGL_PLATFORM"], "glfw")

    def test_darwin_sets_metal_regardless_of_vendor_tools(self):
        self._patched_env(platform="darwin")

        configure_hardware_env()

        self.assertEqual(os.environ["JAX_PLATFORMS"], "METAL,cpu")
        self.assertEqual(os.environ["PYOPENGL_PLATFORM"], "glfw")

    def test_second_call_in_same_process_is_cached_and_skips_redetection(self):
        _make_stub(self.bin_dir, "rocm-smi", exit_code=0)
        self._patched_env(platform="linux")

        first = configure_hardware_env()
        self.assertEqual(first, "amd")

        # Remove the stub entirely; if the second call re-ran detection it
        # would find nothing and return None instead of the cached "amd".
        (self.bin_dir / "rocm-smi").unlink()
        second = configure_hardware_env()

        self.assertEqual(second, "amd")
        self.assertIn(_VENDOR_SENTINEL_ENV_VAR, os.environ)


if __name__ == "__main__":
    unittest.main()
