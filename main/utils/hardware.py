"""OS-level GPU vendor detection and hardware-aware environment configuration.

This module must never import jax or call jax.devices() (except in the
post-init validation branch of configure_hardware_env(), which only runs
after the caller has already imported jax itself - see its docstring).
JAX_PLATFORMS has to be set correctly before JAX initializes at all, so "ask
JAX which backend to use" is not available for detection itself - that part
is done purely via OS-level signals (vendor CLI tools on PATH), the same way
exea.md's own SETUP step checks for a GPU after the fact, just earlier and
vendor-aware.

configure_hardware_env() is the single source of truth for the
detect-and-set-env-vars logic; call it from any entry point that needs
hardware-aware JAX_PLATFORMS/PYOPENGL_PLATFORM/HIP_VISIBLE_DEVICES before
`import jax` happens (currently: run.py's true top, and defensively again in
experiments/angle_1.py for any code path that imports experiments.angle_1
without going through run.py, e.g. tests). It is idempotent and cheap to
call more than once in the same process - the first call's detection result
is recorded in a sentinel env var, so a second call skips re-running
nvidia-smi/rocm-smi and just returns the already-recorded vendor.
"""

import os
import shutil
import subprocess
import sys
from typing import Optional

_SMI_TIMEOUT_SECONDS = 10
_VENDOR_SENTINEL_ENV_VAR = "_ECHOCRITIC_GPU_VENDOR"


def _tool_runs(binary: str) -> bool:
    """True if `binary` is on PATH and exits 0 with no arguments.

    A bare `nvidia-smi` / `rocm-smi` invocation is the standard way both
    tools report "I can actually talk to the driver" - e.g. nvidia-smi exits
    non-zero (and prints "couldn't communicate with the NVIDIA driver") if
    the binary is present but no driver/GPU backs it, which a plain
    `shutil.which` presence check would miss.
    """
    path = shutil.which(binary)
    if path is None:
        return False
    try:
        result = subprocess.run(
            [path],
            capture_output=True,
            timeout=_SMI_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def detect_gpu_vendor() -> Optional[str]:
    """Return 'nvidia', 'amd', or None, using only OS-level signals.

    Checked in that order: if both nvidia-smi and rocm-smi somehow succeed
    on the same machine, 'nvidia' wins, so the existing, already-verified
    NVIDIA path is preferred over an ambiguous mixed signal rather than
    silently switching an NVIDIA box onto the new AMD branch.
    """
    if _tool_runs("nvidia-smi"):
        return "nvidia"
    if _tool_runs("rocm-smi"):
        return "amd"
    return None


def configure_hardware_env() -> Optional[str]:
    """Detects GPU vendor (OS-level only, safe before `import jax`) and sets
    JAX_PLATFORMS / PYOPENGL_PLATFORM / HIP_VISIBLE_DEVICES via
    os.environ.setdefault - never overrides a value an operator/launcher
    already set explicitly. Must be called before `import jax` by whichever
    entry point calls it first.

    AMD: JAX_PLATFORMS=rocm,cpu; PYOPENGL_PLATFORM=egl (headless cloud, the
    only current AMD usage); HIP_VISIBLE_DEVICES mirrored from
    CUDA_VISIBLE_DEVICES when an operator/launcher already set that (this
    repo's existing per-process device-assignment convention - see
    .claude/rules/compute-and-data-safety.md), so existing launch commands
    keep working unchanged on AMD without learning a new env var name.

    NVIDIA / no vendor detected / macOS: unchanged from before AMD support
    existed - PYOPENGL_PLATFORM defaults to 'glfw', and on macOS JAX_PLATFORMS
    defaults to 'METAL,cpu'; on Linux with NVIDIA or no vendor detected,
    JAX_PLATFORMS is left untouched so JAX auto-detects CUDA/CPU normally.

    Idempotent: the detected vendor is recorded in a sentinel env var on the
    first call in this process; subsequent calls (from a second entry point
    in the same process) skip re-running nvidia-smi/rocm-smi and return the
    already-recorded vendor, so calling this from both run.py's top AND
    experiments/angle_1.py's top (defense-in-depth for direct imports) never
    double-detects or risks the two call sites disagreeing.

    Returns the detected vendor: 'nvidia', 'amd', or None.
    """
    if _VENDOR_SENTINEL_ENV_VAR in os.environ:
        recorded = os.environ[_VENDOR_SENTINEL_ENV_VAR]
        return None if recorded == "none" else recorded

    vendor = detect_gpu_vendor()
    os.environ[_VENDOR_SENTINEL_ENV_VAR] = vendor or "none"

    if vendor == "amd":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("JAX_PLATFORMS", "rocm,cpu")
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ.setdefault("HIP_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ.setdefault("PYOPENGL_PLATFORM", "glfw")
        if sys.platform == "darwin":
            os.environ.setdefault("JAX_PLATFORMS", "METAL,cpu")

    return vendor


def validate_rocm_jax_available(vendor: Optional[str]) -> None:
    """Post-`import jax` sanity check: if AMD was detected, confirm JAX
    actually found a ROCm device rather than silently falling back to CPU.
    Must be called AFTER the caller's own `import jax` (this function does
    not import jax itself, to keep this module import-safe pre-jax) - see
    experiments/angle_1.py for the call site, right after
    jax.config.update(...).

    rocm-smi finding an AMD GPU doesn't guarantee the installed jax/jaxlib
    build actually has a working ROCm plugin - if it doesn't, JAX silently
    runs on CPU instead of erroring, which is exactly the "discovered three
    hours later" failure mode exea.md's own SETUP step already guards
    against for the general case. Same check, ROCm-aware, callable from any
    entry point (not just angle_1.py) once jax is imported there.
    """
    if vendor != "amd":
        return
    import jax  # local import: only reached post-jax-import by the caller

    jax_devices = jax.devices()
    # 'rocm' is the documented JAX/PJRT platform name for the ROCm plugin;
    # 'gpu' is included defensively, mirroring exea.md's own SETUP check
    # (`d.platform in ('gpu', 'cuda', 'tpu')`), which already hedges across
    # platform-string naming because it isn't stable across jax versions.
    if not any(d.platform in ("rocm", "gpu") for d in jax_devices):
        print(
            "[hardware] WARNING: rocm-smi detected an AMD GPU, but "
            f"jax.devices() = {jax_devices} contains no rocm/gpu-platform "
            "device - training will silently run on CPU. Verify a "
            "ROCm-matched jax/jaxlib build is installed in this environment."
        )
