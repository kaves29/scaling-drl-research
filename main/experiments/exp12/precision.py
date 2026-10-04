"""Matmul precision for Exp 1/2 (amendment (v)): TF32 for every float32 matmul on GPUs that have it.

jax 0.4.34 documents `Precision.HIGH` (alias 'tensorfloat32') as "On GPU: uses tensorfloat32 where
available, otherwise float32", and precision as having no impact on CPU. Only matmuls and convolutions
are affected; every dtype stays float32. Check 1 and the A0 measurement use a local
`jax.default_matmul_precision("highest")` context instead.
"""

import importlib.metadata
from typing import Dict

import jax

TF32 = "tensorfloat32"


def set_matmul_precision() -> None:
    """Called at the start of every Exp 1/2 GPU entry point; does nothing on CPU."""
    if jax.default_backend() == "gpu":
        jax.config.update("jax_default_matmul_precision", TF32)


def runtime_info() -> Dict:
    """Recorded with every launch in run_metadata.json."""
    import jaxlib
    from jax.extend.backend import get_backend

    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    d = jax.devices()[0]
    return {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "platform": d.platform,
            "device_kind": d.device_kind, "backend_platform_version": get_backend().platform_version,
            "jax_cuda12_plugin": version("jax-cuda12-plugin"), "jax_cuda12_pjrt": version("jax-cuda12-pjrt"),
            "matmul_precision": jax.config.jax_default_matmul_precision}
