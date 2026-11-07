"""Matmul precision for Exp 1/2 (amendment (v)): TF32 for every float32 matmul on GPUs that have it.

jax 0.4.34 documents `Precision.HIGH` (alias 'tensorfloat32') as "On GPU: uses tensorfloat32 where
available, otherwise float32", and precision as having no impact on CPU. Only matmuls and convolutions
are affected; every dtype stays float32. Check 1 and the A0 measurement use a local
`jax.default_matmul_precision("highest")` context instead.
"""

import atexit
import functools
import importlib.metadata
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import jax

TF32 = "tensorfloat32"


def set_matmul_precision() -> None:
    """Called at the start of every Exp 1/2 GPU entry point; does nothing on CPU."""
    if jax.default_backend() == "gpu":
        jax.config.update("jax_default_matmul_precision", TF32)


CACHE_DIR_ENV = "EXP12_JAX_CACHE_DIR"    # exact cache directory ("off" disables); Block B's identity test sets it
CACHE_ROOT_ENV = "EXP12_JAX_CACHE_ROOT"  # parent of the per-GPU-model directories; default <main>/jax_cache
_cache_events: Dict[str, int] = {}


def configure_compilation_cache() -> Optional[str]:
    """Persistent XLA compilation cache, one directory per GPU model (item 8a, 2026-10-05). A cache hit reuses the
    executable compiled by another process, with the same kernels; small compiles are cached too. Called at the
    start of every Exp 1/2 entry point, after set_matmul_precision()."""
    from jax._src import compilation_cache, monitoring

    path = os.environ.get(CACHE_DIR_ENV)
    if not path:
        root = os.environ.get(CACHE_ROOT_ENV) or str(Path(__file__).resolve().parents[2] / "jax_cache")
        path = os.path.join(root, re.sub(r"[^A-Za-z0-9._-]+", "_", jax.devices()[0].device_kind))
    if path == "off":
        return None
    os.makedirs(path, exist_ok=True)
    atomic_cache_writes()
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    compilation_cache.reset_cache()  # the cache is otherwise fixed at the process's first compile
    if not _cache_events:
        _cache_events["registered"] = 1
        monitoring.register_event_listener(
            lambda event, **kw: _cache_events.__setitem__(event, _cache_events.get(event, 0) + 1))
        atexit.register(lambda: print(
            f"[exp12] persistent compilation cache {jax.config.jax_compilation_cache_dir}: "
            f"hits {_cache_events.get('/jax/compilation_cache/cache_hits', 0)}, "
            f"misses {_cache_events.get('/jax/compilation_cache/cache_misses', 0)}", flush=True))
    return path


def atomic_cache_writes() -> None:
    """jax 0.4.34's LRUCache.put writes the entry in place, so a process reading it meanwhile gets a truncated entry
    ("Error -5 while decompressing data"), and a writer killed mid-write leaves one that stays truncated. Each entry
    is written to its own temporary file and renamed into place instead; the bytes written are unchanged."""
    from jax._src import lru_cache

    if getattr(lru_cache.LRUCache.put, "atomic", False):
        return
    put = lru_cache.LRUCache.put

    @functools.wraps(put)
    def atomic_put(self, key: str, val: bytes) -> None:
        if self.eviction_enabled or not lru_cache._is_local_filesystem(str(self.path)):
            return put(self, key, val)
        if not key:
            raise ValueError("key cannot be empty")
        cache_path = self.path / f"{key}{lru_cache._CACHE_SUFFIX}"
        if cache_path.exists():
            return
        fd, tmp = tempfile.mkstemp(dir=self.path, prefix=f".{key}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(val)
            os.replace(tmp, cache_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        (self.path / f"{key}{lru_cache._ATIME_SUFFIX}").write_bytes(time.time_ns().to_bytes(8, "little"))

    atomic_put.atomic = True
    lru_cache.LRUCache.put = atomic_put


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
            "matmul_precision": jax.config.jax_default_matmul_precision,
            "compilation_cache_dir": jax.config.jax_compilation_cache_dir}
