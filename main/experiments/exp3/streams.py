"""Ordered, checksummed, bounded-memory raw arrivals. Replay is not a trajectory."""

import copy
import contextlib
import functools
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from experiments.exp3.artifacts import digest
from utils.atomic_io import atomic_write_text

FIELDS = (
    "observation",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_observation",
)


def write_json(path, value):
    atomic_write_text(Path(path), json.dumps(value, sort_keys=True, indent=2))


def atomic_pickle(path, value):
    """Private checkpoint publication. Never replace a scientific source artifact."""
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with open(temp, "xb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class StreamWriter:
    """Single-writer atomic chunks; an incomplete prefix is never a complete source."""

    def __init__(self, root, provenance, chunk_size, resume=False):
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("stream chunk_size must be explicitly positive")
        self.root = Path(root).resolve()
        self.chunk_size = chunk_size
        self.pending = []
        if resume:
            reader = StreamReader(self.root, require_complete=False)
            if reader.manifest["provenance"] != provenance:
                raise ValueError("stream resume provenance differs")
            if reader.manifest["complete"]:
                raise ValueError("cannot append to a sealed stream")
            # Verify every committed byte before accepting a prefix.
            list_count = sum(1 for _ in reader)
            if list_count != reader.manifest["count"]:
                raise ValueError("incomplete committed stream")
            self.manifest = reader.manifest
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            self.manifest = {
                "schema_version": 1,
                "provenance": provenance,
                "count": 0,
                "chunks": [],
                "complete": False,
            }
            self._commit()

    def _commit(self):
        write_json(self.root / "manifest.json", self.manifest)

    def append(self, step, transition, updates, normalization):
        if self.manifest["complete"]:
            raise ValueError("sealed stream")
        expected = (
            self.manifest["provenance"]["start_step"]
            + self.manifest["count"]
            + len(self.pending)
            + 1
        )
        if type(step) is not int or step != expected:
            raise ValueError(f"transition order: expected step {expected}, got {step}")
        if type(updates) is not int or updates < 0:
            raise ValueError("invalid source update budget")
        if set(transition) != set(FIELDS):
            raise ValueError("raw transition fields differ")
        arrays = {k: np.array(transition[k], copy=True) for k in FIELDS}
        if any(a.dtype.hasobject or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("nonfinite/object transition")
        if any(
            arrays[k].ndim != 2 or arrays[k].shape[0] != 1
            for k in ("observation", "action", "next_observation")
        ):
            raise ValueError("source stream requires one vectorized environment")
        if any(arrays[k].shape != (1,) for k in ("reward", "terminated", "truncated")):
            raise ValueError("invalid transition scalar shape")
        if arrays["observation"].shape != arrays["next_observation"].shape:
            raise ValueError("source observation dimensions differ")
        if any(
            not np.isin(arrays[k], (0, 1)).all() for k in ("terminated", "truncated")
        ):
            raise ValueError("invalid transition terminal mask")
        if set(normalization) != {"mean", "var", "count"}:
            raise ValueError("missing source normalization")
        norm = {k: np.asarray(v) for k, v in normalization.items()}
        if (
            any(not np.isfinite(v).all() for v in norm.values())
            or norm["mean"].shape != norm["var"].shape
            or np.any(norm["var"] < 0)
            or np.any(norm["count"] < 0)
        ):
            raise ValueError("invalid source normalization")
        self.pending.append(
            {
                **arrays,
                "step": step,
                "updates": updates,
                "normalization": copy.deepcopy(normalization),
            }
        )
        if len(self.pending) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        # Generation IDs avoid overwriting an orphan from an interrupted publication.
        import uuid

        name = f"chunk_{len(self.manifest['chunks']):08d}_{uuid.uuid4().hex}.npz"
        values = {k: np.stack([r[k] for r in self.pending]) for k in FIELDS}
        values.update(
            {
                "step": np.asarray([r["step"] for r in self.pending], np.int64),
                "updates": np.asarray([r["updates"] for r in self.pending], np.int64),
            }
        )
        for k in ("mean", "var", "count"):
            values[f"rms_{k}"] = np.stack(
                [np.asarray(r["normalization"][k]) for r in self.pending]
            )
        temp = self.root / (name + ".tmp")
        with open(temp, "xb") as f:
            np.savez(f, **values)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.root / name)
        self.manifest["chunks"].append(
            {
                "name": name,
                "sha256": digest(self.root / name),
                "count": len(self.pending),
                "bytes": (self.root / name).stat().st_size,
            }
        )
        self.manifest["count"] += len(self.pending)
        self.pending.clear()
        self._commit()

    def seal(self):
        self.flush()
        self.manifest["complete"] = True
        self._commit()


class StreamReader:
    def __init__(self, root, require_complete=True):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if (
            m.get("schema_version") != 1
            or type(m.get("count")) is not int
            or m["count"] < 0
        ):
            raise ValueError("invalid stream manifest")
        if require_complete and m.get("complete") is not True:
            raise ValueError("source stream is incomplete")
        if (
            not isinstance(m.get("provenance"), dict)
            or type(m["provenance"].get("start_step")) is not int
        ):
            raise ValueError("missing stream provenance/start step")
        names = [c["name"] for c in m["chunks"]]
        if (
            len(set(names)) != len(names)
            or sum(c["count"] for c in m["chunks"]) != m["count"]
        ):
            raise ValueError("duplicated chunks or incorrect stream length")
        self.fingerprint = digest(self.root / "manifest.json")

    def __iter__(self):
        expected = self.manifest["provenance"]["start_step"] + 1
        for chunk in self.manifest["chunks"]:
            name = chunk["name"]
            path = (self.root / name).resolve()
            if (
                Path(name).name != name
                or path.parent != self.root
                or digest(path) != chunk["sha256"]
            ):
                raise ValueError(f"stream path/checksum mismatch: {name}")
            with np.load(path, allow_pickle=False) as arrays:
                fields = set(FIELDS) | {
                    "step",
                    "updates",
                    "rms_mean",
                    "rms_var",
                    "rms_count",
                }
                if set(arrays.files) != fields or any(
                    len(arrays[k]) != chunk["count"] for k in fields
                ):
                    raise ValueError("invalid stream chunk fields/length")
                for i in range(chunk["count"]):
                    step, updates = arrays["step"][i], arrays["updates"][i]
                    if (
                        step != expected
                        or updates < 0
                        or arrays["step"].dtype != np.int64
                        or arrays["updates"].dtype != np.int64
                    ):
                        raise ValueError("transition order/update budget differs")
                    transition = {k: arrays[k][i].copy() for k in FIELDS}
                    norm = {
                        k: arrays[f"rms_{k}"][i].copy()
                        for k in ("mean", "var", "count")
                    }
                    if any(
                        not np.isfinite(v).all()
                        for v in (*transition.values(), *norm.values())
                    ):
                        raise ValueError("nonfinite stream payload")
                    if (
                        norm["mean"].shape != norm["var"].shape
                        or np.any(norm["var"] < 0)
                        or np.any(norm["count"] < 0)
                    ):
                        raise ValueError("invalid stream normalization")
                    if any(
                        transition[k].ndim != 2 or transition[k].shape[0] != 1
                        for k in ("observation", "next_observation", "action")
                    ) or any(
                        transition[k].shape != (1,)
                        for k in ("reward", "terminated", "truncated")
                    ):
                        raise ValueError("invalid stream transition shape")
                    if any(
                        not np.isin(transition[k], (0, 1)).all()
                        for k in ("terminated", "truncated")
                    ):
                        raise ValueError("invalid stream terminal mask")
                    yield {
                        "step": int(step),
                        "updates": int(updates),
                        "transition": transition,
                        "normalization": norm,
                    }
                    expected += 1


class RecordingBuffer:
    """Exp3-only adapter; records raw add input before n-step replay mutates it."""

    def __init__(self, trainer, writer):
        self.buffer, self.trainer, self.writer = trainer.buffer, trainer, writer
        self.pending = None

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    def add(self, transition):
        if self.pending is not None:
            raise RuntimeError("recording after_step hook was not called")
        self.pending = (
            copy.deepcopy(transition),
            {
                k: copy.deepcopy(getattr(self.trainer.agent.obs_rms, k))
                for k in ("mean", "var", "count")
            },
            self.trainer.update_step,
        )
        self.buffer.add(transition)

    def after_step(self, trainer):
        if self.pending is None:
            raise RuntimeError("missing raw source arrival")
        transition, norm, before = self.pending
        self.writer.append(
            trainer.interaction_step, transition, trainer.update_step - before, norm
        )
        self.pending = None


@contextlib.contextmanager
def benchmark_source(trainer, benchmark):
    """Optional timings around existing calls. No RNG, extra steps, updates or saves."""
    from unittest.mock import patch

    if benchmark is None:
        yield
        return
    with contextlib.ExitStack() as stack:
        for obj, name, label in (
            (trainer.train_env, "step", "environment_step"),
            (trainer.agent, "update_many", "sac_update"),
            (trainer, "_evaluate", "evaluation"),
            (trainer, "save", "checkpoint"),
        ):
            original = getattr(obj, name)
            stack.enter_context(
                patch.object(
                    obj, name, functools.partial(benchmark.measure, label, original)
                )
            )
        yield


def record_training(
    trainer,
    last_step,
    writer,
    after_step=None,
    before_first_update=None,
    benchmark=None,
):
    """Opt-in future source continuation, same loop/RNG/eval/save behavior as Exp12.

    Restore the exact source checkpoint before calling. A stream prefix ahead of
    its source checkpoint is refused, rather than duplicating arrivals on resume.
    """
    expected = writer.manifest["provenance"]["start_step"] + writer.manifest["count"]
    if trainer.interaction_step != expected:
        raise ValueError("source checkpoint and committed stream cursor differ")
    original = trainer.buffer
    adapter = RecordingBuffer(trainer, writer)
    trainer.buffer = adapter
    try:

        def hook(t):
            adapter.after_step(t)
            if after_step:
                after_step(t)

        with benchmark_source(trainer, benchmark):
            trainer.train(
                last_step, after_step=hook, before_first_update=before_first_update
            )
        writer.seal()
    finally:
        trainer.buffer = original


class Benchmark:
    """Explicit synchronized timings; first-call compilation is not isolated compiler time."""

    def __init__(self):
        self.values = {}

    @contextlib.contextmanager
    def compilation(self):
        """Pinned JAX backend-compile call time; excludes Python tracing/lowering."""
        from jax._src import compiler
        from unittest.mock import patch

        original = compiler.backend_compile

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                row = self.values.setdefault(
                    "backend_compile",
                    {"count": 0, "seconds": 0.0, "first_call_seconds": elapsed},
                )
                row["count"] += 1
                row["seconds"] += elapsed

        with patch.object(compiler, "backend_compile", timed):
            yield

    def measure(self, name, function, *args, **kwargs):
        import jax

        t = time.perf_counter()
        result = function(*args, **kwargs)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - t
        row = self.values.setdefault(
            name, {"count": 0, "seconds": 0.0, "first_call_seconds": elapsed}
        )
        row["count"] += 1
        row["seconds"] += elapsed
        return result

    def report(self, root):
        import jax
        import resource

        root = Path(root)
        return {
            "timings": self.values,
            "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "device_memory": [d.memory_stats() for d in jax.devices()],
            "storage_bytes": sum(
                p.stat().st_size for p in root.rglob("*") if p.is_file()
            ),
            "compile_time": "backend_compile timings exclude tracing/lowering; first-call stage times include compilation/execution; nested timings are not additive",
        }
