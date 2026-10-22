"""Immutable numeric tasks and explicit pairing rules for retrospective probes.

This API does not change the operational Method C target or campaign schedule.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.exp12.probe import iqm


def array_digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(json.dumps(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class FrozenTask:
    observations: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    minibatch_indices: np.ndarray
    provenance: dict | str

    def __post_init__(self):
        for field in self.array_names:
            original = np.asarray(getattr(self, field))
            value = np.frombuffer(original.tobytes(), dtype=original.dtype).reshape(
                original.shape
            )
            object.__setattr__(self, field, value)
        o, a, y, ix = (getattr(self, n) for n in self.array_names)
        if o.ndim != 2 or a.ndim != 2 or y.shape != (len(o),) or len(a) != len(o):
            raise ValueError("task input/target shape mismatch")
        if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in (o, a, y)):
            raise ValueError("canonical task inputs must be finite float32")
        if (
            ix.ndim != 2
            or ix.dtype != np.int32
            or ix.size == 0
            or ix.min() < 0
            or ix.max() >= len(o)
        ):
            raise ValueError("invalid fixed minibatch indices")
        metadata = (
            json.loads(self.provenance)
            if isinstance(self.provenance, str)
            else self.provenance
        )
        object.__setattr__(
            self, "provenance", json.dumps(metadata, sort_keys=True, allow_nan=False)
        )

    @property
    def metadata(self):
        """Returns a detached metadata copy; the canonical JSON identity is immutable."""
        return json.loads(self.provenance)

    @property
    def array_names(self):
        return ("observations", "actions", "targets", "minibatch_indices")

    @property
    def manifest(self):
        return {
            "schema": 1,
            "array_sha256": {
                n: array_digest(getattr(self, n)) for n in self.array_names
            },
            "provenance": self.metadata,
        }

    @property
    def task_id(self):
        return hashlib.sha256(
            json.dumps(self.manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "arrays.npz").open("xb") as stream:
            np.savez(stream, **{n: getattr(self, n) for n in self.array_names})
            stream.flush()
            os.fsync(stream.fileno())
        with (directory / "manifest.json").open("x") as stream:
            json.dump({**self.manifest, "task_id": self.task_id}, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        with np.load(directory / "arrays.npz", allow_pickle=False) as values:
            task = cls(
                **{
                    n: values[n]
                    for n in ("observations", "actions", "targets", "minibatch_indices")
                },
                provenance=manifest["provenance"]
            )
        if (
            task.manifest
            != {k: manifest[k] for k in ("schema", "array_sha256", "provenance")}
            or task.task_id != manifest["task_id"]
        ):
            raise ValueError("numeric task or provenance checksum mismatch")
        return task


def fixed_task_paired_summary(
    fresh_mse,
    current_mse,
    variance,
    fresh_task_ids,
    current_task_ids,
    fresh_steps,
    current_steps,
):
    """IQM10 of paired FP32 scores, with identical numeric tasks/horizons required."""
    f, c, b = [np.asarray(x, np.float32) for x in (fresh_mse, current_mse, variance)]
    if any(x.shape != (10,) or not np.isfinite(x).all() for x in (f, c, b)):
        raise ValueError(
            "IQM10 requires ten finite paired targets; screens are not IQM10"
        )
    if np.any(f < 0) or np.any(c < 0) or np.any(b < 0):
        raise ValueError("MSE and variance must be nonnegative")
    if len(fresh_task_ids) != 10 or list(fresh_task_ids) != list(current_task_ids):
        raise ValueError(
            "fixed-task pairing requires identical numeric task identities"
        )
    if len(set(fresh_task_ids)) != 10:
        raise ValueError(
            "ten distinct target tasks are required, not duplicated records"
        )
    if fresh_steps != current_steps:
        raise ValueError("cannot pair different fitting horizons")
    loss = (b - f) - (b - c)
    return {
        "paired_losses": loss,
        "paired_loss_iqm10": iqm(loss),
        "fresh_score_iqm10": iqm(b - f),
        "current_score_iqm10": iqm(b - c),
        "interpretation": "conditional fixed-task finite-budget fitting; no causal actor claim",
    }
