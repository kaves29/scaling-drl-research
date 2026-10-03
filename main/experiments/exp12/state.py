"""Crash-safe on-disk layout for complete Exp 1/2 training states.

Each save goes to a new directory `<root>/step_<n>_<uid>`, and only then is
`<root>/LATEST` atomically repointed to it. A crash mid-save therefore leaves
the previous complete state as LATEST. Large arrays are stored with np.savez
(buffer) or Orbax (agent); only small metadata is pickled.
"""

import pickle
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from utils.atomic_io import atomic_write_text
from utils.paths import require_absolute

LATEST = "LATEST"
BUFFER_ARRAYS = ("_observations", "_actions", "_rewards", "_terminateds", "_truncateds", "_next_observations")


def latest_state_dir(root) -> Optional[Path]:
    pointer = Path(root) / LATEST
    if not pointer.exists():
        return None
    return Path(root) / pointer.read_text().strip()


def new_state_dir(root, interaction_step: int) -> Path:
    """A never-reused directory name: tensorstore caches by path within a process."""
    require_absolute(root, "state root")
    path = Path(root) / f"step_{interaction_step:09d}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)
    return path


def commit_state_dir(root, path: Path, keep_previous: bool = False) -> None:
    """Repoints LATEST to path; then removes older states (incl. uncommitted orphans of a crash)."""
    atomic_write_text(Path(root) / LATEST, path.name)
    if not keep_previous:
        for old in Path(root).glob("step_*"):
            if old != path:
                shutil.rmtree(old)


def save_buffer(buffer, path: Path) -> None:
    n = buffer._num_in_buffer
    np.savez(path / "buffer.npz", **{k.lstrip("_"): getattr(buffer, k)[:n] for k in BUFFER_ARRAYS})
    with open(path / "buffer_meta.pkl", "wb") as f:
        pickle.dump({
            "num_in_buffer": n,
            "current_idx": buffer._current_idx,
            "n_step_transitions": list(buffer._n_step_transitions),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_buffer(buffer, path: Path) -> None:
    with open(path / "buffer_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    n = meta["num_in_buffer"]
    with np.load(path / "buffer.npz") as data:
        for k in BUFFER_ARRAYS:
            getattr(buffer, k)[:n] = data[k.lstrip("_")]
    buffer._num_in_buffer = n
    buffer._current_idx = meta["current_idx"]
    buffer._n_step_transitions.clear()
    buffer._n_step_transitions.extend(meta["n_step_transitions"])


def save_meta(path: Path, meta: Dict[str, Any]) -> None:
    with open(path / "meta.pkl", "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_meta(path: Path) -> Dict[str, Any]:
    with open(path / "meta.pkl", "rb") as f:
        return pickle.load(f)
