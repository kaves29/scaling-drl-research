"""Crash-safe on-disk layout for complete Exp 1/2 training states.

Each save goes to a new directory `<root>/step_<n>_<uid>`, and only then is
`<root>/LATEST` atomically repointed to it. A crash mid-save therefore leaves
the previous complete state as LATEST. Large arrays are stored with np.savez
(buffer) or Orbax (agent); only small metadata is pickled.
"""

import pickle
import shutil
import uuid
import zipfile
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
    name = pointer.read_text().strip()
    path = Path(root) / name
    if not name or Path(name).name != name or name in (".", "..") or not path.is_dir():
        raise ValueError(f"{pointer}: invalid or missing state directory")
    if path.resolve().parent != Path(root).resolve():
        raise ValueError(f"{pointer}: state directory escapes root")
    return path


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
    n, idx = meta["num_in_buffer"], meta["current_idx"]
    capacity = len(buffer._observations)
    if (type(n) is not int or type(idx) is not int or not 0 <= n <= capacity or not 0 <= idx < capacity):
        raise ValueError("invalid replay count or index")
    # Validate all array headers before mutating any replay array; avoid loading
    # a second full replay into RAM just for structural validation.
    with zipfile.ZipFile(path / "buffer.npz") as archive:
        expected = {k.lstrip("_") + ".npy" for k in BUFFER_ARRAYS}
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError("replay array members differ")
        for k in BUFFER_ARRAYS:
            with archive.open(k.lstrip("_") + ".npy") as member:
                version = np.lib.format.read_magic(member)
                shape, _, dtype = np.lib.format._read_array_header(member, version)
            target = getattr(buffer, k)
            if shape != (n, *target.shape[1:]) or dtype != target.dtype:
                raise ValueError(f"replay {k}: shape or dtype differs")
    with np.load(path / "buffer.npz", allow_pickle=False) as data:
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


def _leaves(tree):
    import jax

    return jax.tree_util.tree_leaves_with_path(tree)


def load_agent_tree(state_dir):
    """Read values for exact comparison without requiring the saved device."""
    import jax
    import orbax.checkpoint
    from orbax.checkpoint import aggregate_handlers

    path = Path(state_dir) / "agent_ckpt"
    handler = aggregate_handlers.MsgpackHandler()
    try:
        structure = handler.deserialize(path / "checkpoint")
    finally:
        handler.close()
    restore_args = jax.tree_util.tree_map(
        lambda _: orbax.checkpoint.RestoreArgs(restore_type=np.ndarray), structure
    )
    return orbax.checkpoint.PyTreeCheckpointer().restore(
        str(path), restore_args=restore_args
    )


def state_differences(dir_a, dir_b, ignore_meta=("wandb_run_id",)):
    """Names of every component that differs bit-wise between two saved states."""
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    diffs = []
    tree_a, tree_b = load_agent_tree(dir_a), load_agent_tree(dir_b)
    import jax

    leaves_a, leaves_b = dict(_leaves(tree_a)), dict(_leaves(tree_b))
    if jax.tree_util.tree_structure(tree_a) != jax.tree_util.tree_structure(tree_b):
        diffs.append("agent:structure")
    for path in sorted(leaves_a.keys() & leaves_b.keys(), key=str):
        a, b = leaves_a[path], leaves_b[path]
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            diffs.append("agent:" + "/".join(str(p) for p in path))
    with open(dir_a / "obs_rms.pkl", "rb") as f:
        rms_a = pickle.load(f)
    with open(dir_b / "obs_rms.pkl", "rb") as f:
        rms_b = pickle.load(f)
    for k in sorted(rms_a.keys() | rms_b.keys()):
        if k not in rms_a or k not in rms_b or not np.array_equal(rms_a[k], rms_b[k]):
            diffs.append(f"obs_rms:{k}")
    with np.load(dir_a / "buffer.npz") as ba, np.load(dir_b / "buffer.npz") as bb:
        for k in sorted(set(ba.files) | set(bb.files)):
            if (
                k not in ba.files
                or k not in bb.files
                or not np.array_equal(ba[k], bb[k])
            ):
                diffs.append(f"buffer:{k}")
    with open(dir_a / "buffer_meta.pkl", "rb") as f:
        buffer_a = pickle.load(f)
    with open(dir_b / "buffer_meta.pkl", "rb") as f:
        buffer_b = pickle.load(f)
    for k in sorted(buffer_a.keys() | buffer_b.keys()):
        if (
            k not in buffer_a
            or k not in buffer_b
            or not _deep_equal(buffer_a[k], buffer_b[k])
        ):
            diffs.append(f"buffer_meta:{k}")
    with open(dir_a / "meta.pkl", "rb") as f:
        meta_a = pickle.load(f)
    with open(dir_b / "meta.pkl", "rb") as f:
        meta_b = pickle.load(f)
    for k in sorted(meta_a.keys() | meta_b.keys()):
        if k in ignore_meta:
            continue
        if (
            k not in meta_a
            or k not in meta_b
            or (
                pickle.dumps(meta_a[k]) != pickle.dumps(meta_b[k])
                and not _deep_equal(meta_a[k], meta_b[k])
            )
        ):
            diffs.append(f"meta:{k}")
    return diffs


def _deep_equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b
