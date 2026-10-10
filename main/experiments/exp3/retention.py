"""Opt-in retention of routine post-fork checkpoints that Exp1/Exp2 already write.

Exp1 control and Exp2 arms save a complete state every checkpoint_interval and then
delete the previous one (commit_state_dir). Retention hard-links (or copies) each
committed routine save inside an explicit step window into a separate root before it
is deleted. It performs no extra save, update, evaluation or RNG draw. Which window to
retain, and whether replay files are kept, are explicit inputs with no defaults.
"""

import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path

import numpy as np

from experiments.exp3.artifacts import digest

REPLAY_FILES = ("buffer.npz", "buffer_meta.pkl")
SNAPSHOT = "snapshot.json"


def validate_retention(retain):
    if retain is None:
        return None
    if set(retain) != {"root", "until_step", "replay"}:
        raise ValueError("retention needs exactly root, until_step and replay")
    if not Path(retain["root"]).is_absolute():
        raise ValueError("retention root must be absolute")
    if type(retain["until_step"]) is not int or retain["until_step"] < 0:
        raise ValueError("explicit retention until_step required")
    if retain["replay"] not in ("retain", "omit"):
        raise ValueError("explicit retention replay mode required: retain | omit")
    return retain


def replay_dimensions(state_dir):
    with zipfile.ZipFile(Path(state_dir) / "buffer.npz") as archive:
        dims = {}
        for name in ("observations", "actions"):
            with archive.open(name + ".npy") as member:
                version = np.lib.format.read_magic(member)
                shape, _, _ = np.lib.format._read_array_header(member, version)
            dims[name] = int(shape[-1])
    return dims["observations"], dims["actions"]


def _files(state_dir, replay):
    state_dir = Path(state_dir)
    return sorted(
        p.relative_to(state_dir)
        for p in state_dir.rglob("*")
        if p.is_file() and (replay == "retain" or p.name not in REPLAY_FILES)
    )


def retain_state(state_dir, root, step, replay):
    """Publish one immutable snapshot of a committed state directory; returns its path.

    A second save at the same step (a resumed source) must be byte-identical. A
    differing one is recorded as a conflict marker and never overwrites the first;
    Exp3 adapters refuse conflicted snapshots. The source run is not interrupted.
    """
    state_dir, root = Path(state_dir).resolve(), Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step_{step:09d}"
    files = _files(state_dir, replay)
    hashes = {str(f): digest(state_dir / f) for f in files}
    if target.exists():
        existing = json.loads((target / SNAPSHOT).read_text())
        if existing["files"] != hashes or existing["replay"] != replay:
            conflict = root / f"CONFLICT_{target.name}_{uuid.uuid4().hex}.json"
            conflict.write_text(json.dumps({"step": step, "source_state": str(state_dir),
                                            "files": hashes}, indent=1))
        return target
    obs_dim, act_dim = replay_dimensions(state_dir)
    temporary = root / f".{target.name}.{uuid.uuid4().hex}.pending"
    temporary.mkdir()
    modes = set()
    try:
        for f in files:
            (temporary / f).parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(state_dir / f, temporary / f)
                modes.add("hardlink")
            except OSError:
                shutil.copy2(state_dir / f, temporary / f)
                modes.add("copy")
        (temporary / SNAPSHOT).write_text(json.dumps({
            "schema_version": 1, "step": step, "source_state": str(state_dir),
            "replay": "retained" if replay == "retain" else "omitted",
            "observation_dim": obs_dim, "action_dim": act_dim,
            "files": hashes, "link_modes": sorted(modes),
        }, indent=1, sort_keys=True))
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def read_snapshot(state_dir):
    """None for an ordinary state directory; the verified snapshot record otherwise."""
    state_dir = Path(state_dir).resolve()
    path = state_dir / SNAPSHOT
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("schema_version") != 1 or record.get("replay") not in ("retained", "omitted"):
        raise ValueError("invalid retained snapshot record")
    if any(state_dir.parent.glob(f"CONFLICT_{state_dir.name}_*.json")):
        raise ValueError("retained snapshot has a conflicting re-save; refuse to use it")
    present = {str(p.relative_to(state_dir)) for p in state_dir.rglob("*")
               if p.is_file() and p.name != SNAPSHOT}
    if present != set(record["files"]):
        raise ValueError("retained snapshot files differ from its record")
    for name, sha in record["files"].items():
        if digest(state_dir / name) != sha:
            raise ValueError(f"retained snapshot checksum mismatch: {name}")
    return record
