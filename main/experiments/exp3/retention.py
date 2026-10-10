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
    if retain["until_step"] != "arm_end" and (
        type(retain["until_step"]) is not int or retain["until_step"] < 0
    ):
        raise ValueError("explicit retention until_step or arm_end policy required")
    if retain["replay"] not in ("retain", "omit"):
        raise ValueError("explicit retention replay mode required: retain | omit")
    return retain


def retention_end(retain, plan):
    """Resolve only an explicitly selected storage window from the actual fork."""
    return (
        int(plan["arm_end_step"])
        if retain["until_step"] == "arm_end"
        else retain["until_step"]
    )


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


def _local_device_strings():
    import jax

    found = set()
    for backend in ("cpu", "gpu", "tpu"):
        try:
            found |= {str(d) for d in jax.local_devices(backend=backend)}
        except RuntimeError:
            continue
    return found


def validate_checkpoint(state_dir):
    """Integrity and device-backed restorability of a saved agent checkpoint.

    Independent of value equality (load_agent_tree reads NumPy without the saved
    device, which cannot reveal broken Orbax sharding metadata). Requires: readable
    Orbax tree metadata; a `_sharding` record that parses and covers exactly the
    array leaves; every saved device present on this host; an actual Orbax restore
    with the saved shardings; and restored values bitwise equal to the NumPy read.
    Serialization/layout bytes may differ between valid saves; this checks meaning,
    not bytes. Raises ValueError with an explicit reason.
    """
    import base64

    import jax
    import orbax.checkpoint

    from experiments.exp12.state import _leaves, bitwise_equal, load_agent_tree

    path = Path(state_dir) / "agent_ckpt"
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # The sharding record first: Orbax's own metadata reader consumes it, and its
    # errors would otherwise hide the specific reason.
    try:
        raw = json.loads((path / "_sharding").read_text())
        sharding = {
            base64.b64decode(k, validate=True).decode(): json.loads(v)
            for k, v in raw.items()
        }
    except Exception as exc:
        raise ValueError(f"invalid Orbax sharding metadata: {exc!r}") from exc
    devices = set()
    for spec in sharding.values():
        if (
            not isinstance(spec, dict)
            or spec.get("sharding_type") != "SingleDeviceSharding"
            or not isinstance(spec.get("device_str"), str)
        ):
            raise ValueError(f"incompatible Orbax sharding entry: {spec!r}")
        devices.add(spec["device_str"])
    def dotted(key_path):
        return ".".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in key_path)

    from orbax.checkpoint import aggregate_handlers

    handler = aggregate_handlers.MsgpackHandler()
    try:
        structure = handler.deserialize(path / "checkpoint")
    except Exception as exc:
        raise ValueError(f"invalid Orbax checkpoint structure: {exc!r}") from exc
    finally:
        handler.close()
    leaves = {dotted(k) for k, _ in jax.tree_util.tree_flatten_with_path(structure)[0]}
    if set(sharding) != leaves:
        raise ValueError("Orbax sharding metadata does not cover the checkpoint leaves")
    missing = devices - _local_device_strings()
    if missing:
        raise ValueError(
            f"saved checkpoint devices unavailable here, restore unverifiable: {sorted(missing)}"
        )
    try:
        metadata = checkpointer.metadata(str(path))
    except Exception as exc:
        raise ValueError(f"invalid Orbax checkpoint metadata: {exc!r}") from exc
    if {dotted(k) for k, _ in jax.tree_util.tree_flatten_with_path(metadata)[0]} != leaves:
        raise ValueError("Orbax checkpoint metadata differs from its tree structure")
    try:
        restored = checkpointer.restore(str(path))
    except Exception as exc:
        raise ValueError(f"checkpoint restore with saved sharding failed: {exc!r}") from exc
    values = dict(_leaves(load_agent_tree(state_dir)))
    placed = dict(_leaves(restored))
    if values.keys() != placed.keys() or not all(
        bitwise_equal(np.asarray(placed[k]), values[k]) for k in values
    ):
        raise ValueError("device-backed restore differs from the checkpoint values")
    return {
        "orbax_metadata": "valid",
        "sharding": "SingleDeviceSharding",
        "devices": sorted(devices),
        "restore": "device-backed with saved sharding",
        "leaves": len(values),
    }


def record_failure(root, step, state_dir, error):
    """Visible record that retention of a routine save failed; never a snapshot."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"FAILED_step_{step:09d}_{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps({"step": step, "source_state": str(state_dir), "reason": str(error)}, indent=1)
    )
    return path


def retain_state(state_dir, root, step, replay):
    """Publish one immutable snapshot of a committed state directory; returns its path.

    Every publication first validates the source checkpoint (validate_checkpoint).
    At an existing step, the existing snapshot must pass read_snapshot (record
    checksums, no conflict) before anything is accepted. Identical bytes are then
    accepted; otherwise both checkpoints are validated and compared logically:
    bitwise-equal state is recorded as EQUIVALENT, any difference as CONFLICT. The
    first snapshot is never overwritten. Invalid input raises ValueError.
    """
    state_dir, root = Path(state_dir).resolve(), Path(root).resolve()
    if root == state_dir or root.is_relative_to(state_dir):
        raise ValueError("retention root must be outside the source state")
    routine_root = state_dir.parent
    if (routine_root / "LATEST").exists() and (
        root == routine_root or root.is_relative_to(routine_root)
    ):
        raise ValueError(
            "retention root must be outside the mutable routine checkpoint root"
        )
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step_{step:09d}"
    files = _files(state_dir, replay)
    hashes = {str(f): digest(state_dir / f) for f in files}
    if target.exists():
        try:
            existing = read_snapshot(target)  # current bytes vs record, no conflict
        except ValueError as exc:
            raise ValueError(f"existing retained snapshot failed integrity: {exc}") from exc
        mode = "retained" if replay == "retain" else "omitted"
        if existing["files"] == hashes and existing["replay"] == mode:
            return target
        validate_checkpoint(state_dir)  # never certify an invalid incoming checkpoint
        validate_checkpoint(target)
        # A resumed source can legitimately re-save the same step (e.g. a death
        # between its final save and DONE). Orbax/pickle bytes are not reproducible
        # across saves, so compare the logical state bitwise before declaring a
        # conflict; identical state is recorded, never silently replaced.
        differences = (
            ["replay mode"]
            if existing["replay"] != mode
            else logical_differences(target, state_dir, replay)
        )
        kind = "CONFLICT" if differences else "EQUIVALENT"
        record = root / f"{kind}_{target.name}_{uuid.uuid4().hex}.json"
        record.write_text(
            json.dumps(
                {
                    "step": step,
                    "source_state": str(state_dir),
                    "files": hashes,
                    "logical_differences": differences,
                },
                indent=1,
            )
        )
        return target
    validation = validate_checkpoint(state_dir)
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
        (temporary / SNAPSHOT).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "step": step,
                    "source_state": str(state_dir),
                    "replay": "retained" if replay == "retain" else "omitted",
                    "observation_dim": obs_dim,
                    "action_dim": act_dim,
                    "files": hashes,
                    "link_modes": sorted(modes),
                    "validation": validation,
                },
                indent=1,
                sort_keys=True,
            )
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def logical_differences(snapshot, state_dir, replay):
    """Bitwise differences between a retained snapshot and a source state, using the
    existing exact comparator (shape, dtype and bytes; wandb_run_id ignored as in
    state_differences). Replay files are compared only when they were retained."""
    import pickle

    from experiments.exp12.state import (
        _deep_equal,
        _leaves,
        bitwise_equal,
        load_agent_tree,
        state_differences,
    )

    if replay == "retain":
        return state_differences(snapshot, state_dir)
    import jax

    diffs = []
    tree_a, tree_b = load_agent_tree(snapshot), load_agent_tree(state_dir)
    if jax.tree_util.tree_structure(tree_a) != jax.tree_util.tree_structure(tree_b):
        diffs.append("agent:structure")
    leaves_a, leaves_b = dict(_leaves(tree_a)), dict(_leaves(tree_b))
    for path in sorted(leaves_a.keys() & leaves_b.keys(), key=str):
        if not bitwise_equal(leaves_a[path], leaves_b[path]):
            diffs.append("agent:" + "/".join(str(p) for p in path))
    loaded = {}
    for name in ("obs_rms.pkl", "meta.pkl"):
        with open(Path(snapshot) / name, "rb") as a, open(Path(state_dir) / name, "rb") as b:
            loaded[name] = (pickle.load(a), pickle.load(b))
    rms_a, rms_b = loaded["obs_rms.pkl"]
    for k in sorted(rms_a.keys() | rms_b.keys()):
        if k not in rms_a or k not in rms_b or not bitwise_equal(rms_a[k], rms_b[k]):
            diffs.append(f"obs_rms:{k}")
    meta_a, meta_b = loaded["meta.pkl"]
    for k in sorted(meta_a.keys() | meta_b.keys()):
        if k == "wandb_run_id":
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


def read_snapshot(state_dir):
    """None for an ordinary state directory; the verified snapshot record otherwise."""
    state_dir = Path(state_dir).resolve()
    path = state_dir / SNAPSHOT
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("schema_version") != 1 or record.get("replay") not in (
        "retained",
        "omitted",
    ):
        raise ValueError("invalid retained snapshot record")
    if any(state_dir.parent.glob(f"CONFLICT_{state_dir.name}_*.json")):
        raise ValueError(
            "retained snapshot has a conflicting re-save; refuse to use it"
        )
    present = {
        str(p.relative_to(state_dir))
        for p in state_dir.rglob("*")
        if p.is_file() and p.name != SNAPSHOT
    }
    if present != set(record["files"]):
        raise ValueError("retained snapshot files differ from its record")
    for name, sha in record["files"].items():
        if digest(state_dir / name) != sha:
            raise ValueError(f"retained snapshot checksum mismatch: {name}")
    return record


def retention_report(root, required_steps):
    """Separate retention verdict for a capture: required snapshots must exist and
    pass read_snapshot; any CONFLICT or FAILED record fails the retention verdict."""
    root = Path(root)
    reasons = []
    for step in sorted(set(required_steps)):
        target = root / f"step_{step:09d}"
        if not target.is_dir():
            reasons.append(f"step {step}: required snapshot missing")
            continue
        try:
            read_snapshot(target)
        except ValueError as exc:
            reasons.append(f"step {step}: {exc}")
    for kind in ("CONFLICT", "FAILED"):
        for marker in sorted(root.glob(f"{kind}_*.json")):
            reasons.append(f"{kind} record: {marker.name}")
    return {
        "verdict": "FAILED" if reasons else "COMPLETE",
        "required_steps": sorted(set(required_steps)),
        "reasons": reasons,
    }
