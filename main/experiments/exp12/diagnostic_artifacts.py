"""Read-only, bounded-memory access to frozen Exp96 checkpoint components."""

import ast
import hashlib
import json
import os
import pickle
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tensorstore as ts


def state_dir(checkpoint, kind):
    if kind not in ("early", "latest"):
        raise ValueError("state kind must be early or latest")
    root = Path(checkpoint) / ("fork/state" if kind == "early" else "state")
    name = (root / "LATEST").read_text().strip()
    target = (root / name).resolve()
    if not name or Path(name).name != name or not target.is_relative_to(root.resolve()):
        raise ValueError("LATEST must name a snapshot inside its state directory")
    if not target.is_dir():
        raise ValueError("LATEST snapshot is unavailable")
    return target


def optimizer_binding(row, frozen_source):
    """Read and bind the original resolved optimizer; never infer defaults."""
    from utils.run_metadata import _hash, _strip_locations

    path = Path(row["checkpoint_dir"]) / "run_metadata.json"
    raw = path.read_bytes()
    metadata = json.loads(raw)
    cfg = metadata["resolved_config"]
    if metadata.get("code") != {"commit": frozen_source, "dirty": False}:
        raise ValueError("checkpoint source mismatch")
    for key in ("environment", "architecture", "seed", "run_key"):
        if metadata.get("identity", {}).get(key) != row[key]:
            raise ValueError("checkpoint logical identity mismatch: " + key)
    if metadata.get("config_hash") != row["config_hash"] or metadata[
        "config_hash"
    ] != _hash(_strip_locations(cfg)):
        raise ValueError("checkpoint resolved configuration hash mismatch")
    launches = metadata.get("launches")
    if not launches or any(
        x.get("commit") != frozen_source or x.get("dirty") is not False
        for x in launches
    ):
        raise ValueError("checkpoint launch source mismatch")
    if cfg.get("seed") != row["seed"] or cfg.get("env_name") != row["environment"]:
        raise ValueError("resolved configuration identity mismatch")
    agent = cfg["agent"]
    settings = {
        key: agent[key] for key in ("critic_learning_rate", "critic_weight_decay")
    }
    if any(
        type(v) not in (float, int) or not np.isfinite(v) for v in settings.values()
    ):
        raise ValueError("optimizer settings must be finite numbers")
    if settings["critic_learning_rate"] <= 0 or settings["critic_weight_decay"] < 0:
        raise ValueError("invalid original optimizer settings")
    # Only these residual, single-critic models are reconstructed by this runner.
    from re import fullmatch

    match = fullmatch(r"D(\d+)W(\d+)", row["architecture"])
    if (
        match is None
        or agent.get("critic_block_type") != "residual"
        or (agent.get("critic_num_blocks"), agent.get("critic_hidden_dim"))
        != tuple(map(int, match.groups()))
        or agent.get("critic_use_cdq") is not False
    ):
        raise ValueError("unsupported or inconsistent archived critic architecture")
    return settings, {
        "parent": row["run_key"],
        "metadata_path": str(path),
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "config_hash": metadata["config_hash"],
        "source": frozen_source,
        "optimizer": {
            "algorithm": "optax.adamw",
            **settings,
            "b1": 0.9,
            "b2": 0.999,
            "eps": 1e-8,
            "eps_root": 0.0,
        },
    }


def replay_array(state, name):
    path = Path(state) / "buffer.npz"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("expected uncompressed frozen replay for read-only mmap")
    with path.open("rb") as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        names, extra = struct.unpack_from("<HH", header, 26)
        stream.seek(names + extra, 1)
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        return np.memmap(
            path,
            mode="r",
            offset=stream.tell(),
            dtype=dtype,
            shape=shape,
            order="F" if fortran else "C",
        )


def normalizer(state):
    with (Path(state) / "obs_rms.pkl").open("rb") as stream:
        return pickle.load(stream)


def initial_normalizer(observations):
    from scale_rl.agents.wrappers.utils import RunningMeanStd

    rms = RunningMeanStd(shape=(1, observations.shape[-1]), dtype=observations.dtype)
    # The initial empty timestep skips normalization. At n, observations1..n-1 were seen.
    for observation in observations[1:5000]:
        rms.update(observation.reshape(1, -1))
    return {"mean": rms.mean, "var": rms.var, "count": rms.count}


def load_critic(checkpoint, kind):
    checkpoint = Path(checkpoint)
    path = (
        checkpoint / "fresh_critic"
        if kind == "fresh"
        else state_dir(checkpoint, kind) / "agent_ckpt"
    )
    prefix = ("params",) if kind == "fresh" else ("critic", "params")
    metadata = json.loads((path / "_METADATA").read_text())["tree_metadata"]
    tree, hashes = {}, {}
    for key in sorted(metadata):
        keys = ast.literal_eval(key)
        if keys[: len(prefix)] != prefix:
            continue
        leaf = ".".join(keys)
        array = (
            ts.open(
                {
                    "driver": "zarr",
                    "kvstore": {"driver": "ocdbt", "base": "file://" + str(path) + "/"},
                    "path": leaf,
                },
                open=True,
            )
            .result()
            .read()
            .result()
        )
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"invalid critic leaf {path}:{leaf}")
        hashes[leaf] = hashlib.sha256(array.tobytes()).hexdigest()
        node = tree
        for part in keys[len(prefix) : -1]:
            node = node.setdefault(part, {})
        node[keys[-1]] = array
    if not tree:
        raise ValueError(f"no critic parameters found in {path}")
    return tree, {
        "path": str(path),
        "metadata_sha256": hashlib.sha256(
            (path / "_METADATA").read_bytes()
        ).hexdigest(),
        "leaf_sha256": hashes,
    }


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        __import__("os").fsync(stream.fileno())
    temporary.replace(path)


def atomic_npz(path, **arrays):
    """Publish complete numeric evidence; a killed write cannot replace a receipt."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
