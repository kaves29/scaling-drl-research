"""Read-only, bounded-memory access to frozen Exp96 checkpoint components."""

import ast
import hashlib
import json
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np
import tensorstore as ts


def state_dir(checkpoint, kind):
    root = Path(checkpoint) / ("fork/state" if kind == "early" else "state")
    return root / (root / "LATEST").read_text().strip()


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
