"""Read-only Exp1 replay admission for a proposed matched-data Exp2.

This module prepares a source-bound plan. It does not train, run probes, or
modify the Exp1 parents. The same plan and sample_indices() must be used by
both future critic-size arms.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from experiments.exp12.state import BUFFER_ARRAYS, latest_state_dir, load_meta
from utils.atomic_io import atomic_write_text
from utils.run_metadata import RUN_METADATA_FILENAME, load_run_metadata


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_headers(path):
    """Inspect NPZ headers without loading a full Exp1 replay into memory."""
    names = {name.lstrip("_") for name in BUFFER_ARRAYS}
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {f"{name}.npy" for name in names}:
            raise ValueError(f"{path}: incomplete or unexpected replay arrays")
        headers = {}
        for name in names:
            with archive.open(f"{name}.npy") as member:
                version = np.lib.format.read_magic(member)
                shape, _, dtype = np.lib.format._read_array_header(member, version)
            if len(shape) < 1 or shape[0] < 1 or dtype.hasobject:
                raise ValueError(f"{path}: invalid {name} array")
            headers[name] = {"shape": list(shape), "dtype": str(dtype)}
    if len({row["shape"][0] for row in headers.values()}) != 1:
        raise ValueError(f"{path}: replay arrays have different lengths")
    return headers


def inspect_exp1_run(run_dir):
    """Require a completed Exp1 parent and retain its exact source identity."""
    run_dir = Path(run_dir).resolve(strict=True)
    if not (run_dir / "DONE").is_file():
        raise ValueError(f"{run_dir}: missing DONE; active/partial run is ineligible")
    metadata = load_run_metadata(run_dir / RUN_METADATA_FILENAME)
    if metadata is None or metadata.get("identity", {}).get("experiment") != "exp1":
        raise ValueError(f"{run_dir}: missing Exp1 run metadata")
    config = metadata.get("resolved_config", {})
    probe = config.get("probe", {})
    if (probe.get("rounds"), probe.get("steps"), probe.get("checks")) != (10, 2000, 50):
        raise ValueError(f"{run_dir}: not the official ten-target probe configuration")
    state = latest_state_dir(run_dir / "state")
    if state is None:
        raise ValueError(f"{run_dir}: missing final state")
    done_step = json.loads((run_dir / "DONE").read_text())["interaction_step"]
    saved_step = load_meta(state)["interaction_step"]
    if done_step != saved_step or done_step != int(config["num_interaction_steps"]):
        raise ValueError(f"{run_dir}: DONE, state, and configured final steps differ")
    replay = state / "buffer.npz"
    rms = state / "obs_rms.pkl"
    if not replay.is_file() or not rms.is_file() or not (state / "agent_ckpt").is_dir():
        raise ValueError(f"{state}: missing replay, normalizer, or agent state")
    headers = replay_headers(replay)
    return {
        "run_dir": str(run_dir),
        "state_dir": str(state),
        "identity": metadata["identity"],
        "source_commit": metadata.get("code", {}).get("commit"),
        "config_hash": metadata["config_hash"],
        "protocol_hash": metadata["protocol_hash"],
        "replay": str(replay),
        "replay_sha256": sha256(replay),
        "normalizer_sha256": sha256(rms),
        "headers": headers,
        "rows": headers["observations"]["shape"][0],
    }


def paired_indices(seed, update_step, batch_size, small_rows, large_rows):
    """Draw one balanced, repeatable batch recipe for both future arms."""
    if any(
        type(value) is not int or value < 0
        for value in (seed, update_step, batch_size, small_rows, large_rows)
    ):
        raise ValueError("sampling arguments must be nonnegative integers")
    if batch_size < 2 or batch_size % 2 or not small_rows or not large_rows:
        raise ValueError("sampling needs a positive even batch and nonempty replays")
    rng = np.random.default_rng(np.random.SeedSequence([seed, update_step]))
    half = batch_size // 2
    source = np.concatenate(
        (np.zeros(half, dtype=np.int8), np.ones(half, dtype=np.int8))
    )
    row = np.concatenate(
        (rng.integers(small_rows, size=half), rng.integers(large_rows, size=half))
    )
    order = rng.permutation(batch_size)
    return source[order], row[order]


def make_plan(small_run, large_run, sampling_seed, batch_size):
    small, large = inspect_exp1_run(small_run), inspect_exp1_run(large_run)
    if small["identity"]["environment"] != large["identity"]["environment"]:
        raise ValueError("Exp1 parents must have the same environment")
    if small["identity"]["seed"] != large["identity"]["seed"]:
        raise ValueError("Exp1 parents must have the same seed")
    if small["protocol_hash"] != large["protocol_hash"]:
        raise ValueError("Exp1 parents have different training protocols")
    if (small["identity"]["architecture"], large["identity"]["architecture"]) != (
        "D2W512",
        "D4W1536",
    ):
        raise ValueError("Exp2 comparison requires small D2W512 and large D4W1536")
    for key in small["headers"]:
        if (small["headers"][key]["shape"][1:], small["headers"][key]["dtype"]) != (
            large["headers"][key]["shape"][1:],
            large["headers"][key]["dtype"],
        ):
            raise ValueError(f"incompatible {key} shape or dtype")
    paired_indices(sampling_seed, 0, batch_size, small["rows"], large["rows"])
    return {
        "schema": "exp2-matched-replay-proposal-v1",
        "status": "data-plan-only",
        "launch_approved": False,
        "small": small,
        "large": large,
        "sampler": {
            "seed": sampling_seed,
            "batch_size": batch_size,
            "rule": "equal parent counts; SeedSequence([seed, update_step]); replacement; shuffle",
        },
        "probe": {
            "producer": "experiments.exp12.run_probes",
            "rounds": 10,
            "steps": 2000,
            "checks": 50,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-run", type=Path, required=True)
    parser.add_argument("--large-run", type=Path, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    plan = make_plan(
        args.small_run, args.large_run, args.sampling_seed, args.batch_size
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.out, json.dumps(plan, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
