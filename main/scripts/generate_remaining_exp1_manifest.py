"""Generate only unfinished Exp1 cells; never submit or execute a GPU job."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp12.official_exp1 import ARCHITECTURES, ENVIRONMENTS
from scripts.generate_official_exp1_manifest import build

COMPLETED = (
    Path(__file__).resolve().parents[1] / "configs/exp1_completed_historical.json"
)
HISTORICAL_SOURCE = "f597ee9898e4d401912092c336c3e5330cf2fda1"


def identity(row):
    return row["environment"], row["architecture"], row["seed"]


def select_remaining(plan, completed, seeds):
    """Only explicit completed identities are skipped; partials are not imported."""
    if (
        len(seeds) != 10
        or len(set(seeds)) != 10
        or any(type(s) is not int for s in seeds)
    ):
        raise ValueError("exactly ten distinct integer seed IDs are required")
    expected = {
        (env, arch, seed)
        for env, _ in ENVIRONMENTS
        for arch, _, _ in ARCHITECTURES
        for seed in seeds
    }
    entries = plan["entries"]
    if len(entries) != len(expected) or {identity(row) for row in entries} != expected:
        raise ValueError(
            "manifest population differs from requested seeds; freeze the official seed IDs first"
        )
    if completed.get("schema_version") != 1:
        raise ValueError("unsupported completion registry")
    reused = completed["completed"]
    valid_envs = {env for env, _ in ENVIRONMENTS}
    valid_archs = {arch for arch, _, _ in ARCHITECTURES}
    for row in reused:
        if (
            row["source"] != HISTORICAL_SOURCE
            or row["environment"] not in valid_envs
            or row["architecture"] not in valid_archs
            or type(row["seed"]) is not int
            or row["seed"] not in range(1, 9)
            or not row.get("run_id")
        ):
            raise ValueError("foreign or ambiguous historical completion record")
    if len({identity(row) for row in reused}) != len(reused) or len(
        {row["run_id"] for row in reused}
    ) != len(reused):
        raise ValueError("duplicate historical completion identity")
    historical_envs = ("dog-run", "humanoid-run", "myo-key-turn", "myo-pen-twirl")
    expected_completed = {
        (env, arch, seed)
        for env in historical_envs
        for arch, _, _ in ARCHITECTURES
        for seed in range(1, 9)
        if not (arch == "D4W1536" and seed in (7, 8))
    }
    if {identity(row) for row in reused} != expected_completed:
        raise ValueError(
            "completion registry differs from the 88 completed historical cells"
        )
    reuse_keys = {identity(row) for row in reused} & expected
    pending = [row for row in entries if identity(row) not in reuse_keys]
    for row in pending:
        command = row["command"]
        checkpoint = Path(command[command.index("--checkpoint_dir") + 1])
        if checkpoint.exists():
            raise ValueError(
                f"unfinished cell already has output: {checkpoint}; use a fresh root, not a partial resume"
            )
    counts = {
        arch: {
            "required": sum(row["architecture"] == arch for row in entries),
            "reused": sum(key[1] == arch for key in reuse_keys),
            "remaining": sum(row["architecture"] == arch for row in pending),
        }
        for arch, _, _ in ARCHITECTURES
    }
    return {
        **plan,
        "approved": False,
        "executable": False,
        "submitted": False,
        "seed_ids_explicit": list(seeds),
        "counts": counts,
        "reused": [row for row in reused if identity(row) in expected],
        "historical_artifact_admission_pending": True,
        "partial_policy": "Do not resume/count partials; rerun missing cells in fresh directories",
        "entries": pending,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", required=True, type=int, nargs=10)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--commands-out", type=Path)
    args = parser.parse_args()
    result = select_remaining(
        build(args.root), json.loads(COMPLETED.read_text()), args.seeds
    )
    result["completion_registry_sha256"] = hashlib.sha256(
        COMPLETED.read_bytes()
    ).hexdigest()
    # Refuse all pre-existing outputs before creating either file.
    for path in [args.out, args.commands_out]:
        if path is not None and path.exists():
            raise FileExistsError(path)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    if args.commands_out is not None:
        with args.commands_out.open("x") as stream:
            stream.write(
                "".join(shlex.join(row["command"]) + "\n" for row in result["entries"])
            )
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
