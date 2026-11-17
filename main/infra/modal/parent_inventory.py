"""Verify a source inventory and derive replay parent identities."""
import hashlib
import json
from pathlib import Path

SOURCE = "f597ee9898e4d401912092c336c3e5330cf2fda1"
POPULATION = {"dog-run": (1, 2, 3), "humanoid-run": (1, 2, 3),
              "myo-key-turn": (1, 2), "myo-pen-twirl": (1, 2)}


def verify_transfer(root, source_path, receipt_path):
    root = Path(root).resolve()
    source_bytes = Path(source_path).read_bytes()
    source = json.loads(source_bytes)
    receipt = json.loads(Path(receipt_path).read_text())
    files = source["files"]
    if (source["schema_version"] != 1 or source["source_commit"] != SOURCE
            or source["volume"] != "exp1-pilot-states"
            or source["destination_prefix"] != "exp1-pilot-parents"
            or source["file_count"] != len(files)
            or source["total_bytes"] != sum(f["size_bytes"] for f in files)):
        raise ValueError("invalid source inventory")
    if (receipt["manifest_sha256"] != hashlib.sha256(source_bytes).hexdigest()
            or receipt["volume"] != source["volume"]
            or receipt["prefix"] != source["destination_prefix"]
            or receipt["source_commit"] != SOURCE or receipt["verified"] is not True
            or receipt["extra"] or receipt["missing"]
            or receipt["file_count"] != len(files)
            or receipt["total_bytes"] != source["total_bytes"]):
        raise ValueError("source inventory receipt mismatch")
    results = {r["path"]: r for r in receipt["results"]}
    if len(results) != len(receipt["results"]) or len(results) != len(files):
        raise ValueError("duplicate or incomplete receipt paths")
    expected = {(arch, env, seed) for arch in ("D2W512", "D4W1536")
                for env, seeds in POPULATION.items() for seed in seeds}
    identities, inventory = set(), set()
    for item in files:
        relative = item["destination_relative_path"]
        parts = Path(relative).parts
        identity = item["parent_identity"]
        if identity is None:
            key = None
            valid_identity = relative == "source/exp1_pilot_source_f597ee9_20261116.bundle"
        else:
            key = (identity["regime"], identity["environment"], identity["seed"])
            valid_identity = (key in expected and len(parts) >= 4
                              and parts[:3] == (key[0], key[1], f"seed_{key[2]}"))
        if (not valid_identity or item["source_commit"] != SOURCE
                or relative in inventory or Path(relative).is_absolute() or ".." in parts):
            raise ValueError("invalid inventory identity or path")
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError("unsafe or missing inventory file")
        result = results.get(relative, {})
        if (result.get("status") != "ok" or result.get("expected_sha256") != item["sha256"]
                or result.get("actual_sha256") != item["sha256"]
                or result.get("expected_bytes") != item["size_bytes"]
                or result.get("actual_bytes") != item["size_bytes"]):
            raise ValueError("receipt file mismatch")
        h = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                size += len(block)
                h.update(block)
        if size != item["size_bytes"] or h.hexdigest() != item["sha256"]:
            raise ValueError(f"checksum mismatch: {relative}")
        if key is not None:
            identities.add(key)
        inventory.add(relative)
    if identities != expected:
        raise ValueError("incomplete parent population")
    runs = []
    for arch, env, seed in sorted(expected):
        run = Path(arch) / env / f"seed_{seed}"
        metadata = json.loads((root / run / "run_metadata.json").read_text())
        done = json.loads((root / run / "DONE").read_text())
        state = (root / run / "state/LATEST").read_text().strip()
        if not state.startswith("step_") or Path(state).name != state:
            raise ValueError("invalid state pointer")
        required = [run / "DONE", run / "run_metadata.json", run / "state/LATEST",
                    *(run / "state" / state / name for name in
                      ("meta.pkl", "buffer.npz", "buffer_meta.pkl", "obs_rms.pkl"))]
        agent = (run / "state" / state / "agent_ckpt").as_posix() + "/"
        if any(p.as_posix() not in inventory for p in required) or not any(p.startswith(agent) for p in inventory):
            raise ValueError("inventory omits required parent payload")
        ident = metadata["identity"]
        if ((ident["architecture"], ident["environment"], ident["seed"]) != (arch, env, seed)
                or metadata["code"]["commit"] != SOURCE or done["interaction_step"] != 500000):
            raise ValueError("parent metadata mismatch")
        runs.append(dict(architecture=arch, environment=env, seed=seed, state=state,
                         source_commit=SOURCE, config_hash=metadata["config_hash"], done_step=500000))
    return runs
