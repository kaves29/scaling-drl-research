"""Read-only admission of completed, source-bound Experiment 1 parents."""
import hashlib
import json
from pathlib import Path

SOURCE = "f597ee9898e4d401912092c336c3e5330cf2fda1"


def checkpoints(updates, count=15):
    """Exactly count positive, evenly spaced update endpoints, including final."""
    if type(updates) is not int or updates < count or count < 1:
        raise ValueError("parent update budget cannot supply requested checkpoints")
    return [(i * updates + count - 1) // count for i in range(1, count + 1)]


def admit_parent(manifest_path, identity):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    regime = identity["reference_regime"]
    if regime not in ("reference_small", "reference_large"):
        raise ValueError("unknown reference regime")
    architecture = "D2W512" if regime == "reference_small" else "D4W1536"
    rows = [r for r in manifest["runs"] if
            (r["environment"], r["seed"], r["architecture"]) ==
            (identity["environment"], identity["seed"], architecture)]
    if len(rows) != 1:
        raise ValueError("parent identity must have exactly one manifest row")
    row = rows[0]
    if row["source_commit"] != SOURCE or row["done_step"] != 500000:
        raise ValueError("unapproved parent source or incomplete endpoint")
    state_name = row["state"]
    if not state_name.startswith("step_") or Path(state_name).name != state_name:
        raise ValueError("invalid parent state name")
    run = path.parent / architecture / identity["environment"] / f"seed_{identity['seed']}"
    state = run / "state" / state_name
    if not state.resolve().is_relative_to(path.parent) or not state.is_dir():
        raise ValueError("parent state missing or outside manifest root")
    metadata = json.loads((run / "run_metadata.json").read_text())
    actual = metadata["identity"]
    if (actual["architecture"], actual["environment"], actual["seed"]) != (
            architecture, identity["environment"], identity["seed"]):
        raise ValueError("parent metadata identity mismatch")
    if metadata["code"]["commit"] != SOURCE or metadata["config_hash"] != row["config_hash"]:
        raise ValueError("parent metadata source/config mismatch")
    if json.loads((run / "DONE").read_text())["interaction_step"] != 500000:
        raise ValueError("parent DONE endpoint mismatch")
    if (run / "state" / "LATEST").read_text().strip() != state_name:
        raise ValueError("parent LATEST differs from frozen manifest")
    for name in ("meta.pkl", "buffer.npz", "buffer_meta.pkl", "obs_rms.pkl"):
        if not (state / name).is_file():
            raise ValueError(f"missing parent payload {name}")
    if not (state / "agent_ckpt").is_dir():
        raise ValueError("missing parent agent checkpoint")
    # An upload must be followed by complete byte verification, recorded alongside it.
    receipt = path.parent / "verification.json"
    if not receipt.is_file():
        raise ValueError("missing transfer checksum verification receipt")
    verification = json.loads(receipt.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if verification.get("manifest_sha256") != digest or verification.get("all_files_ok") is not True:
        raise ValueError("transfer verification does not match manifest")
    return state, {**row, "manifest_sha256": digest}


def validate_parent_config(recorded, current):
    """Fail closed on changed SAC/observation/environment semantics.

    Logging, Exp2 probe schedule and the number of reference collection steps
    deliberately differ. All model, optimizer, buffer, and environment settings
    must still match the writer's resolved configuration.
    """
    if not isinstance(recorded, dict):
        raise ValueError("parent resolved_config missing; writer compatibility review required")
    keys = ("agent", "env", "buffer", "gamma", "n_step", "action_repeat",
            "updates_per_interaction_step", "actor_grad_cosine_every")
    mismatches = [key for key in keys if key not in recorded or
                  key not in current or recorded[key] != current[key]]
    if mismatches:
        raise ValueError("parent/current protocol mismatch: " + ", ".join(mismatches))
