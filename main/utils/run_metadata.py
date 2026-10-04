"""Per-run provenance: what a run was produced under, and whether runs may be combined.

A run's metadata is written next to its checkpoint (and next to its persisted
metrics) when it starts. Resuming, and every analysis that combines runs,
compares the stored settings and refuses on a mismatch.

`protocol` is the part that must be identical across every run combined in one
analysis: the whole agent/env/buffer subtrees (minus seed and critic size) plus
the top-level keys in PROTOCOL_KEYS, and the simulator package versions.
`config_hash` covers the entire resolved config except storage locations
(keys ending in "_root"), and must match exactly on resume.
"""

import hashlib
import importlib.metadata
import json
import socket
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from utils.atomic_io import atomic_write_text
from utils.paths import REPO_ROOT

SCHEMA_VERSION = 1
RUN_METADATA_FILENAME = "run_metadata.json"
# Inside a frozen-agent snapshot dir, where Angle 2A writes its own run_metadata.json.
SNAPSHOT_RUN_METADATA_FILENAME = "training_run_metadata.json"

PER_RUN_AGENT_KEYS = ("seed", "critic_num_blocks", "critic_hidden_dim")
PROTOCOL_KEYS = (
    "env_name",
    "num_env_steps",
    "num_interaction_steps",
    "num_train_envs",
    "action_repeat",
    "eff_episode_len",
    "gamma",
    "n_step",
    "updates_per_interaction_step",
    "logging_per_interaction_step",
    "actor_grad_cosine_every",
    "evaluation_per_interaction_step",
    "num_eval_episodes",
    "actor_num_blocks",
    "actor_hidden_dim",
    "actor_sparsity",
    "critic_sparsity",
)
SETTINGS_KEYS = (
    "seed",
    "num_env_steps",
    "num_interaction_steps",
    "updates_per_interaction_step",
    "action_repeat",
    "gamma",
    "logging_per_interaction_step",
    "actor_grad_cosine_every",
    "evaluation_per_interaction_step",
    "num_eval_episodes",
)
SIMULATOR_PACKAGES = {
    "dmc": ("dm_control", "mujoco"),
    "myosuite": ("myosuite", "mujoco"),
    "humanoid_bench": ("humanoid_bench", "dm_control", "mujoco"),
}


class RunMetadataMismatch(ValueError):
    pass


def code_version() -> Dict[str, Optional[object]]:
    def git(*args):
        return subprocess.run(
            ["git", "-C", REPO_ROOT, *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_versions(env_type: str) -> Dict[str, Optional[str]]:
    versions = {}
    for package in SIMULATOR_PACKAGES.get(env_type, ()):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def env_id(env_type: str, env_name: str) -> str:
    if env_type == "myosuite":
        from scale_rl.envs.myosuite import MYOSUITE_TASKS_DICT

        return MYOSUITE_TASKS_DICT[env_name]
    return env_name


def _strip_locations(node):
    if isinstance(node, dict):
        return {k: _strip_locations(v) for k, v in node.items() if not str(k).endswith("_root")}
    if isinstance(node, list):
        return [_strip_locations(v) for v in node]
    return node


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def training_protocol(resolved_cfg: dict) -> dict:
    env_cfg = resolved_cfg["env"]
    protocol = {key: resolved_cfg[key] for key in PROTOCOL_KEYS if key in resolved_cfg}
    protocol["agent"] = {k: v for k, v in resolved_cfg["agent"].items() if k not in PER_RUN_AGENT_KEYS}
    protocol["env"] = {k: v for k, v in env_cfg.items() if k != "seed"}
    protocol["buffer"] = dict(resolved_cfg.get("buffer", {}))
    protocol["env_id"] = env_id(env_cfg["env_type"], env_cfg["env_name"])
    protocol["simulator_versions"] = _package_versions(env_cfg["env_type"])
    return protocol


def build_run_metadata(resolved_cfg: dict, identity: dict, launch: dict) -> dict:
    protocol = training_protocol(resolved_cfg)
    code = code_version()
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code": code,
        "settings": {
            "architecture": identity["architecture"],
            "env_id": protocol["env_id"],
            "simulator_versions": protocol["simulator_versions"],
            "discount": resolved_cfg.get("gamma"),
            **{key: resolved_cfg.get(key) for key in SETTINGS_KEYS},
        },
        "protocol": protocol,
        "protocol_hash": _hash(protocol),
        "resolved_config": resolved_cfg,
        "config_hash": _hash(_strip_locations(resolved_cfg)),
        "launches": [
            {**launch, **code, "hostname": socket.gethostname(), "simulator_versions": protocol["simulator_versions"]}
        ],
    }


def load_run_metadata(path) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_run_metadata(path, metadata: dict) -> None:
    atomic_write_text(Path(path), json.dumps(metadata, indent=2, sort_keys=True, default=str))


def _flatten(node, prefix=""):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: node}


def differing_keys(a: dict, b: dict) -> List[str]:
    fa, fb = _flatten(a), _flatten(b)
    return sorted(k for k in fa.keys() | fb.keys() if fa.get(k, "<missing>") != fb.get(k, "<missing>"))


def check_resume_matches(stored: dict, current: dict, where: str) -> None:
    """Refuses to resume a checkpoint under a different config; warns on a code change."""
    if stored["config_hash"] != current["config_hash"]:
        keys = differing_keys(
            _strip_locations(stored["resolved_config"]), _strip_locations(current["resolved_config"])
        )
        raise RunMetadataMismatch(
            f"Refusing to resume {where}: its checkpoint was produced under a different "
            f"config. Differing keys: {keys}"
        )
    stored_versions = stored["protocol"]["simulator_versions"]
    current_versions = current["protocol"]["simulator_versions"]
    if stored_versions != current_versions:
        warnings.warn(
            f"Resuming {where} with simulator versions {current_versions}, but it was "
            f"started with {stored_versions}; recorded in its launch history."
        )
    if stored["code"]["commit"] != current["code"]["commit"]:
        warnings.warn(
            f"Resuming {where} under code commit {current['code']['commit']}, but it was "
            f"started under {stored['code']['commit']}; recorded in its launch history."
        )


def record_launch(stored: dict, current: dict) -> dict:
    stored = dict(stored)
    launch = {**current["launches"][0], "simulator_versions": current["protocol"]["simulator_versions"]}
    stored["launches"] = list(stored.get("launches", [])) + [launch]
    return stored


def check_runs_comparable(runs: Dict[str, Optional[dict]], what: str) -> List[str]:
    """Raises if any two runs with metadata differ in protocol; returns the run
    keys that have no metadata (produced before metadata was recorded)."""
    present = {key: meta for key, meta in runs.items() if meta is not None}
    missing = sorted(key for key, meta in runs.items() if meta is None)
    if present:
        reference_key = sorted(present)[0]
        reference = present[reference_key]["protocol"]
        mismatched = {
            key: differing_keys(reference, meta["protocol"])
            for key, meta in present.items()
            if meta["protocol_hash"] != present[reference_key]["protocol_hash"]
        }
        if mismatched:
            raise RunMetadataMismatch(
                f"Refusing to combine runs for {what}: settings differ from "
                f"{reference_key}: {mismatched}"
            )
    if missing:
        warnings.warn(
            f"{what}: no run metadata for {missing} (produced before run metadata was "
            f"recorded); their settings are not machine-verified."
        )
    return missing
