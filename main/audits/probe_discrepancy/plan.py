"""Shared structural and scientific-plan validation; imports no training modules."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from audits.echocritic_forensic.evidence import CAMPAIGN


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def plan_hash(manifest):
    """Only readiness flags/status may be sealed after unchanged-plan qualification."""
    value = {
        k: v
        for k, v in manifest.items()
        if k not in ("launch_approved", "executable", "diagnostic_status")
    }
    value["resource_estimate"] = {
        k: v for k, v in value["resource_estimate"].items() if k != "packing_qualified"
    }
    return digest(value)


def validate_design(m):
    if (
        m.get("schema") != "echocritic-discrepancy-pilot-design-v2"
        or m.get("count") != 10
        or len(m["runs"]) != 10
    ):
        raise ValueError("exactly ten v2 pilot runs required")
    schedule = m["schedule"]
    if schedule != {
        "mode": "simultaneous",
        "processes_per_gpu": [5, 5],
        "waves": 1,
        "replacement_runs": False,
        "automatic_retries": False,
        "barrier": "all ten children ready before any adapter is released",
    }:
        raise ValueError("five processes on each GPU, simultaneous single wave only")
    if m["training_source"] != CAMPAIGN or m["allocation_proposal"]["gpus"] != 2:
        raise ValueError("original scientific source and two GPUs required")
    bindings = json.loads(
        Path(__file__).with_name("source_config_bindings.json").read_text()
    )
    require_configs = bindings["scientific_config_sha256_by_original_run_id"]
    runs = m["runs"]
    for field in ("pilot_id", "output_relative", "wandb_id"):
        if len({r[field] for r in runs}) != 10:
            raise ValueError(f"ten distinct {field} values required")
    if Counter(r["gpu"] for r in runs) != {0: 5, 1: 5}:
        raise ValueError("GPU counts must be five/five")
    pairs = defaultdict(list)
    for r in runs:
        for field in ("pilot_id", "pair_id", "wandb_id"):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", r[field]):
                raise ValueError("unsafe run identity")
        if type(r["gpu"]) is not int:
            raise ValueError("GPU assignment must be an integer")
        if (
            require_configs.get(r["original_run_id"])
            != r["source_scientific_config_sha256"]
        ):
            raise ValueError(
                "scientific settings differ from pinned original source config"
            )
        path = PurePosixPath(r["output_relative"])
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 2
            or path.parts[0] != "runs"
        ):
            raise ValueError("each output must be a separate confined runs directory")
        c = r["source_scientific_config"]
        if digest(c) != r["source_scientific_config_sha256"]:
            raise ValueError("scientific config hash mismatch")
        if r["training_source"] != CAMPAIGN or (
            c["probe"]["steps"],
            c["probe"]["rounds"],
            c["probe"]["checks"],
            c["probe"]["pool_size"],
        ) != (2000, 10, 50, 25600):
            raise ValueError("scientific source/probe settings changed")
        if c["num_interaction_steps"] != 500000 or (
            c["agent"]["critic_learning_rate"],
            c["agent"]["critic_weight_decay"],
        ) != (1e-4, 0.01):
            raise ValueError("training budget/optimizer changed")
        if not r["seed"] == c["seed"] == c["agent"]["seed"] == c["env"]["seed"]:
            raise ValueError("seed metadata mismatch")
        if (
            r["environment"] != c["env_name"]
            or r["architecture"]
            != f"D{c['agent']['critic_num_blocks']}W{c['agent']['critic_hidden_dim']}"
        ):
            raise ValueError("condition identity mismatch")
        pairs[r["pair_id"]].append(r)
    if len(pairs) != 5:
        raise ValueError("five matched GPU pairs required")
    for values in pairs.values():
        if len(values) != 2 or {r["gpu"] for r in values} != {0, 1}:
            raise ValueError("each condition must appear once per GPU")
        for key in (
            "environment",
            "architecture",
            "seed",
            "original_run_id",
            "source_scientific_config_sha256",
        ):
            if values[0][key] != values[1][key]:
                raise ValueError("condition/seed confounded with GPU")
    if {r["environment"] for r in runs} != {
        "dog-run",
        "humanoid-run",
        "myo-key-turn",
        "myo-pen-twirl",
    } or {r["architecture"] for r in runs} != {"D2W512", "D4W1024", "D4W1536"}:
        raise ValueError("all four tasks and three sizes required")
