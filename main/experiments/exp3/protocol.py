"""Explicit owner choices and reproducible per-cell dependency manifests."""

import json
from pathlib import Path

import yaml

from generate_manifest import EXP12_ENVS, EXP12_SEEDS

MYO = tuple(e for e, g in EXP12_ENVS if g == "myosuite_simba")


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"{name}: explicit positive integer required; no scientific default"
        )
    return value


def validate_protocol(spec):
    if spec.get("schema_version") != 1 or spec.get("run_role") != "dev":
        raise ValueError("only versioned development pilots are supported")
    if spec.get("approved") is not True:
        raise ValueError(
            "protocol scientific choices must be explicitly approved by the owner"
        )
    pilot = spec["pilot"]
    if pilot not in (1, 2, 3):
        raise ValueError("unknown Exp3 pilot")
    for k in ("chunk_size", "rng_seed"):
        if k not in spec:
            raise ValueError(f"missing explicit {k}")
    positive(spec["chunk_size"], "chunk_size")
    if type(spec["rng_seed"]) is not int or spec["rng_seed"] < 0:
        raise ValueError("explicit nonnegative diagnostic RNG seed required")
    if pilot == 1:
        for k in ("panel_size", "outcome_eval_episodes"):
            positive(spec[k], k)
        if spec["normalization"] not in ("common_fork", "artifact_specific") or spec[
            "alpha_source"
        ] not in ("fork", "actor_specific"):
            raise ValueError("explicit panel normalization/alpha choice required")
        if spec["intervention_space"] not in ("action", "parameter_per_state"):
            raise ValueError(
                "explicit action or parameter_per_state intervention space required"
            )
        if (
            not spec["interventions"]
            or len(set(spec["interventions"])) != len(spec["interventions"])
            or not set(spec["interventions"])
            <= {"sanity", "full", "direction", "magnitude"}
        ):
            raise ValueError("explicit nonduplicated intervention list required")
        if spec["zero_signal_policy"] not in ("error", "keep_baseline"):
            raise ValueError("explicit zero-signal policy required")
    elif pilot == 2:
        for k in (
            "arrivals",
            "evaluation_every",
            "evaluation_episodes",
            "checkpoint_every",
            "updates_per_arrival",
        ):
            positive(spec[k], k)
        if spec["normalization"] not in ("source_statistics", "frozen_fork"):
            raise ValueError("explicit passive normalization policy required")
        if type(spec["stage_b"]) is not bool:
            raise ValueError("explicit Stage B opt-in required")
        if (
            spec["injection_m"] not in ("last", "half", "all")
            or type(spec["injection_seed"]) is not int
        ):
            raise ValueError("explicit existing injection label/seed required")
    else:
        if spec.get("enabled") is not True:
            raise ValueError("Pilot3 requires explicit opt-in")
        for k in ("updates", "batch_size", "panel_size", "checkpoint_every"):
            positive(spec[k], k)
        if spec["normalization"] not in ("common_fork", "artifact_specific"):
            raise ValueError("explicit frozen-target normalization choice required")
        for k in ("alpha_source", "critic_source", "evaluator_source"):
            if spec[k] not in ("fork", "untreated", "injected"):
                raise ValueError(f"explicit {k} required")
        if spec.get("alternative_evaluator") not in (
            None,
            "fork",
            "untreated",
            "injected",
        ):
            raise ValueError("unknown alternative evaluator")
    return spec


def manifest(config, artifact_root):
    """Resolve repository IDs and seeds; never invent the MyoSuite selection."""
    cfg = yaml.safe_load(Path(config).read_text())
    selected = cfg["environments"]["myosuite"]
    if (
        not isinstance(selected, list)
        or len(selected) != 2
        or len(set(selected)) != 2
        or not set(selected) <= set(MYO)
    ):
        raise ValueError(f"owner must select TWO MyoSuite tasks from {MYO}")
    if cfg["seeds"] != EXP12_SEEDS or cfg["environments"]["dmc"] != [
        "dog-run",
        "humanoid-walk",
    ]:
        raise ValueError(
            "pilot environment/seed conventions differ from requested repository population"
        )
    root = Path(artifact_root).resolve()
    cells = []
    for env in [*cfg["environments"]["dmc"], *selected]:
        for seed in EXP12_SEEDS:
            d = root / env / f"seed_{seed}"
            dependencies = {
                k: str(d / k)
                for k in ("fork", "untreated", "injected", "stream_u", "stream_i")
            }
            cells.append(
                {
                    "environment": env,
                    "env_group": dict(EXP12_ENVS)[env],
                    "seed": seed,
                    "run_role": "dev",
                    "dependencies": dependencies,
                    "available_paths": {
                        k: Path(v).exists() for k, v in dependencies.items()
                    },
                    "qualification": "UNVERIFIED; path existence is not compatibility validation",
                    "required": {
                        "pilot1": ["fork", "untreated", "injected"],
                        "pilot2_a": ["fork", "stream_u"],
                        "pilot2_b": ["fork", "stream_u", "stream_i"],
                        "pilot3": ["fork", "untreated", "injected"],
                    },
                }
            )
    return {"schema_version": 1, "config": cfg, "cells": cells}


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    return validate_protocol(spec)
