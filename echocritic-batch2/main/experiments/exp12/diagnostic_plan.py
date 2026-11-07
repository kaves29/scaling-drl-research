"""Explicit, bounded retrospective diagnostic plans; never authorizes execution."""

import hashlib
import json
from pathlib import Path

ENVS = ("dog-run", "humanoid-run", "myo-key-turn", "myo-pen-twirl")
ARCHITECTURES = ("D2W512", "D4W1024", "D4W1536")
KINDS = ("fresh", "early", "latest")


def load_plan(path):
    """Reject unknown parameters rather than silently adopting scientific settings."""
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    if (
        set(plan) != {"schema", "description", "conditions"}
        or type(plan["schema"]) is not int
        or plan["schema"] != 1
    ):
        raise ValueError("unsupported diagnostic plan schema")
    conditions = plan["conditions"]
    if not isinstance(conditions, list) or not conditions or len(conditions) > 64:
        raise ValueError(
            "select 1..64 explicit conditions; no implicit experiment matrix"
        )
    names = set()
    for c in conditions:
        common = {"id", "type", "environment", "seed", "round"}
        if c.get("type") == "original":
            required = common | {"architecture"}
        elif c.get("type") == "comparison":
            required = common | {
                "architectures",
                "learners",
                "offset_modes",
                "horizons",
                "pool",
                "shared_reference",
            }
        else:
            raise ValueError("unknown condition type")
        if set(c) != required:
            raise ValueError("missing or unknown condition field")
        name = c["id"]
        if (
            not isinstance(name, str)
            or not name
            or any(
                x
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for x in name
            )
            or name in names
        ):
            raise ValueError("condition ids must be unique safe directory names")
        names.add(name)
        if (
            c["environment"] not in ENVS
            or type(c["seed"]) is not int
            or c["seed"] not in range(1, 9)
            or type(c["round"]) is not int
            or c["round"] not in range(10)
        ):
            raise ValueError(
                "condition must select a canonical parent and target round"
            )
        if c["type"] == "original":
            if c["architecture"] not in ARCHITECTURES:
                raise ValueError("unknown architecture")
            continue
        for key, allowed in (
            ("architectures", ARCHITECTURES),
            ("learners", KINDS),
            ("offset_modes", ("own", "zero", "shared", "incremental")),
            ("horizons", (2000, 4000, 8000)),
        ):
            values = c[key]
            if (
                not isinstance(values, list)
                or not values
                or len(set(values)) != len(values)
                or any(v not in allowed for v in values)
                or key == "horizons"
                and any(type(v) is not int for v in values)
            ):
                raise ValueError("invalid explicit selection: " + key)
        pool = c["pool"]
        if (
            set(pool) != {"period", "normalizer", "support", "target_normalizer"}
            or pool["period"] not in ("early", "late")
            or pool["normalizer"] not in ("latest", "early", "initial_reconstructed")
            or pool["target_normalizer"]
            not in ("same", "latest", "early", "initial_reconstructed")
            or (
                pool["support"] is not None
                and (
                    type(pool["support"]) is not int
                    or not 1
                    <= pool["support"]
                    <= (5000 if pool["period"] == "early" else 500000)
                )
            )
        ):
            raise ValueError("invalid replay/normalization crossover")
        ref = c["shared_reference"]
        if (
            not isinstance(ref, dict)
            or set(ref) != {"architecture", "kind"}
            or ref["architecture"] not in ARCHITECTURES
            or ref["kind"] not in KINDS
        ):
            raise ValueError("shared targets need an explicit reference learner")
    return plan, hashlib.sha256(raw).hexdigest()
