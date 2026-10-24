"""Read exports without extraction. Record IDs, configs and explicit evidence only."""

import hashlib
import json
from pathlib import PurePosixPath, Path
import zipfile
import math


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def ingest(path):
    records, issues = [], []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive member names")
        for n in names:
            p = PurePosixPath(n)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("unsafe archive member")
        for name in sorted(
            n for n in names if n.endswith("/metadata.json") or n == "metadata.json"
        ):
            meta = json.loads(z.read(name))
            root = name.rsplit("/", 1)[0] + "/" if "/" in name else ""
            rid = meta.get("id") or meta.get("run_id")

            def flag(code, detail=None):
                issues.append(
                    {"record": name, "run_id": rid, "code": code, "detail": detail}
                )

            if not isinstance(rid, str) or not rid:
                flag("missing_run_id")
                rid = "unidentified:" + name
            config = meta.get("config", meta.get("resolved_config", {}))
            config = {
                k: v["value"] if isinstance(v, dict) and set(v) >= {"value"} else v
                for k, v in config.items()
            }
            identity = meta.get("identity", {})
            env = identity.get("environment", config.get("env_name"))
            arch = identity.get("architecture", config.get("architecture"))
            if (
                arch is None
                and "critic_num_blocks" in config
                and "critic_hidden_dim" in config
            ):
                arch = f"D{config['critic_num_blocks']}W{config['critic_hidden_dim']}"
            seed = identity.get("seed", config.get("seed"))
            if type(seed) is not int:
                flag("missing_or_invalid_seed")
            if not env or not arch:
                flag("missing_condition_identity")
            code = meta.get("code", {})
            if not code.get("commit") or code.get("dirty") is not False:
                flag("unverified_source")
            history = []
            hname = root + "history.jsonl"
            if hname not in names:
                flag("missing_history")
            else:
                for i, line in enumerate(z.read(hname).decode().splitlines(), 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for k, v in row.items():
                        if isinstance(v, (int, float)) and not math.isfinite(v):
                            flag("nonfinite_metric", {"line": i, "metric": k})
                    history.append(row)
            if not history:
                flag("empty_history")
            step_key = meta.get("step_key", "interaction_step")
            steps = [r[step_key] for r in history if step_key in r]
            if len(steps) != len(history):
                flag("missing_step_counter", step_key)
            if any(
                not isinstance(s, (float, int)) or not math.isfinite(s) for s in steps
            ):
                flag("invalid_step_counter")
            elif any(b <= a for a, b in zip(steps, steps[1:])):
                flag("nonincreasing_steps")
            status = meta.get("state", meta.get("status", "unknown"))
            if status not in ("finished", "complete"):
                flag("nonfinished_status", status)
            horizon = config.get("num_interaction_steps")
            if horizon is None:
                flag("missing_horizon")
            elif not steps or steps[-1] < horizon:
                flag("partial_history")
            for checkpoint in meta.get("expected_checkpoints", []):
                if root + checkpoint not in names:
                    flag("missing_checkpoint", checkpoint)
            for metric in meta.get("expected_metrics", []):
                if not any(metric in row for row in history):
                    flag("missing_metric_history", metric)
            group = meta.get("campaign")
            if group not in ("exp96", "pilot"):
                flag("unverified_campaign_membership")
            if not meta.get("provenance"):
                flag("incomplete_provenance")
            record = {
                "id": rid,
                "name": meta.get("name"),
                "environment": env,
                "architecture": arch,
                "seed": seed,
                "campaign": group,
                "status": status,
                "config": config,
                "config_hash": digest(config),
                "source": code,
                "history": history,
                "step_key": step_key,
                "horizon": horizon,
                "metadata": meta,
                "archive_member": name,
            }
            records.append(record)
    ids = {}
    conditions = {}
    for r in records:
        ids.setdefault(r["id"], []).append(r["archive_member"])
        key = (r["campaign"], r["environment"], r["architecture"], r["seed"])
        conditions.setdefault(key, []).append(r)
    for rid, members in ids.items():
        if len(members) > 1:
            issues.append(
                {"code": "duplicate_run_id", "run_id": rid, "members": members}
            )
    relationships = []
    for key, rs in conditions.items():
        if len(rs) > 1:
            hashes = {r["config_hash"] for r in rs}
            relationships.append(
                {
                    "condition": key,
                    "ids": [r["id"] for r in rs],
                    "classification": "same_nominal_condition_unresolved",
                    "same_exact_config": len(hashes) == 1,
                    "declared_resumes": [r["metadata"].get("resumed_from") for r in rs],
                }
            )
            issues.append({"code": "multiple_records_per_condition", "condition": key})
    return {
        "schema": 1,
        "input_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "records": records,
        "issues": issues,
        "relationships": relationships,
        "coverage": {
            "records": len(records),
            "exp96_verified_records": sum(r["campaign"] == "exp96" for r in records),
            "pilot_verified_records": sum(r["campaign"] == "pilot" for r in records),
        },
    }


def local_artifacts(path):
    """Lossless numeric adapters. No inferred success, pairing or run identities."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    p = Path(path)
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as a:
            return {k: a[k].tolist() for k in a.files}
    if p.suffix == ".csv":
        frame = pd.read_csv(p).astype(object)
        return frame.where(pd.notna(frame), None).to_dict("records")
    if p.suffix == ".json":
        return json.loads(p.read_text())
    raise ValueError("supported local artifacts: CSV JSON NPZ")
