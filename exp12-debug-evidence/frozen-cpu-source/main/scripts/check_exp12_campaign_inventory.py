#!/usr/bin/env python3
"""Read-only census of195 parent markers before queueing; not scientific certification."""

import argparse
import json
import re
from pathlib import Path

ARCHITECTURES = ("D2W512", "D4W1024", "D4W1536")
ENVIRONMENTS = (
    "dog-run",
    "dog-trot",
    "humanoid-run",
    "humanoid-walk",
    "humanoid-stand",
    "swimmer-swimmer15",
    "hopper-hop",
    "myo-key-turn",
    "myo-pen-twirl",
    "myo-pose-hard",
    "myo-reach",
    "h1-reach-v0",
    "h1-run-v0",
)


def read_json(path):
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def budget(environment):
    return (
        2_000_000
        if environment.startswith("h1-")
        else (
            500_000 if environment in ("swimmer-swimmer15", "hopper-hop") else 1_000_000
        )
    )


def inspect_parent(parent, key, nominal, commit):
    """Check marker/header structure only; never restore or repair a checkpoint."""
    row = dict(
        run_key=key,
        directory=str(parent),
        status="fresh",
        errors=[],
        payload_integrity="not_assessed",
    )
    if not parent.exists():
        return row
    row["status"] = "incomplete"
    try:
        if not parent.is_dir():
            raise ValueError("parent path is not a directory")
        metadata = parent / "run_metadata.json"
        if metadata.exists():
            meta = read_json(metadata)
            if (
                meta["identity"]["run_key"] != key
                or meta["identity"]["run_role"] != "confirmatory"
            ):
                raise ValueError("metadata identity/role mismatch")
            if meta["code"]["commit"] != commit or meta["code"]["dirty"] is not False:
                raise ValueError("metadata source revision is incompatible")
            if not meta["launches"]:
                raise ValueError("missing resume launch provenance")
            for launch in meta["launches"]:
                if launch["commit"] != commit or launch["dirty"] is not False:
                    raise ValueError("resume launch source revision is incompatible")
        ready, fork_json = parent / "fork/FORK_READY", parent / "fork/fork.json"
        end = nominal
        if ready.exists():
            if key.split("_", 2)[1] == "D2W512":
                raise ValueError("default critic has an unexpected fork")
            fork = read_json(fork_json)
            if ready.read_text().strip() != key or fork["run_key"] != key:
                raise ValueError("fork marker/run-key mismatch")
            f, k = fork["fork_step"], fork["fork_check_index"]
            if (
                type(f) is not int
                or type(k) is not int
                or not 2 <= k <= 19
                or f != k * nominal // 20
            ):
                raise ValueError(
                    "fork step/check is incompatible with the approved schedule"
                )
            end = max(nominal, f + nominal // 4)
            if (
                fork["num_interaction_steps"] != nominal
                or fork["horizon_steps"] != nominal // 4
                or fork["arm_end_step"] != f + nominal // 4
                or fork["control_end_step"] != end
            ):
                raise ValueError("fork horizon/header mismatch")
            kind = fork.get("device", {}).get("device_kind")
            if not isinstance(kind, str) or not kind or kind == "unknown":
                raise ValueError("fork lacks a recorded GPU model")
            row["fork_discovery"] = (
                "present; payload/trigger certification still required"
            )
        done = parent / "DONE"
        if done.exists():
            if fork_json.exists() and not ready.exists():
                raise ValueError("DONE includes an unready fork")
            if not metadata.is_file():
                raise ValueError("DONE without run metadata")
            terminal = read_json(done)
            if (
                type(terminal.get("interaction_step")) is not int
                or terminal["interaction_step"] != end
            ):
                raise ValueError("DONE terminal step is incompatible or stale")
            pointer = parent / "state/LATEST"
            name = pointer.read_text().strip()
            state = parent / "state" / name
            if (
                not re.fullmatch(r"step_[0-9]{9}_[0-9a-f]{8}", name)
                or not state.is_dir()
                or state.resolve().parent != pointer.parent.resolve()
                or int(name.split("_")[1]) != end
            ):
                raise ValueError("DONE has an invalid/missing/stale LATEST state")
            for name in ("meta.pkl", "buffer.npz", "buffer_meta.pkl", "obs_rms.pkl"):
                path = state / name
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError(f"DONE state is missing {name}")
            if not any(p.is_file() for p in (state / "agent_ckpt").rglob("*")):
                raise ValueError("DONE state is missing the agent checkpoint")
            row["status"] = "complete_marker_structure_only"
        if (parent / "claim_guard").exists():
            row["ownership_note"] = (
                "transition guard present; inspect only after all relevant owners stop"
            )
        if (parent / "claim").exists():
            row["ownership_note"] = (
                "claim present; ownership was not queried or changed"
            )
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        row["status"] = "blocked"
        row["errors"].append(str(exc))
    return row


def inventory(root, commit):
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("checkpoint root must be absolute")
    if root.exists() and not root.is_dir():
        raise ValueError("checkpoint root is not a directory")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("expected commit must be a full lowercase40-character hash")
    rows, locations = [], set()
    for architecture in ARCHITECTURES:
        for environment in ENVIRONMENTS:
            for seed in range(1, 6):
                parent = root / "exp1" / architecture / environment / f"seed_{seed}"
                key = f"exp1_{architecture}_{environment}_seed{seed}"
                resolved = parent.resolve()
                if not resolved.is_relative_to(root.resolve()) or resolved in locations:
                    rows.append(
                        dict(
                            run_key=key,
                            directory=str(parent),
                            status="blocked",
                            errors=[
                                "checkpoint directory escapes root or aliases another parent"
                            ],
                            payload_integrity="not_assessed",
                        )
                    )
                else:
                    rows.append(
                        inspect_parent(parent, key, budget(environment) // 2, commit)
                    )
                locations.add(resolved)
    counts = {
        s: sum(r["status"] == s for r in rows)
        for s in sorted({r["status"] for r in rows})
    }
    return dict(
        schema_version=1,
        expected_commit=commit,
        parent_count=len(rows),
        counts=counts,
        blocked=counts.get("blocked", 0),
        parents=rows,
        qualification_status="not_assessed",
        warning="Marker structure is not complete-state integrity, trigger validity, arm completeness, runtime compatibility or scientific certification. No files or claims were repaired.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = inventory(args.checkpoint_root, args.expected_commit)
        with args.out.open("x") as handle:
            json.dump(report, handle, indent=2)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Inventory refused: {exc}\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("parent_count", "counts", "blocked", "qualification_status")
            }
        )
    )
    return 1 if report["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
