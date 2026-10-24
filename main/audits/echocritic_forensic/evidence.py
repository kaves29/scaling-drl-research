"""Independent reconstruction from raw JSON, never from supplied pickle/functions."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

CAMPAIGN = "f597ee9898e4d401912092c336c3e5330cf2fda1"
PILOT = "f8619dd5f494399d9febc79fc9092010234c714d"


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def source_iqm(values, rounds):
    """Independent rank-bin integration for ten; historical whole trimming for five."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    if len(values) != rounds or not np.isfinite(values).all():
        raise ValueError("complete finite round array required")
    if rounds == 10:
        left, right = rounds / 4, 3 * rounds / 4
        weights = [max(0, min(i + 1, right) - max(i, left)) for i in range(rounds)]
        return math.fsum(float(v) * w for v, w in zip(values, weights)) / (right - left)
    trim = rounds // 4
    return math.fsum(float(v) for v in values[trim : rounds - trim]) / (
        rounds - 2 * trim
    )


def close_error(actual, expected):
    """Report error and a fixed float64 arithmetic bound; no experiment tolerance changes."""
    error = abs(float(actual) - float(expected))
    bound = (
        32
        * np.finfo(np.float64).eps
        * max(1.0, abs(float(actual)), abs(float(expected)))
    )
    return error, error <= bound


def verify_raw_manifest(package):
    from pathlib import PurePosixPath

    rows = json.loads((Path(package) / "raw_api_manifest.json").read_text())
    for row in rows:
        name = PurePosixPath(row["path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("unsafe raw manifest path")
        data = (Path(package) / "raw_api" / row["path"]).read_bytes()
        if (
            len(data) != row["bytes"]
            or hashlib.sha256(data).hexdigest() != row["sha256"]
        ):
            raise ValueError("raw evidence hash mismatch")
    return len(rows)


def prepare_output(output, *inputs):
    output = Path(output).resolve()
    for source in (*inputs, Path(__file__).resolve().parents[3]):
        source = Path(source).resolve()
        if output == source or source in output.parents:
            raise ValueError("audit output must be outside inputs and repository")
    output.mkdir(parents=True, exist_ok=False)
    return output


def read_records(package):
    verify_raw_manifest(package)
    records = []
    for path in sorted((Path(package) / "raw_api").glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        history = [
            json.loads(s)
            for s in path.with_name("history.jsonl").read_text().splitlines()
            if s.strip()
        ]
        cfg = meta["config"]
        source = meta["attrs"]["commit"]
        if source not in (CAMPAIGN, PILOT):
            raise ValueError("unknown source revision")
        if (
            meta["id"] != path.parent.name
            or len(history) != meta["attrs"]["historyLineCount"]
        ):
            raise ValueError("identity/history count mismatch")
        if not cfg["seed"] == cfg["agent"]["seed"] == cfg["env"]["seed"]:
            raise ValueError("seed binding mismatch")
        steps = [r.get("_step") for r in history]
        if not all(finite(s) for s in steps) or any(
            b <= a for a, b in zip(steps, steps[1:])
        ):
            raise ValueError("invalid logging counters")
        records.append(
            {
                "meta": meta,
                "history": history,
                "config": cfg,
                "id": meta["id"],
                "source": source,
                "cohort": "exp96" if source == CAMPAIGN else "pilot",
                "environment": cfg["env_name"],
                "architecture": f"D{cfg['agent']['critic_num_blocks']}W{cfg['agent']['critic_hidden_dim']}",
                "seed": cfg["seed"],
            }
        )
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate IDs")
    return records


def reconstruct_probes(records):
    rows, errors = [], []
    for run in records:
        cfg, seen = run["config"], set()
        rounds = cfg["probe"]["rounds"]
        for raw in run["history"]:
            k = raw.get("probe/check_index")
            if not finite(k):
                continue
            k = int(k)
            if k in seen:
                raise ValueError("duplicate probe check")
            seen.add(k)
            fresh = [raw.get(f"probe/score_fresh_r{r}") for r in range(rounds)]
            if not all(finite(v) for v in fresh):
                raise ValueError("incomplete fresh rounds")
            point = {
                key: run[key]
                for key in ("id", "environment", "architecture", "seed", "cohort")
            }
            point.update(
                check=k,
                interaction_step=int(raw["probe/interaction_step"]),
                raw_environment_step=int(raw["probe/env_step"]),
                logging_step=raw["_step"],
                fresh_iqm=source_iqm(fresh, rounds),
            )
            if (
                point["raw_environment_step"]
                != point["interaction_step"] * cfg["action_repeat"]
            ):
                raise ValueError("counter-unit mismatch")
            if k == 0:
                expected_step = cfg["buffer"]["min_length"]
                error, passed = close_error(
                    raw["probe/score_fresh_iqm"], point["fresh_iqm"]
                )
                errors.append(
                    {
                        "id": run["id"],
                        "check": 0,
                        "kind": "initial_fresh_iqm",
                        "error": error,
                        "passed": passed,
                    }
                )
            else:
                expected_step = (
                    k * cfg["num_interaction_steps"] // cfg["probe"]["checks"]
                )
                current = [raw.get(f"probe/score_current_r{r}") for r in range(rounds)]
                logged_loss = [raw.get(f"probe/loss_r{r}") for r in range(rounds)]
                if not all(finite(v) for v in current + logged_loss):
                    raise ValueError("incomplete paired rounds")
                # Recorded operands are FP32. Preserve source subtraction before float64 IQM.
                paired = np.asarray(fresh, np.float32) - np.asarray(current, np.float32)
                if not np.array_equal(
                    paired.astype(np.float64), np.asarray(logged_loss, np.float64)
                ):
                    raise ValueError("source FP32 pairing mismatch")
                point.update(
                    current_iqm=source_iqm(current, rounds),
                    paired_deficit=source_iqm(paired, rounds),
                )
                point["marginal_iqm_gap"] = point["fresh_iqm"] - point["current_iqm"]
                point["iqm_nonadditivity"] = (
                    point["marginal_iqm_gap"] - point["paired_deficit"]
                )
                error, passed = close_error(
                    raw["probe/loss_iqm"], point["paired_deficit"]
                )
                errors.append(
                    {
                        "id": run["id"],
                        "check": k,
                        "kind": "paired_iqm",
                        "error": error,
                        "passed": passed,
                    }
                )
            if point["interaction_step"] != expected_step:
                raise ValueError("probe schedule mismatch")
            point["fraction"] = point["interaction_step"] / cfg["num_interaction_steps"]
            rows.append(point)
    return rows, errors


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def verify_source_snapshots(package, repository):
    results = []
    for entry in json.loads((package / "source/manifest.json").read_text()):
        data = (package / "source" / entry["commit"] / entry["path"]).read_bytes()
        original = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "show",
                entry["commit"] + ":" + entry["path"],
            ]
        )
        results.append(
            {
                **entry,
                "passed": data == original
                and hashlib.sha256(data).hexdigest() == entry["sha256"],
            }
        )
    return results


def check_derived(package, metric_package, points, records):
    point_by_key = {(p["id"], p["check"]): p for p in points}
    checks = []
    for row in read_csv(package / "derived/probe_trajectories.csv"):
        p = point_by_key[row["id"], int(row["check"])]
        for field, original in (
            ("fresh_iqm", "probe/score_fresh_iqm"),
            ("current_iqm", "derived_current_score_iqm"),
            ("paired_deficit", "probe/loss_iqm"),
        ):
            if field in p:
                error, passed = close_error(p[field], float(row[original]))
                checks.append(
                    {
                        "id": p["id"],
                        "check": p["check"],
                        "table": "probe_trajectories",
                        "field": field,
                        "error": error,
                        "passed": passed,
                    }
                )
    initial = {p["id"]: p["fresh_iqm"] for p in points if p["check"] == 0}
    runs = {r["id"]: r for r in records}
    for row in read_csv(metric_package / "aligned_probe_metrics.csv"):
        run = runs[row["id"]]
        step = int(float(row["interaction_step"]))
        p = next(
            p for p in points if p["id"] == row["id"] and p["interaction_step"] == step
        )
        expected = {
            "trained_score": p["current_iqm"],
            "anchored_deficit": initial[p["id"]] - p["current_iqm"],
            "source_paired_deficit": p["paired_deficit"],
        }
        for key in (k for k in row if k.startswith("train/")):
            candidates = [
                (raw["_step"] / run["config"]["action_repeat"], raw[key])
                for raw in run["history"]
                if finite(raw.get(key))
                and raw["_step"] / run["config"]["action_repeat"] <= step
            ]
            previous = max(candidates, default=None, key=lambda pair: pair[0])
            expected[key] = (
                previous[1] if previous and step - previous[0] <= 2000 else None
            )
        for key, value in expected.items():
            if value is None:
                passed, error = not row[key], None
            else:
                error, passed = close_error(value, float(row[key]))
            checks.append(
                {
                    "id": p["id"],
                    "check": p["check"],
                    "table": "aligned_probe_metrics",
                    "field": key,
                    "error": error,
                    "passed": passed,
                }
            )
    return checks


def duplicate_trajectories(records):
    canonical = {
        (r["environment"], r["architecture"], r["seed"]): r
        for r in records
        if r["cohort"] == "exp96"
    }
    checks = []
    for pilot in (r for r in records if r["cohort"] == "pilot"):
        campaign = canonical[pilot["environment"], pilot["architecture"], pilot["seed"]]
        shared, errors = 0, []
        later = {r["_step"]: r for r in campaign["history"]}
        for row in pilot["history"]:
            other = later.get(row["_step"], {})
            for key, value in row.items():
                if (
                    key.startswith(("train/", "eval/"))
                    and finite(value)
                    and finite(other.get(key))
                ):
                    shared += 1
                    if value != other[key]:
                        errors.append(
                            {
                                "step": row["_step"],
                                "metric": key,
                                "pilot": value,
                                "campaign": other[key],
                            }
                        )
        checks.append(
            {
                "pilot": pilot["id"],
                "campaign": campaign["id"],
                "shared_values": shared,
                "exact_matches": shared - len(errors),
                "differences": errors,
            }
        )
    return checks


def archived_probe(package, points, records):
    path = next((package / "raw_archived").glob("*.npz"))
    run = next(
        r
        for r in records
        if r["cohort"] == "exp96"
        and (r["environment"], r["architecture"], r["seed"])
        == ("myo-pen-twirl", "D2W512", 5)
    )
    raw = next(r for r in run["history"] if r.get("probe/check_index") == 50)
    for name in ("archived_probe_origin.json", "archived_probe_provenance.json"):
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != json.loads((package / name).read_text())["sha256"]
        ):
            raise ValueError("archived probe hash does not match supplied provenance")
    out = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "run_id": run["id"],
        "check": 50,
    }
    with np.load(path, allow_pickle=False) as data:
        out["arrays"] = {
            k: {"shape": data[k].shape, "dtype": str(data[k].dtype)} for k in data.files
        }
        out["shared_variances_exact"] = bool(
            np.array_equal(data["fresh_b"], data["current_b"])
        )
        for kind in ("fresh", "current"):
            scores = data[kind + "_b"] - data[kind + "_final_loss"]
            out[kind] = {
                "score_identity_exact": bool(
                    np.array_equal(scores, data[kind + "_score"])
                ),
                "wandb_score_max_error": max(
                    abs(float(scores[i]) - raw[f"probe/score_{kind}_r{i}"])
                    for i in range(10)
                ),
                "median_initial_minibatch_mse": float(
                    np.median(data[kind + "_losses"][:, 0])
                ),
                "median_final_pool_mse": float(np.median(data[kind + "_final_loss"])),
                "median_offset": float(np.median(data[kind + "_offset"])),
                "last_minibatch_score_differs_from_actual_max": float(
                    np.max(
                        np.abs(
                            (data[kind + "_b"] - data[kind + "_losses"][:, -1])
                            - data[kind + "_score"]
                        )
                    )
                ),
                "curve_finite": bool(np.isfinite(data[kind + "_losses"]).all()),
            }
        out["can_reconstruct"] = [
            "stored FP32 b-minus-final-MSE scores",
            "W&B round binding",
            "2000 changing-minibatch pre-update losses",
            "per-round offsets and shared variance",
        ]
        out["cannot_reconstruct"] = [
            "actual parameter hashes",
            "original replay/normalizer",
            "numeric targets/indices",
            "full-pool initial/intermediate MSE",
            "optimizer states or exact GPU execution",
        ]
    return out


def summarize_points(points):
    initial = {p["id"]: p["fresh_iqm"] for p in points if p["check"] == 0}
    common = []
    for p in points:
        if p["check"] == 35:
            common.append(
                {
                    **p,
                    "initial_fresh_iqm": initial[p["id"]],
                    "fresh_change_from_initial": p["fresh_iqm"] - initial[p["id"]],
                    "trained_change_from_initial": p["current_iqm"] - initial[p["id"]],
                    "anchored_deficit": initial[p["id"]] - p["current_iqm"],
                }
            )
    groups = defaultdict(list)
    for p in common:
        groups[p["environment"], p["architecture"]].append(p)
    summary = []
    for (env, arch), rows in sorted(groups.items()):
        if sorted(p["seed"] for p in rows) != list(range(1, 9)):
            raise ValueError("common horizon does not contain all independent seeds")
        summary.append(
            {
                "environment": env,
                "architecture": arch,
                "n_seeds": len(rows),
                **{
                    key: math.fsum(p[key] for p in rows) / len(rows)
                    for key in (
                        "initial_fresh_iqm",
                        "fresh_iqm",
                        "current_iqm",
                        "fresh_change_from_initial",
                        "trained_change_from_initial",
                        "paired_deficit",
                        "anchored_deficit",
                    )
                },
            }
        )
    return common, summary


def run(package, metric_package, output, repository):
    package, metric_package, output = map(Path, (package, metric_package, output))
    output = prepare_output(output, package, metric_package, repository)
    records = read_records(package)
    points, numerical = reconstruct_probes(records)
    derived = check_derived(package, metric_package, points, records)
    duplicates = duplicate_trajectories(records)
    source = verify_source_snapshots(package, repository)
    archived = archived_probe(package, points, records)
    common, groups = summarize_points(points)
    for name, rows in (
        ("independent_probe_points", points),
        ("common70_per_seed", common),
        ("common70_groups", groups),
        ("numerical_checks", numerical),
        ("derived_point_checks", derived),
    ):
        write_csv(output / (name + ".csv"), rows)
    report = {
        "records": len(records),
        "campaign": sum(r["cohort"] == "exp96" for r in records),
        "pilot": sum(r["cohort"] == "pilot" for r in records),
        "states": dict(Counter(r["meta"]["state"] for r in records)),
        "probe_points": len(points),
        "nonzero_probe_points": sum(p["check"] > 0 for p in points),
        "common70_runs": len(common),
        "fresh_declines_at70": sum(p["fresh_change_from_initial"] < 0 for p in common),
        "numerical_checks": len(numerical),
        "numerical_failures": [r for r in numerical if not r["passed"]],
        "max_source_iqm_error": max(r["error"] for r in numerical),
        "derived_point_checks": len(derived),
        "derived_point_failures": [r for r in derived if not r["passed"]],
        "max_derived_point_error": max(r["error"] or 0 for r in derived),
        "duplicate_pilot_trajectories": duplicates,
        "source_snapshots": source,
        "archived_probe": archived,
        "initial_reference_directory_present": (
            package / "initial_reference_comparison"
        ).exists(),
        "pairing_nonadditivity_max": max(
            abs(p.get("iqm_nonadditivity", 0)) for p in points
        ),
        "interpretation": "Source and numeric data validation only; no causal mechanism or production checkpoint identity inferred",
    }
    (output / "checks.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "records",
                    "campaign",
                    "pilot",
                    "probe_points",
                    "common70_runs",
                    "fresh_declines_at70",
                    "numerical_checks",
                    "max_source_iqm_error",
                    "derived_point_checks",
                    "max_derived_point_error",
                )
            },
            indent=2,
        )
    )
    if (
        report["numerical_failures"]
        or report["derived_point_failures"]
        or not all(r["passed"] for r in source)
    ):
        raise SystemExit(
            "independent validation mismatch; inspect retained check records"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--metric-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()
    run(args.package, args.metric_package, args.out, args.repository)


if __name__ == "__main__":
    main()
