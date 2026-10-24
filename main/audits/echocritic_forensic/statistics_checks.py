"""Independent raw-record operational-CI and seed-aggregate verification."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .evidence import (
    close_error,
    finite,
    read_csv,
    read_records,
    reconstruct_probes,
    write_csv,
    prepare_output,
)

METRICS = (
    "final_return",
    "final_window",
    "peak_return",
    "auc70",
    "return70",
    "final_deficit",
    "common70_deficit",
    "fresh_score_change",
    "fresh_score_change70",
    "late_q_abs",
    "late_td_loss",
)


def operational_checks(records):
    checks, schedules = [], []
    for run in (r for r in records if r["cohort"] == "exp96"):
        settings = run["config"]["trigger"]
        firing, streak, first = [], 0, None
        for row in run["history"]:
            k = row.get("probe/check_index")
            if not finite(k) or k == 0:
                continue
            k = int(k)
            x = np.asarray([row[f"probe/loss_r{i}"] for i in range(10)], np.float64)
            generator = np.random.default_rng([run["seed"], 0x424F4F54, k])
            sample = x[generator.integers(0, 10, size=(settings["resamples"], 10))]
            sample.sort(axis=1)
            # Independent rank-bin weighted reduction, not an imported analysis function.
            weights = np.asarray([0, 0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0, 0])
            values = np.sum(sample * weights, axis=1)
            tail = (1 - settings["confidence"]) / 2
            low, high = np.quantile(values, [tail, 1 - tail])
            for name, expected in (("ci_low", low), ("ci_high", high)):
                error, passed = close_error(row["probe/" + name], expected)
                checks.append(
                    {
                        "id": run["id"],
                        "check": k,
                        "field": name,
                        "error": error,
                        "passed": passed,
                    }
                )
            decision = row["probe/ci_low"] > settings["null_threshold"]
            if bool(row["probe/triggered"]) != decision:
                raise ValueError(
                    "recorded trigger does not implement strict logged lower-bound rule"
                )
            streak = streak + 1 if decision else 0
            if (
                first is None
                and streak >= settings["consecutive_checks"]
                and k
                <= math.floor(
                    settings["eligible_fraction"] * run["config"]["probe"]["checks"]
                    + 1e-9
                )
            ):
                first = k
            firing.append({"check": k, "triggered": decision})
        schedules.append(
            {
                "id": run["id"],
                "first_eligible_two_consecutive_check": first,
                "disk_ledger_not_available": True,
            }
        )
    return checks, schedules


def endpoints(records, points):
    by_run = defaultdict(dict)
    for p in points:
        by_run[p["id"]][p["check"]] = p
    result = []
    for run in (r for r in records if r["cohort"] == "exp96"):
        cfg, history = run["config"], run["history"]
        evaluations = [r for r in history if finite(r.get("eval/avg_return"))]
        x = np.asarray([r["_step"] / cfg["num_env_steps"] for r in evaluations])
        y = np.asarray([r["eval/avg_return"] for r in evaluations], np.float64)
        complete = x[-1] == 1
        common = x <= 0.7
        auc = math.fsum(
            float((x1 - x0) * (y0 + y1) / 2)
            for x0, x1, y0, y1 in zip(
                x[common][:-1], x[common][1:], y[common][:-1], y[common][1:]
            )
        )
        p = by_run[run["id"]]
        item = {key: run[key] for key in ("id", "environment", "architecture", "seed")}
        item.update(
            final_return=float(y[-1]) if complete else None,
            final_window=float(y[x >= 0.9].mean()) if complete else None,
            peak_return=float(y.max()),
            auc70=auc,
            return70=next(float(v) for t, v in zip(x, y) if t == 0.7),
            final_deficit=p[50]["paired_deficit"] if 50 in p else None,
            common70_deficit=p[35]["paired_deficit"],
            fresh_score_change=p[max(p)]["fresh_iqm"] - p[0]["fresh_iqm"],
            fresh_score_change70=p[35]["fresh_iqm"] - p[0]["fresh_iqm"],
        )
        for field, name in (
            ("train/q1_mean", "late_q_abs"),
            ("train/critic_loss", "late_td_loss"),
        ):
            values = [
                r[field]
                for r in history
                if r["_step"] / cfg["num_env_steps"] >= 0.7 and finite(r.get(field))
            ]
            item[name] = float(
                np.median(np.abs(values) if name == "late_q_abs" else values)
            )
        result.append(item)
    return result


def seed_aggregate_checks(package, rows):
    supplied = {
        (r["environment"], r["architecture"], r["metric"]): r
        for r in read_csv(package / "derived/architecture_summaries.csv")
    }
    groups = defaultdict(list)
    for row in rows:
        groups[row["environment"], row["architecture"]].append(row)
    generator = np.random.default_rng(20261024)
    checks = []

    def estimate(values, kind, axis=None):
        if kind == "mean":
            return np.mean(values, axis=axis)
        ordered = np.sort(values, axis=axis)
        cut = ordered.shape[0 if axis is None else axis] // 4
        return np.mean(
            (
                ordered[cut : len(ordered) - cut]
                if axis is None
                else ordered[:, cut : ordered.shape[1] - cut]
            ),
            axis=axis,
        )

    for (environment, architecture), group in sorted(groups.items()):
        for metric in METRICS:
            x = np.asarray(
                [r[metric] for r in group if r[metric] is not None], np.float64
            )
            expected = {
                "n": len(x),
                "median": float(np.median(x)),
                "sd": float(np.std(x, ddof=1)),
            }
            for kind, lower, upper in (
                ("mean", "ci_low", "ci_high"),
                ("iqm", "iqm_low", "iqm_high"),
            ):
                expected[kind] = float(estimate(x, kind))
                sample = x[generator.integers(0, len(x), (10000, len(x)))]
                interval = np.quantile(estimate(sample, kind, axis=1), [0.025, 0.975])
                expected[lower], expected[upper] = map(float, interval)
            row = supplied[environment, architecture, metric]
            for field, value in expected.items():
                error, passed = close_error(float(row[field]), value)
                checks.append(
                    {
                        "environment": environment,
                        "architecture": architecture,
                        "metric": metric,
                        "field": field,
                        "n_independent_seeds": len(x),
                        "error": error,
                        "passed": passed,
                    }
                )
    return checks


def metric_window_checks(metric_package, records):
    supplied = read_csv(metric_package / "per_seed_summaries.csv")
    runs = {r["id"]: r for r in records}
    metrics = (
        "train/critic_DR0.1",
        "train/critic_DR0.2",
        "train/critic_srank",
        "train/critic_gnorm",
        "train/critic_fnorm",
        "train/critic_wnorm",
        "train/critic_loss",
        "train/td_error_var",
        "train/q1_mean",
    )
    checks = []
    initial_times = []
    effective = []
    for row in supplied:
        run = runs[row["id"]]
        history = run["history"]
        budget = run["config"]["num_env_steps"]
        for metric in metrics:
            observed = [r for r in history if finite(r.get(metric))]
            initial_times.append(
                {
                    "id": run["id"],
                    "metric": metric,
                    "first_raw_environment_step": observed[0]["_step"],
                    "first_training_interaction": observed[0]["_step"]
                    / run["config"]["action_repeat"],
                }
            )
            expected = {"initial": observed[0][metric]}
            for label, lo, hi in (
                ("early", 0.1, 0.2),
                ("common70", 0.6, 0.7),
                ("late_observed", 0.8, 1),
            ):
                values = [
                    r[metric] for r in observed if lo <= r["_step"] / budget <= hi
                ]
                expected[label] = float(np.median(values)) if values else None
                effective.append(
                    {
                        "id": run["id"],
                        "environment": run["environment"],
                        "architecture": run["architecture"],
                        "metric": metric,
                        "window": label,
                        "has_observations": bool(values),
                    }
                )
            for field, value in expected.items():
                if value is None:
                    passed, error = not row[metric + "_" + field], None
                else:
                    error, passed = close_error(float(row[metric + "_" + field]), value)
                checks.append(
                    {
                        "id": run["id"],
                        "metric": metric,
                        "field": field,
                        "error": error,
                        "passed": passed,
                    }
                )
    grouped = defaultdict(list)
    for row in effective:
        grouped[
            row["environment"], row["architecture"], row["metric"], row["window"]
        ].append(row)
    finite_counts = [
        {
            "environment": e,
            "architecture": a,
            "metric": m,
            "window": w,
            "nominal_seed_records": len(v),
            "effective_seed_n": sum(r["has_observations"] for r in v),
        }
        for (e, a, m, w), v in sorted(grouped.items())
    ]
    return checks, initial_times, finite_counts


def run(package, metric_package, output):
    package, metric_package, output = map(Path, (package, metric_package, output))
    output = prepare_output(
        output, package, metric_package, Path(__file__).resolve().parents[3]
    )
    records = read_records(package)
    points, _ = reconstruct_probes(records)
    operational, schedules = operational_checks(records)
    endpoint_rows = endpoints(records, points)
    aggregates = seed_aggregate_checks(package, endpoint_rows)
    windows, first_times, effective = metric_window_checks(metric_package, records)
    for name, rows in (
        ("operational_ci_checks", operational),
        ("reconstructed_seed_endpoints", endpoint_rows),
        ("seed_aggregate_checks", aggregates),
        ("metric_window_checks", windows),
        ("metric_first_logged_times", first_times),
        ("metric_effective_seed_counts", effective),
        ("inferred_trigger_schedules", schedules),
    ):
        write_csv(output / (name + ".csv"), rows)
    summary = {}
    for name, rows in (
        ("operational", operational),
        ("seed_aggregate", aggregates),
        ("metric_windows", windows),
    ):
        summary[name] = {
            "checks": len(rows),
            "failures": [r for r in rows if not r["passed"]],
            "max_error": max((r["error"] or 0) for r in rows),
        }
    summary["late_metric_rows_with_less_than_eight_seeds"] = [
        r for r in effective if r["effective_seed_n"] < 8
    ]
    summary["first_metric_is_post_warmup"] = all(
        r["first_training_interaction"] > 5000 for r in first_times
    )
    (output / "checks.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    )
    print(
        json.dumps(
            {
                k: {a: b for a, b in v.items() if a != "failures"}
                for k, v in summary.items()
                if isinstance(v, dict)
            },
            indent=2,
        )
    )
    if any(
        summary[k]["failures"]
        for k in ("operational", "seed_aggregate", "metric_windows")
    ):
        raise SystemExit(
            "independent statistical comparison mismatch; inspect retained rows"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", required=True)
    p.add_argument("--metric-package", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.package, a.metric_package, a.out)


if __name__ == "__main__":
    main()
