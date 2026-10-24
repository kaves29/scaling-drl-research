"""Read raw stored metrics and source blobs only; no model imports or fitting."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .evidence import (
    CAMPAIGN,
    close_error,
    finite,
    prepare_output,
    read_csv,
    read_records,
    reconstruct_probes,
    write_csv,
)

METRICS = (
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


def source_comparison(repository, current):
    paths = (
        "scale_rl/networks/layers.py",
        "scale_rl/networks/metrics.py",
        "scale_rl/networks/trainer.py",
        "scale_rl/networks/utils.py",
        "scale_rl/agents/sac/sac_network.py",
        "scale_rl/agents/sac/sac_update.py",
        "scale_rl/agents/sac/sac_agent.py",
        "scale_rl/agents/wrappers/normalization.py",
        "scale_rl/common/logger.py",
        "experiments/angle_1.py",
        "experiments/exp12/diagnostics.py",
        "experiments/exp12/probe.py",
        "experiments/exp12/run_probes.py",
        "experiments/exp12/trainer.py",
    )
    rows = []
    for path in paths:
        blobs = [
            subprocess.check_output(
                ["git", "-C", str(repository), "show", revision + ":main/" + path]
            )
            for revision in (CAMPAIGN, current)
        ]
        rows.append(
            {
                "path": "main/" + path,
                "campaign_sha256": hashlib.sha256(blobs[0]).hexdigest(),
                "current_sha256": hashlib.sha256(blobs[1]).hexdigest(),
                "identical": blobs[0] == blobs[1],
            }
        )
    return rows


def run(package, metric_package, repository, current, output):
    out = prepare_output(output, package, metric_package, repository)
    records = [r for r in read_records(package) if r["cohort"] == "exp96"]
    points, _ = reconstruct_probes(records)
    probes = {(p["id"], p["check"]): p for p in points}
    supplied = {
        r["id"]: r for r in read_csv(Path(metric_package) / "per_seed_summaries.csv")
    }
    seed_rows, checks = [], []
    coverage, q_checks, metric_errors = Counter(), 0, []
    first_times = defaultdict(set)
    for record in records:
        rid, cfg, history = record["id"], record["config"], record["history"]
        row = {k: record[k] for k in ("id", "environment", "architecture", "seed")}
        for metric in METRICS:
            observations = [h for h in history if finite(h.get(metric))]
            coverage[metric] += len(observations)
            first_times[metric].add(observations[0]["_step"] / cfg["action_repeat"])
            values = {"initial": float(observations[0][metric])}
            for label, lower, upper in (("early", 0.1, 0.2), ("common70", 0.6, 0.7)):
                values[label] = float(
                    np.median(
                        [
                            h[metric]
                            for h in observations
                            if lower <= h["_step"] / cfg["num_env_steps"] <= upper
                        ]
                    )
                )
            for label, value in values.items():
                row[metric + "_" + label] = value
                error, passed = close_error(
                    value, float(supplied[rid][metric + "_" + label])
                )
                checks.append(
                    {
                        "id": rid,
                        "metric": metric,
                        "field": label,
                        "error": error,
                        "passed": passed,
                    }
                )
        for h in history:
            for metric in METRICS:
                value = h.get(metric)
                if type(value) in (int, float) and not finite(value):
                    metric_errors.append(
                        {
                            "id": rid,
                            "step": h["_step"],
                            "issue": "nonfinite_numeric_metric",
                            "metric": metric,
                        }
                    )
            if finite(h.get("train/q1_mean")) and finite(h.get("train/q2_mean")):
                q_checks += 1
                if h["train/q1_mean"] != h["train/q2_mean"]:
                    metric_errors.append(
                        {
                            "id": rid,
                            "step": h["_step"],
                            "issue": "single_critic_q_labels_differ",
                        }
                    )
            if (
                finite(h.get("train/critic_srank"))
                and not 1 <= h["train/critic_srank"] <= 256
            ):
                metric_errors.append(
                    {"id": rid, "step": h["_step"], "issue": "rank_out_of_range"}
                )
            if finite(h.get("train/critic_DR0.1")) and finite(
                h.get("train/critic_DR0.2")
            ):
                if not 0 <= h["train/critic_DR0.1"] <= h["train/critic_DR0.2"] <= 100:
                    metric_errors.append(
                        {
                            "id": rid,
                            "step": h["_step"],
                            "issue": "dormancy_bounds_or_threshold_order",
                        }
                    )
        point = probes[rid, 35]
        row["anchored_gap70"] = probes[rid, 0]["fresh_iqm"] - point["current_iqm"]
        row["trained_score70"] = point["current_iqm"]
        for field in ("anchored_gap70", "trained_score70"):
            error, passed = close_error(row[field], float(supplied[rid][field]))
            checks.append(
                {
                    "id": rid,
                    "metric": field,
                    "field": "check35",
                    "error": error,
                    "passed": passed,
                }
            )
        seed_rows.append(row)
    groups = defaultdict(list)
    for row in seed_rows:
        groups[row["environment"], row["architecture"]].append(row)
    group_rows = []
    for (env, arch), rows in sorted(groups.items()):
        rank = "train/critic_srank"
        gradient = "train/critic_gnorm"
        dormancy = "train/critic_DR0.2"

        def mean(field):
            return float(np.mean([r[field] for r in rows]))

        group_rows.append(
            {
                "environment": env,
                "architecture": arch,
                "n_seeds": len(rows),
                "rank_initial": mean(rank + "_initial"),
                "rank_common70": mean(rank + "_common70"),
                "rank_down_seeds": sum(
                    r[rank + "_common70"] < r[rank + "_initial"] for r in rows
                ),
                "gradient_early": mean(gradient + "_early"),
                "gradient_common70": mean(gradient + "_common70"),
                "gradient_up_seeds": sum(
                    r[gradient + "_common70"] > r[gradient + "_early"] for r in rows
                ),
                "dormancy_early": mean(dormancy + "_early"),
                "dormancy_common70": mean(dormancy + "_common70"),
                "dormancy_down_seeds": sum(
                    r[dormancy + "_common70"] < r[dormancy + "_early"] for r in rows
                ),
                "anchored_gap70": mean("anchored_gap70"),
            }
        )
    sources = source_comparison(repository, current)
    for name, rows in (
        ("per_seed", seed_rows),
        ("groups", group_rows),
        ("published_checks", checks),
        ("source_comparison", sources),
    ):
        write_csv(out / (name + ".csv"), rows)
    receipt = {
        "campaign": CAMPAIGN,
        "current_comparison": current,
        "evidence_type": "read-only stored raw JSON and Git blobs; no fits or model execution",
        "records": len(records),
        "checks": len(checks),
        "failures": [r for r in checks if not r["passed"]],
        "max_error": max(r["error"] for r in checks),
        "metric_coverage": dict(coverage),
        "first_interaction_per_metric": {k: sorted(v) for k, v in first_times.items()},
        "q_label_exact_matches": q_checks,
        "metric_invariant_errors": metric_errors,
        "source_paths_identical": sum(r["identical"] for r in sources),
        "source_paths_checked": len(sources),
        "groups": group_rows,
        "analyzed_document_sha256": {
            (
                str(p.relative_to(Path(package)))
                if Path(package) in p.parents
                else p.name
            ): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(package) / "REPORT.md",
                Path(package) / "DISCOVERY_LEDGER.md",
                Path(metric_package) / "FINDINGS.md",
            )
        },
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["failures"] or metric_errors:
        raise SystemExit("raw diagnostic comparison mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "metric-package", "repository", "current", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.package, args.metric_package, args.repository, args.current, args.out)


if __name__ == "__main__":
    main()
