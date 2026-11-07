"""CPU offline CLI. Ambiguous conditions are inventoried, never auto-stitched."""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from .ingest import ingest
from .statistics import summary, trajectory


def json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def run(archive, output, metric="eval/avg_return", reps=1000):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    data = ingest(archive)
    (out / "validation.json").write_text(
        json.dumps(json_safe(data), sort_keys=True, indent=2, allow_nan=False)
    )
    multiplicity = Counter(
        (r["campaign"], r["environment"], r["architecture"], r["seed"])
        for r in data["records"]
    )
    ids = Counter(r["id"] for r in data["records"])
    rows = []
    groups = defaultdict(list)
    for r in data["records"]:
        key = (r["campaign"], r["environment"], r["architecture"], r["seed"])
        reasons = []
        if ids[r["id"]] != 1:
            reasons.append("duplicate_id")
        if multiplicity[key] != 1:
            reasons.append("ambiguous_condition")
        if r["campaign"] not in ("exp96", "pilot"):
            reasons.append("unverified_membership")
        if not r["source"].get("commit") or r["source"].get("dirty") is not False:
            reasons.append("unverified_source")
        if any(
            i.get("record") == r["archive_member"]
            and i["code"]
            in ("nonincreasing_steps", "invalid_step_counter", "nonfinite_metric")
            for i in data["issues"]
        ):
            reasons.append("invalid_history")
        if type(r["seed"]) is not int or not r["environment"] or not r["architecture"]:
            reasons.append("missing_identity")
        points = [
            (h[r["step_key"]], h[metric])
            for h in r["history"]
            if r["step_key"] in h and metric in h
        ]
        stats = None
        try:
            stats = trajectory(points, r["horizon"])
        except (ValueError, TypeError):
            reasons.append("invalid_or_missing_trajectory")
        if r["status"] not in ("finished", "complete"):
            reasons.append("nonfinished")
        if stats and not stats["complete"]:
            reasons.append("incomplete_horizon")
        row = {
            "id": r["id"],
            "environment": r["environment"],
            "architecture": r["architecture"],
            "seed": r["seed"],
            "statistics": stats,
            "aggregate_exclusions": reasons,
        }
        rows.append(row)
        if not reasons:
            # Exact config minus explicit replicate identity only. All other settings/source retained.
            cfg = {k: v for k, v in r["config"].items() if k != "seed"}
            from .ingest import digest

            group = (
                r["campaign"],
                r["environment"],
                r["architecture"],
                digest(cfg),
                json.dumps(r["source"], sort_keys=True),
            )
            groups[group].append((r, points, stats))
    aggregate = []
    for key, rs in sorted(groups.items()):
        result = {"group": key, "seed_ids": [r["seed"] for r, _, _ in rs]}
        for value in ("last_observed", "peak", "final_window", "observed_auc"):
            result[value] = summary([s[value] for _, _, s in rs], reps=reps)
        aggregate.append(result)
    (out / "statistics.json").write_text(
        json.dumps(
            {
                "runs": rows,
                "groups": aggregate,
                "missing_data_sensitivity": {
                    "complete_only_n": sum(not r["aggregate_exclusions"] for r in rows),
                    "excluded_n": sum(bool(r["aggregate_exclusions"]) for r in rows),
                    "note": "Excluded trajectories retain observed summaries; no zero imputation or population bound without a prespecified return range.",
                },
            },
            indent=2,
        )
    )
    figures(groups, out, metric, reps)
    return data


def figures(groups, out, metric, reps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for i, (key, rs) in enumerate(sorted(groups.items())):
        fig, ax = plt.subplots(figsize=(7, 4))
        steps = sorted(set(t for _, p, _ in rs for t, v in p))
        means = []
        lo = []
        hi = []
        ns = []
        for r, p, s in rs:
            x, y = zip(*p)
            ax.plot(
                x, y, alpha=0.35, label=f"seed {r['seed']}" if len(rs) <= 12 else None
            )
        for t in steps:
            vals = [v for _, p, _ in rs for x, v in p if x == t]
            a = summary(vals, reps=reps)
            means.append(a["mean"])
            ns.append(len(vals))
            ci = a["mean_ci95"]
            lo.append(ci[0] if ci else np.nan)
            hi.append(ci[1] if ci else np.nan)
        ax.plot(steps, means, color="black", label="observed-step mean")
        ax.fill_between(steps, lo, hi, color="black", alpha=0.15)
        ax.set(
            xlabel="Interaction steps",
            ylabel=metric,
            title=f"{key[1]} {key[2]}; n={len(rs)} independent seeds",
        )
        ax.text(
            0.01,
            0.01,
            "Pointwise 95% seed bootstrap; no interpolation; n per step: " + str(ns),
            transform=ax.transAxes,
            fontsize=6,
        )
        ax.legend(
            fontsize=7,
            labels=["seed trajectories and observed mean"] if len(rs) > 12 else None,
        )
        fig.tight_layout()
        fig.savefig(out / f"learning_{i:03d}.pdf")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive")
    p.add_argument("--out", required=True)
    p.add_argument("--metric", default="eval/avg_return")
    p.add_argument("--bootstrap-reps", type=int, default=1000)
    a = p.parse_args()
    run(a.archive, a.out, a.metric, a.bootstrap_reps)


if __name__ == "__main__":
    main()
