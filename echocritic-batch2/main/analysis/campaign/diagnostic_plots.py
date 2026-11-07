"""Explicit paired-score validation and unsmoothed diagnostic figures."""

import numpy as np


def validate_pair(fresh, trained):
    for key in (
        "run_id",
        "training_seed",
        "round",
        "checkpoint",
        "task_id",
        "target_hash",
        "optimizer_hash",
        "completed_updates",
    ):
        if key not in fresh or key not in trained or fresh[key] != trained[key]:
            raise ValueError("incompatible or missing pairing provenance: " + key)
    if type(fresh["completed_updates"]) is not int or fresh["completed_updates"] <= 0:
        raise ValueError("invalid fitting horizon")
    for r in (fresh, trained):
        if r.get("status") != "complete":
            raise ValueError("incomplete measurement")
        if not np.isfinite([r["baseline"], r["final_pool_mse"], r["score"]]).all():
            raise ValueError("nonfinite score")
        if not np.isclose(
            r["score"], r["baseline"] - r["final_pool_mse"], rtol=1e-6, atol=0
        ):
            raise ValueError("invalid score")
    return float(fresh["score"] - trained["score"])


def plot_series(
    series, path, metric, environment, units, step_unit="interaction steps"
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for s in series:
        if s["environment"] != environment:
            raise ValueError("mixed environments")
        ax.plot(
            s["steps"],
            s["values"],
            label=f"{s['architecture']} seed{s['seed']} {s['run_id']}",
        )
    n = len({(s["architecture"], s["seed"]) for s in series})
    ax.set(
        xlabel=step_unit,
        ylabel=f"{metric} ({units})",
        title=f"{environment}; {n} architecture/seed units",
    )
    ax.text(
        0.01,
        0.01,
        "Observed values; no interpolation or smoothing",
        transform=ax.transAxes,
        fontsize=7,
    )
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(rows, path, environment, metric, units):
    """Seed dots and mean; compatibility hashes required, separate architecture axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows or any(r["environment"] != environment for r in rows):
        raise ValueError("one environment required")
    if len({r["protocol_hash"] for r in rows}) != 1:
        raise ValueError("incompatible protocols")
    archs = sorted({r["architecture"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, a in enumerate(archs):
        sub = [r for r in rows if r["architecture"] == a]
        if len({r["seed"] for r in sub}) != len(sub):
            raise ValueError("repeated training seed")
        y = [r[metric] for r in sub]
        ax.scatter([i] * len(y), y, label=f"{a} n={len(y)}")
        ax.plot([i - 0.15, i + 0.15], [np.mean(y)] * 2, color="black")
    ax.set(
        xticks=range(len(archs)),
        xticklabels=archs,
        ylabel=f"{metric} ({units})",
        title=environment,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_relationship(rows, path, environment, x, y, x_units, y_units):
    """Within-environment observations; descriptive, without causal regression."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if any(r["environment"] != environment for r in rows):
        raise ValueError("mixed environments")
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in rows:
        ax.scatter(r[x], r[y], label=f"{r['run_id']} checkpoint {r['checkpoint']}")
    ax.set(
        xlabel=f"{x} ({x_units})",
        ylabel=f"{y} ({y_units})",
        title=environment + "; repeated observations, descriptive only",
    )
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
