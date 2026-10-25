"""Scientific figures from retained-data accounting; no fitting/model imports."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from audits.echocritic_forensic.evidence import read_csv, prepare_output
from .accounting import ARCHITECTURES, ENVIRONMENTS, seed_interval


def numeric(value):
    return float(value) if value not in (None, "") else None


def condition_figures(source, output):
    points = read_csv(source / "probe_points.csv")
    intervals = read_csv(source / "condition_intervals.csv")
    lookup = defaultdict(list)
    for row in intervals:
        lookup[row["environment"], row["architecture"], row["metric"]].append(row)
    fields = (
        ("fresh_iqm", "Fresh score (not identified fitting difficulty)"),
        ("current_iqm", "Trained score"),
        ("paired_iqm", "Contemporary paired deficit"),
        ("anchored_gap", "Initial fresh minus trained score"),
        ("b_iqm", "Base-target variance b"),
        ("fresh_final_pool_mse_iqm", "Fresh final full-pool MSE"),
        ("current_final_pool_mse_iqm", "Trained final full-pool MSE"),
        ("fresh_mse_over_b_iqm", "Fresh full-pool MSE/b"),
        ("current_mse_over_b_iqm", "Trained full-pool MSE/b"),
    )
    with PdfPages(output / "all12_condition_trajectories.pdf") as pdf:
        for env in ENVIRONMENTS:
            for arch in ARCHITECTURES:
                fig, axes = plt.subplots(3, 3, figsize=(15, 11))
                selected = [
                    p
                    for p in points
                    if (p["environment"], p["architecture"]) == (env, arch)
                ]
                for ax, (field, label) in zip(axes.flat, fields):
                    by_seed = defaultdict(list)
                    for p in selected:
                        if numeric(p[field]) is not None:
                            by_seed[p["seed"]].append(p)
                    for rows in by_seed.values():
                        rows.sort(key=lambda r: int(r["check"]))
                        ax.plot(
                            [int(r["interaction_step"]) / 1000 for r in rows],
                            [float(r[field]) for r in rows],
                            color="C0",
                            alpha=0.18,
                            lw=0.7,
                            marker="o" if len(rows) == 1 else None,
                        )
                    rows = [
                        r
                        for r in lookup[env, arch, field]
                        if numeric(r["mean"]) is not None
                    ]
                    rows.sort(key=lambda r: int(r["check"]))
                    if rows:
                        x = np.array([int(r["interaction_step"]) / 1000 for r in rows])
                        ax.plot(
                            x,
                            [float(r["mean"]) for r in rows],
                            color="C0",
                            marker="o" if len(rows) == 1 else None,
                        )
                        valid = [r for r in rows if numeric(r["ci_low"]) is not None]
                        if valid:
                            ax.fill_between(
                                [int(r["interaction_step"]) / 1000 for r in valid],
                                [float(r["ci_low"]) for r in valid],
                                [float(r["ci_high"]) for r in valid],
                                color="C0",
                                alpha=0.18,
                            )
                        ax.text(
                            0.02,
                            0.03,
                            f"observed parents n={rows[0]['n_seeds']}→{rows[-1]['n_seeds']}",
                            transform=ax.transAxes,
                            fontsize=8,
                        )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "Not retained\nNo trajectory reconstructed",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            color="dimgray",
                        )
                        ax.set_facecolor("#f4f4f4")
                    ax.set_title(label, fontsize=9)
                    ax.set_xlabel("Training interactions (thousands)")
                    ax.grid(alpha=0.2)
                fig.suptitle(
                    f"{env} / {arch}\nUnsmoothed observed points; faint individual seeds; 95% pointwise seed bootstrap.\nComponents: one original seed/check only; no round-as-seed uncertainty.",
                    fontsize=11,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.94])
                pdf.savefig(fig)
                fig.savefig(output / f"{env}_{arch}.png", dpi=110)
                plt.close(fig)


def domain_figures(source, output):
    rows = read_csv(source / "domain_seed_contrasts.csv")
    fields = (
        ("fresh_iqm", "Fresh score"),
        ("current_iqm", "Trained score"),
        ("paired_iqm", "Contemporary paired deficit"),
        ("anchored_gap", "Initial-reference gap"),
    )
    fig, axes = plt.subplots(4, 3, figsize=(15, 13))
    for j, arch in enumerate(ARCHITECTURES):
        for i, (field, label) in enumerate(fields):
            ax = axes[i, j]
            data = [
                r for r in rows if r["architecture"] == arch and r["metric"] == field
            ]
            for domain, color in (("dmc", "C0"), ("myo", "C1")):
                groups = defaultdict(list)
                by_seed = defaultdict(list)
                for r in data:
                    groups[int(r["check"])].append(
                        float(r[domain + "_equal_task_mean"])
                    )
                    by_seed[r["seed"]].append(r)
                for values in by_seed.values():
                    ax.plot(
                        [int(r["interaction_step"]) / 1000 for r in values],
                        [float(r[domain + "_equal_task_mean"]) for r in values],
                        color=color,
                        alpha=0.13,
                        lw=0.6,
                    )
                x, mean, low, high = [], [], [], []
                for k, values in sorted(groups.items()):
                    interval = seed_interval(values, 900000 + j * 100 + k)
                    x.append(k * 10)
                    mean.append(interval["mean"])
                    low.append(interval["ci_low"])
                    high.append(interval["ci_high"])
                ax.plot(x, mean, color=color, label=domain.upper())
                ax.fill_between(x, low, high, color=color, alpha=0.15)
            ax.set_title(arch + " / " + label, fontsize=9)
            ax.set_xlabel("Training interactions (thousands)")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(
        "Observed common checkpoints only: all96 parents through check39 (78%)\nEqual-task means per domain, eight seed-label clusters; two tasks/domain are not independent environment replication",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output / "DMC_vs_Myo_common.pdf")
    fig.savefig(output / "DMC_vs_Myo_common.png", dpi=120)
    plt.close(fig)


def archive_curve_figure(package, output):
    path = Path(package) / "raw_archived/D2W512_myo-pen-twirl_seed5_check50.npz"
    with np.load(path, allow_pickle=False) as data:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, kind in zip(axes, ("fresh", "current")):
            losses = data[kind + "_losses"]
            for rep in range(10):
                ax.plot(np.arange(2000), losses[rep], alpha=0.14, lw=0.5, color="C0")
                ax.scatter(
                    [2000],
                    [data[kind + "_final_loss"][rep]],
                    color="C1",
                    alpha=0.5,
                    s=18,
                )
            ax.plot(
                np.arange(2000),
                np.median(losses, axis=0),
                color="C0",
                lw=1,
                label="Median changing-minibatch pre-update loss",
            )
            ax.scatter(
                [2000],
                [np.median(data[kind + "_final_loss"])],
                color="C1",
                s=35,
                label="Post-fit full-pool MSE",
            )
            ax.set_yscale("log")
            ax.set_xlabel("Fitting updates completed")
            ax.set_ylabel("Squared-output loss (log axis)")
            ax.set_title(kind)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        fig.suptitle(
            "Original pen-twirl / D2W512 / seed5 / check50\nTen rounds of one parent; no intermediate full-pool curve and no seed-level CI"
        )
        fig.tight_layout()
        fig.savefig(output / "original_archived_curves.pdf")
        fig.savefig(output / "original_archived_curves.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = prepare_output(args.out, args.analysis, args.package)
    condition_figures(args.analysis, output)
    domain_figures(args.analysis, output)
    archive_curve_figure(args.package, output)


if __name__ == "__main__":
    main()
