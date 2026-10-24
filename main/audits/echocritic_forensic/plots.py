"""Unsmoothed raw-reconstructed score plots with matched observed-seed anchors."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evidence import prepare_output, read_csv


def run(points_file, output):
    out = prepare_output(output, Path(points_file).parent)
    points = read_csv(points_file)
    initial = {p["id"]: float(p["fresh_iqm"]) for p in points if int(p["check"]) == 0}
    groups = defaultdict(list)
    for p in points:
        if int(p["check"]) > 0:
            groups[p["environment"], p["architecture"]].append(p)
    envs = ("dog-run", "humanoid-run", "myo-key-turn", "myo-pen-twirl")
    architectures = ("D2W512", "D4W1024", "D4W1536")
    for scale in ("linear", "symlog"):
        fig, axes = plt.subplots(4, 3, figsize=(15, 13), squeeze=False)
        for i, env in enumerate(envs):
            for j, arch in enumerate(architectures):
                ax, rows = axes[i, j], groups[env, arch]
                by_check, by_seed = defaultdict(list), defaultdict(list)
                for row in rows:
                    by_check[int(row["check"])].append(row)
                    by_seed[row["id"]].append(row)
                for records in by_seed.values():
                    records.sort(key=lambda r: int(r["check"]))
                    ax.plot(
                        [int(r["check"]) / 50 for r in records],
                        [float(r["current_iqm"]) for r in records],
                        color="C0",
                        alpha=0.16,
                        lw=0.7,
                    )
                for field, label, color in (
                    ("current_iqm", "Trained", "C0"),
                    ("fresh_iqm", "Fresh", "C1"),
                    (None, "Initial fresh (matched seeds)", "C2"),
                ):
                    x, y = [], []
                    for check, records in sorted(by_check.items()):
                        x.append(check / 50)
                        y.append(
                            np.mean(
                                [
                                    (
                                        initial[r["id"]]
                                        if field is None
                                        else float(r[field])
                                    )
                                    for r in records
                                ]
                            )
                        )
                    ax.plot(x, y, label=label, color=color)
                ax.set_title(f"{env} / {arch}")
                ax.set_yscale(scale)
                ax.set_xlabel("Recorded training fraction")
                ax.set_ylabel("Source-defined round IQM score")
                ax.grid(alpha=0.2)
                ax.text(
                    0.02,
                    0.03,
                    f"n70=8; n100={len(by_check.get(50, []))}",
                    transform=ax.transAxes,
                    fontsize=8,
                )
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            "Raw JSON reconstruction; no smoothing or imputation; faint lines = trained seeds"
        )
        fig.tight_layout()
        fig.savefig(out / f"scores_{scale}.pdf")
        fig.savefig(out / f"scores_{scale}.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.points, args.out)


if __name__ == "__main__":
    main()
