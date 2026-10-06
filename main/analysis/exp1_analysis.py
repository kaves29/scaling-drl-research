"""Experiment 1 analysis (.claude/methodology-exp1-exp2.md; decisions C1-C3).

Primary endpoint: plasticity loss at the final scheduled check (20/20),
difference = IQM(scaled) - IQM(D2W512) over all runs, pooled across
environments with rliable's stratified bootstrap (strata = environments;
the two architectures are resampled independently, i.e. runs are unpaired) and
percentile 95% intervals. Descriptive outputs: plasticity-loss trajectories on
a budget-fraction axis, every seed in every environment, the per-run f*_run
table, and probe learning curves. Development runs are excluded by default.

    python -m analysis.exp1_analysis --out /abs/path/exp1_analysis [--include-dev] [--results-root ...]
"""

import argparse
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics

from experiments.exp12 import ledger
from utils.paths import require_absolute

DEFAULT = "D2W512"
SCALED = ("D4W1024", "D4W1536")
REPS = 50_000
SEED = 0
# Categorical slots 1-3 of the reference palette (dataviz skill), one per architecture.
ARCH_COLORS = {"D2W512": "#2a78d6", "D4W1024": "#eb6834", "D4W1536": "#1baf7a"}


def interval_estimate(scores, func, reps: int, seed: int):
    """rliable.get_interval_estimates, reproducibly. rliable 1.2.0's
    StratifiedIndependentBootstrap draws indices with the global np.random.choice
    and ignores random_state, so the global stream is seeded as well (this is
    offline analysis; no training stream is involved)."""
    np.random.seed(seed)
    point, interval = rly.get_interval_estimates(
        {"x": scores}, func, reps=reps, random_state=np.random.RandomState(seed)
    )
    return float(point["x"][0]), float(interval["x"][0, 0]), float(interval["x"][1, 0])


def score_matrix(runs: pd.DataFrame, architecture: str, value: str = "final_loss_iqm") -> np.ndarray:
    """(num_seeds, num_envs) matrix with environments in sorted order; refuses gaps."""
    sub = runs[runs.architecture == architecture]
    table = sub.pivot_table(index="seed", columns="environment", values=value, aggfunc="first")
    table = table.reindex(columns=sorted(runs.environment.unique()))
    if table.isna().any().any() or table.shape[0] == 0:
        missing = [(s, e) for s in table.index for e in table.columns if pd.isna(table.loc[s, e])]
        raise ValueError(f"{architecture}: missing or non-finite {value} for (seed, env) {missing or 'all'}")
    return table.to_numpy(dtype=np.float64)


def primary_endpoint(runs: pd.DataFrame, default: str = DEFAULT, scaled: Sequence[str] = SCALED,
                     reps: int = REPS, seed: int = SEED) -> pd.DataFrame:
    y = score_matrix(runs, default)
    rows = []
    for arch in scaled:
        x = score_matrix(runs, arch)
        point, low, high = interval_estimate(
            (x, y), lambda a, b: np.array([metrics.aggregate_iqm(a) - metrics.aggregate_iqm(b)]), reps, seed
        )
        rows.append({
            "comparison": f"{arch} - {default}", "statistic": "IQM(scaled) - IQM(default) of final-check L",
            "point": point, "ci_low": low, "ci_high": high, "runs_scaled": x.size, "runs_default": y.size,
            "environments": x.shape[1], "reps": reps,
        })
    return pd.DataFrame(rows)


def per_architecture_iqm(runs: pd.DataFrame, reps: int = REPS, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for arch in sorted(runs.architecture.unique()):
        x = score_matrix(runs, arch)
        point, low, high = interval_estimate(x, lambda a: np.array([metrics.aggregate_iqm(a)]), reps, seed)
        rows.append({"architecture": arch, "final_loss_iqm": point, "ci_low": low, "ci_high": high, "runs": x.size})
    return pd.DataFrame(rows)


def trajectories(checks: pd.DataFrame, reps: int = 2_000, seed: int = SEED) -> pd.DataFrame:
    """Descriptive: per architecture and check, IQM of L over runs with a stratified-bootstrap band."""
    rows = []
    checks = checks[checks.check_index > 0]
    for (arch, k), sub in checks.groupby(["architecture", "check_index"]):
        x = sub.pivot_table(index="seed", columns="environment", values="loss_iqm", aggfunc="first")
        x = x.to_numpy(dtype=np.float64)
        if np.isnan(x).any():
            point, low, high = np.nan, np.nan, np.nan
        else:
            point, low, high = interval_estimate(x, lambda a: np.array([metrics.aggregate_iqm(a)]), reps, seed)
        rows.append({"architecture": arch, "check_index": k, "budget_fraction": float(sub.budget_fraction.iloc[0]),
                     "loss_iqm_over_runs": point, "ci_low": low, "ci_high": high, "runs": int(np.isfinite(x).sum())})
    return pd.DataFrame(rows)


def f_star_table(runs: pd.DataFrame) -> pd.DataFrame:
    return runs[["run_key", "run_role", "architecture", "environment", "seed", "f_star_check",
                 "f_star_interaction_step", "f_star_fraction"]].sort_values(["architecture", "environment", "seed"])


def _plot_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    return plt


def plot_trajectories(traj: pd.DataFrame, path: Path) -> None:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for arch, sub in traj.groupby("architecture"):
        color = ARCH_COLORS.get(arch, "#52514e")
        ax.fill_between(sub.budget_fraction, sub.ci_low, sub.ci_high, color=color, alpha=0.15, linewidth=0)
        ax.plot(sub.budget_fraction, sub.loss_iqm_over_runs, color=color, linewidth=2, label=arch)
    ax.axhline(0, color="#52514e", linewidth=1)
    ax.set_xlabel("fraction of training budget")
    ax.set_ylabel("plasticity loss L (IQM over runs)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_every_seed(checks: pd.DataFrame, path: Path) -> None:
    plt = _plot_setup()
    checks = checks[checks.check_index > 0]
    envs = sorted(checks.environment.unique())
    cols = min(4, len(envs))
    rows = int(np.ceil(len(envs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False, sharex=True)
    for ax, env in zip(axes.flat, envs):
        for (arch, seed), sub in checks[checks.environment == env].groupby(["architecture", "seed"]):
            ax.plot(sub.budget_fraction, sub.loss_iqm, color=ARCH_COLORS.get(arch, "#52514e"), linewidth=1,
                    alpha=0.8, label=arch if seed == sub.seed.min() else None)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_title(env, loc="left")
    for ax in list(axes.flat)[len(envs):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=c, linewidth=2) for c in ARCH_COLORS.values()]
    fig.legend(handles, list(ARCH_COLORS), frameon=False, loc="upper right")
    fig.supxlabel("fraction of training budget")
    fig.supylabel("plasticity loss L per seed")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(run_key: str, path: Path, results_root=None, check_indices=(1, 10, 20)) -> None:
    """Descriptive probe learning curves (current vs fresh, mean of rounds) for one run."""
    plt = _plot_setup()
    curves = np.load(ledger.ledger_root(results_root) / run_key / "probe_curves.npz")
    fig, axes = plt.subplots(1, len(check_indices), figsize=(3.6 * len(check_indices), 2.8), squeeze=False)
    for ax, k in zip(axes[0], check_indices):
        for name, color in (("fresh", "#2a78d6"), ("current", "#eb6834")):
            losses = curves[f"check_{k:02d}/{name}_losses"]
            ax.plot(losses.mean(0), color=color, linewidth=1.5, label=name)
        ax.set_yscale("log")
        ax.set_title(f"{run_key} check {k}", loc="left")
        ax.set_xlabel("probe step")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_analysis(out_dir: str, results_root=None, include_dev: bool = False, curves_for: Sequence[str] = (),
                 default: str = DEFAULT, scaled: Sequence[str] = SCALED) -> Dict:
    out = Path(require_absolute(out_dir, "--out"))
    out.mkdir(parents=True, exist_ok=True)
    runs, checks = ledger.load(results_root, include_dev=include_dev)
    if runs.empty:
        raise ValueError("no complete runs in the ledger")
    outputs = {}
    primary = primary_endpoint(runs, default, scaled)
    primary.to_csv(out / "primary_endpoint.csv", index=False)
    per_architecture_iqm(runs).to_csv(out / "final_loss_by_architecture.csv", index=False)
    traj = trajectories(checks)
    traj.to_csv(out / "trajectories.csv", index=False)
    f_star_table(runs).to_csv(out / "f_star_table.csv", index=False)
    checks.to_csv(out / "every_seed_checks.csv", index=False)
    plot_trajectories(traj, out / "trajectories.png")
    plot_every_seed(checks, out / "every_seed.png")
    for run_key in curves_for:
        plot_learning_curves(run_key, out / f"learning_curves_{run_key}.png", results_root)
    outputs["primary"] = primary
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--include-dev", action="store_true")
    parser.add_argument("--curves", nargs="*", default=[], help="run keys to plot probe learning curves for")
    args = parser.parse_args()
    print(run_analysis(args.out, args.results_root, args.include_dev, args.curves)["primary"].to_string(index=False))


if __name__ == "__main__":
    main()
