"""Experiment 2 analysis (Methodology "Main experiment"; amendments (g), (m), (n); decisions F1-F4).

Graphs only, no scalar summary and no normalisation (normalize(env, values) defaults
to identity). Per scaled architecture:
  - PRIMARY: per-environment paired seed-level differences in raw return (injected
    minus control, each the mean over an evaluation's episodes), aligned by steps
    since fork, with every seed's line and a percentile bootstrap band over seeds;
  - actor diagnostics (I1-I4) of both arms after the fork;
  - critic plasticity loss of both arms after the fork;
  - Check 1 and Check 2 tables;
  - critic plasticity loss and actor-side diagnostics on a shared time axis (the
    whole Exp 1 run, every seed, f*_run marked), descriptive only;
  - SECONDARY: the primary graphs restricted to forks whose Check 2 passed.
Development runs and the identity (validation) arm are excluded by default. A fork
enters the paired graphs only when both arms have every post-fork evaluation.

    python -m analysis.exp2_analysis --out /abs/path/exp2_analysis --study-manifest /abs/study.json
    Add --exploratory for unvalidated progress output or custom summaries.
"""

import argparse
import json
import zlib
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from analysis.exp1_analysis import _plot_setup
from experiments.exp12 import exp2_ledger, ledger
from utils.paths import require_absolute

SCALED = ("D4W1024", "D4W1536")
ARM_COLORS = {"control": "#2a78d6", "injected": "#eb6834"}
STATISTICS = {"mean": lambda x, axis: np.mean(x, axis=axis),
              "iqm": lambda x, axis: trim_mean(x, proportiontocut=0.25, axis=axis)}
DIAGNOSTICS = ("train/policy_kl", "train/churn", "train/actor_gnorm", "train/actor_gnorm_std",
               "train/actor_action", "train/actor_saturation")
SHARED_AXIS = ("loss_iqm", "train/policy_kl", "train/actor_gnorm", "train/actor_gnorm_std",
               "train/actor_saturation", "train/actor_action", "eval/avg_return")
REPS, CONFIDENCE, SEED = 10_000, 0.95, 0


def identity(env: str, values: np.ndarray) -> np.ndarray:
    return values


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def load(results_root=None, include_dev: bool = False) -> Dict[str, pd.DataFrame]:
    """Uncertified raw/progress loader; run_analysis performs confirmatory validation."""
    runs, _ = ledger.load(results_root, include_dev=include_dev, require_complete=False)
    root = Path(exp2_ledger.run_root("x", results_root)).parent
    forks, evals, checks, metrics = [], [], [], []
    for d in sorted(p for p in root.glob("*") if p.is_dir()):
        run = runs[runs.run_key == d.name]
        plan = _read_json(d / "fork.json")
        if run.empty or plan is None:
            continue
        run = run.iloc[0]
        ident = {k: run[k] for k in ("run_key", "run_role", "architecture", "environment", "seed")}
        c1, c2 = _read_json(d / "check1_injected.json"), _read_json(d / "check2.json")
        n_evals = plan["horizon_steps"] // plan["eval_every_steps"] + 1
        row = {**ident, "fork_step": plan["fork_step"], "fork_check_index": plan["fork_check_index"],
               "check1_pass": None if c1 is None else c1["pass"],
               "check1_max_eps_units": None if c1 is None else c1.get("max_eps_units"),
               "check2_pass": None if c2 is None else c2["pass"],
               "check2_paired_difference_iqm": None if c2 is None else c2["paired_difference_iqm"],
               "check2_ci_low": None if c2 is None else c2["paired_difference_ci_low"],
               "check2_ci_high": None if c2 is None else c2["paired_difference_ci_high"]}
        for arm in ("control", "injected"):
            frames = {name: _read_csv(d / f"arm_{arm}" / f"{name}.csv")
                      for name in ("eval_episodes", "checks", "metrics")}
            e = frames["eval_episodes"]
            row[f"{arm}_evals"] = int(e.eval_index.nunique()) if not e.empty else 0
            for name, store in (("eval_episodes", evals), ("checks", checks), ("metrics", metrics)):
                if not frames[name].empty:
                    store.append(frames[name].assign(**ident, arm=arm))
        row["complete"] = row["control_evals"] == n_evals and row["injected_evals"] == n_evals
        forks.append(row)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return {"runs": runs, "forks": pd.DataFrame(forks), "evals": cat(evals), "checks": cat(checks),
            "metrics": cat(metrics)}


def paired_returns(evals: pd.DataFrame, forks: pd.DataFrame,
                   normalize: Callable[[str, np.ndarray], np.ndarray] = identity) -> pd.DataFrame:
    """One row per (fork, evaluation): mean (normalised) return of each arm and injected - control."""
    complete = forks[forks.complete.astype(bool)].run_key
    e = evals[evals.run_key.isin(complete)].copy()
    if e.empty:
        return pd.DataFrame(columns=["run_key", "architecture", "environment", "seed", "eval_index",
                                     "steps_since_fork", "control", "injected", "difference"])
    e["value"] = e["return"].astype(np.float64)
    for env, index in e.groupby("environment").groups.items():
        e.loc[index, "value"] = normalize(env, e.loc[index, "return"].to_numpy(np.float64))
    keys = ["run_key", "architecture", "environment", "seed", "eval_index", "steps_since_fork"]
    per_eval = e.groupby(keys + ["arm"]).value.mean().unstack("arm").reset_index()
    per_eval["difference"] = per_eval["injected"] - per_eval["control"]
    return per_eval


def bootstrap_band(matrix: np.ndarray, statistic: str, reps: int = REPS, confidence: float = CONFIDENCE,
                   seed: int = SEED) -> Dict[str, np.ndarray]:
    """Percentile bootstrap over seeds (rows) of `statistic` at every point (column); the same
    resampled seed sets are used at every point."""
    stat = STATISTICS[statistic]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, matrix.shape[0], size=(reps, matrix.shape[0]))
    boot = stat(matrix[idx], axis=1)
    alpha = (1 - confidence) / 2 * 100
    low, high = np.percentile(boot, [alpha, 100 - alpha], axis=0)
    return {"point": stat(matrix, axis=0), "low": low, "high": high}


def _band_seed(*parts) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode())


def paired_bands(paired: pd.DataFrame, statistic: str) -> pd.DataFrame:
    rows = []
    for (arch, env), g in paired.groupby(["architecture", "environment"]):
        m = g.pivot(index="run_key", columns="steps_since_fork", values="difference")
        band = bootstrap_band(m.to_numpy(np.float64), statistic, seed=_band_seed(arch, env))
        for i, step in enumerate(m.columns):
            rows.append({"architecture": arch, "environment": env, "steps_since_fork": step, "n_seeds": len(m),
                         "statistic": statistic, "point": band["point"][i], "ci_low": band["low"][i],
                         "ci_high": band["high"][i]})
    return pd.DataFrame(rows)


def _grid(plt, n, width=3.2, height=2.4):
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), squeeze=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return fig, list(axes.flat)[:n]


def plot_paired(paired: pd.DataFrame, bands: pd.DataFrame, arch: str, path: Path, title: str) -> None:
    plt = _plot_setup()
    envs = sorted(bands[bands.architecture == arch].environment.unique())
    if not envs:
        return
    fig, axes = _grid(plt, len(envs))
    for ax, env in zip(axes, envs):
        for _, g in paired[(paired.architecture == arch) & (paired.environment == env)].groupby("run_key"):
            ax.plot(g.steps_since_fork, g.difference, color="#898781", linewidth=0.8, alpha=0.8)
        b = bands[(bands.architecture == arch) & (bands.environment == env)]
        ax.fill_between(b.steps_since_fork, b.ci_low, b.ci_high, color=ARM_COLORS["injected"], alpha=0.2, linewidth=0)
        ax.plot(b.steps_since_fork, b.point, color=ARM_COLORS["injected"], linewidth=2)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_title(f"{env} (n={int(b.n_seeds.iloc[0])})", loc="left")
    fig.suptitle(title, x=0.01, ha="left")
    fig.supxlabel("interaction steps since fork")
    fig.supylabel("return: injected - control (raw)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_both_arms(df: pd.DataFrame, arch: str, value: str, path: Path, ylabel: str) -> None:
    """Every seed's line for both arms, one panel per environment."""
    if df.empty:
        return
    plt = _plot_setup()
    df = df[(df.architecture == arch) & df[value].notna()] if value in df else df.iloc[0:0]
    envs = sorted(df.environment.unique())
    if not envs:
        return
    fig, axes = _grid(plt, len(envs))
    for ax, env in zip(axes, envs):
        for (arm, _), g in df[df.environment == env].groupby(["arm", "run_key"]):
            ax.plot(g.steps_since_fork, g[value], color=ARM_COLORS[arm], linewidth=1, alpha=0.8)
        ax.set_title(env, loc="left")
    handles = [plt.Line2D([], [], color=c, linewidth=2) for c in ARM_COLORS.values()]
    fig.legend(handles, list(ARM_COLORS), frameon=False, loc="upper right")
    fig.supxlabel("interaction steps since fork")
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shared_time_axis(runs: pd.DataFrame, checks: pd.DataFrame, metrics: pd.DataFrame, arch: str, env: str,
                     path: Path) -> None:
    """Descriptive: critic plasticity loss and actor-side diagnostics of every seed's Exp 1 run (the
    control trajectory after a fork) on one budget-fraction axis, f*_run marked per seed."""
    plt = _plot_setup()
    sel = runs[(runs.architecture == arch) & (runs.environment == env)]
    if sel.empty:
        return
    panels = [v for v in SHARED_AXIS if v == "loss_iqm" or (v in metrics and metrics[v].notna().any())]
    fig, axes = plt.subplots(len(panels), 1, figsize=(6, 1.6 * len(panels)), sharex=True, squeeze=False)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59d0", "#d64a7a"]
    for i, (_, run) in enumerate(sel.sort_values("seed").iterrows()):
        color = palette[i % len(palette)]
        c = checks[(checks.run_key == run.run_key) & (checks.check_index > 0)]
        m = metrics[metrics.run_key == run.run_key] if not metrics.empty else metrics
        for ax, v in zip(axes[:, 0], panels):
            src = c if v == "loss_iqm" else m
            if not src.empty and v in src:
                ok = src[v].notna()
                ax.plot(src.budget_fraction[ok], src[v][ok], color=color, linewidth=1, label=f"seed {run.seed}")
            if pd.notna(run.f_star_fraction):
                ax.axvline(run.f_star_fraction, color=color, linewidth=0.8, linestyle="--")
    for ax, v in zip(axes[:, 0], panels):
        ax.set_ylabel("L (IQM)" if v == "loss_iqm" else v.split("/")[-1], fontsize=7)
    axes[0, 0].axhline(0, color="#52514e", linewidth=0.8)
    axes[0, 0].legend(frameon=False, fontsize=6, ncol=5)
    axes[0, 0].set_title(f"{arch} {env}: descriptive (dashed: f*_run)", loc="left")
    axes[-1, 0].set_xlabel("fraction of training budget")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_analysis(out_dir: str, statistic: str = "iqm", results_root=None, include_dev: bool = False,
                    normalize: Callable[[str, np.ndarray], np.ndarray] = identity, scaled=SCALED,
                    certification=None) -> Dict:
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {sorted(STATISTICS)}")
    out = Path(require_absolute(out_dir, "--out"))
    out.mkdir(parents=True, exist_ok=True)
    data = load(results_root, include_dev)
    forks = data["forks"]
    if forks.empty and certification is None:
        raise ValueError("no forks in results/exp12/exp2")
    if forks.empty:
        forks = pd.DataFrame(columns=["run_key", "architecture", "environment", "seed", "check1_pass",
                                      "check1_max_eps_units", "check2_pass", "check2_paired_difference_iqm",
                                      "check2_ci_low", "check2_ci_high", "complete"])
        data["evals"] = pd.DataFrame(columns=["run_key"])
    if certification is not None:
        census = pd.DataFrame(certification["census"])
        census.to_csv(out / "candidate_census.csv", index=False)
        population = census.groupby(["architecture", "environment"])
        summary = population.status.agg(candidates="size", eligible=lambda x: (x == "eligible_trigger").sum(),
                                        valid_no_trigger=lambda x: (x == "valid_no_trigger").sum()).reset_index()
        summary["zero_eligible"] = summary.eligible == 0
        summary.to_csv(out / "eligibility_by_environment.csv", index=False)
    forks.to_csv(out / "forks.csv", index=False)
    forks[["run_key", "architecture", "environment", "seed", "check1_pass", "check1_max_eps_units"]].to_csv(
        out / "check1_table.csv", index=False)
    forks[["run_key", "architecture", "environment", "seed", "check2_paired_difference_iqm", "check2_ci_low",
           "check2_ci_high", "check2_pass"]].to_csv(out / "check2_table.csv", index=False)
    paired = paired_returns(data["evals"], forks, normalize)
    success = forks[forks.check2_pass == True].run_key  # noqa: E712 (None = no Check 2 record)
    outputs = {"forks": forks, "paired": paired}
    for label, sub in (("", paired), ("_check2_success_only", paired[paired.run_key.isin(success)])):
        bands = paired_bands(sub, statistic) if not sub.empty else pd.DataFrame(
            columns=["architecture", "environment", "steps_since_fork", "n_seeds", "statistic", "point",
                     "ci_low", "ci_high"])
        sub.to_csv(out / f"paired_returns{label}.csv", index=False)
        bands.to_csv(out / f"paired_bands{label}.csv", index=False)
        outputs[f"bands{label}"] = bands
        kind = "SECONDARY (Check 2 passed only)" if label else "PRIMARY"
        for arch in scaled:
            plot_paired(sub, bands, arch, out / f"paired_returns_{arch}{label}.png",
                        f"{arch}: {kind}, {statistic} over seeds, {int(CONFIDENCE * 100)}% percentile bootstrap band")
    post = data["checks"]
    if not post.empty:
        post.to_csv(out / "plasticity_post_fork.csv", index=False)
    if not data["metrics"].empty:
        data["metrics"].to_csv(out / "diagnostics_post_fork.csv", index=False)
    for arch in scaled:
        plot_both_arms(post, arch, "loss_iqm", out / f"plasticity_post_fork_{arch}.png", "plasticity loss L (IQM)")
        for q in (1, 2):
            value = f"loss_q{q}_iqm"
            if value in post and post[value].notna().any():
                plot_both_arms(post, arch, value, out / f"plasticity_post_fork_{arch}_q{q}.png",
                               f"Q{q} plasticity loss L (IQM)")
        for v in DIAGNOSTICS:
            plot_both_arms(data["metrics"], arch, v, out / f"diagnostics_{v.split('/')[-1]}_{arch}.png", v)
    _, exp1_checks = ledger.load(results_root, include_dev=include_dev, require_complete=False)
    scaled_runs = data["runs"][data["runs"].architecture.isin(scaled)]
    exp1_metrics = ledger.load_metrics(scaled_runs.run_key, results_root)
    for (arch, env), _ in scaled_runs.groupby(["architecture", "environment"]):
        shared_time_axis(scaled_runs, exp1_checks, exp1_metrics, arch, env,
                         out / f"shared_time_axis_{arch}_{env}.png")
    return outputs


def run_analysis(out_dir: str, statistic: str = "iqm", results_root=None, include_dev: bool = False,
                 normalize: Callable[[str, np.ndarray], np.ndarray] = identity, scaled=SCALED,
                 *, study_manifest=None, exploratory: bool = False) -> Dict:
    from analysis.exp12_validation import ReportingDecisionRequired, publish

    require_absolute(out_dir, "--out")
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {sorted(STATISTICS)}")
    if not exploratory and (include_dev or statistic != "iqm" or normalize is not identity or tuple(scaled) != SCALED):
        raise ValueError("custom analysis requires the explicitly exploratory path")

    def write(out, certification):
        if not exploratory:
            eligible = pd.DataFrame(certification["census"])
            eligible = eligible[eligible.status == "eligible_trigger"]
            counts = eligible.groupby(["architecture", "environment"]).size()
            secondary = eligible[eligible.get("check2_pass", pd.Series(False, index=eligible.index)) == True]
            secondary_counts = secondary.groupby(["architecture", "environment"]).size()
            unresolved = list(counts[counts == 1].index) + list(secondary_counts[secondary_counts == 1].index)
            if unresolved:
                raise ReportingDecisionRequired([f"one-seed uncertainty decision required (primary/secondary): {unresolved}"], certification["census"])
        return _write_analysis(str(out), statistic, results_root, include_dev, normalize, scaled,
                               None if exploratory else certification)

    return publish(out_dir, write, results_root, "exp2", study_manifest, exploratory)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--statistic", default="iqm", choices=sorted(STATISTICS),
                        help="statistic over seeds for the bands (IQM, amendment (r))")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--include-dev", action="store_true")
    parser.add_argument("--study-manifest", default=None)
    parser.add_argument("--exploratory", action="store_true", help="unvalidated progress output; never confirmatory")
    args = parser.parse_args()
    outputs = run_analysis(args.out, args.statistic, args.results_root, args.include_dev,
                           study_manifest=args.study_manifest, exploratory=args.exploratory)
    print(outputs["forks"].to_string(index=False))


if __name__ == "__main__":
    main()
