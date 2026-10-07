"""Structured Exp 1 ledger: one directory per run, so concurrent jobs never share a file.

    <results_root>/exp12/exp1/runs/<run_key>/run.csv          one row: identity, budget, role, f*_run, fork step
    <results_root>/exp12/exp1/runs/<run_key>/checks.csv       one row per check (0 = fresh-critic probe)
    <results_root>/exp12/exp1/runs/<run_key>/probe_curves.npz learning curves of every check
    <results_root>/exp12/exp1/runs/<run_key>/metrics.csv      training metrics per logging window (actor
                                                              diagnostics, Q values, returns), for the
                                                              shared-time-axis plot

Files are written atomically and rewritten at every save, so an interrupted run
shows its progress (status "running") and a finished one is "complete".
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.atomic_io import atomic_write_bytes, atomic_write_text
from utils.paths import results_path

EXPERIMENT = "exp1"
ROUNDS = 5
RUN_COLUMNS = [
    "run_key", "experiment", "run_role", "architecture", "environment", "seed", "budget_env_steps",
    "num_interaction_steps", "num_checks", "status", "initial_fresh_score_iqm", "final_loss_iqm",
    "f_star_check", "f_star_interaction_step", "f_star_fraction", "fork_interaction_step", "code_commit",
]
LEGACY_CHECK_COLUMNS = [
    "run_key", "experiment", "run_role", "architecture", "environment", "seed", "check_index",
    "interaction_step", "budget_fraction",
    *[f"score_current_r{r}" for r in range(ROUNDS)], *[f"score_fresh_r{r}" for r in range(ROUNDS)],
    *[f"loss_r{r}" for r in range(ROUNDS)],
    "score_current_iqm", "score_fresh_iqm", "loss_iqm", "ci_low", "ci_high", "triggered", "valid",
]


TWIN_CHECK_COLUMNS = [
    *[f"score_{name}_q{q}_r{r}" for q in (1, 2) for name in ("current", "fresh") for r in range(ROUNDS)],
    *[f"loss_q{q}_r{r}" for q in (1, 2) for r in range(ROUNDS)],
    *[f"loss_q{q}_iqm" for q in (1, 2)],
]
CHECK_COLUMNS = LEGACY_CHECK_COLUMNS + TWIN_CHECK_COLUMNS


class LedgerSchemaError(ValueError):
    pass


def ledger_root(results_root: Optional[str] = None) -> Path:
    return Path(results_path("exp12", EXPERIMENT, "runs", results_root=results_root))


def write_run(identity: Dict, records: List[Dict], f_star: Optional[Dict], status: str, probe_dir: Path,
              results_root: Optional[str] = None, fork_step: Optional[int] = None,
              metrics_rows: Optional[List[Dict]] = None, action_repeat: int = 1) -> Path:
    """identity: run_key, run_role, architecture, environment, seed, budget_env_steps,
    num_interaction_steps, num_checks, code_commit."""
    out = ledger_root(results_root) / identity["run_key"]
    n, checks = identity["num_interaction_steps"], identity["num_checks"]
    base = {k: identity[k] for k in ("run_key", "run_role", "architecture", "environment", "seed")}
    base["experiment"] = EXPERIMENT
    rows = [{**base, **r, "budget_fraction": r["interaction_step"] / n} for r in records]
    checks_df = pd.DataFrame(rows).reindex(columns=CHECK_COLUMNS)
    by_check = {r["check_index"]: r for r in records}
    run = {
        **base,
        "budget_env_steps": identity["budget_env_steps"],
        "num_interaction_steps": n,
        "num_checks": checks,
        "status": status,
        "initial_fresh_score_iqm": by_check.get(0, {}).get("score_fresh_iqm", np.nan),
        "final_loss_iqm": by_check.get(checks, {}).get("loss_iqm", np.nan),
        "f_star_check": f_star["check_index"] if f_star else None,
        "f_star_interaction_step": f_star["interaction_step"] if f_star else None,
        "f_star_fraction": f_star["check_index"] / checks if f_star else None,
        "fork_interaction_step": fork_step,
        "code_commit": identity.get("code_commit"),
    }
    atomic_write_text(out / "checks.csv", checks_df.to_csv(index=False))
    if metrics_rows is not None:
        metrics = pd.DataFrame(metrics_rows)
        if not metrics.empty:
            metrics.insert(0, "run_key", identity["run_key"])
            metrics["interaction_step"] = metrics.env_step / action_repeat
            metrics["budget_fraction"] = metrics.interaction_step / n
        atomic_write_text(out / "metrics.csv", metrics.to_csv(index=False))
    atomic_write_text(out / "run.csv", pd.DataFrame([run]).reindex(columns=RUN_COLUMNS).to_csv(index=False))
    curves = {}
    for path in sorted(Path(probe_dir).glob("check_*.npz")):
        with np.load(path) as d:
            for key in d.files:
                curves[f"{path.stem}/{key}"] = d[key]
    buf = io.BytesIO()
    np.savez_compressed(buf, **curves)
    atomic_write_bytes(out / "probe_curves.npz", buf.getvalue())
    return out


def load(results_root: Optional[str] = None, include_dev: bool = False,
         require_complete: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(runs, checks) for every run under the ledger root. Development runs are
    excluded unless include_dev; unfinished runs unless require_complete=False."""
    root = ledger_root(results_root)
    runs, checks = [], []
    for run_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        run = pd.read_csv(run_dir / "run.csv")
        chk = pd.read_csv(run_dir / "checks.csv")
        for df, cols, name in ((run, RUN_COLUMNS, "run.csv"), (chk, CHECK_COLUMNS, "checks.csv")):
            if list(df.columns) != cols and not (name == "checks.csv" and list(df.columns) == LEGACY_CHECK_COLUMNS):
                raise LedgerSchemaError(f"{run_dir / name}: columns {list(df.columns)} != {cols}")
        if (run.experiment != EXPERIMENT).any():
            raise LedgerSchemaError(f"{run_dir}: not an {EXPERIMENT} run")
        runs.append(run)
        checks.append(chk.reindex(columns=CHECK_COLUMNS))
    if not runs:
        return pd.DataFrame(columns=RUN_COLUMNS), pd.DataFrame(columns=CHECK_COLUMNS)
    runs, checks = pd.concat(runs, ignore_index=True), pd.concat(checks, ignore_index=True)
    keep = runs.run_role.isin(["confirmatory", "dev"] if include_dev else ["confirmatory"])
    if require_complete:
        keep &= runs.status == "complete"
    runs = runs[keep].reset_index(drop=True)
    checks = checks[checks.run_key.isin(runs.run_key)].reset_index(drop=True)
    return runs, checks


def load_metrics(run_keys, results_root: Optional[str] = None) -> pd.DataFrame:
    """metrics.csv of the given runs (training metrics per logging window), concatenated."""
    frames = [pd.read_csv(ledger_root(results_root) / k / "metrics.csv") for k in run_keys
              if (ledger_root(results_root) / k / "metrics.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
