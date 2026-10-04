"""Experiment 2 records: one directory per forked run, one sub-directory per arm.

    <results_root>/exp12/exp2/<run_key>/fork.json | check1.json | check2.json
    <results_root>/exp12/exp2/<run_key>/arm_<arm>/checks.csv          post-fork probe checks
    <results_root>/exp12/exp2/<run_key>/arm_<arm>/eval_episodes.csv   raw per-episode post-fork evaluations
    <results_root>/exp12/exp2/<run_key>/arm_<arm>/metrics.csv         training metrics (Q values, actor
                                                                      updates, ...) from the fork on
Written atomically at every save of the arm.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from utils.atomic_io import atomic_write_text
from utils.paths import results_path

ARMS = ("control", "injected", "identity")


def run_root(run_key: str, results_root: Optional[str] = None) -> Path:
    return Path(results_path("exp12", "exp2", run_key, results_root=results_root))


def write_json(run_key: str, name: str, payload: Dict, results_root: Optional[str] = None) -> None:
    atomic_write_text(run_root(run_key, results_root) / name, json.dumps(payload, indent=2, default=float))


def write_arm(run_key: str, arm: str, plan: Dict, probe_records: List[Dict], eval_rows: List[Dict],
              metrics_rows: List[Dict], action_repeat: int, results_root: Optional[str] = None) -> Path:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    out = run_root(run_key, results_root) / f"arm_{arm}"
    fork_step, k0 = plan["fork_step"], plan["fork_check_index"]
    checks = pd.DataFrame([r for r in probe_records if r["check_index"] >= k0])
    if not checks.empty:
        checks.insert(0, "arm", arm)
        checks["steps_since_fork"] = checks.interaction_step - fork_step
    metrics = pd.DataFrame([r for r in metrics_rows if r.get("env_step", -1) >= fork_step * action_repeat])
    if not metrics.empty:
        metrics.insert(0, "arm", arm)
        metrics["steps_since_fork"] = metrics.env_step / action_repeat - fork_step
    atomic_write_text(out / "checks.csv", checks.to_csv(index=False))
    atomic_write_text(out / "eval_episodes.csv", pd.DataFrame(eval_rows).to_csv(index=False))
    atomic_write_text(out / "metrics.csv", metrics.to_csv(index=False))
    return out
