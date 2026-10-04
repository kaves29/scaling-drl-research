"""Operational plasticity-loss trigger (Methodology Exp 1; decisions B1-B5).

Per check: a percentile bootstrap of the IQM over the paired per-round
plasticity losses L_r = P_r(fresh) - P_r(current). A check fires when its
interval lies entirely above `null_threshold` (lower bound > threshold,
strictly; threshold 0 is the Methodology's "distinguishable from zero").
f*_run is the check completing the first run of `consecutive_checks` firing
checks, provided it is at or before `eligible_fraction` of the budget;
otherwise f*_run is null. A check with any non-finite round never fires.
Every setting is a config value (configs/base_exp12.yaml `trigger:`).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import trim_mean

BOOT_STREAM = 0x424F4F54  # "BOOT"


@dataclass(frozen=True)
class TriggerConfig:
    resamples: int
    confidence: float
    consecutive_checks: int
    null_threshold: float
    eligible_fraction: float


def trigger_config(cfg) -> TriggerConfig:
    t = cfg.trigger
    out = TriggerConfig(
        resamples=int(t.resamples), confidence=float(t.confidence), consecutive_checks=int(t.consecutive_checks),
        null_threshold=float(t.null_threshold), eligible_fraction=float(t.eligible_fraction),
    )
    if out.consecutive_checks < 1 or not 0 < out.confidence < 1 or not 0 < out.eligible_fraction <= 1:
        raise ValueError(f"invalid trigger config {out}")
    return out


def bootstrap_interval(loss_rounds, seed: int, check_index: int, reps: int, confidence: float) -> Tuple[float, float]:
    x = np.asarray(loss_rounds, dtype=np.float64)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return float("nan"), float("nan")
    rng = np.random.default_rng([seed, BOOT_STREAM, check_index])
    resamples = x[rng.integers(0, x.size, size=(reps, x.size))]
    stats = trim_mean(resamples, proportiontocut=0.25, axis=1)
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(stats, [tail, 100 - tail])
    return float(low), float(high)


def triggered(ci_low: float, null_threshold: float) -> bool:
    return bool(np.isfinite(ci_low) and ci_low > null_threshold)


def last_eligible_check(checks: int, eligible_fraction: float) -> int:
    """Largest k with k / checks <= eligible_fraction (19 of 20 at 0.95)."""
    return int(np.floor(eligible_fraction * checks + 1e-9))


def f_star(records: List[Dict], checks: int, consecutive_checks: int, eligible_fraction: float) -> Optional[Dict]:
    last = last_eligible_check(checks, eligible_fraction)
    run, previous = 0, None
    for r in sorted((r for r in records if r["check_index"] >= 1), key=lambda r: r["check_index"]):
        contiguous = previous is not None and r["check_index"] == previous + 1
        run = (run + 1 if contiguous else 1) if r.get("triggered") else 0
        previous = r["check_index"]
        if run >= consecutive_checks:
            if r["check_index"] > last:
                return None
            return {"check_index": r["check_index"], "interaction_step": r["interaction_step"]}
    return None
