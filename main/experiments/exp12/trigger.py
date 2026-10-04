"""Operational plasticity-loss trigger (Methodology Exp 1; decisions B1-B5).

Per check: a percentile bootstrap of the IQM over the 5 paired per-round
plasticity losses L_r = P_r(fresh) - P_r(current). The check triggers when the
95% interval lies entirely above zero (lower bound > 0, strictly). f*_run is the
first triggering check at or before 95% of the budget (checks 1..19 of 20);
otherwise f*_run is null. A check with any non-finite round is invalid and
never triggers.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import trim_mean

BOOT_STREAM = 0x424F4F54  # "BOOT"
REPS = 10_000
CONFIDENCE = 0.95


def bootstrap_interval(loss_rounds, seed: int, check_index: int, reps: int = REPS,
                       confidence: float = CONFIDENCE) -> Tuple[float, float]:
    x = np.asarray(loss_rounds, dtype=np.float64)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return float("nan"), float("nan")
    rng = np.random.default_rng([seed, BOOT_STREAM, check_index])
    resamples = x[rng.integers(0, x.size, size=(reps, x.size))]
    stats = trim_mean(resamples, proportiontocut=0.25, axis=1)
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(stats, [tail, 100 - tail])
    return float(low), float(high)


def triggered(ci_low: float) -> bool:
    return bool(np.isfinite(ci_low) and ci_low > 0)


def last_eligible_check(checks: int, eligible_fraction: float = 0.95) -> int:
    """Checks k/checks with k/checks <= eligible_fraction (19 of 20)."""
    return int(np.floor(eligible_fraction * checks + 1e-9))


def f_star(records: List[Dict], checks: int) -> Optional[Dict]:
    last = last_eligible_check(checks)
    for r in sorted(records, key=lambda r: r["check_index"]):
        if 1 <= r["check_index"] <= last and r.get("triggered"):
            return {"check_index": r["check_index"], "interaction_step": r["interaction_step"]}
    return None
