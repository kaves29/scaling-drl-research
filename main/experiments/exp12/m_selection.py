"""Positive-control arithmetic: healthy reference, recovery, probe noise and the m rule (amendments (d), (f))."""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from experiments.exp12.injection import M_LABELS
from experiments.exp12.probe import iqm

HEALTHY_REFERENCES = ("iqm_pre_trigger", "last_pre_trigger")  # primary (decision 3, Phase 4), sensitivity


class Stop(Exception):
    """The Methodology says to stop and consult the project lead."""


def loss_rounds(record: Dict) -> np.ndarray:
    return np.array([record[k] for k in sorted((k for k in record if k.startswith("loss_r")),
                                               key=lambda k: int(k[len("loss_r"):]))], dtype=np.float64)


def healthy_reference(records: Sequence[Dict], trigger_check: int, definition: str) -> Tuple[float, List[int]]:
    """L of the same critic before the trigger: its last pre-trigger check, or the IQM of the
    per-round L pooled over all pre-trigger checks."""
    pre = sorted((r for r in records if 1 <= r["check_index"] < trigger_check), key=lambda r: r["check_index"])
    if not pre:
        raise Stop(f"the run triggered at check {trigger_check}; there is no pre-trigger check to serve as "
                   "the healthy reference")
    if definition == "last_pre_trigger":
        return float(pre[-1]["loss_iqm"]), [pre[-1]["check_index"]]
    if definition == "iqm_pre_trigger":
        return iqm(np.concatenate([loss_rounds(r) for r in pre])), [r["check_index"] for r in pre]
    raise ValueError(f"healthy reference must be one of {HEALTHY_REFERENCES}")


def recovery(l_trigger: float, l_injected: float, l_healthy: float) -> float:
    """(d): 0 = no change, 1 = fully back to the healthy reference."""
    return (l_trigger - l_injected) / (l_trigger - l_healthy)


def pooled_sd(series: Dict[str, np.ndarray]) -> float:
    """Pooled within-series SD: sqrt(sum_i sum_r (x_ir - mean_i)^2 / sum_i (n_i - 1))."""
    ss = sum(float(np.sum((np.asarray(x, np.float64) - np.mean(x)) ** 2)) for x in series.values())
    dof = sum(len(x) - 1 for x in series.values())
    return float(np.sqrt(ss / dof))


def select_m(recoveries: Dict[str, float], similar_within: float) -> str:
    """The smallest m (labels ordered last <= half <= all) whose recovery is within `similar_within` of the best."""
    best = max(recoveries[m] for m in M_LABELS)
    return next(m for m in M_LABELS if recoveries[m] >= best - similar_within)


def evaluate(loss: Dict[str, np.ndarray], l_healthy: float, noise_series: Dict[str, np.ndarray],
             noise_threshold: float, similar_within: float) -> Dict:
    """loss: per-round L for 'degraded' and 'injected_<m>' on one shared pool. noise_series: every
    per-round L series the script probed, pooled for the noise SD. A stop per (d) and decisions 3-4
    leaves chosen_m None and states the reason; every number computed so far is kept."""
    l_trigger = iqm(loss["degraded"])
    denominator = l_trigger - l_healthy
    noise_sd = pooled_sd(noise_series)
    out = {"l_trigger": l_trigger, "l_healthy": l_healthy, "denominator": denominator,
           "l_injected": {m: iqm(loss[f"injected_{m}"]) for m in M_LABELS},
           "noise_sd": noise_sd, "noise_series": sorted(noise_series), "chosen_m": None, "stop": None}
    if not denominator > 0:
        out["stop"] = (f"L at the trigger ({l_trigger:.6g}) is not above the healthy reference ({l_healthy:.6g}); "
                       "recovery is undefined")
        return out
    out["recovery"] = {m: recovery(l_trigger, out["l_injected"][m], l_healthy) for m in M_LABELS}
    out["noise"] = noise_sd / denominator  # recovery units
    if denominator < noise_sd:
        out["stop"] = (f"L_trigger - L_healthy ({denominator:.6g}) is below the probe noise "
                       f"(pooled SD {noise_sd:.6g})")
        return out
    if out["noise"] >= noise_threshold:
        out["stop"] = f"probe noise {out['noise']:.4f} (pooled SD in recovery units) is >= {noise_threshold}"
        return out
    out["chosen_m"] = select_m(out["recovery"], similar_within)
    return out
