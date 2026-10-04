"""Positive-control arithmetic: healthy reference, recovery, probe noise and the m rule (amendments (d), (f))."""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from experiments.exp12.injection import M_LABELS
from experiments.exp12.probe import iqm

HEALTHY_REFERENCES = ("last_pre_trigger", "iqm_pre_trigger")
NOISE_STATISTICS = ("range", "std")


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


def spread(values, statistic: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    if statistic == "range":
        return float(values.max() - values.min())
    if statistic == "std":
        return float(np.std(values, ddof=1))
    raise ValueError(f"noise statistic must be one of {NOISE_STATISTICS}")


def select_m(recoveries: Dict[str, float], similar_within: float) -> str:
    """The smallest m (labels ordered last <= half <= all) whose recovery is within `similar_within` of the best."""
    best = max(recoveries[m] for m in M_LABELS)
    return next(m for m in M_LABELS if recoveries[m] >= best - similar_within)


def evaluate(loss: Dict[str, np.ndarray], l_healthy: float, noise_statistic: str, noise_threshold: float,
             similar_within: float) -> Dict:
    """loss: per-round L for 'degraded' and 'injected_<m>' on one shared pool. A stop per (d)
    leaves chosen_m None and states the reason; every number computed so far is kept."""
    l_trigger = iqm(loss["degraded"])
    denominator = l_trigger - l_healthy
    out = {"l_trigger": l_trigger, "l_healthy": l_healthy, "denominator": denominator,
           "l_injected": {m: iqm(loss[f"injected_{m}"]) for m in M_LABELS}, "chosen_m": None, "stop": None}
    if not denominator > 0:
        out["stop"] = (f"L at the trigger ({l_trigger:.6g}) is not above the healthy reference ({l_healthy:.6g}); "
                       "recovery is undefined")
        return out
    out["recovery"] = {m: recovery(l_trigger, out["l_injected"][m], l_healthy) for m in M_LABELS}
    raw = {name: spread(v, noise_statistic) for name, v in loss.items()}
    out["noise_raw"] = raw
    out["noise_recovery_units"] = {name: v / denominator for name, v in raw.items()}
    out["noise"] = max(out["noise_recovery_units"].values())
    if out["noise"] > noise_threshold:
        out["stop"] = (f"probe noise {out['noise']:.4f} (recovery units, {noise_statistic} over rounds) "
                       f"exceeds {noise_threshold}")
        return out
    out["chosen_m"] = select_m(out["recovery"], similar_within)
    return out
