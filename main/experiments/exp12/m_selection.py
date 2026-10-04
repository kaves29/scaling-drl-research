"""Positive-control arithmetic: recovery, probe noise and the m rule (amendments (d), (q)).

The healthy reference is the fresh critic (check 0). L is measured against it, so
L_healthy = 0 and recovery(m) = (L_trigger - L_injected(m)) / L_trigger.
"""

from typing import Dict

import numpy as np

from experiments.exp12.injection import M_LABELS
from experiments.exp12.probe import iqm


def loss_rounds(record: Dict) -> np.ndarray:
    return np.array([record[k] for k in sorted((k for k in record if k.startswith("loss_r")),
                                               key=lambda k: int(k[len("loss_r"):]))], dtype=np.float64)


def recovery(l_trigger: float, l_injected: float) -> float:
    """0 = no change, 1 = fully back to the fresh critic's level."""
    return (l_trigger - l_injected) / l_trigger


def pooled_sd(series: Dict[str, np.ndarray]) -> float:
    """Pooled within-series SD: sqrt(sum_i sum_r (x_ir - mean_i)^2 / sum_i (n_i - 1))."""
    ss = sum(float(np.sum((np.asarray(x, np.float64) - np.mean(x)) ** 2)) for x in series.values())
    dof = sum(len(x) - 1 for x in series.values())
    return float(np.sqrt(ss / dof))


def select_m(recoveries: Dict[str, float], similar_within: float) -> str:
    """The smallest m (labels ordered last <= half <= all) whose recovery is within `similar_within` of the best."""
    best = max(recoveries[m] for m in M_LABELS)
    return next(m for m in M_LABELS if recoveries[m] >= best - similar_within)


def evaluate(loss: Dict[str, np.ndarray], noise_threshold: float, similar_within: float) -> Dict:
    """loss: per-round L for 'degraded' and 'injected_<m>' on one shared pool, all with the real probe
    settings; the noise is their pooled SD. A stop leaves chosen_m None and states the reason."""
    l_trigger = iqm(loss["degraded"])
    noise_sd = pooled_sd(loss)
    out = {"l_trigger": l_trigger, "l_injected": {m: iqm(loss[f"injected_{m}"]) for m in M_LABELS},
           "noise_sd": noise_sd, "noise_series": sorted(loss), "chosen_m": None, "stop": None}
    if not l_trigger > 0:
        out["stop"] = f"L at the trigger ({l_trigger:.6g}) is not above 0; recovery is undefined"
        return out
    out["recovery"] = {m: recovery(l_trigger, out["l_injected"][m]) for m in M_LABELS}
    out["noise"] = noise_sd / l_trigger  # recovery units
    if out["noise"] >= noise_threshold:
        out["stop"] = f"probe noise {out['noise']:.4f} (pooled SD / L_trigger) is >= {noise_threshold}"
        return out
    out["chosen_m"] = select_m(out["recovery"], similar_within)
    return out
