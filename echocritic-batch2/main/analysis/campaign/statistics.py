"""Seed-level estimands. Callers must partition compatible protocols first."""

import numpy as np
from scipy.stats import trim_mean


def summary(values, reps=10000, seed=0):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("requires one finite observation per independent seed")
    if type(reps) is not int or reps < 1:
        raise ValueError("reps must be a positive integer")
    funcs = dict(mean=np.mean, median=np.median, iqm=lambda a: trim_mean(a, 0.25))
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), (reps, len(x)))] if len(x) > 1 else None
    result = {"n_seeds": len(x), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None}
    for key, fn in funcs.items():
        result[key] = float(fn(x))
        result[key + "_ci95"] = (
            np.quantile([fn(d) for d in draws], [0.025, 0.975]).tolist()
            if draws is not None
            else None
        )
    return result


def paired(left, right, **kwargs):
    """Seed-keyed values; explicit intersection, unmatched seeds reported."""
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError("no paired seeds")
    return {
        "matched_seeds": common,
        "left_only": sorted(set(left) - set(right)),
        "right_only": sorted(set(right) - set(left)),
        "difference": summary([left[s] - right[s] for s in common], **kwargs),
    }


def trajectory(points, horizon, window=0.1):
    """AUC is the observed trapezoid integral; no missing endpoint extrapolation."""
    a = np.asarray(points, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError("finite step/value pairs required")
    if np.any(np.diff(a[:, 0]) <= 0) or horizon <= 0 or not 0 < window <= 1:
        raise ValueError(
            "strictly increasing steps, positive horizon and window required"
        )
    late = a[(a[:, 0] >= horizon * (1 - window)) & (a[:, 0] <= horizon), 1]
    return {
        "observed_start": float(a[0, 0]),
        "observed_end": float(a[-1, 0]),
        "complete": bool(a[0, 0] == 0 and a[-1, 0] == horizon),
        "final_window_n": len(late),
        "final_window": float(late.mean()) if len(late) else None,
        "last_observed": float(a[-1, 1]),
        "peak": float(a[:, 1].max()),
        "observed_auc": float(np.sum(np.diff(a[:, 0]) * (a[1:, 1] + a[:-1, 1]) / 2)),
    }


def independent_seed_summary(rows, value, **kwargs):
    """Explicitly partitioned rows: reject checkpoint/repetition pseudoreplication."""
    keys = [r["seed"] for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("multiple observations per training seed")
    return summary([r[value] for r in rows], **kwargs)


def longitudinal(rows, **kwargs):
    """Rows have seed, step, value; missing steps remain absent, n is per step."""
    steps = sorted({r["step"] for r in rows})
    return [
        {
            "step": step,
            **independent_seed_summary(
                [r for r in rows if r["step"] == step], "value", **kwargs
            ),
        }
        for step in steps
    ]


def architecture_environment(rows, value, **kwargs):
    groups = sorted(
        {(r["environment"], r["architecture"], r["protocol_hash"]) for r in rows}
    )
    return [
        {
            "environment": e,
            "architecture": a,
            "protocol_hash": p,
            **independent_seed_summary(
                [
                    r
                    for r in rows
                    if (r["environment"], r["architecture"], r["protocol_hash"])
                    == (e, a, p)
                ],
                value,
                **kwargs
            ),
        }
        for e, a, p in groups
    ]
