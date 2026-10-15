"""Unactivated reference calculation for a proposed ten-function Method C.

This module is intentionally outside the production probe/trigger/config paths.
It defines the proposed 25%-75% empirical-quantile integral (fractional
boundary weights) and a paired percentile bootstrap for ten function losses.
It is exploratory until a complete, source-bound protocol is approved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.exp12.probe import probe_key
from experiments.exp12.trigger import BOOT_STREAM


VERSION = "method_c_v2_10targets_draft"


@dataclass(frozen=True)
class TenFunctionDecision:
    loss_rounds: tuple[float, ...]
    loss_iqm: float
    ci_low: float
    ci_high: float
    fires: bool


def empirical_iqm10(values) -> float:
    """Integral of the inverse empirical CDF over [0.25, 0.75], / 0.5.

    For ten ordered observations x1..x10 the weights are
    (0, 0, .1, .2, .2, .2, .2, .1, 0, 0). There is no interpolation between
    observations in the empirical quantile function; boundary bins contribute
    half their probability mass. The input must contain exactly ten finite
    observations and each function is one resampling unit.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (10,) or not np.all(np.isfinite(x)):
        raise ValueError("ten finite paired function observations are required")
    ordered = np.sort(x)
    return float((ordered[2] + ordered[7]) / 10 + np.sum(ordered[3:7]) / 5)


def bootstrap_iqm10(loss_rounds, seed: int, check_index: int,
                    resamples: int = 10_000, confidence: float = .95) -> TenFunctionDecision:
    """Paired-function percentile bootstrap, with the original seed recipe.

    The reporter subtracts fresh and trained scores within each function
    *before* calling this function. The returned lower bound is strictly
    compared with zero, matching the existing sign and trigger convention.
    The bootstrap percentile uses NumPy's linear (Hyndman-Fan type 7)
    interpolation. This is a proposed specification, not a calibration claim.
    """
    x = np.asarray(loss_rounds, dtype=np.float64)
    if x.shape != (10,) or not np.all(np.isfinite(x)):
        raise ValueError("ten finite paired function observations are required")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap specification")
    rng = np.random.default_rng([seed, BOOT_STREAM, check_index])
    samples = np.sort(x[rng.integers(0, 10, size=(resamples, 10))], axis=1)
    stats = (samples[:, 2] + samples[:, 7]) / 10 + np.sum(samples[:, 3:7], axis=1) / 5
    tail = (1 - confidence) / 2
    low, high = np.quantile(stats, [tail, 1 - tail], method="linear")
    return TenFunctionDecision(
        tuple(map(float, x)), empirical_iqm10(x), float(low), float(high), bool(low > 0),
    )


def target_round_key(seed: int, check_index: int, set_index: int, round_index: int):
    """Set zero reuses the official five keys; later sets never recycle them."""
    if set_index < 0 or not 0 <= round_index < 10:
        raise ValueError("invalid target set or round")
    return probe_key(seed, check_index, 10 * set_index + round_index)
