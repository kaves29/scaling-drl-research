"""ACF/CCF calibration of N (sustain window) and W (propagation/lag window)
from the default-SimBa baseline data, per environment - see
research-methodology.md's Onset Definitions for the full procedure.
Companion to analysis/baseline_calibration.py (percentile threshold) - that
module's onset/propagation percentile threshold is intentionally NOT
expanded by the 2026-09-08 shared-pool change (see
analysis/baseline_calibration_pool.py): Angle 1's 150 already-completed runs'
onset-detection results must never be silently re-thresholded (see
research-methodology.md's "Things Claude Must Never Change Silently"), so
baseline_calibration.py keeps its original, unchanged 5-seed-only input.
Window calibration is not a "must never change silently" threshold in that
same sense (N/W are windowing parameters, not the pathology-onset
determination itself), and unlike a pairwise null distribution, N/W's
per-seed-averaged design (see calibrate_window_parameters's docstring)
extends cleanly to more seeds without changing what each seed's own
computation means - hence it accepts the expanded 10-seed pool.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from analysis.baseline_calibration import compute_baseline_source_fingerprint
from analysis.metrics_store import RunIdentity, load_metrics
from utils.atomic_io import atomic_write_text

# Both values are valid: 5 is Angle 1's original locked baseline seed count
# (existing callers/configs keep working unchanged); 10 is the expanded
# shared calibration pool (see analysis/baseline_calibration_pool.py),
# added 2026-09-08. Anything else is very likely a caller bug (e.g. a
# mismatched seed list), not a deliberate choice, so it still fails loudly.
VALID_BASELINE_SEED_COUNTS = (5, 10)
WINDOW_CALIBRATION_ROOT = "results/baselines"

# Practical conventions, not statistically derived - see research-methodology.md.
BURN_IN_FRACTION = 0.25
ACF_DECAY_THRESHOLD = 1.0 / np.e
CCF_LAG_FRACTION_CAP = 0.20


@dataclass
class WindowCalibrationResult:
    architecture: str
    environment: str
    n_sustain_window_points: int
    w_propagation_window_steps: int
    logging_per_interaction_step: int
    per_seed_decorrelation_points: List[float] = field(default_factory=list)
    acf_capped_seeds: List[str] = field(default_factory=list)
    per_seed_lag_points_used_for_w: List[float] = field(default_factory=list)
    excluded_seeds_for_w: List[str] = field(default_factory=list)
    segment_length_points: int = 0
    max_abs_ccf_lag_points: int = 0


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation at lags 0..max_lag (inclusive); r[0] == 1
    (or the array is all-ones if the segment has zero variance)."""
    x = x - x.mean()
    denom = np.sum(x**2)
    r = np.empty(max_lag + 1)
    r[0] = 1.0
    if denom == 0:
        r[:] = 1.0
        return r
    for k in range(1, max_lag + 1):
        r[k] = np.sum(x[: len(x) - k] * x[k:]) / denom
    return r


def _decorrelation_time(x: np.ndarray, run_key: str) -> "tuple[float, bool]":
    """Returns (decorrelation_lag_in_points, was_capped)."""
    max_lag = len(x) - 1
    if max_lag < 1:
        raise ValueError(
            f"Segment for {run_key} has fewer than 2 points after the "
            f"{BURN_IN_FRACTION:.0%} burn-in trim; cannot compute an ACF."
        )
    r = _acf(x, max_lag)
    below = np.where(r < ACF_DECAY_THRESHOLD)[0]
    if len(below) == 0:
        warnings.warn(
            f"ACF for {run_key} never decayed below 1/e within the available "
            f"post-burn-in segment ({max_lag + 1} points); capping the "
            f"decorrelation-time estimate at the full segment length rather "
            f"than treating this as a rare-pathology signal (see "
            f"research-methodology.md's Onset Definitions: this can arise "
            f"from ordinary residual non-stationarity given the burn-in "
            f"fraction is a practical, not statistically derived, cutoff)."
        )
        return float(max_lag), True
    return float(below[0]), False


def _ccf(x: np.ndarray, y: np.ndarray, max_abs_lag: int) -> np.ndarray:
    """Normalized cross-correlation at lags -max_abs_lag..+max_abs_lag,
    indexed 0..2*max_abs_lag (index i => lag = i - max_abs_lag). Positive lag
    k means x leads y by k points: corr(x[t], y[t+k])."""
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    lags = list(range(-max_abs_lag, max_abs_lag + 1))
    result = np.zeros(len(lags))
    if denom == 0:
        return result
    for i, k in enumerate(lags):
        if k >= 0:
            a, b = (x[: len(x) - k], y[k:]) if k > 0 else (x, y)
        else:
            kk = -k
            a, b = x[kk:], y[: len(y) - kk]
        result[i] = np.sum(a * b) / denom
    return result


def calibrate_window_parameters(
    baseline_identities: List[RunIdentity],
    logging_per_interaction_step: int,
    metrics_root: str = "results/metrics",
    burn_in_fraction: float = BURN_IN_FRACTION,
    ccf_lag_fraction_cap: float = CCF_LAG_FRACTION_CAP,
) -> WindowCalibrationResult:
    """N via ACF decorrelation time (1/e rule) of td_error_variance; W via
    CCF peak lag (not ACF) between td_error_variance and actor_grad_cosine -
    see research-methodology.md's Onset Definitions. Per seed, then averaged
    across all seeds (5 for Angle 1's original baseline, 10 for the expanded
    shared calibration pool - see VALID_BASELINE_SEED_COUNTS)."""
    if len(baseline_identities) not in VALID_BASELINE_SEED_COUNTS:
        raise ValueError(
            f"Window calibration requires exactly one of "
            f"{VALID_BASELINE_SEED_COUNTS} default-SimBa baseline seeds; "
            f"got {len(baseline_identities)}."
        )

    architectures = {i.architecture for i in baseline_identities}
    environments = {i.environment for i in baseline_identities}
    if len(architectures) != 1 or len(environments) != 1:
        raise ValueError(
            "All baseline_identities must share the same architecture and "
            f"environment; got architectures={architectures}, environments={environments}. "
            "N/W are calibrated strictly per-environment, never pooled."
        )

    series_td, series_actor, run_keys = [], [], []
    for ident in baseline_identities:
        df = load_metrics(ident, root=metrics_root)
        if df.empty or "td_error_variance" not in df.columns or "actor_grad_cosine" not in df.columns:
            raise ValueError(
                f"No persisted td_error_variance/actor_grad_cosine metrics for "
                f"baseline run '{ident.run_key}'. Did training run with "
                f"critic_degradation=true for this baseline seed?"
            )
        df = df.set_index("interaction_step").sort_index()
        series_td.append(df["td_error_variance"].rename(ident.run_key))
        series_actor.append(df["actor_grad_cosine"].rename(ident.run_key))
        run_keys.append(ident.run_key)

    aligned_td = pd.concat(series_td, axis=1, join="inner").sort_index()
    aligned_actor = pd.concat(series_actor, axis=1, join="inner").sort_index()
    if not aligned_td.index.equals(aligned_actor.index):
        raise ValueError(
            "td_error_variance and actor_grad_cosine baseline series do not "
            "share the same aligned interaction_step index - both are logged "
            "together every logging_per_interaction_step, so this should be "
            "impossible; refusing to guess an alignment."
        )

    n_points_total = len(aligned_td)
    burn_in_start = int(np.floor(n_points_total * burn_in_fraction))
    segment_len = n_points_total - burn_in_start
    if segment_len < 4:
        raise ValueError(
            f"Only {segment_len} recorded points remain after the "
            f"{burn_in_fraction:.0%} burn-in trim (from {n_points_total} "
            f"aligned points across all {len(run_keys)} baseline seeds) - too "
            f"few to compute a meaningful ACF/CCF."
        )

    max_abs_lag = max(1, int(np.ceil(segment_len * ccf_lag_fraction_cap)))

    per_seed_decorrelation, acf_capped_seeds = [], []
    per_seed_lag_for_w, excluded_seeds_for_w = [], []

    for run_key in run_keys:
        td_segment = aligned_td[run_key].to_numpy()[burn_in_start:]
        actor_segment = aligned_actor[run_key].to_numpy()[burn_in_start:]

        dc_time, was_capped = _decorrelation_time(td_segment, run_key)
        per_seed_decorrelation.append(dc_time)
        if was_capped:
            acf_capped_seeds.append(run_key)

        # Zero variance makes CCF undefined (0/0); _ccf returns an all-zero
        # array, and argmax would silently pick the most-negative-lag index -
        # explicit exclusion avoids that false "negative lag" reading.
        if np.std(td_segment) == 0 or np.std(actor_segment) == 0:
            excluded_seeds_for_w.append(run_key)
            warnings.warn(
                f"Seed {run_key}: td_error_variance or actor_grad_cosine is "
                f"exactly constant over the analyzed segment - cross-"
                f"correlation is undefined; excluded from this environment's "
                f"W average."
            )
            continue

        ccf_vals = _ccf(td_segment, actor_segment, max_abs_lag)
        lags = np.arange(-max_abs_lag, max_abs_lag + 1)
        peak_lag = int(lags[np.argmax(np.abs(ccf_vals))])
        if peak_lag < 0:
            excluded_seeds_for_w.append(run_key)
            warnings.warn(
                f"Seed {run_key}: CCF peak between td_error_variance and "
                f"actor_grad_cosine is at NEGATIVE lag {peak_lag} (actor "
                f"metric appears to lead the critic metric) - excluded from "
                f"this environment's W average as a likely logging/indexing "
                f"misalignment, not treated as a genuine causal-direction "
                f"reversal (critic-leads-actor holds by construction of how "
                f"the actor gradient is computed from the critic's Q-output "
                f"at the same update step). See research-methodology.md's "
                f"Onset Definitions."
            )
        else:
            per_seed_lag_for_w.append(float(peak_lag))

    if not per_seed_lag_for_w:
        raise ValueError(
            f"All {len(run_keys)} baseline seeds for environment "
            f"'{baseline_identities[0].environment}' were excluded from W's "
            f"calibration (negative-lag CCF peak or zero-variance segment; "
            f"excluded: {excluded_seeds_for_w}) - cannot calibrate W "
            f"from zero valid seeds. This strongly suggests a systemic "
            f"logging/indexing misalignment between td_error_variance and "
            f"actor_grad_cosine for this environment's baseline data, not an "
            f"isolated per-seed anomaly - requires manual review."
        )

    n_sustain_window_points = max(1, round(float(np.mean(per_seed_decorrelation))))
    w_lag_points = float(np.mean(per_seed_lag_for_w))
    w_propagation_window_steps = max(1, round(w_lag_points * logging_per_interaction_step))

    return WindowCalibrationResult(
        architecture=baseline_identities[0].architecture,
        environment=baseline_identities[0].environment,
        n_sustain_window_points=n_sustain_window_points,
        w_propagation_window_steps=w_propagation_window_steps,
        logging_per_interaction_step=logging_per_interaction_step,
        per_seed_decorrelation_points=per_seed_decorrelation,
        acf_capped_seeds=acf_capped_seeds,
        per_seed_lag_points_used_for_w=per_seed_lag_for_w,
        excluded_seeds_for_w=excluded_seeds_for_w,
        segment_length_points=segment_len,
        max_abs_ccf_lag_points=max_abs_lag,
    )


def _window_cache_path(architecture: str, environment: str, root: str) -> Path:
    return Path(root) / architecture / environment / "window_params.json"


def save_window_calibration(result: WindowCalibrationResult, source_fingerprint: str, root: str = WINDOW_CALIBRATION_ROOT) -> Path:
    path = _window_cache_path(result.architecture, result.environment, root)
    payload = {
        "architecture": result.architecture,
        "environment": result.environment,
        "n_sustain_window_points": result.n_sustain_window_points,
        "w_propagation_window_steps": result.w_propagation_window_steps,
        "logging_per_interaction_step": result.logging_per_interaction_step,
        "per_seed_decorrelation_points": result.per_seed_decorrelation_points,
        "acf_capped_seeds": result.acf_capped_seeds,
        "per_seed_lag_points_used_for_w": result.per_seed_lag_points_used_for_w,
        "excluded_seeds_for_w": result.excluded_seeds_for_w,
        "segment_length_points": result.segment_length_points,
        "max_abs_ccf_lag_points": result.max_abs_ccf_lag_points,
        "source_fingerprint": source_fingerprint,
    }
    atomic_write_text(path, json.dumps(payload, indent=2))
    return path


def load_window_calibration(architecture: str, environment: str, root: str = WINDOW_CALIBRATION_ROOT) -> Optional[WindowCalibrationResult]:
    path = _window_cache_path(architecture, environment, root)
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    return WindowCalibrationResult(
        architecture=payload["architecture"],
        environment=payload["environment"],
        n_sustain_window_points=payload["n_sustain_window_points"],
        w_propagation_window_steps=payload["w_propagation_window_steps"],
        logging_per_interaction_step=payload["logging_per_interaction_step"],
        per_seed_decorrelation_points=payload.get("per_seed_decorrelation_points", []),
        acf_capped_seeds=payload.get("acf_capped_seeds", []),
        per_seed_lag_points_used_for_w=payload.get("per_seed_lag_points_used_for_w", []),
        excluded_seeds_for_w=payload.get("excluded_seeds_for_w", []),
        segment_length_points=payload.get("segment_length_points", 0),
        max_abs_ccf_lag_points=payload.get("max_abs_ccf_lag_points", 0),
    )


def _load_cached_fingerprint(architecture: str, environment: str, root: str) -> Optional[str]:
    path = _window_cache_path(architecture, environment, root)
    meta_path = path.with_suffix(".fingerprint.json")
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f).get("source_fingerprint")
    except (json.JSONDecodeError, OSError):
        return None


def load_or_calibrate_window_parameters(
    baseline_identities: List[RunIdentity],
    logging_per_interaction_step: int,
    metrics_root: str = "results/metrics",
    window_root: str = WINDOW_CALIBRATION_ROOT,
    force_recompute: bool = False,
    burn_in_fraction: float = BURN_IN_FRACTION,
    ccf_lag_fraction_cap: float = CCF_LAG_FRACTION_CAP,
) -> WindowCalibrationResult:
    """Mirrors baseline_calibration.load_or_calibrate_baseline's cache/
    fingerprint pattern."""
    current_fingerprint = compute_baseline_source_fingerprint(baseline_identities, metrics_root)

    if not force_recompute:
        cached = load_window_calibration(
            baseline_identities[0].architecture, baseline_identities[0].environment, root=window_root,
        )
        if cached is not None:
            cached_fingerprint = _load_cached_fingerprint(
                baseline_identities[0].architecture, baseline_identities[0].environment, root=window_root,
            )
            if cached_fingerprint == current_fingerprint:
                return cached
            warnings.warn(
                f"Cached window calibration (N/W) for architecture="
                f"'{baseline_identities[0].architecture}', environment="
                f"'{baseline_identities[0].environment}' is stale (underlying "
                f"baseline seed metrics changed since it was computed); "
                f"recalibrating automatically."
            )

    result = calibrate_window_parameters(
        baseline_identities, logging_per_interaction_step, metrics_root=metrics_root,
        burn_in_fraction=burn_in_fraction, ccf_lag_fraction_cap=ccf_lag_fraction_cap,
    )
    save_window_calibration(result, current_fingerprint, root=window_root)
    fp_path = _window_cache_path(result.architecture, result.environment, window_root).with_suffix(".fingerprint.json")
    atomic_write_text(fp_path, json.dumps({"source_fingerprint": current_fingerprint}))
    return result
