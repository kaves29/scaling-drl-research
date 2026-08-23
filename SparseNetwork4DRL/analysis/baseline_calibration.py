"""Baseline calibration for onset detection.

Builds the stepwise 95th-percentile threshold curves — for TD-error variance
and actor-gradient cosine — from the 5-seed default-SimBa baseline runs, at
"equivalent interaction steps" (see module docstring in onset_detection.py for
what "equivalent" means given interval-based logging).

Deliberately has no knowledge of what a threshold *means* scientifically
(that's analysis/onset_detection.py) — this module only aligns and summarizes
baseline distributions.
"""

import hashlib
import io
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from analysis.metrics_store import RunIdentity, load_metrics, metrics_path
from utils.atomic_io import atomic_write_text

BASELINE_ROOT = "results/baselines"

# The scientific protocol requires exactly 5 default-SimBa baseline seeds -
# not "at least 2", not a configurable count. Enforced in calibrate_baseline().
REQUIRED_BASELINE_SEED_COUNT = 5


@dataclass
class BaselineThresholds:
    architecture: str
    environment: str
    metric: str
    percentile: int
    num_seeds: int
    steps: np.ndarray
    thresholds: np.ndarray

    def at(self, interaction_step: int) -> Optional[float]:
        """Threshold at an exact recorded interaction_step, or None if the
        baseline has no coverage at that step (steps must match exactly —
        see module docstring for the alignment assumption this implies)."""
        idx = np.searchsorted(self.steps, interaction_step)
        if idx < len(self.steps) and self.steps[idx] == interaction_step:
            return float(self.thresholds[idx])
        return None


def _cache_path(architecture: str, environment: str, metric: str, percentile: int, root: str) -> Path:
    return Path(root) / architecture / environment / f"{metric}_p{percentile}.csv"


def _infer_logging_interval(steps: np.ndarray) -> Optional[int]:
    """Infers the recording cadence from a sorted array of interaction_step
    values, as the most common consecutive difference (mode), so a single
    dropped/extra point doesn't throw off the inferred interval. Returns
    None if fewer than 2 points are available (cadence can't be inferred)."""
    if len(steps) < 2:
        return None
    diffs = np.diff(np.sort(steps))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    values, counts = np.unique(diffs, return_counts=True)
    return int(values[np.argmax(counts)])


def calibrate_baseline(
    baseline_identities: List[RunIdentity],
    metric_column: str,
    percentile: int = 95,
    metrics_root: str = "results/metrics",
    expected_logging_interval: Optional[int] = None,
) -> BaselineThresholds:
    """Computes stepwise percentile thresholds from the 5 baseline seed runs.

    Alignment assumption: baseline runs and the run(s) being evaluated all use
    the same `logging_per_interaction_step` cadence, so "equivalent
    interaction step" reduces to an exact join on `interaction_step`. Steps
    recorded by only a subset of the baseline seeds (e.g. a run that stopped
    early) are dropped from the threshold curve rather than guessed at, and
    this is what determines the walkable range of `BaselineThresholds.at(...)`.

    If `expected_logging_interval` is given (normally the analyzed run's own
    `cfg.logging_per_interaction_step`), each baseline seed's *actual*
    recorded cadence is checked against it and calibration fails loudly on a
    mismatch, rather than silently degrading into "no overlap" further down
    the pipeline with no indication of the real cause.
    """
    if len(baseline_identities) != REQUIRED_BASELINE_SEED_COUNT:
        raise ValueError(
            f"Baseline calibration requires exactly {REQUIRED_BASELINE_SEED_COUNT} "
            f"default-SimBa seeds (the scientific protocol, not a configurable "
            f"count); got {len(baseline_identities)}: "
            f"{[i.run_key for i in baseline_identities]}. Check "
            f"onset_detection.baseline_seeds in your config."
        )

    architectures = {ident.architecture for ident in baseline_identities}
    environments = {ident.environment for ident in baseline_identities}
    if len(architectures) != 1 or len(environments) != 1:
        raise ValueError(
            "All baseline_identities must share the same architecture and "
            f"environment; got architectures={architectures}, environments={environments}"
        )

    series = []
    for ident in baseline_identities:
        df = load_metrics(ident, root=metrics_root)
        if df.empty or metric_column not in df.columns:
            raise ValueError(
                f"No persisted '{metric_column}' metrics found for baseline run "
                f"'{ident.run_key}'. Did training run with critic_degradation=true "
                f"(or pathology_prop=true) for this baseline seed?"
            )

        if expected_logging_interval is not None:
            actual_interval = _infer_logging_interval(df["interaction_step"].to_numpy())
            if actual_interval is not None and actual_interval != expected_logging_interval:
                raise ValueError(
                    "Baseline logging interval and analyzed-run logging interval "
                    f"do not match: baseline seed '{ident.run_key}' was recorded "
                    f"every {actual_interval} interaction steps, but the run being "
                    f"analyzed uses logging_per_interaction_step="
                    f"{expected_logging_interval}. Onset thresholds computed from "
                    f"a mismatched grid are not meaningful at 'equivalent "
                    f"interaction steps'. Re-run this baseline seed with a matching "
                    f"logging_per_interaction_step, or explicitly resample if that "
                    f"is ever intentionally supported."
                )

        series.append(df.set_index("interaction_step")[metric_column].rename(ident.run_key))

    aligned = pd.concat(series, axis=1, join="inner").sort_index()
    if aligned.empty:
        raise ValueError(
            "Baseline seed runs share no common interaction_step values; cannot "
            "calibrate. Check that all seeds used the same logging_per_interaction_step."
        )
    if len(aligned) < max(len(s) for s in series) * 0.5:
        warnings.warn(
            f"Only {len(aligned)} interaction_step points are common across all "
            f"{len(baseline_identities)} baseline seeds; calibration coverage may be sparse."
        )

    thresholds = aligned.quantile(percentile / 100.0, axis=1)

    return BaselineThresholds(
        architecture=baseline_identities[0].architecture,
        environment=baseline_identities[0].environment,
        metric=metric_column,
        percentile=percentile,
        num_seeds=len(baseline_identities),
        steps=aligned.index.to_numpy(),
        thresholds=thresholds.to_numpy(),
    )


def _fingerprint_path(architecture: str, environment: str, metric: str, percentile: int, root: str) -> Path:
    return Path(root) / architecture / environment / f"{metric}_p{percentile}.meta.json"


def compute_baseline_source_fingerprint(baseline_identities: List[RunIdentity], metrics_root: str) -> str:
    """A content-based fingerprint of the baseline seeds' persisted metrics
    files (path + mtime + size), used to detect a stale cached baseline
    automatically: if any seed's metrics file has changed (or a seed's
    identity list itself changed) since the cache was written, the
    fingerprint changes and load_or_calibrate_baseline recomputes.
    """
    parts = []
    for ident in sorted(baseline_identities, key=lambda i: i.run_key):
        path = metrics_path(ident, root=metrics_root)
        if path.exists():
            stat = path.stat()
            parts.append(f"{ident.run_key}:{stat.st_mtime_ns}:{stat.st_size}")
        else:
            parts.append(f"{ident.run_key}:MISSING")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def save_baseline(thresholds: BaselineThresholds, source_fingerprint: Optional[str] = None, root: str = BASELINE_ROOT) -> Path:
    path = _cache_path(thresholds.architecture, thresholds.environment, thresholds.metric, thresholds.percentile, root)
    df = pd.DataFrame(
        {
            "interaction_step": thresholds.steps,
            "threshold": thresholds.thresholds,
            "num_seeds": thresholds.num_seeds,
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    atomic_write_text(path, buf.getvalue())

    if source_fingerprint is not None:
        meta_path = _fingerprint_path(thresholds.architecture, thresholds.environment, thresholds.metric, thresholds.percentile, root)
        atomic_write_text(meta_path, json.dumps({"source_fingerprint": source_fingerprint}))

    return path


def load_baseline(
    architecture: str,
    environment: str,
    metric_column: str,
    percentile: int,
    root: str = BASELINE_ROOT,
) -> Optional[BaselineThresholds]:
    path = _cache_path(architecture, environment, metric_column, percentile, root)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return BaselineThresholds(
        architecture=architecture,
        environment=environment,
        metric=metric_column,
        percentile=percentile,
        num_seeds=int(df["num_seeds"].iloc[0]),
        steps=df["interaction_step"].to_numpy(),
        thresholds=df["threshold"].to_numpy(),
    )


def _load_cached_fingerprint(architecture: str, environment: str, metric_column: str, percentile: int, root: str) -> Optional[str]:
    meta_path = _fingerprint_path(architecture, environment, metric_column, percentile, root)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f).get("source_fingerprint")
    except (json.JSONDecodeError, OSError):
        return None


def load_or_calibrate_baseline(
    baseline_identities: List[RunIdentity],
    metric_column: str,
    percentile: int = 95,
    metrics_root: str = "results/metrics",
    baseline_root: str = BASELINE_ROOT,
    force_recompute: bool = False,
    expected_logging_interval: Optional[int] = None,
) -> BaselineThresholds:
    """Convenience wrapper: use the cached threshold curve if present AND
    still fresh, else (re)calibrate from `baseline_identities` and cache the
    result.

    "Still fresh" is checked automatically via a content-based fingerprint
    of the baseline seeds' underlying metrics files (see
    compute_baseline_source_fingerprint): if any seed's persisted metrics
    changed since the cache was written, this recomputes without being
    asked. `force_recompute=True` recomputes unconditionally regardless of
    the fingerprint (e.g. after changing `percentile` isn't itself covered
    by the fingerprint, since percentile is part of the cache file's own
    path/key already, so this is mainly for deliberate manual invalidation).
    """
    current_fingerprint = compute_baseline_source_fingerprint(baseline_identities, metrics_root)

    if not force_recompute:
        cached = load_baseline(
            baseline_identities[0].architecture,
            baseline_identities[0].environment,
            metric_column,
            percentile,
            root=baseline_root,
        )
        if cached is not None:
            cached_fingerprint = _load_cached_fingerprint(
                baseline_identities[0].architecture, baseline_identities[0].environment,
                metric_column, percentile, root=baseline_root,
            )
            if cached_fingerprint == current_fingerprint:
                return cached
            warnings.warn(
                f"Cached baseline for architecture="
                f"'{baseline_identities[0].architecture}', metric='{metric_column}' "
                f"is stale (underlying baseline seed metrics changed since it was "
                f"computed); recalibrating automatically."
            )

    thresholds = calibrate_baseline(
        baseline_identities, metric_column, percentile,
        metrics_root=metrics_root, expected_logging_interval=expected_logging_interval,
    )
    save_baseline(thresholds, source_fingerprint=current_fingerprint, root=baseline_root)
    return thresholds
