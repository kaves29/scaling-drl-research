"""(mean + 2*std) null-exceedance comparison - the statistical criterion
research-methodology.md's Angle 2B null_distribution section specifies for
this exact comparison.

As of the 2026-08-28 audit, this is a DIFFERENT rule from Angle 1's own
onset/propagation threshold (analysis/baseline_calibration.py /
onset_detection.py, a 95th percentile of the baseline distribution) -
research-methodology.md previously (incorrectly) described both as "the
same criterion." (mean + 2*std) remains the right choice specifically here:
Angle 2B's null distribution has very few points (at most one A/B pair per
baseline seed, <=5 total), where an empirical percentile would be poorly
defined/unstable, unlike Angle 1's baseline curves (many logged timesteps
across all 5 seeds). Do not change this to match Angle 1's percentile rule,
and do not invent a third threshold rule for this comparison - see
"Redefining or re-thresholding the onset/propagation criteria... without
recalibrating" in research-methodology.md's "Things Claude Must Never
Change Silently".
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class NullComparisonResult:
    metric_name: str
    observed_value: float
    null_mean: float
    null_std: float
    null_n: int
    threshold: float  # null_mean + 2 * null_std
    exceeds_null: bool


def compare_to_null(metric_name: str, observed_value: float, null_values: List[float]) -> NullComparisonResult:
    values = np.asarray(null_values, dtype=np.float64)
    null_mean = float(np.mean(values))
    null_std = float(np.std(values))
    threshold = null_mean + 2.0 * null_std
    return NullComparisonResult(
        metric_name=metric_name,
        observed_value=float(observed_value),
        null_mean=null_mean,
        null_std=null_std,
        null_n=int(values.shape[0]),
        threshold=threshold,
        exceeds_null=bool(observed_value > threshold),
    )
