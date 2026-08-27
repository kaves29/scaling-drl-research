"""(mean + 2*std) null-exceedance comparison - the same statistical
criterion research-methodology.md's Onset Definitions section uses
throughout this study (critic-degradation onset, pathology-propagation
onset both use "exceeds (mean + 2 sigma) of the baseline... distribution").
Angle 2B must not invent a new threshold rule for this comparison - see
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
