"""Angle 2C's three candidate distortion properties - pure numpy functions
over already-computed nabla_a Q / Q(s,a) arrays (direction, magnitude/bias)
or already-computed per-(s,a) instability variances (see perturbation.py).
No I/O, no JAX critic calls here - see loader.py/perturbation.py for where
those live.

Each property's LITERAL formula (as specified in research-methodology.md)
is reported for scientific transparency. Where that literal formula isn't
compatible with experiments/angle_2b/statistics.py's compare_to_null (a
ONE-SIDED, upper-tail (mean + 2*std) rule - "reuse Angle 2B's null
infrastructure directly", per this task's explicit instruction), a
monotonic transform is applied ONLY for the null-exceedance verdict, never
for the reported value itself. See each function's docstring for why.
"""

import numpy as np

_EPS = 1e-8


def directional_corruption(grad_aq_d: np.ndarray, grad_aq_r: np.ndarray) -> np.ndarray:
    """D_direction(s,a) = cos_sim(nabla_a Q_D, nabla_a Q_R) - literal
    formula. A value near 1 means similar direction (little corruption); a
    value well below 1 (or negative) means the critics disagree on which
    way to move the action - LOWER is more corrupted, unlike Angle 2B's
    D_dir (=1-cos, where HIGHER is more distorted). See
    directional_corruption_for_null_comparison for the transform used only
    for the null-exceedance verdict.
    """
    dots = np.sum(grad_aq_d * grad_aq_r, axis=-1)
    norms = np.linalg.norm(grad_aq_d, axis=-1) * np.linalg.norm(grad_aq_r, axis=-1)
    return dots / (norms + _EPS)


def directional_corruption_for_null_comparison(cos_sim: np.ndarray) -> np.ndarray:
    """1 - cos_sim: turns "lower cosine = more corrupted" into "higher
    value = more corrupted", matching compare_to_null's one-sided,
    upper-tail convention - exactly Angle 2B's own D_dir formula, reused
    here for the identical reason it was chosen there (see
    experiments/angle_2b/gradients.py's compute_distortion_metrics). Used
    ONLY to decide the null-exceedance verdict; directional_corruption's raw
    cosine is still the reported value.
    """
    return 1.0 - cos_sim


def magnitude_ratio(grad_aq_d: np.ndarray, grad_aq_r: np.ndarray) -> np.ndarray:
    """D_magnitude(s,a) = ||nabla_a Q_D|| / ||nabla_a Q_R|| - literal
    formula. A value far from 1.0 in EITHER direction indicates
    miscalibration (two-sided). See magnitude_ratio_for_null_comparison.
    """
    norm_d = np.linalg.norm(grad_aq_d, axis=-1)
    norm_r = np.linalg.norm(grad_aq_r, axis=-1)
    return (norm_d + _EPS) / (norm_r + _EPS)


def magnitude_ratio_for_null_comparison(ratio: np.ndarray) -> np.ndarray:
    """|log(ratio)|: a ratio of 1.0 (perfectly calibrated) maps to 0; any
    departure from 1.0 in either direction maps to a positive value growing
    with miscalibration - compatible with compare_to_null's one-sided
    upper-tail convention, unlike the raw (two-sided) ratio.

    This is a DIFFERENT transform from Angle 2B's D_mag (a SIGNED log-ratio,
    no absolute value, per that angle's explicit locked-methodology
    decision - see experiments/angle_2b/gradients.py). Angle 2C's
    D_magnitude is a genuinely different, unsigned-ratio formula by the
    methodology doc's own design (not the same metric under a different
    name), so this is not a re-litigation of that earlier Angle 2B decision
    - just the transform this metric's own two-sided departure question
    requires.
    """
    return np.abs(np.log(ratio + _EPS))


def raw_offset(q_d: np.ndarray, q_r: np.ndarray) -> np.ndarray:
    """raw_offset(s,a) = Q_D(s,a) - Q_R(s,a) - literal formula. Reported and
    null-compared AS-IS (signed, one-sided, no transform): the study's
    overestimation hypothesis specifically expects D > R (a MORE positive
    offset), which is already the "higher = more anomalous" direction
    compare_to_null assumes.
    """
    return q_d - q_r


def instability_ratio(var_d: np.ndarray, var_r: np.ndarray) -> np.ndarray:
    """D_instability(s,a) = var_D(s,a) / var_R(s,a). Not itself named in
    research-methodology.md (which only specifies comparing the two
    variances) - constructed as a ratio to mirror magnitude_ratio's
    convention. "Elevated variance in the degraded critic" (the doc's own
    framing) means D's variance is expected to be the larger one when this
    property is real - a one-sided departure (ratio > 1), directly usable
    with compare_to_null's upper-tail convention as-is, no transform needed
    (unlike magnitude_ratio, which is two-sided by nature).
    """
    return (var_d + _EPS) / (var_r + _EPS)
