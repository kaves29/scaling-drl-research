"""Formal onset-detection logic for critic degradation and pathology propagation.

Pure functions over (steps, values, baseline) — no I/O, no WandB, no training
state. This is what makes recalibrating N/W or rerunning detection possible
without retraining (see analysis/pipeline.py for the orchestration that wires
this to persisted metrics + the ledger).

=== Alignment of interaction steps ===
Metrics are only recorded every `logging_per_interaction_step` interaction
steps (see analysis/metrics_store.py). N/W come from
analysis/window_calibration.py's ACF/CCF calibration, already in the units
this module expects:

  - `sustain_window_points` counts consecutive *recorded* points, matching
    window_calibration.py's ACF estimate directly (no conversion needed).
  - `propagation_window` is in raw interaction steps: the search considers
    every recorded point whose interaction_step falls in [onset, onset + W]
    inclusive, matching window_calibration.py's CCF-lag-times-
    logging-interval conversion.

=== Degradation onset ===
First recorded interaction_step of the first run of consecutive points whose
TD-error variance exceeds the baseline's 95th percentile at that same step,
where the run's length is strictly greater than `sustain_window_points`.

=== Propagation onset ===
First recorded interaction_step, within [t_degradation, t_degradation + W]
inclusive, at which actor-gradient cosine exceeds the baseline's 95th
percentile at that step. Unlike degradation, no sustain requirement applies
here — a single qualifying point is sufficient, per spec section 13.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from analysis.baseline_calibration import BaselineThresholds

STATUS_SUCCESS = "success"
STATUS_NO_ONSET = "no_onset_detected"
STATUS_NEEDS_REVIEW = "needs_manual_review"
# No "ambiguous" status: anything not confidently success/no_onset_detected
# uses needs_manual_review - see utils/onset_ledger.py's VALID_STATUSES.


@dataclass(frozen=True)
class OnsetResult:
    onset_step: Optional[int]
    status: str
    notes: str


def _aligned_pairs(steps: Sequence[int], values: Sequence[float], baseline: BaselineThresholds):
    """Keeps only (step, value) pairs the baseline actually has coverage for."""
    return [(s, v) for s, v in zip(steps, values) if v == v and baseline.at(s) is not None]


def detect_critic_degradation_onset(
    steps: Sequence[int],
    td_error_variance: Sequence[float],
    baseline: BaselineThresholds,
    sustain_window_points: int,
) -> OnsetResult:
    """`sustain_window_points` must already be in units of consecutive
    *recorded* points (see module docstring) - convert from the raw
    interaction-step `onset_detection.sustain_window` config value by
    dividing by `logging_per_interaction_step` before calling this."""
    if len(steps) == 0:
        return OnsetResult(None, STATUS_NEEDS_REVIEW, "no recorded metric points for this run")

    aligned = _aligned_pairs(steps, td_error_variance, baseline)
    if not aligned:
        return OnsetResult(
            None,
            STATUS_NEEDS_REVIEW,
            "no overlap between this run's recorded interaction steps and the "
            "baseline calibration steps; cannot evaluate exceedance",
        )

    coverage_ratio = len(aligned) / len(steps)
    a_steps = [s for s, _ in aligned]
    exceeds = [v > baseline.at(s) for s, v in aligned]

    onset_step = None
    run_start = None
    run_len = 0
    for i, exceeded in enumerate(exceeds):
        if exceeded:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > sustain_window_points:
                onset_step = a_steps[run_start]
                break
        else:
            run_len = 0
            run_start = None

    coverage_note = (
        f"; baseline coverage {coverage_ratio:.0%} of this run's recorded steps"
        if coverage_ratio < 1.0
        else ""
    )

    if onset_step is not None:
        return OnsetResult(
            onset_step,
            STATUS_SUCCESS,
            f"sustained exceedance run (> {sustain_window_points} consecutive recorded "
            f"points) starting at interaction_step={onset_step}{coverage_note}",
        )

    if coverage_ratio < 0.5:
        return OnsetResult(
            None,
            STATUS_NEEDS_REVIEW,
            f"no sustained exceedance found, but baseline coverage was only "
            f"{coverage_ratio:.0%} of recorded steps; result is unreliable{coverage_note}",
        )

    return OnsetResult(
        None,
        STATUS_NO_ONSET,
        f"no run of consecutive exceedance longer than sustain_window_points="
        f"{sustain_window_points} was found{coverage_note}",
    )


def detect_propagation_onset(
    steps: Sequence[int],
    actor_grad_cosine: Sequence[float],
    baseline: BaselineThresholds,
    degradation_onset_step: Optional[int],
    propagation_window: int,
) -> OnsetResult:
    if degradation_onset_step is None:
        return OnsetResult(
            None,
            STATUS_NO_ONSET,
            "critic degradation was not detected for this run; propagation is "
            "undefined without a degradation onset (no propagation analysis performed)",
        )

    if len(steps) == 0:
        return OnsetResult(None, STATUS_NEEDS_REVIEW, "no recorded metric points for this run")

    window_lo = degradation_onset_step
    window_hi = degradation_onset_step + propagation_window
    in_window = [(s, v) for s, v in zip(steps, actor_grad_cosine) if window_lo <= s <= window_hi]

    if not in_window:
        return OnsetResult(
            None,
            STATUS_NEEDS_REVIEW,
            f"no recorded interaction steps fall within the propagation window "
            f"[{window_lo}, {window_hi}]",
        )

    aligned = [(s, v) for s, v in in_window if v == v and baseline.at(s) is not None]
    if not aligned:
        return OnsetResult(
            None,
            STATUS_NEEDS_REVIEW,
            f"no baseline coverage within the propagation window [{window_lo}, {window_hi}]",
        )

    for s, v in aligned:
        if v > baseline.at(s):
            return OnsetResult(
                s,
                STATUS_SUCCESS,
                f"first exceedance within propagation window [{window_lo}, {window_hi}] "
                f"at interaction_step={s}",
            )

    return OnsetResult(
        None,
        STATUS_NO_ONSET,
        f"no exceedance found within propagation window [{window_lo}, {window_hi}]",
    )
