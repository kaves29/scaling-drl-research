"""Post-hoc integration: persisted metrics -> baseline -> onset detection -> ledger.

This is the only place that calls both analysis/onset_detection.py and
utils/onset_ledger.py; everything else stays decoupled so detection can be
rerun offline (see utils.onset_ledger, analysis.baseline_calibration).

Called once, after training finishes (never during the training loop — see
experiments/angle_1.py).
"""

import warnings
from typing import List, Optional

from analysis.baseline_calibration import load_or_calibrate_baseline
from analysis.metrics_store import RunIdentity, load_metrics
from analysis.window_calibration import load_or_calibrate_window_parameters
from analysis.onset_detection import (
    STATUS_NEEDS_REVIEW,
    STATUS_NO_ONSET,
    STATUS_SUCCESS,
    OnsetResult,
    detect_critic_degradation_onset,
    detect_propagation_onset,
)
from utils.onset_ledger import REQUIRED_COLUMNS, WandbIdentity, log_onset_event

_STATUS_PRIORITY = [STATUS_NEEDS_REVIEW, STATUS_SUCCESS, STATUS_NO_ONSET]


def _combine_status(results: List[OnsetResult]) -> str:
    statuses = {r.status for r in results if r is not None}
    for status in _STATUS_PRIORITY:
        if status in statuses:
            return status
    return STATUS_NO_ONSET


def run_post_hoc_onset_analysis(
    run_identity: RunIdentity,
    critic_degradation_enabled: bool,
    pathology_prop_enabled: bool,
    baseline_identities: List[RunIdentity],
    onset_cfg: dict,
    logging_per_interaction_step: int,
    wandb_identity: Optional[WandbIdentity] = None,
    metrics_root: str = "results/metrics",
    baseline_root: str = "results/baselines",
    ledger_root: str = "results/ledgers",
):
    """Runs onset detection for one finished run and upserts its ledger row.

    `logging_per_interaction_step` feeds window_calibration's N/W ACF/CCF
    calibration and validates the baseline seeds share this run's cadence.
    Returns the ledger CSV path, or None if neither tracking flag was enabled.
    """
    if not (critic_degradation_enabled or pathology_prop_enabled):
        return None

    row = {c: None for c in REQUIRED_COLUMNS}
    row.update(
        {
            "run_key": run_identity.run_key,
            "exact_run_name": wandb_identity.run_obj.name if wandb_identity and wandb_identity.run_obj else None,
            "wandb_run_id": wandb_identity.run_obj.id if wandb_identity and wandb_identity.run_obj else None,
            "architecture": run_identity.architecture,
            "environment": run_identity.environment,
            "seed": run_identity.seed,
        }
    )

    metrics_df = load_metrics(run_identity, root=metrics_root)
    if metrics_df.empty:
        row["status"] = STATUS_NEEDS_REVIEW
        row["detection_notes"] = (
            "no persisted metrics found for this run; cannot run onset analysis "
            "(training may have crashed before the first logging interval, or "
            "critic_degradation/pathology_prop were disabled during training "
            "for this run)"
        )
        return log_onset_event(
            run_identity.experiment, run_identity.architecture, row,
            identity=wandb_identity, root=ledger_root,
        )

    steps = metrics_df["interaction_step"].tolist()
    notes = []
    results = []
    force_baseline_recompute = bool(onset_cfg.get("force_baseline_recompute", False))

    degradation_result: Optional[OnsetResult] = None
    # N and W are calibrated together; reused for propagation's W below.
    window_calibration = None
    if critic_degradation_enabled:
        try:
            baseline = load_or_calibrate_baseline(
                baseline_identities,
                "td_error_variance",
                percentile=onset_cfg["baseline_percentile"],
                metrics_root=metrics_root,
                baseline_root=baseline_root,
                force_recompute=force_baseline_recompute,
                expected_logging_interval=logging_per_interaction_step,
            )
            window_calibration = load_or_calibrate_window_parameters(
                baseline_identities,
                logging_per_interaction_step=logging_per_interaction_step,
                metrics_root=metrics_root,
                window_root=baseline_root,
                force_recompute=force_baseline_recompute,
                burn_in_fraction=onset_cfg["acf_ccf_burn_in_fraction"],
                ccf_lag_fraction_cap=onset_cfg["ccf_lag_fraction_cap"],
            )
            degradation_result = detect_critic_degradation_onset(
                steps,
                metrics_df["td_error_variance"].tolist(),
                baseline,
                window_calibration.n_sustain_window_points,
            )
        except Exception as e:
            warnings.warn(f"Critic degradation onset analysis failed for {run_identity.run_key}: {e}")
            degradation_result = OnsetResult(None, STATUS_NEEDS_REVIEW, f"analysis error: {e}")

        row["critic_degradation_onset_step"] = degradation_result.onset_step
        row["critic_degradation_method"] = onset_cfg["critic_degradation_method_version"]
        notes.append(f"critic_degradation[{degradation_result.status}]: {degradation_result.notes}")
        results.append(degradation_result)

    propagation_result: Optional[OnsetResult] = None
    if pathology_prop_enabled:
        if not critic_degradation_enabled:
            propagation_result = OnsetResult(
                None,
                STATUS_NEEDS_REVIEW,
                "pathology_prop=true requires critic_degradation=true in the same "
                "run: propagation is only meaningful relative to a degradation "
                "onset, and this run did not compute one",
            )
        elif window_calibration is None:
            propagation_result = OnsetResult(
                None,
                STATUS_NEEDS_REVIEW,
                "window calibration (N/W) failed alongside critic degradation "
                "analysis - see the critic_degradation note above for the "
                "underlying error; propagation's W could not be obtained",
            )
        else:
            try:
                baseline = load_or_calibrate_baseline(
                    baseline_identities,
                    "actor_grad_cosine",
                    percentile=onset_cfg["baseline_percentile"],
                    metrics_root=metrics_root,
                    baseline_root=baseline_root,
                    force_recompute=force_baseline_recompute,
                    expected_logging_interval=logging_per_interaction_step,
                )
                propagation_result = detect_propagation_onset(
                    steps,
                    metrics_df["actor_grad_cosine"].tolist(),
                    baseline,
                    degradation_result.onset_step if degradation_result else None,
                    window_calibration.w_propagation_window_steps,
                )
            except Exception as e:
                warnings.warn(f"Propagation onset analysis failed for {run_identity.run_key}: {e}")
                propagation_result = OnsetResult(None, STATUS_NEEDS_REVIEW, f"analysis error: {e}")

        row["propagation_onset_step"] = propagation_result.onset_step
        row["propagation_method"] = onset_cfg["propagation_method_version"]
        notes.append(f"propagation[{propagation_result.status}]: {propagation_result.notes}")
        results.append(propagation_result)

    if (
        degradation_result is not None
        and propagation_result is not None
        and degradation_result.onset_step is not None
        and propagation_result.onset_step is not None
    ):
        row["propagation_lag"] = propagation_result.onset_step - degradation_result.onset_step
    else:
        row["propagation_lag"] = None

    row["status"] = _combine_status(results)
    row["detection_notes"] = " || ".join(notes)

    return log_onset_event(
        run_identity.experiment, run_identity.architecture, row,
        identity=wandb_identity, root=ledger_root,
    )
