"""Angle 1 onset-timing comparison for Angle 2C's closing-the-loop
requirement.

Per explicit decision (see End-of-Task Summary): Angle 2A/2B only ever
produce a SINGLE frozen snapshot per matchup (at the scaled critic's own
t*), never a time series - a true "first departure from null across the
degradation window" onset-timing curve is not constructible from existing
artifacts without new training/snapshots, which this task rules out. This
module implements the agreed single-point PROXY instead.

IMPORTANT HONESTY NOTE, not just an implementation detail: t* (this
matchup's stopping step) and Angle 1's own logged critic-degradation onset
step are THE SAME NUMBER BY CONSTRUCTION - Angle 2A's run_matchup() trains D
to stop_step=onset.onset_step, i.e. exactly Angle 1's own ledger value (see
experiments/angle_2a/onset_lookup.py, experiments/angle_2a/matchup.py).
There is no possible "timing disagreement" for this proxy to detect between
two independently-measured quantities - both loaded values will always be
identical. The only question with real content is "had the dominant
property already diverged from its null by that (necessarily shared)
reference point" - this is reported honestly as such, not dressed up as an
independent timing-consistency check.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from experiments.angle_2a.onset_lookup import lookup_critic_degradation_onset
from experiments.angle_2a.storage import DEFAULT_OUTPUT_ROOT as ANGLE_2A_ROOT
from experiments.angle_2c.errors import Angle2CDataError


def load_scaled_architecture_label(
    environment: str,
    seed: int,
    matchup_name: str,
    root: str = ANGLE_2A_ROOT,
) -> str:
    """Reads the scaled architecture label Angle 2A already recorded for
    this matchup (e.g. "D5W768") from its own run_metadata.json - the
    source of truth for architecture identity, never re-derived here."""
    metadata_path = Path(root) / environment / f"seed{seed}" / matchup_name / "run_metadata.json"
    if not metadata_path.exists():
        raise Angle2CDataError(
            f"Missing Angle 2A run_metadata.json at '{metadata_path}' - "
            f"cannot determine this matchup's scaled architecture label for "
            f"the Angle 1 onset-timing lookup."
        )
    with open(metadata_path) as f:
        metadata = json.load(f)
    return metadata["scaled_architecture"]


@dataclass(frozen=True)
class OnsetTimingComparison:
    dominant_property: str
    exceeds_null_at_t_star: bool
    t_star: int
    angle1_degradation_onset_step: int
    consistent: bool
    note: str


def compare_onset_timing(
    dominant_property: str,
    exceeds_null_at_t_star: bool,
    t_star: int,
    environment: str,
    seed: int,
    matchup_name: str,
    onset_source_experiment: str,
    onset_ledger_root: str,
    angle_2a_root: str = ANGLE_2A_ROOT,
) -> OnsetTimingComparison:
    """Single-point proxy comparison - see module docstring for what this
    can and cannot actually establish. "consistent" means the dominant
    property had already departed from null by t* - it is NOT a comparison
    of two independently-measured onset times (t_star and
    angle1_degradation_onset_step are the same number by construction; both
    are reported for completeness/transparency, not because they might
    disagree).
    """
    architecture = load_scaled_architecture_label(environment, seed, matchup_name, root=angle_2a_root)
    onset = lookup_critic_degradation_onset(
        architecture=architecture,
        environment=environment,
        seed=seed,
        source_experiment=onset_source_experiment,
        ledger_root=onset_ledger_root,
    )

    return OnsetTimingComparison(
        dominant_property=dominant_property,
        exceeds_null_at_t_star=exceeds_null_at_t_star,
        t_star=t_star,
        angle1_degradation_onset_step=onset.onset_step,
        consistent=bool(exceeds_null_at_t_star),
        note=(
            "PROXY ONLY: t_star and angle1_degradation_onset_step are the "
            "SAME number by construction (Angle 2A trains D to exactly "
            "Angle 1's own logged onset step) - this is not an independent "
            "timing comparison. The only real content here is whether the "
            "dominant property had already diverged from its null by that "
            "shared reference point; a full onset-timing curve would "
            "require Angle 2A snapshots across a window, which do not "
            "exist and were out of scope for this task."
        ),
    )
