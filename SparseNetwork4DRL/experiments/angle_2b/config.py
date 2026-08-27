"""Angle 2B configuration validation.

Angle 2B reads architecture, onset step, and agent configuration entirely
from Angle 2A's already-persisted snapshots (see checkpoint_io.py) - it
never needs architecture fields supplied directly, unlike Angle 2A's
config.py. What IS required explicitly here (no hidden defaults, matching
this repo's established convention): which environment/seed/matchup to
analyze, and which seeds' null-baseline snapshots to pool into the null
distribution - both are methodologically significant choices this project
was explicitly told not to silently default.
"""

from dataclasses import dataclass
from typing import List

from experiments.angle_2b.errors import Angle2BConfigError

REQUIRED_BLOCK_FIELDS = ("matchup_names", "null_seeds")
VALID_MATCHUP_NAMES = ("matchup_1", "matchup_2")


@dataclass(frozen=True)
class Angle2BRunConfig:
    environment: str
    seed: int
    matchup_names: List[str]
    null_seeds: List[int]
    analysis_seed: int
    num_states_per_source: int
    angle_2a_results_root: str
    output_root: str


def validate_angle2b_config(cfg) -> Angle2BRunConfig:
    if "angle_2_b" not in cfg:
        raise Angle2BConfigError(
            "Missing 'angle_2_b' config block. Use --config_name base_angle2b "
            "(or a config that includes its angle_2_b: section) with "
            "--experiment angle_2_b."
        )

    block = cfg.angle_2_b
    missing = [f for f in REQUIRED_BLOCK_FIELDS if f not in block or block[f] is None]
    if missing:
        raise Angle2BConfigError(
            "Angle 2B requires the following angle_2_b.* fields to be "
            "explicitly supplied; no defaults are assumed. Missing/unset "
            f"required field(s): {[f'angle_2_b.{f}' for f in missing]}. Example:\n"
            "  --overrides angle_2_b.matchup_names=[matchup_1,matchup_2] "
            "angle_2_b.null_seeds=[1,2,3,4,5]"
        )

    matchup_names = list(block.matchup_names)
    if not matchup_names:
        raise Angle2BConfigError("angle_2_b.matchup_names must be non-empty.")
    invalid = [n for n in matchup_names if n not in VALID_MATCHUP_NAMES]
    if invalid:
        raise Angle2BConfigError(
            f"angle_2_b.matchup_names entries must be one of "
            f"{VALID_MATCHUP_NAMES} (Angle 2A's own matchup-naming "
            f"convention), got invalid entries: {invalid}."
        )

    null_seeds = [int(s) for s in block.null_seeds]
    if not null_seeds:
        raise Angle2BConfigError("angle_2_b.null_seeds must be non-empty.")

    seed = int(cfg.seed)
    analysis_seed = int(block.analysis_seed) if "analysis_seed" in block and block.analysis_seed is not None else seed

    return Angle2BRunConfig(
        environment=str(cfg.env_name),
        seed=seed,
        matchup_names=matchup_names,
        null_seeds=null_seeds,
        analysis_seed=analysis_seed,
        num_states_per_source=int(block.get("num_states_per_source", 30)),
        angle_2a_results_root=str(block.get("angle_2a_results_root", "results/angle_2a")),
        output_root=str(block.get("output_root", "results/angle_2b")),
    )
