"""Angle 2C configuration validation.

Angle 2C reads everything it needs from Angle 2B's already-persisted output
(states, actions, nabla_a Q, Q values, actor-parameter gradients - see
loader.py) and Angle 1's onset ledger (see onset_lookup.py) - it never needs
architecture fields supplied directly, unlike Angle 2A's config.py. What IS
required explicitly here (no hidden defaults, matching this repo's
established convention): which environment/seed/matchup to analyze - a
methodologically significant choice this project was explicitly told not to
silently default.
"""

from dataclasses import dataclass
from typing import List

from experiments.angle_2c.errors import Angle2CConfigError

REQUIRED_BLOCK_FIELDS = ("matchup_names",)
VALID_MATCHUP_NAMES = ("matchup_1", "matchup_2")

# Local-instability perturbation defaults - kept configurable for
# testability, but the scientific protocol requires exactly these (see
# End-of-Task Summary for the reasoning): K=20 small Gaussian offsets,
# sigma=0.01 in the tanh-squashed [-1,1] action space, clipped back into
# [-1,1] so perturbed actions stay within the space the critic was ever
# actually trained on.
DEFAULT_NUM_PERTURBATIONS = 20
DEFAULT_PERTURBATION_SIGMA = 0.01


@dataclass(frozen=True)
class Angle2CRunConfig:
    environment: str
    seed: int
    matchup_names: List[str]
    analysis_seed: int
    num_perturbations: int
    perturbation_sigma: float
    onset_source_experiment: str
    onset_ledger_root: str
    angle_2b_results_root: str
    output_root: str


def validate_angle2c_config(cfg) -> Angle2CRunConfig:
    if "angle_2_c" not in cfg:
        raise Angle2CConfigError(
            "Missing 'angle_2_c' config block. Use --config_name base_angle2c "
            "(or a config that includes its angle_2_c: section) with "
            "--experiment angle_2_c."
        )

    block = cfg.angle_2_c
    missing = [f for f in REQUIRED_BLOCK_FIELDS if f not in block or block[f] is None]
    if missing:
        raise Angle2CConfigError(
            "Angle 2C requires the following angle_2_c.* fields to be "
            "explicitly supplied; no defaults are assumed. Missing/unset "
            f"required field(s): {[f'angle_2_c.{f}' for f in missing]}. Example:\n"
            "  --overrides angle_2_c.matchup_names=[matchup_1,matchup_2]"
        )

    matchup_names = list(block.matchup_names)
    if not matchup_names:
        raise Angle2CConfigError("angle_2_c.matchup_names must be non-empty.")
    invalid = [n for n in matchup_names if n not in VALID_MATCHUP_NAMES]
    if invalid:
        raise Angle2CConfigError(
            f"angle_2_c.matchup_names entries must be one of "
            f"{VALID_MATCHUP_NAMES} (Angle 2A/2B's own matchup-naming "
            f"convention), got invalid entries: {invalid}."
        )

    seed = int(cfg.seed)
    analysis_seed = int(block.analysis_seed) if "analysis_seed" in block and block.analysis_seed is not None else seed

    return Angle2CRunConfig(
        environment=str(cfg.env_name),
        seed=seed,
        matchup_names=matchup_names,
        analysis_seed=analysis_seed,
        num_perturbations=int(block.get("num_perturbations", DEFAULT_NUM_PERTURBATIONS)),
        perturbation_sigma=float(block.get("perturbation_sigma", DEFAULT_PERTURBATION_SIGMA)),
        onset_source_experiment=str(block.get("onset_source_experiment", "angle_1")),
        onset_ledger_root=str(block.get("onset_ledger_root", "results/ledgers")),
        angle_2b_results_root=str(block.get("angle_2b_results_root", "results/angle_2b")),
        output_root=str(block.get("output_root", "results/angle_2c")),
    )
