"""Angle 2A configuration validation and per-role agent-config construction.

The scientific protocol requires every participant's critic architecture
(scaled_a, scaled_b, reference) to be an *explicit* config input with no
implicit default (see configs/base_angle2a.yaml, where these fields are `???`
- OmegaConf's mandatory-value marker). This module is the single place that
reads those fields, so there is exactly one place that could accidentally
grow a hardcoded fallback - and it deliberately doesn't have one.
"""

import copy
from dataclasses import dataclass
from typing import Dict

from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from experiments.angle_2a.errors import Angle2AConfigError

REQUIRED_ROLES = ("scaled_a", "scaled_b", "reference")
REQUIRED_ARCH_FIELDS = ("critic_num_blocks", "critic_hidden_dim")


@dataclass(frozen=True)
class RoleArchitecture:
    role: str  # "scaled_a" | "scaled_b" | "reference"
    critic_num_blocks: int
    critic_hidden_dim: int


def validate_angle2a_config(cfg) -> Dict[str, RoleArchitecture]:
    """Validates that angle_2_a.{scaled_a,scaled_b,reference}.{critic_num_blocks,
    critic_hidden_dim} were all explicitly supplied, and returns them.

    Fails with a single Angle2AConfigError listing every missing field at
    once (rather than stopping at the first OmegaConf MissingMandatoryValue),
    since a user is most likely to be missing several of these overrides on
    a first attempt.
    """
    if "angle_2_a" not in cfg:
        raise Angle2AConfigError(
            "Missing 'angle_2_a' config block. Use --config_name base_angle2a "
            "(or a config that includes its angle_2_a: section) with "
            "--experiment angle_2_a."
        )

    missing = []
    architectures: Dict[str, RoleArchitecture] = {}

    for role in REQUIRED_ROLES:
        if role not in cfg.angle_2_a:
            missing.append(f"angle_2_a.{role}")
            continue

        role_cfg = cfg.angle_2_a[role]
        values = {}
        for field in REQUIRED_ARCH_FIELDS:
            try:
                value = role_cfg[field]
                if value is None:
                    raise Angle2AConfigError("null is not a valid architecture value")
            except MissingMandatoryValue:
                missing.append(f"angle_2_a.{role}.{field}")
                continue
            values[field] = value

        if len(values) == len(REQUIRED_ARCH_FIELDS):
            architectures[role] = RoleArchitecture(
                role=role,
                critic_num_blocks=int(values["critic_num_blocks"]),
                critic_hidden_dim=int(values["critic_hidden_dim"]),
            )

    if missing:
        raise Angle2AConfigError(
            "Angle 2A requires every participant's critic architecture to be "
            "explicitly supplied; no defaults are assumed. Missing/unset "
            f"required field(s): {missing}. Example:\n"
            "  --overrides angle_2_a.scaled_a.critic_num_blocks=5 "
            "angle_2_a.scaled_a.critic_hidden_dim=768 "
            "angle_2_a.scaled_b.critic_num_blocks=7 "
            "angle_2_a.scaled_b.critic_hidden_dim=1024 "
            "angle_2_a.reference.critic_num_blocks=2 "
            "angle_2_a.reference.critic_hidden_dim=512"
        )

    return architectures


def architecture_label(role_arch: RoleArchitecture) -> str:
    """Derives the same architecture id string Angle 1 uses (see
    scale_rl.common.logger.get_architecture_id), so Angle 2A's ledger lookups
    and output directory naming stay consistent with how Angle 1 already
    names architectures - reusing that single source of truth rather than
    re-implementing the (hidden_dim, num_blocks) -> label mapping here."""
    from omegaconf import OmegaConf as _OmegaConf

    from scale_rl.common.logger import get_architecture_id

    fake_cfg = _OmegaConf.create(
        {
            "agent": {
                "critic_hidden_dim": role_arch.critic_hidden_dim,
                "critic_num_blocks": role_arch.critic_num_blocks,
            }
        }
    )
    return get_architecture_id(fake_cfg)


def build_role_agent_cfg(base_agent_cfg, architecture: RoleArchitecture) -> dict:
    """Returns a fully-resolved plain dict for create_agent(), with this
    role's critic_num_blocks/critic_hidden_dim substituted in place of
    whatever unresolved/mandatory top-level interpolation the shared agent
    template (configs/agent/sac_simba.yaml) would otherwise point at.

    Mutating a deep copy of the DictConfig *before* resolving is required:
    the top-level critic_num_blocks/critic_hidden_dim in base_angle2a.yaml
    are deliberately `???` (see that file's comments), so resolving the
    agent subtree as-is would raise before we ever get a chance to apply
    the per-role override.
    """
    agent_node = copy.deepcopy(base_agent_cfg)
    OmegaConf.set_struct(agent_node, False)
    agent_node.critic_num_blocks = architecture.critic_num_blocks
    agent_node.critic_hidden_dim = architecture.critic_hidden_dim
    OmegaConf.set_struct(agent_node, True)

    return OmegaConf.to_container(agent_node, resolve=True, throw_on_missing=True)
