"""Loads a frozen Angle 2A agent snapshot (checkpoint + probe-capture
states/actions + resolved agent config) with zero training and zero
environment interaction.

This is the sole read path Angle 2B uses to obtain pi_D/Q_D, pi_R/Q_R, and
healthy-critic null-baseline agents - see
experiments/angle_2a/storage.py:save_frozen_agent_snapshot for what is
persisted and why. Nothing here re-runs training or touches a real
gym/dm_control environment: observation/action dimensionality is recovered
from the saved probe-capture arrays' shapes, and a real env is never
constructed just to read its `.observation_space`/`.action_space` (SACAgent
and ObservationNormalizer only ever use `.shape[-1]`/`.dtype` from those
spaces - see scale_rl/agents/sac/sac_agent.py and
scale_rl/agents/wrappers/normalization.py - so a placeholder gym.spaces.Box
with the right shape is sufficient and exact, not an approximation).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.storage import matchup_dir
from experiments.angle_2b.errors import Angle2BSnapshotError
from scale_rl.agents import create_agent

DEFAULT_ANGLE_2A_ROOT = "results/angle_2a"


@dataclass
class FrozenAgentSnapshot:
    role: str  # "D" or "R" (as labeled by the Angle 2A matchup this came from)
    environment: str
    seed: int
    matchup_name: str
    agent: Any  # SACAgent, possibly ObservationNormalizer-wrapped
    critic_use_cdq: bool
    states: np.ndarray  # (N, obs_dim), raw/unnormalized - see agent_runner.py
    actions: np.ndarray  # (N, act_dim)


def _require_exists(path: Path, what: str) -> Path:
    if not path.exists():
        raise Angle2BSnapshotError(
            f"Missing {what} at '{path}'. Angle 2B never retrains or "
            f"reconstructs a missing Angle 2A snapshot - re-run Angle 2A "
            f"for this (environment, seed, matchup) first, or point "
            f"angle_2a_results_root at wherever those results actually live."
        )
    return path


def load_frozen_agent_snapshot(
    environment: str,
    seed: int,
    matchup_name: str,
    role: str,
    root: str = DEFAULT_ANGLE_2A_ROOT,
) -> FrozenAgentSnapshot:
    if role not in ("D", "R"):
        raise ValueError(f"role must be 'D' or 'R', got {role!r}")

    out_dir = matchup_dir(environment, seed, matchup_name, root=root)

    agent_cfg_path = _require_exists(out_dir / f"agent_cfg_{role}.json", "agent config snapshot")
    with open(agent_cfg_path) as f:
        agent_cfg_dict: Dict[str, Any] = json.load(f)

    probe_capture_path = _require_exists(out_dir / f"probe_capture_{role}.npz", "probe-capture snapshot")
    with np.load(probe_capture_path, allow_pickle=False) as npz:
        states = npz["states"]
        actions = npz["actions"]

    if states.ndim != 2 or states.shape[0] == 0:
        raise Angle2BSnapshotError(
            f"probe_capture_{role}.npz at '{probe_capture_path}' has no usable "
            f"states (shape={states.shape}); Angle 2A must have collected at "
            f"least one transition before this snapshot was taken."
        )

    checkpoint_dir = _require_exists(out_dir / "checkpoints" / role, "agent checkpoint")

    obs_dim = int(states.shape[-1])
    act_dim = int(actions.shape[-1])
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        cfg=OmegaConf.create(agent_cfg_dict),
    )
    agent.load_checkpoint(str(checkpoint_dir))

    return FrozenAgentSnapshot(
        role=role,
        environment=environment,
        seed=seed,
        matchup_name=matchup_name,
        agent=agent,
        critic_use_cdq=bool(agent_cfg_dict["critic_use_cdq"]),
        states=states,
        actions=actions,
    )


def apply_agent_normalization(agent, raw_states: np.ndarray) -> np.ndarray:
    """Applies `agent`'s own obs_rms normalization (if ObservationNormalizer-
    wrapped) to an arbitrary batch of raw states, matching the input
    distribution `agent`'s actor/critic were actually trained on (see
    scale_rl.agents.wrappers.normalization.ObservationNormalizer._normalize).
    Returns the states unchanged for an unwrapped (non-normalized) agent.

    This must be applied using the HELD-FIXED actor's own normalization -
    never per-source (e.g. D-sourced states normalized by D, R-sourced
    states normalized by R, then concatenated) - because
    gradients.py's _actor_loss feeds ONE shared `observations` array to both
    the actor.apply() call (to sample actions) and the critic call (to
    evaluate Q): whichever critic is swapped in must see exactly the
    observation representation the fixed actor itself operates in, not its
    own preferred normalization. Concretely: for the primary analysis
    (pi_D held fixed), the WHOLE batch - both D-sourced and R-sourced raw
    states - is normalized using D's own obs_rms before either Q_D or Q_R
    ever sees it; for the secondary analysis (pi_R held fixed), the same
    raw batch is instead normalized using R's own obs_rms. Without this,
    feeding a state through a critic normalized under a *different* agent's
    statistics would introduce a normalization-mismatch artifact that could
    masquerade as "distortion" having nothing to do with critic pathology.
    """
    if hasattr(agent, "_normalize"):
        return np.asarray(agent._normalize(raw_states))
    return raw_states
