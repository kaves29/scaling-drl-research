"""Probe sampling, cross-critic Q evaluation, and exact-state Monte Carlo
return estimation for one Angle 2A matchup.

Kept free of any "which matchup is this" framing - it only knows about a
`TrainedAgentHandle` for D and one for R, so it is exactly reused, unchanged,
for Matchup 1, Matchup 2, and the null baseline (see matchup.py).
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.angle_2a.agent_runner import TrainedAgentHandle
from experiments.angle_2a.env_state import restore_env_state
from utils.atomic_io import atomic_write_bytes

SOURCE_D = "D"
SOURCE_R = "R"


@dataclass
class Probe:
    probe_id: str
    source: str  # "D" or "R"
    state: np.ndarray
    action: np.ndarray
    # captured dm_control state this probe's (state, action) was drawn from;
    # internal-only, never persisted to the tabular probe index (see
    # storage.py) since it is neither compact nor human-meaningful.
    env_state: Dict[str, Any] = field(repr=False, default=None)
    q_d: Optional[float] = None
    q_r: Optional[float] = None
    mc_rollout_returns: List[float] = field(default_factory=list)
    mc_return: Optional[float] = None
    mc_return_se: Optional[float] = None  # sigma_rollout_sample / sqrt(R) - see r_calibration.py
    diagonal_error: Optional[float] = None

    @property
    def q_source(self) -> Optional[float]:
        return self.q_d if self.source == SOURCE_D else self.q_r

    @property
    def q_other(self) -> Optional[float]:
        return self.q_r if self.source == SOURCE_D else self.q_d


def sample_probes(
    matchup_name: str,
    D: TrainedAgentHandle,
    R: TrainedAgentHandle,
    num_probes_per_source: int,
    rng: np.random.Generator,
) -> List[Probe]:
    """Samples `num_probes_per_source` transitions from D's OWN buffer and
    `num_probes_per_source` from R's OWN buffer. Each probe is permanently tagged with its source."""
    probes: List[Probe] = []

    d_idxs, d_states, d_actions, d_env_states = D.probe_capture.sample(num_probes_per_source, rng)
    for i, idx in enumerate(d_idxs):
        probes.append(
            Probe(
                probe_id=f"{matchup_name}_D_{int(idx)}",
                source=SOURCE_D,
                state=d_states[i],
                action=d_actions[i],
                env_state=d_env_states[i],
            )
        )

    r_idxs, r_states, r_actions, r_env_states = R.probe_capture.sample(num_probes_per_source, rng)
    for i, idx in enumerate(r_idxs):
        probes.append(
            Probe(
                probe_id=f"{matchup_name}_R_{int(idx)}",
                source=SOURCE_R,
                state=r_states[i],
                action=r_actions[i],
                env_state=r_env_states[i],
            )
        )

    return probes


def sample_single_source_probes(
    probe_id_prefix: str,
    probe_capture,
    num_probes: int,
    rng: np.random.Generator,
) -> List[Probe]:
    """Like sample_probes, but for a single agent with no counterpart to pair
    against (see experiments/angle_2a/prereq_check.py) - every probe is
    tagged source=SOURCE_D so evaluate_both_critics/run_monte_carlo_rollouts/
    compute_diagonal_errors work unmodified when passed the same
    TrainedAgentHandle for both their D and R parameters."""
    idxs, states, actions, env_states = probe_capture.sample(num_probes, rng)
    return [
        Probe(
            probe_id=f"{probe_id_prefix}_{int(idx)}",
            source=SOURCE_D,
            state=states[i],
            action=actions[i],
            env_state=env_states[i],
        )
        for i, idx in enumerate(idxs)
    ]


def evaluate_both_critics(probes: List[Probe], D: TrainedAgentHandle, R: TrainedAgentHandle) -> None:
    """Evaluates BOTH agents' critics on EVERY probe (diagonal + off-diagonal)."""
    if not probes:
        return

    states = np.stack([p.state for p in probes])
    actions = np.stack([p.action for p in probes])

    q_d_values = D.agent.get_q_value(states, actions)
    q_r_values = R.agent.get_q_value(states, actions)

    for probe, q_d, q_r in zip(probes, q_d_values, q_r_values):
        probe.q_d = float(q_d)
        probe.q_r = float(q_r)


def _load_mc_progress(checkpoint_path: Optional[str]) -> Dict[str, List[float]]:
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return {}
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def _save_mc_progress(checkpoint_path: str, progress: Dict[str, List[float]]) -> None:
    atomic_write_bytes(checkpoint_path, pickle.dumps(progress, protocol=pickle.HIGHEST_PROTOCOL))


def run_monte_carlo_rollouts(
    probes: List[Probe],
    D: TrainedAgentHandle,
    R: TrainedAgentHandle,
    num_rollouts: int,
    gamma: float,
    max_rollout_steps: int,
    checkpoint_path: Optional[str] = None,
) -> None:
    """For every probe, runs exactly `num_rollouts` independent rollouts:
    reset the probe's SOURCE agent's own environment to the probe's exact
    captured state, force the probe's action, then continue with the SOURCE
    agent's *stochastic* actor only (never the other agent's actor, and
    never its deterministic mean action - see run_rollouts_for_probe) until
    termination/truncation. Fills in probe.mc_rollout_returns and
    probe.mc_return (their mean) with the discounted return gamma^t * r_t
    summed to episode end - matching what a Q-function estimates, not the
    undiscounted return evaluation.py reports for benchmarking.

    checkpoint_path (optional): per-probe-id progress (a plain
    {probe_id: [completed returns so far]} dict) is persisted after every
    single rollout, not just every probe - resuming with the same
    checkpoint_path picks back up from the exact interrupted rollout for a
    partially-done probe, and skips re-running any already-fully-done probe
    entirely, rather than restarting a probe's full run or this whole phase.
    """
    progress = _load_mc_progress(checkpoint_path)
    for probe in probes:
        agent_handle = D if probe.source == SOURCE_D else R
        run_rollouts_for_probe(
            probe, agent_handle, num_rollouts, gamma, max_rollout_steps,
            already_completed_returns=progress.get(probe.probe_id),
            checkpoint_path=checkpoint_path,
            progress=progress,
        )


def run_rollouts_for_probe(
    probe: Probe,
    agent_handle: TrainedAgentHandle,
    num_rollouts: int,
    gamma: float,
    max_rollout_steps: int,
    already_completed_returns: Optional[List[float]] = None,
    checkpoint_path: Optional[str] = None,
    progress: Optional[Dict[str, List[float]]] = None,
) -> None:
    env = agent_handle.single_env
    agent = agent_handle.agent
    returns = list(already_completed_returns) if already_completed_returns else []
    num_remaining = num_rollouts - len(returns)

    for _ in range(num_remaining):
        restore_env_state(env, probe.env_state)
        action = np.asarray(probe.action)
        total_return = 0.0
        discount = 1.0

        for _step in range(max_rollout_steps):
            next_obs, reward, terminated, truncated, _info = env.step(action)
            total_return += discount * float(reward)
            discount *= gamma

            if terminated or truncated:
                break

            prev_timestep = {"next_observation": np.asarray(next_obs)[None, :]}
            # training=True (temperature=1, not 0): Q_MC^pi denotes the
            # return under the actual stochastic trained policy pi, not its
            # deterministic mean action - temperature=0 makes the policy's
            # scale_diag exactly zero (scale_rl/networks/policies.py),
            # making dm_control's otherwise-deterministic dynamics produce
            # bit-identical returns across repeated rollouts (confirmed
            # directly: sigma=0.0 exactly) - i.e. no genuine MC estimate at
            # all. Fixed 2026-09-06; see research-methodology.md's Angle 2A
            # section for the full history of this correction.
            action = np.asarray(
                agent.sample_actions(interaction_step=0, prev_timestep=prev_timestep, training=True)
            )[0]

        returns.append(total_return)

        if checkpoint_path is not None and progress is not None:
            progress[probe.probe_id] = list(returns)
            _save_mc_progress(checkpoint_path, progress)

    probe.mc_rollout_returns = returns
    probe.mc_return = float(np.mean(returns))
    # ddof=1: sample std (Bessel's correction), consistent with
    # r_calibration.py's sigma_rollout estimate. Requires R>=2 - R=1 is
    # never valid here (see r_calibration.py's calibrated_r floor).
    probe.mc_return_se = float(np.std(returns, ddof=1) / np.sqrt(len(returns))) if len(returns) > 1 else None


def mean_and_se(values: List[float]) -> "tuple[float, Optional[float]]":
    """SE here is the standard error of the mean *across* the given values
    (e.g. across probes, or across pre/post-checkpoint probes) - a
    different, additional source of uncertainty from any one value's own
    mc_return_se (which reflects only that single estimate's own R-rollout
    MC noise). See research-methodology.md's Angle 2A section."""
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return mean, se


def compute_diagonal_errors(probes: List[Probe]) -> None:
    """E_D = |Q_D(s,a) - MC^pi_D(s,a)| for D-source probes,
    E_R = |Q_R(s,a) - MC^pi_R(s,a)| for R-source probes.
    Never crosses the actor/critic pairing."""
    for probe in probes:
        if probe.q_source is None or probe.mc_return is None:
            raise ValueError(
                f"Probe {probe.probe_id} is missing q_source or mc_return; "
                f"evaluate_both_critics() and run_monte_carlo_rollouts() must "
                f"both run before compute_diagonal_errors()."
            )
        probe.diagonal_error = abs(probe.q_source - probe.mc_return)
