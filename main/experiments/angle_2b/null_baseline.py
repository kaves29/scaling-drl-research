"""Empirical healthy-critic null distribution for Angle 2B.

Per research-methodology.md's Null Baseline section, refined for Angle 2B
as follows: select two independently-trained default-architecture (D2W512)
agents A and B (different seeds; neither critic has trained on the other's
actor's data). Holding pi_A fixed, compute:

    g_{A|A} = grad J(pi_A; Q_A)   -- real signal, A's own critic
    g_{A|B} = grad J(pi_A; Q_B)   -- foreign healthy critic swapped in

using the identical actor-held-fixed, critic-swapped structure as the
primary analysis (see gradients.py) - not a raw comparison of the two
critics' outputs. Repeat across multiple independent A/B pairs to build a
real distribution, not a single point estimate.

Source of A/B pairs: Angle 2A's own null-baseline matchups
(null_matchup_1 / null_matchup_2), one independent pair per seed, each
already matched-timestep to that seed's own scaled-architecture onset (see
experiments/angle_2a/matchup.py and experiments/angle_2_a.py). Angle 2A
already trains these agents for its own null baseline; the only change this
project made was to persist their checkpoints (see
experiments/angle_2a/storage.py:save_frozen_agent_snapshot), so Angle 2B
performs zero additional training or environment interaction to obtain
them. The "D"/"R" role labels on a null_matchup are just Angle 2A's
matchup-role bookkeeping - both are equally healthy default-architecture
agents here, relabeled A/B.

LIMITATION - see the End-of-Task Summary for full discussion: this gives at
most one independent A/B pair per seed (up to 5 pairs total per
environment+matchup_name), not exhaustive pairwise combinations across
seeds. Cross-seed pairing was not used because Angle 1's baseline seeds are
each trained to a fixed full-length step budget with no intermediate
snapshots, so two different seeds' agents cannot currently be paired at the
SAME t* without either breaking the matched-timestep requirement this null
exists to enforce, or extending Angle 1's own checkpointing infrastructure -
which was out of the scope authorized for this change (only Angle 2A's
checkpoint-persistence gap was authorized to be closed).
"""

from dataclasses import dataclass, field
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from experiments.angle_2a.storage import DEFAULT_OUTPUT_ROOT, matchup_dir
from experiments.angle_2b.checkpoint_io import apply_agent_normalization, load_frozen_agent_snapshot
from experiments.angle_2b.errors import Angle2BSnapshotError
from experiments.angle_2b.gradients import (
    compute_action_gradient,
    compute_counterfactual_actor_gradients,
    compute_distortion_metrics,
    compute_q_value,
    sample_actor_actions,
)
from experiments.angle_2b.sampling import NUM_STATES_PER_SOURCE, sample_state_batch


@dataclass
class NullPairResult:
    environment: str
    seed: int
    null_matchup_name: str
    d_dir: float
    d_mag: float
    d_grad: float
    # Raw (s,a) and per-action critic gradients at A's own point - added for
    # Angle 2C, which needs nabla_a Q_A/Q_B at the same points this null
    # pair's g_{A|A}/g_{A|B} actor-parameter gradients were taken at. Never
    # written to null_distribution.csv (see storage.py's NULL_PAIR_COLUMNS,
    # unchanged) - these are array-valued, not scalars; only consumed via
    # Angle 2C's own loader off gradients.npz (see matchup_2b.py).
    states: np.ndarray = field(default=None, repr=False)
    actions: np.ndarray = field(default=None, repr=False)
    grad_aq_a_at_a: np.ndarray = field(default=None, repr=False)
    grad_aq_b_at_a: np.ndarray = field(default=None, repr=False)
    q_a_at_a: np.ndarray = field(default=None, repr=False)
    q_b_at_a: np.ndarray = field(default=None, repr=False)


def discover_null_seeds(
    environment: str,
    null_matchup_name: str,
    seeds: List[int],
    root: str = DEFAULT_OUTPUT_ROOT,
) -> List[int]:
    """Returns the subset of `seeds` that have a complete, persisted
    null-baseline snapshot (both roles) for (environment, null_matchup_name)."""
    available = []
    for seed in seeds:
        out_dir = matchup_dir(environment, seed, null_matchup_name, root=root)
        has_role = lambda role: (out_dir / "checkpoints" / role).exists() and (
            out_dir / f"agent_cfg_{role}.json"
        ).exists()
        if has_role("D") and has_role("R"):
            available.append(seed)
    return available


def compute_null_pair_distortion(
    environment: str,
    seed: int,
    null_matchup_name: str,
    analysis_seed: int,
    root: str = DEFAULT_OUTPUT_ROOT,
    num_states_per_source: int = NUM_STATES_PER_SOURCE,
) -> NullPairResult:
    """One null A/B pair's g_{A|A} vs g_{A|B} distortion."""
    snap_a = load_frozen_agent_snapshot(environment, seed, null_matchup_name, "D", root=root)
    snap_b = load_frozen_agent_snapshot(environment, seed, null_matchup_name, "R", root=root)

    if snap_a.critic_use_cdq != snap_b.critic_use_cdq:
        raise Angle2BSnapshotError(
            f"Null pair (environment='{environment}', seed={seed}, "
            f"null_matchup_name='{null_matchup_name}') has mismatched "
            f"critic_use_cdq between A and B ({snap_a.critic_use_cdq} vs "
            f"{snap_b.critic_use_cdq}); both should be the same default "
            f"architecture in the same environment."
        )

    # Own-buffer-only sourcing (see sampling.py): pi_A is the fixed actor
    # here, so the batch is drawn exclusively from A's own probe-capture
    # buffer - B's states are never used for batch construction, only B's
    # critic (as the swapped-in Q). Normalized using A's own obs_rms - see
    # apply_agent_normalization's docstring for why it must be A's, not a
    # mix.
    raw_batch = sample_state_batch(
        snap_a.states,
        seed=analysis_seed,
        context=f"null:{environment}:seed{seed}:{null_matchup_name}",
        num_states_per_source=num_states_per_source,
    )
    batch = jnp.asarray(apply_agent_normalization(snap_a.agent, raw_batch))

    key = jax.random.PRNGKey(analysis_seed)
    grad_aa, grad_ab = compute_counterfactual_actor_gradients(
        key,
        snap_a.agent.actor,
        snap_a.agent.critic,
        snap_b.agent.critic,
        snap_a.agent.temperature,
        batch,
        snap_a.critic_use_cdq,
    )
    metrics = compute_distortion_metrics(grad_aa, grad_ab)

    # Same (s,a) recovery as matchup_2b.py's primary/secondary - see
    # sample_actor_actions's docstring for why this is a deterministic
    # recomputation, not a new sample.
    actions_a = sample_actor_actions(snap_a.agent.actor, batch, key)
    grad_aq_a_at_a = compute_action_gradient(snap_a.agent.critic, batch, actions_a, snap_a.critic_use_cdq)
    grad_aq_b_at_a = compute_action_gradient(snap_b.agent.critic, batch, actions_a, snap_a.critic_use_cdq)
    q_a_at_a = compute_q_value(snap_a.agent.critic, batch, actions_a, snap_a.critic_use_cdq)
    q_b_at_a = compute_q_value(snap_b.agent.critic, batch, actions_a, snap_a.critic_use_cdq)

    return NullPairResult(
        environment=environment,
        seed=seed,
        null_matchup_name=null_matchup_name,
        d_dir=float(metrics["d_dir"]),
        d_mag=float(metrics["d_mag"]),
        d_grad=float(metrics["d_grad"]),
        states=np.asarray(batch),
        actions=np.asarray(actions_a),
        grad_aq_a_at_a=np.asarray(grad_aq_a_at_a),
        grad_aq_b_at_a=np.asarray(grad_aq_b_at_a),
        q_a_at_a=np.asarray(q_a_at_a),
        q_b_at_a=np.asarray(q_b_at_a),
    )


def build_null_distribution(
    environment: str,
    null_matchup_name: str,
    seeds: List[int],
    analysis_seed: int,
    root: str = DEFAULT_OUTPUT_ROOT,
    num_states_per_source: int = NUM_STATES_PER_SOURCE,
) -> List[NullPairResult]:
    available_seeds = discover_null_seeds(environment, null_matchup_name, seeds, root=root)
    if not available_seeds:
        raise Angle2BSnapshotError(
            f"No persisted null-baseline snapshots found for environment="
            f"'{environment}', null_matchup_name='{null_matchup_name}' among "
            f"seeds={seeds}. Angle 2B requires at least one completed Angle "
            f"2A run with run_null_baseline=true for this environment before "
            f"a null distribution can be built; it never trains one itself."
        )
    return [
        compute_null_pair_distortion(
            environment, seed, null_matchup_name, analysis_seed,
            root=root, num_states_per_source=num_states_per_source,
        )
        for seed in available_seeds
    ]
