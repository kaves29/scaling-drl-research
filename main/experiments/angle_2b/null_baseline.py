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

Source of A/B pairs (refactored 2026-09-08): the shared baseline-calibration
pool (see analysis/baseline_calibration_pool.py) - 10 independent default-
architecture agents per environment (Angle 1's own 5 locked baseline seeds
plus 5 dedicated calibration-pool seeds), all C(10,2)=45 unique pairs used,
not just a disjoint subset. This replaces the previous design, which sourced
A/B pairs from Angle 2A's own null_matchup snapshots (at most one pair per
Angle 2A seed, <=5 total) - Angle 2A no longer trains its own null_matchup
pairs at all (see experiments/angle_2a/pool_null_baseline.py), so this
module now depends on the shared pool directly, not on Angle 2A's null-
baseline output.
"""

from dataclasses import dataclass, field
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from analysis.baseline_calibration_pool import (
    POOL_MATCHUP_NAME,
    POOL_ROLE,
    all_unique_pairs,
    get_baseline_calibration_pool,
)
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
    seed_a: int
    seed_b: int
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


def compute_null_pair_distortion(
    environment: str,
    seed_a: int,
    seed_b: int,
    analysis_seed: int,
    root: str,
    num_states_per_source: int = NUM_STATES_PER_SOURCE,
) -> NullPairResult:
    """One null A/B pair's g_{A|A} vs g_{A|B} distortion, both drawn from
    the shared calibration pool (role="pool", matchup_name="baseline_pool" -
    see analysis/baseline_calibration_pool.py and
    experiments/angle_2a/pool_null_baseline.py)."""
    snap_a = load_frozen_agent_snapshot(environment, seed_a, POOL_MATCHUP_NAME, POOL_ROLE, root=root)
    snap_b = load_frozen_agent_snapshot(environment, seed_b, POOL_MATCHUP_NAME, POOL_ROLE, root=root)

    if snap_a.critic_use_cdq != snap_b.critic_use_cdq:
        raise Angle2BSnapshotError(
            f"Pool pair (environment='{environment}', seed_a={seed_a}, "
            f"seed_b={seed_b}) has mismatched critic_use_cdq between A and B "
            f"({snap_a.critic_use_cdq} vs {snap_b.critic_use_cdq}); both "
            f"should be the same default architecture in the same environment."
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
        context=f"null:{environment}:seed{seed_a}_seed{seed_b}",
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
        seed_a=seed_a,
        seed_b=seed_b,
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
    analysis_seed: int,
    root: str,
    num_states_per_source: int = NUM_STATES_PER_SOURCE,
) -> List[NullPairResult]:
    """All C(10,2)=45 unique pairs among the shared pool's 10 agents for
    `environment` (see module docstring). Unlike the pre-2026-09-08 design,
    this no longer takes a null_matchup_name or a caller-supplied seed list -
    the pool composition is fixed, shared infrastructure (see
    analysis/baseline_calibration_pool.get_baseline_calibration_pool)."""
    identities = get_baseline_calibration_pool(environment)
    seed_pairs = all_unique_pairs(sorted(ident.seed for ident in identities))
    if not seed_pairs:
        raise Angle2BSnapshotError(
            f"No baseline-calibration-pool seeds configured for environment="
            f"'{environment}'; cannot build a null distribution."
        )
    return [
        compute_null_pair_distortion(
            environment, seed_a, seed_b, analysis_seed, root=root, num_states_per_source=num_states_per_source,
        )
        for seed_a, seed_b in seed_pairs
    ]
