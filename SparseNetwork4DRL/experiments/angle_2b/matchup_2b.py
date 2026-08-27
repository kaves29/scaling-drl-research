"""Orchestrates one Angle 2B analysis end-to-end for a single (environment,
seed, matchup_name):

  load frozen pi_D/Q_D and pi_R/Q_R snapshots (zero training/env interaction
    - see checkpoint_io.py)
  build one shared 60-state batch (30 D-source + 30 R-source - see
    sampling.py), reused for every gradient computation below so primary and
    secondary are measured on identical inputs and are directly comparable
    side by side
  primary:   g_{D|D} vs g_{D|R}, holding pi_D fixed (the real question:
    does the degraded critic distort the actor's real training signal?)
  secondary: g_{R|R} vs g_{R|D}, holding pi_R fixed - a diagnostic
    robustness check only, never averaged with the primary result (see
    research-methodology.md's Angle 2B section)
  null:      g_{A|A} vs g_{A|B} across every available matched-timestep
    healthy pair for this (environment, matchup_name) - see null_baseline.py
  compare primary's three distortion metrics against the null distribution
    via (null_mean + 2*null_std) - the same criterion used throughout this
    study, never a new threshold invented for this step (see statistics.py)
  persist everything (see storage.py)

Scope boundary (see research-methodology.md's Angle 2B section): this
establishes only that the degraded critic generates an unusually altered
actor-facing optimization signal at a single frozen point, relative to the
healthy-critic null. It does NOT establish a downstream training-outcome
effect - that causal claim belongs to Angle 3 alone. Nothing here continues
training, forks a branch, or takes more than one step.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import jax
import jax.numpy as jnp

from experiments.angle_2a.storage import DEFAULT_OUTPUT_ROOT as ANGLE_2A_ROOT
from experiments.angle_2b.checkpoint_io import apply_agent_normalization, load_frozen_agent_snapshot
from experiments.angle_2b.gradients import compute_counterfactual_actor_gradients, compute_distortion_metrics
from experiments.angle_2b.null_baseline import NullPairResult, build_null_distribution
from experiments.angle_2b.sampling import NUM_STATES_PER_SOURCE, sample_state_batch
from experiments.angle_2b.statistics import NullComparisonResult, compare_to_null
from experiments.angle_2b.storage import DEFAULT_OUTPUT_ROOT as ANGLE_2B_ROOT
from experiments.angle_2b.storage import save_angle_2b_result

METRIC_NAMES = ("d_dir", "d_mag", "d_grad")


@dataclass
class Angle2BResult:
    environment: str
    seed: int
    matchup_name: str
    null_matchup_name: str
    primary: Dict[str, float]
    secondary: Dict[str, float]
    null_pairs: List[NullPairResult]
    null_comparison: Dict[str, NullComparisonResult]
    run_metadata: Dict[str, Any] = field(default_factory=dict)
    output_paths: Dict[str, Any] = field(default_factory=dict)


def _metrics_to_float(metrics: Dict[str, jnp.ndarray]) -> Dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


def run_angle_2b_analysis(
    environment: str,
    seed: int,
    matchup_name: str,
    null_seeds: List[int],
    analysis_seed: int,
    num_states_per_source: int = NUM_STATES_PER_SOURCE,
    angle_2a_root: str = ANGLE_2A_ROOT,
    output_root: str = ANGLE_2B_ROOT,
) -> Angle2BResult:
    null_matchup_name = f"null_{matchup_name}"

    snap_d = load_frozen_agent_snapshot(environment, seed, matchup_name, "D", root=angle_2a_root)
    snap_r = load_frozen_agent_snapshot(environment, seed, matchup_name, "R", root=angle_2a_root)

    if snap_d.critic_use_cdq != snap_r.critic_use_cdq:
        raise ValueError(
            f"D and R snapshots for (environment='{environment}', seed={seed}, "
            f"matchup_name='{matchup_name}') disagree on critic_use_cdq "
            f"({snap_d.critic_use_cdq} vs {snap_r.critic_use_cdq}); both come "
            f"from the same env config and should agree."
        )
    critic_use_cdq = snap_d.critic_use_cdq

    # One shared batch of RAW states (30 D-sourced + 30 R-sourced) reused,
    # unchanged, for both primary and secondary below - so the two probe
    # the same underlying scenes and are directly comparable side by side.
    # Normalization is applied separately per analysis (see
    # apply_agent_normalization's docstring): each analysis's fixed actor
    # must see its OWN normalization applied uniformly across the whole
    # batch, not a per-source mix, since gradients.py's _actor_loss feeds
    # one shared `observations` array to both the actor and whichever
    # critic is swapped in.
    raw_batch = sample_state_batch(
        snap_d.states,
        snap_r.states,
        seed=analysis_seed,
        context=f"primary:{environment}:seed{seed}:{matchup_name}",
        num_states_per_source=num_states_per_source,
    )
    key = jax.random.PRNGKey(analysis_seed)

    # Primary: hold pi_D fixed, swap Q_D <-> Q_R. Batch normalized under D's
    # own obs_rms, since pi_D is the fixed actor consuming it.
    batch_d = jnp.asarray(apply_agent_normalization(snap_d.agent, raw_batch))
    g_dd, g_dr = compute_counterfactual_actor_gradients(
        key, snap_d.agent.actor, snap_d.agent.critic, snap_r.agent.critic,
        snap_d.agent.temperature, batch_d, critic_use_cdq,
    )
    primary = _metrics_to_float(compute_distortion_metrics(g_dd, g_dr))

    # Secondary (diagnostic robustness check only - see module docstring):
    # hold pi_R fixed, swap Q_R <-> Q_D. Same raw scenes as primary, but
    # normalized under R's own obs_rms, since pi_R is the fixed actor here.
    batch_r = jnp.asarray(apply_agent_normalization(snap_r.agent, raw_batch))
    g_rr, g_rd = compute_counterfactual_actor_gradients(
        key, snap_r.agent.actor, snap_r.agent.critic, snap_d.agent.critic,
        snap_r.agent.temperature, batch_r, critic_use_cdq,
    )
    secondary = _metrics_to_float(compute_distortion_metrics(g_rr, g_rd))

    null_pairs = build_null_distribution(
        environment, null_matchup_name, null_seeds, analysis_seed,
        root=angle_2a_root, num_states_per_source=num_states_per_source,
    )

    null_comparison = {
        metric: compare_to_null(metric, primary[metric], [getattr(p, metric) for p in null_pairs])
        for metric in METRIC_NAMES
    }

    run_metadata = {
        "matchup_name": matchup_name,
        "null_matchup_name": null_matchup_name,
        "num_states_per_source": num_states_per_source,
        "analysis_seed": analysis_seed,
        "null_seeds_requested": null_seeds,
        "null_seeds_used": [p.seed for p in null_pairs],
        "critic_use_cdq": critic_use_cdq,
        "primary": primary,
        "secondary": secondary,
        "null_comparison": {
            metric: {
                "observed_value": r.observed_value,
                "null_mean": r.null_mean,
                "null_std": r.null_std,
                "null_n": r.null_n,
                "threshold": r.threshold,
                "exceeds_null": r.exceeds_null,
            }
            for metric, r in null_comparison.items()
        },
        "scope_note": (
            "Primary vs secondary are reported side by side and never "
            "combined into a single aggregate statistic. This analysis "
            "establishes only that the degraded critic distorts pi_D's "
            "actor-facing gradient beyond the healthy-critic null at this "
            "single frozen point - it does NOT establish a downstream "
            "training-outcome effect (see Angle 3)."
        ),
    }

    # Flatten each gradient pytree into one vector for storage (mirrors
    # gradients.py's _flatten, kept local here to avoid a private import).
    def _flatten(pytree):
        leaves, _ = jax.tree_util.tree_flatten(pytree)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

    gradients = {
        "g_D_given_D": _flatten(g_dd),
        "g_D_given_R": _flatten(g_dr),
        "g_R_given_R": _flatten(g_rr),
        "g_R_given_D": _flatten(g_rd),
    }

    output_paths = save_angle_2b_result(
        environment, seed, matchup_name, run_metadata, null_pairs, gradients, root=output_root,
    )

    return Angle2BResult(
        environment=environment,
        seed=seed,
        matchup_name=matchup_name,
        null_matchup_name=null_matchup_name,
        primary=primary,
        secondary=secondary,
        null_pairs=null_pairs,
        null_comparison=null_comparison,
        run_metadata=run_metadata,
        output_paths=output_paths,
    )
