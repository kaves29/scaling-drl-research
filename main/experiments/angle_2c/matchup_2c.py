"""Orchestrates one Angle 2C analysis end-to-end for a single (environment,
seed, matchup_name):

  load Angle 2B's already-persisted primary/secondary/null-pair (s,a),
    nabla_a Q, Q(s,a), and actor-parameter-gradient arrays (see loader.py) -
    zero new state-action sampling
  load Angle 2A's frozen D/R checkpoints directly (and each null pair's A/B
    checkpoints - see experiments/angle_2b/checkpoint_io.py, reused
    unchanged) - needed only for local instability's fresh perturbed-action
    critic evaluations; zero new training, zero new environment interaction
  compute the three candidate properties (directional corruption,
    magnitude/bias shift, local instability) at PRIMARY's own (s,a) point
    (D's point - the real question, matching Angle 2B's own primary/
    secondary framing) and again at SECONDARY's point as a robustness
    check, and again for every available null A/B pair to build each
    property's null distribution
  compare primary's three properties against their respective null
    distributions via (mean + 2*std) - reusing
    experiments/angle_2b/statistics.py's compare_to_null directly (per
    explicit instruction: "reuse Angle 2B's null infrastructure directly")
  if MORE THAN ONE property exceeds its null simultaneously: run the
    reconstruction test (see reconstruction.py) to determine which of
    direction/magnitude is doing more of the causal work
  if NONE exceed null: report this explicitly as a legitimate, reportable
    non-result (see research-methodology.md's Angle 2C section) - never
    silently pick the closest-looking candidate
  for whichever property is dominant (or the sole diverging one), look up
    Angle 1's onset-timing PROXY (see onset_lookup.py - single frozen point
    only, not a true time-series comparison)
  persist everything (see storage.py)

Scope boundary: Angle 2C explains WHICH property is responsible for the
distortion Angle 2B already established exists; it does not, on its own,
establish a downstream training-outcome effect (that belongs to Angle 3
alone - see research-methodology.md's Angle 2C purpose and Angle 3 scope
boundary).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from experiments.angle_2a.storage import DEFAULT_OUTPUT_ROOT as ANGLE_2A_ROOT
from experiments.angle_2b.checkpoint_io import load_frozen_agent_snapshot
from experiments.angle_2b.gradients import compute_action_gradient
from experiments.angle_2b.statistics import NullComparisonResult, compare_to_null
from experiments.angle_2c.loader import Angle2BArtifacts, load_angle_2b_artifacts
from experiments.angle_2c.onset_lookup import OnsetTimingComparison, compare_onset_timing
from experiments.angle_2c.perturbation import local_instability_variance, perturb_actions
from experiments.angle_2c.properties import (
    directional_corruption,
    directional_corruption_for_null_comparison,
    instability_ratio,
    magnitude_ratio,
    magnitude_ratio_for_null_comparison,
    raw_offset,
)
from experiments.angle_2c.reconstruction import run_reconstruction_test
from experiments.angle_2c.storage import DEFAULT_OUTPUT_ROOT as ANGLE_2C_ROOT
from experiments.angle_2c.storage import save_angle_2c_result

PROPERTY_NAMES = ("direction", "magnitude", "instability")


@dataclass
class Angle2CResult:
    environment: str
    seed: int
    matchup_name: str
    primary_properties: Dict[str, float]
    secondary_properties: Dict[str, float]
    null_comparison: Dict[str, NullComparisonResult]
    diverging_properties: List[str]
    dominant_property: Optional[str]
    reconstruction: Optional[Dict[str, Dict[str, float]]]
    onset_timing: Optional[OnsetTimingComparison]
    non_result: bool
    run_metadata: Dict[str, Any] = field(default_factory=dict)
    output_paths: Dict[str, Any] = field(default_factory=dict)


def _make_grad_aq_evaluator(critic, states: np.ndarray, critic_use_cdq: bool) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a callable perturbed_actions(K,N,act_dim) -> nabla_a Q(K,N,act_dim),
    evaluating `critic` at the FIXED, already-normalized `states` (from
    Angle 2B's persisted output) tiled to match each of the K perturbation
    copies. See perturbation.py's module docstring for why this counts as
    new computation but not new sampling."""
    states_jnp = jnp.asarray(states)
    n = states.shape[0]

    def evaluator(perturbed_actions_kxnxd: np.ndarray) -> np.ndarray:
        k = perturbed_actions_kxnxd.shape[0]
        states_tiled = jnp.tile(states_jnp, (k, 1))
        actions_flat = jnp.asarray(perturbed_actions_kxnxd.reshape(k * n, -1))
        grad_flat = compute_action_gradient(critic, states_tiled, actions_flat, critic_use_cdq)
        return np.asarray(grad_flat).reshape(k, n, -1)

    return evaluator


def _compute_properties_at_point(
    grad_aq_self: np.ndarray,
    grad_aq_other: np.ndarray,
    q_self: np.ndarray,
    q_other: np.ndarray,
    var_self: np.ndarray,
    var_other: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-(s,a) property arrays at one fixed point - "self" plays the role
    of the fixed actor's own critic (D at D's point, A at A's point, etc.),
    "other" the swapped-in one."""
    cos_sim = directional_corruption(grad_aq_self, grad_aq_other)
    ratio = magnitude_ratio(grad_aq_self, grad_aq_other)
    return {
        "direction_raw": cos_sim,
        "direction_for_null": directional_corruption_for_null_comparison(cos_sim),
        "magnitude_raw": ratio,
        "magnitude_for_null": magnitude_ratio_for_null_comparison(ratio),
        "raw_offset": raw_offset(q_self, q_other),
        "instability_ratio": instability_ratio(var_self, var_other),
    }


def run_angle_2c_analysis(
    environment: str,
    seed: int,
    matchup_name: str,
    num_perturbations: int,
    perturbation_sigma: float,
    analysis_seed: int,
    onset_source_experiment: str,
    onset_ledger_root: str,
    angle_2a_root: str = ANGLE_2A_ROOT,
    angle_2b_root: str = "results/angle_2b",
    output_root: str = ANGLE_2C_ROOT,
) -> Angle2CResult:
    artifacts: Angle2BArtifacts = load_angle_2b_artifacts(environment, seed, matchup_name, root=angle_2b_root)
    critic_use_cdq = artifacts.critic_use_cdq

    snap_d = load_frozen_agent_snapshot(environment, seed, matchup_name, "D", root=angle_2a_root)
    snap_r = load_frozen_agent_snapshot(environment, seed, matchup_name, "R", root=angle_2a_root)

    # --- Primary: at D's own (s,a) point ---
    perturbed_d = perturb_actions(artifacts.actions_d, num_perturbations, perturbation_sigma, seed=analysis_seed)
    var_d_at_d = local_instability_variance(
        _make_grad_aq_evaluator(snap_d.agent.critic, artifacts.states_d, critic_use_cdq), perturbed_d,
    )
    var_r_at_d = local_instability_variance(
        _make_grad_aq_evaluator(snap_r.agent.critic, artifacts.states_d, critic_use_cdq), perturbed_d,
    )
    primary_arrays = _compute_properties_at_point(
        artifacts.grad_aq_d_at_d, artifacts.grad_aq_r_at_d,
        artifacts.q_d_at_d, artifacts.q_r_at_d,
        var_d_at_d, var_r_at_d,
    )
    primary_properties = {k: float(np.mean(v)) for k, v in primary_arrays.items()}

    # --- Secondary: at R's own (s,a) point - diagnostic robustness check only ---
    perturbed_r = perturb_actions(artifacts.actions_r, num_perturbations, perturbation_sigma, seed=analysis_seed)
    var_r_at_r = local_instability_variance(
        _make_grad_aq_evaluator(snap_r.agent.critic, artifacts.states_r, critic_use_cdq), perturbed_r,
    )
    var_d_at_r = local_instability_variance(
        _make_grad_aq_evaluator(snap_d.agent.critic, artifacts.states_r, critic_use_cdq), perturbed_r,
    )
    secondary_arrays = _compute_properties_at_point(
        artifacts.grad_aq_r_at_r, artifacts.grad_aq_d_at_r,
        artifacts.q_r_at_r, artifacts.q_d_at_r,
        var_r_at_r, var_d_at_r,
    )
    secondary_properties = {k: float(np.mean(v)) for k, v in secondary_arrays.items()}

    # --- Null: every available A/B pair, at A's own point ---
    null_matchup_name = f"null_{matchup_name}"
    null_pair_rows = []
    null_direction, null_magnitude, null_offset, null_instability = [], [], [], []
    for ns, pair in sorted(artifacts.null_pairs.items()):
        snap_a = load_frozen_agent_snapshot(environment, ns, null_matchup_name, "D", root=angle_2a_root)
        snap_b = load_frozen_agent_snapshot(environment, ns, null_matchup_name, "R", root=angle_2a_root)
        perturbed_null = perturb_actions(pair.actions, num_perturbations, perturbation_sigma, seed=analysis_seed)
        var_a = local_instability_variance(
            _make_grad_aq_evaluator(snap_a.agent.critic, pair.states, critic_use_cdq), perturbed_null,
        )
        var_b = local_instability_variance(
            _make_grad_aq_evaluator(snap_b.agent.critic, pair.states, critic_use_cdq), perturbed_null,
        )
        pair_arrays = _compute_properties_at_point(
            pair.grad_aq_a_at_a, pair.grad_aq_b_at_a, pair.q_a_at_a, pair.q_b_at_a, var_a, var_b,
        )
        pair_means = {k: float(np.mean(v)) for k, v in pair_arrays.items()}
        null_direction.append(pair_means["direction_for_null"])
        null_magnitude.append(pair_means["magnitude_for_null"])
        null_offset.append(pair_means["raw_offset"])
        null_instability.append(pair_means["instability_ratio"])
        null_pair_rows.append({
            "environment": environment, "seed": ns, "null_matchup_name": null_matchup_name,
            "direction_for_null": pair_means["direction_for_null"],
            "magnitude_for_null": pair_means["magnitude_for_null"],
            "raw_offset": pair_means["raw_offset"],
            "instability_ratio": pair_means["instability_ratio"],
        })

    null_comparison = {
        "direction": compare_to_null("direction", primary_properties["direction_for_null"], null_direction),
        "magnitude": compare_to_null("magnitude", primary_properties["magnitude_for_null"], null_magnitude),
        "instability": compare_to_null("instability", primary_properties["instability_ratio"], null_instability),
    }

    # --- Co-occurrence / dominance determination ---
    diverging_properties = [name for name in PROPERTY_NAMES if null_comparison[name].exceeds_null]
    non_result = len(diverging_properties) == 0
    dominant_property: Optional[str] = None
    reconstruction: Optional[Dict[str, Dict[str, float]]] = None

    if non_result:
        # Legitimate, reportable outcome per research-methodology.md's Angle
        # 2C section - never silently pick the closest-looking candidate.
        pass
    elif len(diverging_properties) == 1:
        dominant_property = diverging_properties[0]
    else:
        # Reconstruction test only ever adjudicates direction vs magnitude
        # (per research-methodology.md's own co-occurrence procedure) - if
        # instability co-occurs alongside them, that is reported as a
        # caveat, not resolved by this test.
        reconstruction = run_reconstruction_test(
            actor=snap_d.agent.actor,
            temperature=snap_d.agent.temperature,
            observations=jnp.asarray(artifacts.states_d),
            key=jax.random.PRNGKey(analysis_seed),
            grad_aq_d_at_d=jnp.asarray(artifacts.grad_aq_d_at_d),
            grad_aq_r_at_d=jnp.asarray(artifacts.grad_aq_r_at_d),
            target_g_d_given_r=jnp.asarray(artifacts.g_d_given_r),
        )
        d_dir_r_mag_score = reconstruction["d_direction_r_magnitude"]["cosine_similarity_to_target"]
        r_dir_d_mag_score = reconstruction["r_direction_d_magnitude"]["cosine_similarity_to_target"]
        dominant_property = "direction" if r_dir_d_mag_score > d_dir_r_mag_score else "magnitude"
        # r_dir_d_mag closer to target => keeping D's magnitude but taking
        # R's direction reproduces the real distortion better => direction
        # is doing more of the causal work (and vice versa).
        #
        # NOTE (scope limitation inherited directly from
        # research-methodology.md's own design, not introduced here): the
        # reconstruction test only ever adjudicates direction vs magnitude,
        # structurally, regardless of which properties actually co-occurred.
        # If e.g. magnitude+instability diverged (not direction), this can
        # still name "direction" dominant even though direction itself never
        # exceeded its own null - exceeds_null_at_t_star below reflects that
        # distinction honestly rather than assuming the reconstruction
        # winner always already diverged.

    onset_timing: Optional[OnsetTimingComparison] = None
    if dominant_property is not None:
        onset_timing = compare_onset_timing(
            dominant_property=dominant_property,
            exceeds_null_at_t_star=dominant_property in diverging_properties,
            t_star=_lookup_t_star(environment, seed, matchup_name, angle_2a_root),
            environment=environment,
            seed=seed,
            matchup_name=matchup_name,
            onset_source_experiment=onset_source_experiment,
            onset_ledger_root=onset_ledger_root,
            angle_2a_root=angle_2a_root,
        )

    run_metadata = {
        "matchup_name": matchup_name,
        "null_matchup_name": null_matchup_name,
        "num_perturbations": num_perturbations,
        "perturbation_sigma": perturbation_sigma,
        "analysis_seed": analysis_seed,
        "critic_use_cdq": critic_use_cdq,
        "primary": primary_properties,
        "secondary": secondary_properties,
        "null_comparison": {
            name: {
                "observed_value": r.observed_value, "null_mean": r.null_mean, "null_std": r.null_std,
                "null_n": r.null_n, "threshold": r.threshold, "exceeds_null": r.exceeds_null,
            }
            for name, r in null_comparison.items()
        },
        "diverging_properties": diverging_properties,
        "non_result": non_result,
        "dominant_property": dominant_property,
        "reconstruction": reconstruction,
        "onset_timing": (
            {
                "dominant_property": onset_timing.dominant_property,
                "exceeds_null_at_t_star": onset_timing.exceeds_null_at_t_star,
                "t_star": onset_timing.t_star,
                "angle1_degradation_onset_step": onset_timing.angle1_degradation_onset_step,
                "consistent": onset_timing.consistent,
                "note": onset_timing.note,
            }
            if onset_timing is not None else None
        ),
        "scope_note": (
            "Angle 2C explains WHICH property is responsible for the "
            "distortion Angle 2B already established - it does not, on its "
            "own, establish a downstream training-outcome effect; that "
            "causal claim belongs to Angle 3 alone."
        ),
        "non_result_note": (
            "None of the three candidate properties cleanly separated from "
            "its null despite Angle 2B confirming a real distortion exists. "
            "This is reported as a legitimate finding: the corruption is "
            "not well explained by these three interpretable properties, "
            "not papered over by selecting the closest-looking candidate."
            if non_result else None
        ),
    }

    property_arrays = {
        "primary_direction_raw": primary_arrays["direction_raw"],
        "primary_magnitude_raw": primary_arrays["magnitude_raw"],
        "primary_raw_offset": primary_arrays["raw_offset"],
        "primary_instability_ratio": primary_arrays["instability_ratio"],
        "secondary_direction_raw": secondary_arrays["direction_raw"],
        "secondary_magnitude_raw": secondary_arrays["magnitude_raw"],
        "secondary_raw_offset": secondary_arrays["raw_offset"],
        "secondary_instability_ratio": secondary_arrays["instability_ratio"],
    }

    output_paths = save_angle_2c_result(
        environment, seed, matchup_name, run_metadata, null_pair_rows, property_arrays, root=output_root,
    )

    return Angle2CResult(
        environment=environment,
        seed=seed,
        matchup_name=matchup_name,
        primary_properties=primary_properties,
        secondary_properties=secondary_properties,
        null_comparison=null_comparison,
        diverging_properties=diverging_properties,
        dominant_property=dominant_property,
        reconstruction=reconstruction,
        onset_timing=onset_timing,
        non_result=non_result,
        run_metadata=run_metadata,
        output_paths=output_paths,
    )


def _lookup_t_star(environment: str, seed: int, matchup_name: str, angle_2a_root: str) -> int:
    """t* = the onset_step Angle 2A trained this matchup's D to (same value
    as Angle 1's own logged onset - see onset_lookup.py's honesty note)."""
    metadata_path = Path(angle_2a_root) / environment / f"seed{seed}" / matchup_name / "run_metadata.json"
    with open(metadata_path) as f:
        return int(json.load(f)["scaled_onset_step"])
