"""Angle 2A Monte Carlo rollout-count (R) calibration.

Replaces the fixed, previously-unvalidated R=15 default for all Angle 2A MC
rollout evaluation (the main diagonal comparison and the pre/post
construct-validity check) with a per-environment, data-derived value: the
smallest R such that the standard error of the MC estimate (sigma_rollout /
sqrt(R)) stays comfortably below the environment's existing Angle 2A null
distribution's 95th-percentile threshold - see research-methodology.md's
Angle 2A section for the full procedure and every design-choice rationale.

No such "null distribution + 95th-percentile threshold" artifact exists
anywhere in Angle 2A prior to this module (only raw per-probe diagonal
errors from optional null-baseline matchups, never aggregated into a
distribution), and using those would create a circularity anyway (R
calibration must run *before* the main comparison, but that's what produces
that data). Resolved by training a dedicated, one-time-per-environment
healthy-vs-healthy reference pair here, decoupled from any specific main-
comparison seed - project decision, confirmed 2026-09-06.

sigma_rollout is measured correctly by construction, not just by convention:
for each of a small number of representative (s,a) pairs, many repeated
rollouts are run from that exact pair, and the standard deviation is taken
WITHIN that one pair's own returns before ever combining anything across
pairs (see calibrate_r's per_pair_sigmas - pooling raw returns across pairs
first would conflate between-state value differences with genuine
rollout-to-rollout noise, inflating the estimate).
"""

import copy
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.agent_runner import (
    ProbeCapture,
    TrainedAgentHandle,
    build_agent_and_env,
    check_single_env_type,
    derive_rng_seed,
    run_training_loop,
    seed_global_rng_for_agent,
)
from experiments.angle_2a.config import RoleArchitecture
from experiments.angle_2a.errors import Angle2AConfigError
from experiments.angle_2a.probes import (
    Probe,
    SOURCE_D,
    compute_diagonal_errors,
    evaluate_both_critics,
    run_monte_carlo_rollouts,
    run_rollouts_for_probe,
    sample_probes,
)
from utils.atomic_io import atomic_write_text

DEFAULT_OUTPUT_ROOT = "results/angle_2a_r_calibration"

NUM_REPRESENTATIVE_PAIRS = 5
NUM_ROLLOUTS_FOR_SIGMA = 50
SE_MARGIN_DIVISOR = 2.0  # target: SE <= threshold_95 / this
R_CAP = 35
# The dedicated null pair's OWN diagonal errors (used only to derive the
# threshold_95 target, not reported as an MC estimate anyone relies on)
# bootstrap with the historical fixed count - calibrating R for the thing
# that defines R's own calibration target would be circular.
NULL_PAIR_BOOTSTRAP_ROLLOUTS = 15
NULL_PAIR_NUM_PROBES_PER_SOURCE = 10
MAX_ROLLOUT_STEPS_MULTIPLIER = 3


def sigma_rollout_from_per_pair_returns(per_pair_returns: List[List[float]]) -> "tuple[float, List[float]]":
    """Standard deviation WITHIN each representative pair's own repeated
    rollouts, computed separately per pair and NEVER pooled together before
    that - pooling raw returns across pairs first would conflate between-
    state value differences with genuine rollout-to-rollout noise, inflating
    the estimate (see module docstring). Combined conservatively via max
    across pairs (sized for the noisiest observed state), not the average."""
    per_pair_sigmas = [float(np.std(returns, ddof=1)) for returns in per_pair_returns]
    return max(per_pair_sigmas), per_pair_sigmas


def solve_calibrated_r(sigma_rollout: float, threshold_95: float, se_margin_divisor: float, r_cap: int) -> "tuple[int, int, bool]":
    """Smallest integer R such that sigma_rollout/sqrt(R) <= threshold_95 /
    se_margin_divisor, capped at r_cap. Returns (calibrated_r, ideal_r,
    underpowered) - underpowered is True iff the cap actually bound (the
    ideal, uncapped R would have exceeded it)."""
    target_se = threshold_95 / se_margin_divisor
    if target_se <= 0:
        raise Angle2AConfigError(
            f"threshold_95={threshold_95} is non-positive - cannot solve for "
            f"a target SE."
        )
    ideal_r = max(1, math.ceil((sigma_rollout / target_se) ** 2))
    calibrated_r = min(ideal_r, r_cap)
    underpowered = ideal_r > r_cap
    return calibrated_r, ideal_r, underpowered


@dataclass
class RCalibrationResult:
    environment: str
    calibrated_r: int
    ideal_r: int
    underpowered: bool
    sigma_rollout: float
    per_pair_sigmas: List[float]
    threshold_95: float
    calibration_seed: int
    stop_step: int


def _cache_path(environment: str, root: str) -> Path:
    return Path(root) / environment / "r_calibration.json"


def load_r_calibration(environment: str, root: str = DEFAULT_OUTPUT_ROOT) -> Optional[RCalibrationResult]:
    path = _cache_path(environment, root)
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    return RCalibrationResult(**payload)


def save_r_calibration(result: RCalibrationResult, root: str = DEFAULT_OUTPUT_ROOT) -> Path:
    path = _cache_path(result.environment, root)
    atomic_write_text(path, json.dumps(asdict(result), indent=2))
    return path


def calibrate_r(
    reference_architecture: RoleArchitecture,
    reference_architecture_label: str,
    base_cfg,
    environment: str,
    calibration_seed: int,
    burn_in_fraction: float,
    num_representative_pairs: int = NUM_REPRESENTATIVE_PAIRS,
    num_rollouts_for_sigma: int = NUM_ROLLOUTS_FOR_SIGMA,
    se_margin_divisor: float = SE_MARGIN_DIVISOR,
    r_cap: int = R_CAP,
    null_pair_num_probes_per_source: int = NULL_PAIR_NUM_PROBES_PER_SOURCE,
    null_pair_bootstrap_rollouts: int = NULL_PAIR_BOOTSTRAP_ROLLOUTS,
) -> RCalibrationResult:
    """Trains a dedicated healthy-vs-healthy reference pair (A, B), one time
    for this environment, to measure both the null-distribution 95th-
    percentile threshold and sigma_rollout - see module docstring."""
    calibration_cfg = copy.deepcopy(base_cfg)
    OmegaConf.set_struct(calibration_cfg, False)
    calibration_cfg.seed = calibration_seed
    OmegaConf.set_struct(calibration_cfg, True)

    check_single_env_type(calibration_cfg)

    num_interaction_steps = int(calibration_cfg.num_interaction_steps)
    stop_step = int(np.floor(burn_in_fraction * num_interaction_steps))
    if stop_step <= 0:
        raise Angle2AConfigError(
            f"R calibration for environment='{environment}': burn_in_fraction "
            f"({burn_in_fraction:.0%}) of num_interaction_steps "
            f"({num_interaction_steps}) rounds down to a non-positive stop "
            f"step; cannot train a calibration pair."
        )

    gamma = float(calibration_cfg.gamma)
    max_rollout_steps = int(calibration_cfg.env.max_episode_steps) * MAX_ROLLOUT_STEPS_MULTIPLIER

    a_train_env, a_eval_env, a_single_env, a_buffer, a_agent, obs_space, act_space = build_agent_and_env(
        reference_architecture, calibration_cfg
    )
    b_train_env, b_eval_env, b_single_env, b_buffer, b_agent, _, _ = build_agent_and_env(
        reference_architecture, calibration_cfg
    )
    try:
        capacity = min(int(calibration_cfg.buffer.max_length), stop_step)
        a_probe_capture = ProbeCapture(
            capacity=capacity, observation_shape=obs_space.shape[-1:], action_shape=act_space.shape[-1:]
        )
        b_probe_capture = ProbeCapture(
            capacity=capacity, observation_shape=obs_space.shape[-1:], action_shape=act_space.shape[-1:]
        )

        seed_global_rng_for_agent(calibration_seed, f"r_calibration:A:{reference_architecture_label}")
        for _ in run_training_loop(a_agent, a_buffer, a_train_env, a_single_env, a_probe_capture, calibration_cfg, stop_step):
            pass
        seed_global_rng_for_agent(calibration_seed, f"r_calibration:B:{reference_architecture_label}")
        for _ in run_training_loop(b_agent, b_buffer, b_train_env, b_single_env, b_probe_capture, calibration_cfg, stop_step):
            pass

        a_handle = TrainedAgentHandle(
            role="D", architecture_label=reference_architecture_label, architecture=reference_architecture,
            agent=a_agent, buffer=a_buffer, train_env=a_train_env, eval_env=a_eval_env,
            single_env=a_single_env, stop_step=stop_step, probe_capture=a_probe_capture,
        )
        b_handle = TrainedAgentHandle(
            role="R", architecture_label=reference_architecture_label, architecture=reference_architecture,
            agent=b_agent, buffer=b_buffer, train_env=b_train_env, eval_env=b_eval_env,
            single_env=b_single_env, stop_step=stop_step, probe_capture=b_probe_capture,
        )

        null_rng = np.random.default_rng(seed=derive_rng_seed(calibration_seed, "r_calibration:null_probes"))
        null_probes = sample_probes("r_calibration_null", a_handle, b_handle, null_pair_num_probes_per_source, null_rng)
        evaluate_both_critics(null_probes, a_handle, b_handle)
        run_monte_carlo_rollouts(null_probes, a_handle, b_handle, null_pair_bootstrap_rollouts, gamma, max_rollout_steps)
        compute_diagonal_errors(null_probes)
        null_diagonal_errors = [p.diagonal_error for p in null_probes]
        threshold_95 = float(np.percentile(null_diagonal_errors, 95))

        # sigma_rollout: within-pair std of repeated rollouts, per
        # representative pair, never pooled across pairs (see module
        # docstring). Conservative: max across pairs, not average - a
        # calibration meant to protect every probe should be sized for the
        # noisiest representative state observed, not a typical one.
        sigma_rng = np.random.default_rng(seed=derive_rng_seed(calibration_seed, "r_calibration:sigma_pairs"))
        pair_idxs, pair_states, pair_actions, pair_env_states = a_handle.probe_capture.sample(
            num_representative_pairs, sigma_rng
        )
        per_pair_returns = []
        for i in range(num_representative_pairs):
            pair_probe = Probe(
                probe_id=f"r_calibration_sigma_pair_{i}", source=SOURCE_D,
                state=pair_states[i], action=pair_actions[i], env_state=pair_env_states[i],
            )
            run_rollouts_for_probe(pair_probe, a_handle, num_rollouts_for_sigma, gamma, max_rollout_steps)
            per_pair_returns.append(pair_probe.mc_rollout_returns)
        sigma_rollout, per_pair_sigmas = sigma_rollout_from_per_pair_returns(per_pair_returns)
    finally:
        a_train_env.close()
        a_eval_env.close()
        b_train_env.close()
        b_eval_env.close()

    try:
        calibrated_r, ideal_r, underpowered = solve_calibrated_r(sigma_rollout, threshold_95, se_margin_divisor, r_cap)
    except Angle2AConfigError as e:
        raise Angle2AConfigError(f"R calibration for environment='{environment}': {e}")

    return RCalibrationResult(
        environment=environment,
        calibrated_r=calibrated_r,
        ideal_r=ideal_r,
        underpowered=underpowered,
        sigma_rollout=sigma_rollout,
        per_pair_sigmas=per_pair_sigmas,
        threshold_95=threshold_95,
        calibration_seed=calibration_seed,
        stop_step=stop_step,
    )


def load_or_calibrate_r(
    reference_architecture: RoleArchitecture,
    reference_architecture_label: str,
    base_cfg,
    environment: str,
    calibration_seed: int,
    burn_in_fraction: float,
    root: str = DEFAULT_OUTPUT_ROOT,
    force_recompute: bool = False,
    **kwargs,
) -> RCalibrationResult:
    if not force_recompute:
        cached = load_r_calibration(environment, root=root)
        if cached is not None:
            return cached

    result = calibrate_r(
        reference_architecture, reference_architecture_label, base_cfg, environment,
        calibration_seed, burn_in_fraction, **kwargs,
    )
    save_r_calibration(result, root=root)

    r_cap = kwargs.get("r_cap", R_CAP)
    se_margin_divisor = kwargs.get("se_margin_divisor", SE_MARGIN_DIVISOR)
    print(
        f"[angle_2a_r_calibration] environment={environment}: sigma_rollout="
        f"{result.sigma_rollout:.4f} threshold_95={result.threshold_95:.4f} "
        f"-> ideal_R={result.ideal_r} calibrated_R={result.calibrated_r}"
    )
    if result.underpowered:
        warnings.warn(
            f"R calibration for environment='{environment}': ideal R="
            f"{result.ideal_r} exceeds the hard cap of {r_cap} - using the "
            f"capped R={result.calibrated_r} instead. This environment's MC "
            f"estimates are UNDERPOWERED relative to the SE <= threshold_95/"
            f"{se_margin_divisor:g} target (sigma_rollout={result.sigma_rollout:.4f}, "
            f"threshold_95={result.threshold_95:.4f}) - flagged in every "
            f"persisted result for this environment; review before trusting "
            f"precision-sensitive conclusions here."
        )
    return result
