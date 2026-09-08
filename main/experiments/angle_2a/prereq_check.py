"""Angle 2A construct-validity prerequisite check.

Runs BEFORE the main Angle 2A diagonal comparison, on 3 dedicated seeds never
reused in (or drawn from) the main 5-seed comparison. Purpose: verify that
the td_error_var-based degradation threshold used to define t* corresponds
to an independently-measured decline in functional accuracy, via a different
method (MC-validated diagonal error) than the one that defined t* - see
research-methodology.md's Angle 2A section for the full procedure and why
seed independence matters here (avoiding any appearance of cherry-picking
seeds after seeing main-analysis results).

Per seed, per scaled architecture: train one fresh D agent, save a real
checkpoint at the burn-in cutoff already used for ACF/CCF window calibration
(configs/base_angle2a.yaml's angle_2_a.prereq_check.burn_in_fraction - should
match configs/base_sac.yaml's onset_detection.acf_ccf_burn_in_fraction, not
auto-synced), evaluate E_D there ("pre"), continue training to that seed's
own t* (from the onset ledger, same lookup the main analysis uses), save
another checkpoint and evaluate E_D again ("post") - with entirely
independent, freshly-sampled probes at each checkpoint.

This is a descriptive report, not an automated gate: it always runs to
completion and the main comparison always proceeds regardless of what it
finds (project decision - the small n=3 sample doesn't support a principled
statistical threshold, so results are surfaced for human judgment instead of
an algorithmic pass/fail).
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
from experiments.angle_2a.env_state import capture_env_state, restore_env_state
from experiments.angle_2a.errors import Angle2AConfigError
from experiments.angle_2a.onset_lookup import lookup_critic_degradation_onset
from experiments.angle_2a.prereq_storage import DEFAULT_OUTPUT_ROOT, prereq_seed_dir, save_prereq_seed_result
from experiments.angle_2a.probes import (
    Probe,
    compute_diagonal_errors,
    evaluate_both_critics,
    mean_and_se,
    run_monte_carlo_rollouts,
    sample_single_source_probes,
)

MAX_ROLLOUT_STEPS_MULTIPLIER = 3  # mirrors matchup.py's own safety cap


@dataclass
class PrereqSeedResult:
    architecture_label: str
    seed: int
    pre_step: int
    post_step: int
    e_d_pre: float
    e_d_pre_se: Optional[float]
    e_d_post: float
    e_d_post_se: Optional[float]
    direction_consistent_with_decline: bool
    output_paths: Dict


def _evaluate_checkpoint(
    handle: TrainedAgentHandle,
    probe_id_prefix: str,
    num_probes: int,
    num_mc_rollouts: int,
    gamma: float,
    max_rollout_steps: int,
    rng: np.random.Generator,
) -> List[Probe]:
    """D-only evaluation: passes the same handle as both the "D" and "R"
    parameter of these dual-source functions, which works unmodified because
    every probe here is tagged source=SOURCE_D (see sample_single_source_probes)
    - evaluate_both_critics/run_monte_carlo_rollouts only ever read the R
    side for source=SOURCE_R probes, which never occur here."""
    probes = sample_single_source_probes(probe_id_prefix, handle.probe_capture, num_probes, rng)
    evaluate_both_critics(probes, handle, handle)
    run_monte_carlo_rollouts(probes, handle, handle, num_mc_rollouts, gamma, max_rollout_steps)
    compute_diagonal_errors(probes)
    return probes


def run_one_prereq_seed(
    architecture: RoleArchitecture,
    architecture_label: str,
    base_cfg,
    seed: int,
    environment: str,
    onset_source_experiment: str,
    onset_ledger_root: str,
    burn_in_fraction: float,
    num_probes: int,
    num_mc_rollouts: int,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> PrereqSeedResult:
    onset = lookup_critic_degradation_onset(
        architecture=architecture_label,
        environment=environment,
        seed=seed,
        source_experiment=onset_source_experiment,
        ledger_root=onset_ledger_root,
    )
    post_step = onset.onset_step

    # Independent copy: this agent trains under `seed` (one of the 3
    # dedicated prereq seeds), never the main comparison's cfg.seed - mirrors
    # config.build_role_agent_cfg's deep-copy-then-override pattern.
    prereq_cfg = copy.deepcopy(base_cfg)
    OmegaConf.set_struct(prereq_cfg, False)
    prereq_cfg.seed = seed
    OmegaConf.set_struct(prereq_cfg, True)

    check_single_env_type(prereq_cfg)

    num_interaction_steps = int(prereq_cfg.num_interaction_steps)
    pre_step = int(np.floor(burn_in_fraction * num_interaction_steps))

    if pre_step <= 0 or pre_step >= post_step:
        raise Angle2AConfigError(
            f"Prereq check for architecture='{architecture_label}' seed={seed}: "
            f"pre-checkpoint step ({pre_step} = {burn_in_fraction:.0%} of "
            f"num_interaction_steps={num_interaction_steps}) is not strictly "
            f"before the post-checkpoint step (t*={post_step}); cannot run a "
            f"pre-vs-post comparison. Check the onset ledger entry for this "
            f"seed and this config's num_interaction_steps."
        )

    out_dir = prereq_seed_dir(environment, seed, architecture_label, root=output_root)
    pre_checkpoint_dir = out_dir / "checkpoints" / "pre"
    post_checkpoint_dir = out_dir / "checkpoints" / "post"

    train_env, eval_env, single_env, buffer, agent, obs_space, act_space = build_agent_and_env(
        architecture, prereq_cfg
    )
    try:
        probe_capacity = min(int(prereq_cfg.buffer.max_length), post_step)
        probe_capture = ProbeCapture(
            capacity=probe_capacity,
            observation_shape=obs_space.shape[-1:],
            action_shape=act_space.shape[-1:],
        )

        gamma = float(prereq_cfg.gamma)
        max_rollout_steps = int(prereq_cfg.env.max_episode_steps) * MAX_ROLLOUT_STEPS_MULTIPLIER
        seed_context = f"prereq:{architecture_label}:D:seed{seed}"
        seed_global_rng_for_agent(seed, seed_context)

        pre_probes: List[Probe] = []
        for step in run_training_loop(agent, buffer, train_env, single_env, probe_capture, prereq_cfg, post_step):
            if step != pre_step:
                continue

            # Save the real "pre" checkpoint the spec requires, and capture
            # everything evaluation is about to mutate (env physics state)
            # so training resumes exactly as if this check never ran.
            pre_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            agent.save_checkpoint(str(pre_checkpoint_dir))
            captured_env_state = capture_env_state(single_env, prereq_cfg.env.env_type)

            probe_rng = np.random.default_rng(seed=derive_rng_seed(seed, f"{seed_context}:pre_probes"))
            pre_handle = TrainedAgentHandle(
                role="D", architecture_label=architecture_label, architecture=architecture,
                agent=agent, buffer=buffer, train_env=train_env, eval_env=eval_env,
                single_env=single_env, stop_step=pre_step, probe_capture=probe_capture,
            )
            pre_probes = _evaluate_checkpoint(
                pre_handle,
                probe_id_prefix=f"prereq_{architecture_label}_seed{seed}_pre",
                num_probes=num_probes, num_mc_rollouts=num_mc_rollouts,
                gamma=gamma, max_rollout_steps=max_rollout_steps, rng=probe_rng,
            )

            # Undo evaluation's side effects (agent._rng consumption from
            # sample_actions during MC rollouts, env physics state from
            # restore_env_state+step) before letting training resume -
            # reloading the checkpoint just saved restores rng exactly;
            # restore_env_state puts physics back where training left it.
            agent.load_checkpoint(str(pre_checkpoint_dir))
            restore_env_state(single_env, captured_env_state)

        # generator above ran to post_step; agent/buffer/probe_capture now
        # reflect training complete through t*.
        post_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        agent.save_checkpoint(str(post_checkpoint_dir))

        probe_rng = np.random.default_rng(seed=derive_rng_seed(seed, f"{seed_context}:post_probes"))
        post_handle = TrainedAgentHandle(
            role="D", architecture_label=architecture_label, architecture=architecture,
            agent=agent, buffer=buffer, train_env=train_env, eval_env=eval_env,
            single_env=single_env, stop_step=post_step, probe_capture=probe_capture,
        )
        post_probes = _evaluate_checkpoint(
            post_handle,
            probe_id_prefix=f"prereq_{architecture_label}_seed{seed}_post",
            num_probes=num_probes, num_mc_rollouts=num_mc_rollouts,
            gamma=gamma, max_rollout_steps=max_rollout_steps, rng=probe_rng,
        )
    finally:
        train_env.close()
        eval_env.close()

    e_d_pre, e_d_pre_se = mean_and_se([p.diagonal_error for p in pre_probes])
    e_d_post, e_d_post_se = mean_and_se([p.diagonal_error for p in post_probes])

    output_paths = save_prereq_seed_result(
        environment=environment, seed=seed, architecture_label=architecture_label,
        pre_step=pre_step, post_step=post_step,
        pre_probes=pre_probes, post_probes=post_probes,
        e_d_pre=e_d_pre, e_d_pre_se=e_d_pre_se, e_d_post=e_d_post, e_d_post_se=e_d_post_se,
        pre_checkpoint_dir=pre_checkpoint_dir, post_checkpoint_dir=post_checkpoint_dir,
        root=output_root,
    )

    return PrereqSeedResult(
        architecture_label=architecture_label,
        seed=seed,
        pre_step=pre_step,
        post_step=post_step,
        e_d_pre=e_d_pre,
        e_d_pre_se=e_d_pre_se,
        e_d_post=e_d_post,
        e_d_post_se=e_d_post_se,
        direction_consistent_with_decline=e_d_post > e_d_pre,
        output_paths=output_paths,
    )


def run_prereq_check(
    scaled_architectures: Dict[str, RoleArchitecture],
    base_cfg,
    environment: str,
    dedicated_seeds: List[int],
    onset_source_experiment: str,
    onset_ledger_root: str,
    burn_in_fraction: float,
    num_probes: int,
    num_mc_rollouts: int,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> List[PrereqSeedResult]:
    """Runs the construct-validity check for every (scaled architecture,
    dedicated seed) pair and prints a clear summary. Always completes and
    always lets the caller's main comparison proceed regardless of what it
    finds - see module docstring for why this is descriptive, not a gate."""
    results: List[PrereqSeedResult] = []
    print(
        f"[angle_2a_prereq] construct-validity check: {len(scaled_architectures)} "
        f"architecture(s) x {len(dedicated_seeds)} dedicated seed(s) "
        f"env={environment}"
    )
    for architecture_label, architecture in scaled_architectures.items():
        for seed in dedicated_seeds:
            result = run_one_prereq_seed(
                architecture=architecture,
                architecture_label=architecture_label,
                base_cfg=base_cfg,
                seed=seed,
                environment=environment,
                onset_source_experiment=onset_source_experiment,
                onset_ledger_root=onset_ledger_root,
                burn_in_fraction=burn_in_fraction,
                num_probes=num_probes,
                num_mc_rollouts=num_mc_rollouts,
                output_root=output_root,
            )
            results.append(result)
            direction = "CONSISTENT with decline" if result.direction_consistent_with_decline else "NOT consistent with decline"
            pre_se_str = f"{result.e_d_pre_se:.4f}" if result.e_d_pre_se is not None else "n/a"
            post_se_str = f"{result.e_d_post_se:.4f}" if result.e_d_post_se is not None else "n/a"
            print(
                f"[angle_2a_prereq] {architecture_label} seed={seed}: "
                f"E_D(pre@{result.pre_step})={result.e_d_pre:.4f}±{pre_se_str} "
                f"E_D(post@{result.post_step})={result.e_d_post:.4f}±{post_se_str} "
                f"diff={result.e_d_post - result.e_d_pre:+.4f} -> {direction}"
            )

    num_consistent = sum(1 for r in results if r.direction_consistent_with_decline)
    print(
        f"[angle_2a_prereq] done: {num_consistent}/{len(results)} (architecture, seed) "
        f"combinations showed E_D(post) > E_D(pre). This is informational only - "
        f"no automated pass/fail; the main comparison proceeds regardless. "
        f"Review results/angle_2a_prereq/ for full detail."
    )
    return results
