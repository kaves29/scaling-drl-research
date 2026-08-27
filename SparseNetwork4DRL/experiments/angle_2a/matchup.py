"""Orchestrates one Angle 2A matchup end-to-end:

    train D (own actor/critic/buffer/env/RNG)
    obtain R (either trained fresh here, own actor/critic/buffer/env/RNG - the
        null-baseline path - or supplied as an already-trained snapshot from
        the single shared reference trajectory - the real-matchup path; see
        `reference_handle` below)
    sample 10 probes from D's own buffer + 10 from R's own buffer
    evaluate BOTH critics on every probe (diagonal + off-diagonal)
    15 exact-state Monte Carlo rollouts per probe, using each probe's SOURCE actor only
    diagonal errors (E_D, E_R)
    persist (utils: experiments/angle_2a/storage.py)
    mirror a run-level summary to WandB

This module is reused for Matchup 1, Matchup 2, and the null baseline: the
independence/no-sharing guarantees for D are identical in all three cases.
For R, Matchup 1 and Matchup 2 pass in `reference_handle` - a snapshot of
the ONE shared R_2x512 trajectory (see
experiments.angle_2a.agent_runner.train_reference_agent_with_snapshots) -
taken at that matchup's own onset step, so a second reference training run
is never created. The null baseline (unchanged) omits `reference_handle`
and gets a fresh, independently trained R, exactly as before.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import wandb

from experiments.angle_2a.agent_runner import TrainedAgentHandle, derive_rng_seed, train_agent_to_step
from experiments.angle_2a.config import RoleArchitecture, build_role_agent_cfg
from experiments.angle_2a.probes import (
    Probe,
    compute_diagonal_errors,
    evaluate_both_critics,
    run_monte_carlo_rollouts,
    sample_probes,
)
from experiments.angle_2a.storage import save_frozen_agent_snapshot, save_matchup_result

MAX_ROLLOUT_STEPS_MULTIPLIER = 3  # safety cap only; natural termination is via terminated/truncated


@dataclass
class MatchupResult:
    matchup_name: str
    probes: List[Probe]
    run_metadata: Dict[str, Any]
    output_paths: Dict[str, Any]


def _aggregate_metrics(probes: List[Probe]) -> Dict[str, float]:
    d_errors = [p.diagonal_error for p in probes if p.source == "D"]
    r_errors = [p.diagonal_error for p in probes if p.source == "R"]
    metrics = {}
    if d_errors:
        metrics["mean_diagonal_error_D"] = float(np.mean(d_errors))
        metrics["median_diagonal_error_D"] = float(np.median(d_errors))
    if r_errors:
        metrics["mean_diagonal_error_R"] = float(np.mean(r_errors))
        metrics["median_diagonal_error_R"] = float(np.median(r_errors))
    return metrics


def run_matchup(
    matchup_name: str,
    scaled_architecture: RoleArchitecture,
    scaled_architecture_label: str,
    reference_architecture: RoleArchitecture,
    reference_architecture_label: str,
    onset_step: int,
    onset_source_run_key: str,
    base_cfg,
    seed: int,
    environment: str,
    experiment_name: str,
    num_probes_per_source: int,
    num_mc_rollouts: int,
    output_root: str,
    wandb_project: Optional[str] = None,
    wandb_enabled: bool = True,
    reference_handle: Optional[TrainedAgentHandle] = None,
    reference_run_key: Optional[str] = None,
) -> MatchupResult:
    """Runs one D-vs-R matchup.

    reference_handle: if given, this pre-trained snapshot is used as R
        as-is (never trained here, never closed here - the caller owns its
        lifecycle, since it is shared with another matchup). If omitted
        (the null-baseline path), R is trained fresh here, exactly as
        before, and closed here too.
    reference_run_key: an identifier for the shared reference trajectory,
        recorded in run_metadata so two matchups' persisted results can be
        matched back to "the same underlying R_2x512 run". Only meaningful
        (and only ever passed) alongside `reference_handle`.
    """
    # derive_rng_seed (hashlib-based) rather than Python's built-in hash():
    # hash() of a str/tuple-containing-str is randomized per-process by
    # default (PYTHONHASHSEED), which would silently break reproducibility
    # of probe sampling across separate invocations of the same command.
    rng = np.random.default_rng(seed=derive_rng_seed(seed, f"probes:{matchup_name}"))
    gamma = float(base_cfg.gamma)
    max_rollout_steps = int(base_cfg.env.max_episode_steps) * MAX_ROLLOUT_STEPS_MULTIPLIER

    started_at = time.time()

    D: Optional[TrainedAgentHandle] = None
    R: Optional[TrainedAgentHandle] = None
    owns_reference = reference_handle is None
    try:
        D = train_agent_to_step(
            role="D",
            architecture=scaled_architecture,
            architecture_label=scaled_architecture_label,
            base_cfg=base_cfg,
            stop_step=onset_step,
            seed_context=f"{matchup_name}:D:{scaled_architecture_label}",
        )
        if reference_handle is not None:
            R = reference_handle
        else:
            R = train_agent_to_step(
                role="R",
                architecture=reference_architecture,
                architecture_label=reference_architecture_label,
                base_cfg=base_cfg,
                stop_step=onset_step,
                seed_context=f"{matchup_name}:R:{reference_architecture_label}",
            )

        probes = sample_probes(matchup_name, D, R, num_probes_per_source, rng)
        evaluate_both_critics(probes, D, R)
        run_monte_carlo_rollouts(probes, D, R, num_mc_rollouts, gamma, max_rollout_steps)
        compute_diagonal_errors(probes)

        # Persist frozen agent snapshots (checkpoint + full probe-capture
        # data) for both roles, before anything is closed below. This is
        # unconditional for every matchup type (real + null-baseline) since
        # run_matchup() is shared code - see storage.py's module docstring
        # for why downstream consumers (Angle 2B) need this.
        snapshot_paths = {
            "D": save_frozen_agent_snapshot(
                environment, seed, matchup_name, "D", D.agent, D.probe_capture,
                agent_cfg=build_role_agent_cfg(base_cfg.agent, scaled_architecture),
                root=output_root,
            ),
            "R": save_frozen_agent_snapshot(
                environment, seed, matchup_name, "R", R.agent, R.probe_capture,
                agent_cfg=build_role_agent_cfg(base_cfg.agent, reference_architecture),
                root=output_root,
            ),
        }

        aggregate = _aggregate_metrics(probes)

        run_metadata = {
            "experiment": experiment_name,
            "seed": seed,
            "environment": environment,
            "matchup": matchup_name,
            "scaled_architecture": scaled_architecture_label,
            "scaled_architecture_config": {
                "critic_num_blocks": scaled_architecture.critic_num_blocks,
                "critic_hidden_dim": scaled_architecture.critic_hidden_dim,
            },
            "reference_architecture": reference_architecture_label,
            "reference_architecture_config": {
                "critic_num_blocks": reference_architecture.critic_num_blocks,
                "critic_hidden_dim": reference_architecture.critic_hidden_dim,
            },
            "scaled_onset_step": onset_step,
            "onset_source_run_key": onset_source_run_key,
            "stopping_step": onset_step,
            "num_probes_per_source": num_probes_per_source,
            "num_mc_rollouts": num_mc_rollouts,
            "gamma": gamma,
            "duration_seconds": time.time() - started_at,
            "snapshots": {
                role: {k: str(v) for k, v in paths.items()}
                for role, paths in snapshot_paths.items()
            },
            **aggregate,
        }
        if reference_run_key is not None:
            # identifies the shared R_2x512 trajectory this snapshot came
            # from - constant across Matchup 1 and Matchup 2 for the same
            # seed, even though `stopping_step` (the snapshot step) differs.
            run_metadata["reference_run_key"] = reference_run_key
            run_metadata["reference_snapshot_step"] = onset_step

        output_paths = save_matchup_result(
            environment=environment,
            seed=seed,
            matchup_name=matchup_name,
            run_metadata=run_metadata,
            probes=probes,
            root=output_root,
        )
        output_paths["snapshots"] = snapshot_paths

        if wandb_enabled:
            _log_to_wandb(matchup_name, run_metadata, wandb_project, output_paths)

        return MatchupResult(
            matchup_name=matchup_name,
            probes=probes,
            run_metadata=run_metadata,
            output_paths=output_paths,
        )
    finally:
        if D is not None:
            D.close()
        # a supplied reference_handle is shared with another matchup and
        # owned by the caller (see experiments/angle_2_a.py); only close a
        # reference agent that this call trained itself (the null-baseline
        # path).
        if R is not None and owns_reference:
            R.close()


def _log_to_wandb(matchup_name: str, run_metadata: Dict[str, Any], wandb_project: Optional[str], output_paths: Dict[str, Any]) -> None:
    run = wandb.init(
        project=wandb_project,
        group=f"angle_2a_{run_metadata['environment']}_seed{run_metadata['seed']}",
        job_type=matchup_name,
        name=(
            f"angle2a-{matchup_name}-{run_metadata['scaled_architecture']}"
            f"-vs-{run_metadata['reference_architecture']}-{run_metadata['environment']}"
            f"-seed{run_metadata['seed']}"
        ),
        config=run_metadata,
        reinit=True,
    )
    try:
        run.summary.update(
            {
                "probes_csv_path": str(output_paths["probes_csv"]),
                "probes_arrays_path": str(output_paths["probes_arrays"]),
                "run_metadata_path": str(output_paths["metadata"]),
            }
        )
        run.log(run_metadata)
    finally:
        run.finish()
