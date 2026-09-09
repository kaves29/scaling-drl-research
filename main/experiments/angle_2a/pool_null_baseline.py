"""Angle 2A's null-baseline distribution, built from the shared
baseline-calibration pool (added 2026-09-08 - see
analysis/baseline_calibration_pool.py), replacing the previous per-seed
fresh-training design (see the End-of-Task Summary for why: research-
methodology.md's "Null Baseline" section always intended "reusable, at no
extra compute cost" reuse of existing baseline agents, but no prior
mechanism existed to persist them in Angle-2A-consumable form; Angle 1's
save_probe_capture_snapshot opt-in flag now provides that).

Procedure, per environment: load all 10 pool agents (checkpoint + full
env_state-inclusive ProbeCapture), share ONE live environment across all of
them (env_state capture/restore is purely a property of the physics
instance, not of which agent is being evaluated, so no per-agent environment
is needed), form all C(10,2)=45 unique pairs, and for each pair run the
IDENTICAL diagonal-error procedure the main D-vs-R comparison uses (see
experiments/angle_2a/probes.py): sample probes from both sides' own buffers,
evaluate both critics, run MC rollouts, compute diagonal-only errors. Each
pair contributes one E_A and one E_B value (mirroring E_D/E_R) - both are
"a healthy reference-architecture critic's own diagonal error", so all 90
resulting values (45 pairs x 2 sides) are pooled into one null distribution,
not kept as two separate D-side/R-side nulls (there is no scaled/reference
role distinction between two pool agents - both are the same architecture).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from analysis.baseline_calibration_pool import (
    POOL_MATCHUP_NAME,
    POOL_ROLE,
    POOL_STORAGE_ROOT,
    all_unique_pairs,
    get_baseline_calibration_pool,
)
from experiments.angle_2a.agent_runner import ProbeCapture, TrainedAgentHandle, derive_rng_seed
from experiments.angle_2a.probes import (
    Probe,
    compute_diagonal_errors,
    evaluate_both_critics,
    run_monte_carlo_rollouts,
    sample_probes,
)
from experiments.angle_2b.checkpoint_io import load_frozen_agent_snapshot
from experiments.angle_2a.storage import matchup_dir
from scale_rl.envs import create_envs
from utils.atomic_io import atomic_write_text

POOL_NULL_CACHE_ROOT = "results/angle_2a_pool_null"


@dataclass
class PoolNullResult:
    environment: str
    seed_a: int
    seed_b: int
    e_a: float
    e_b: float


@dataclass
class PoolNullDistribution:
    """The full cached result for one environment - computed once (see
    load_or_compute_pool_null_distribution), reused by every Angle 2A seed's
    invocation for that environment rather than recomputing 45 pairs' worth
    of real MC rollouts every single run."""

    environment: str
    pairs: List[PoolNullResult] = field(default_factory=list)
    null_mean: float = 0.0
    null_std: float = 0.0
    null_n: int = 0

    @property
    def values(self) -> List[float]:
        return pool_null_values(self.pairs)


def _load_pool_handle(environment: str, seed: int, single_env, root: str) -> TrainedAgentHandle:
    snapshot = load_frozen_agent_snapshot(environment, seed, POOL_MATCHUP_NAME, POOL_ROLE, root=root)
    checkpoint_dir = matchup_dir(environment, seed, POOL_MATCHUP_NAME, root=root) / "checkpoints" / POOL_ROLE
    probe_capture = ProbeCapture.load_fresh(str(checkpoint_dir))
    return TrainedAgentHandle(
        role=POOL_ROLE,
        architecture_label=f"pool_seed{seed}",
        architecture=None,
        agent=snapshot.agent,
        buffer=None,
        train_env=None,
        eval_env=None,
        single_env=single_env,
        stop_step=len(probe_capture),
        probe_capture=probe_capture,
    )


def compute_pool_null_distribution(
    environment: str,
    base_cfg,
    num_probes_per_source: int,
    num_mc_rollouts: int,
    analysis_seed: int,
    angle_2a_root: str,
) -> List[PoolNullResult]:
    """Live-instantiates ONE environment (shared across every pair - see
    module docstring) and runs the diagonal-error procedure on all 45 unique
    pool-agent pairs for `environment`."""
    identities = get_baseline_calibration_pool(environment)

    train_env, eval_env = create_envs(**base_cfg.env)
    try:
        single_env = train_env.envs[0]
        handles = {
            ident.seed: _load_pool_handle(environment, ident.seed, single_env, angle_2a_root)
            for ident in identities
        }

        gamma = float(base_cfg.gamma)
        max_rollout_steps = int(base_cfg.env.max_episode_steps) * 3  # matches matchup.py's own safety-cap convention

        results: List[PoolNullResult] = []
        for seed_a, seed_b in all_unique_pairs(sorted(handles.keys())):
            handle_a, handle_b = handles[seed_a], handles[seed_b]
            pair_name = f"pool_pair_seed{seed_a}_seed{seed_b}"
            rng = np.random.default_rng(seed=derive_rng_seed(analysis_seed, pair_name))

            probes: List[Probe] = sample_probes(pair_name, handle_a, handle_b, num_probes_per_source, rng)
            evaluate_both_critics(probes, handle_a, handle_b)
            run_monte_carlo_rollouts(probes, handle_a, handle_b, num_mc_rollouts, gamma, max_rollout_steps)
            compute_diagonal_errors(probes)

            e_a_values = [p.diagonal_error for p in probes if p.source == "D"]
            e_b_values = [p.diagonal_error for p in probes if p.source == "R"]
            results.append(
                PoolNullResult(
                    environment=environment,
                    seed_a=seed_a,
                    seed_b=seed_b,
                    e_a=float(np.mean(e_a_values)),
                    e_b=float(np.mean(e_b_values)),
                )
            )
        return results
    finally:
        train_env.close()
        eval_env.close()


def pool_null_values(results: List[PoolNullResult]) -> List[float]:
    """Pools E_A and E_B from every pair into one flat null distribution -
    see module docstring for why these are not kept separate."""
    values: List[float] = []
    for r in results:
        values.append(r.e_a)
        values.append(r.e_b)
    return values


def _cache_path(environment: str, root: str) -> Path:
    return Path(root) / environment / "pool_null_distribution.json"


def load_pool_null_distribution(environment: str, root: str = POOL_NULL_CACHE_ROOT) -> Optional[PoolNullDistribution]:
    path = _cache_path(environment, root)
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    return PoolNullDistribution(
        environment=payload["environment"],
        pairs=[PoolNullResult(**p) for p in payload["pairs"]],
        null_mean=payload["null_mean"],
        null_std=payload["null_std"],
        null_n=payload["null_n"],
    )


def save_pool_null_distribution(result: PoolNullDistribution, root: str = POOL_NULL_CACHE_ROOT) -> Path:
    path = _cache_path(result.environment, root)
    atomic_write_text(path, json.dumps(asdict(result), indent=2))
    return path


def load_or_compute_pool_null_distribution(
    environment: str,
    base_cfg,
    num_probes_per_source: int,
    num_mc_rollouts: int,
    analysis_seed: int,
    angle_2a_root: str = POOL_STORAGE_ROOT,
    cache_root: str = POOL_NULL_CACHE_ROOT,
    force_recompute: bool = False,
) -> PoolNullDistribution:
    """Computed once per environment (all C(10,2)=45 real MC-rollout-based
    pairs), cached, then reused by every Angle 2A seed's own invocation for
    that environment - avoids recomputing the entire pool null every single
    run."""
    if not force_recompute:
        cached = load_pool_null_distribution(environment, root=cache_root)
        if cached is not None:
            return cached

    pairs = compute_pool_null_distribution(
        environment, base_cfg, num_probes_per_source, num_mc_rollouts, analysis_seed, angle_2a_root,
    )
    values = pool_null_values(pairs)
    result = PoolNullDistribution(
        environment=environment,
        pairs=pairs,
        null_mean=float(np.mean(values)),
        null_std=float(np.std(values, ddof=1)),
        null_n=len(values),
    )
    save_pool_null_distribution(result, root=cache_root)
    print(
        f"[angle_2a_pool_null] environment={environment}: {len(pairs)} pairs, "
        f"null_n={result.null_n} null_mean={result.null_mean:.4f} "
        f"null_std={result.null_std:.4f}"
    )
    return result
