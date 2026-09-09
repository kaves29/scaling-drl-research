"""Shared baseline-agent calibration pool (added 2026-09-08).

Single source of truth for the 10-agent default-architecture (D2W512) pool
used to build every null/baseline distribution in this study: Angle 2A's own
null-baseline comparison, Angle 2B's null distribution (which Angle 2C
inherits), and the N/W ACF/CCF window calibration. Built once, reused
everywhere - see research-methodology.md's Null Baseline / Angle 2B / Onset
Definitions sections for why a single shared pool replaces what were
previously three separately-capped (<=5-point) constructions.

Composition: Angle 1's own 5 locked baseline seeds (experiment="angle_1",
seeds 1-5 - see configs/base_sac.yaml's onset_detection.baseline_seeds) PLUS
5 new, dedicated calibration-pool agents (experiment="baseline_calibration_pool",
seeds 6-10), trained via the identical procedure (see experiments/angle_1.py's
save_probe_capture_snapshot opt-in flag) but under a distinct experiment name
and seed range - RunIdentity.run_key is keyed by (experiment, architecture,
environment, seed), so this is a structurally separate storage subtree with
zero path collision with Angle 1's own 150 locked runs, never mixed with or
overwriting them.

Pairing: every consumer of this pool that needs pairwise comparisons (Angle
2A's refactored null, Angle 2B/2C's null distribution) uses ALL C(10,2)=45
unique unordered pairs among the 10 agents (project decision, 2026-09-08) -
not disjoint/non-overlapping pairs. This means the 45 resulting values are
NOT mutually independent (each agent appears in 9 of the 45 pairs), but
gives a substantially larger empirical sample than the previous <=5-point
null, which is the explicit tradeoff accepted for this change. The N/W
window calibration is NOT pair-based (each seed contributes its own
independent ACF/CCF estimate, then all are averaged) - see
get_baseline_calibration_pool's docstring for why it uses all 10 seeds
directly instead of a pairing scheme.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, TypeVar

from analysis.metrics_store import RunIdentity

ANGLE1_EXPERIMENT = "angle_1"
ANGLE1_BASELINE_SEEDS = (1, 2, 3, 4, 5)

POOL_EXPERIMENT = "baseline_calibration_pool"
POOL_SEEDS = (6, 7, 8, 9, 10)

BASELINE_ARCHITECTURE = "D2W512"  # matches configs/base_sac.yaml's onset_detection.default_architecture

# Where the 5 new pool agents' checkpoints/probe-capture snapshots are
# persisted (see experiments/angle_1.py's save_probe_capture_snapshot flag
# and experiments/angle_2a/storage.py's save_frozen_agent_snapshot, called
# with root=POOL_STORAGE_ROOT) - a distinct root from both results/angle_1
# (raw per-run logs/checkpoints, untouched by this pool) and results/angle_2a
# (real matchup outputs), so the pool is never confused with either.
POOL_STORAGE_ROOT = "results/baseline_calibration_pool"

# matchup_name/role used with experiments/angle_2a/storage.py's
# save_frozen_agent_snapshot/load_frozen_agent_snapshot for every pool
# agent - a standalone agent, not part of a D-vs-R matchup, so these are
# fixed placeholders (see save_frozen_agent_snapshot's role="pool" doc).
POOL_MATCHUP_NAME = "baseline_pool"
POOL_ROLE = "pool"

TOTAL_POOL_SIZE = len(ANGLE1_BASELINE_SEEDS) + len(POOL_SEEDS)


def get_baseline_calibration_pool(environment: str, architecture: str = BASELINE_ARCHITECTURE) -> List[RunIdentity]:
    """Returns all 10 baseline-architecture RunIdentity's for `environment`:
    Angle 1's own 5 locked seeds, plus this project's 5 dedicated
    calibration-pool seeds. Order is deterministic (Angle 1's 5 first, then
    the pool's 5) so callers that pair by index (see all_unique_pairs) get
    reproducible pairings run to run."""
    identities = [
        RunIdentity(experiment=ANGLE1_EXPERIMENT, architecture=architecture, environment=environment, seed=s)
        for s in ANGLE1_BASELINE_SEEDS
    ]
    identities += [
        RunIdentity(experiment=POOL_EXPERIMENT, architecture=architecture, environment=environment, seed=s)
        for s in POOL_SEEDS
    ]
    return identities


T = TypeVar("T")


def all_unique_pairs(items: List[T]) -> List[Tuple[T, T]]:
    """All C(n,2) unique unordered pairs, in a fixed, deterministic order
    (itertools.combinations preserves input order) - not disjoint pairing.
    See module docstring's Pairing section for why this is the accepted
    design for Angle 2A/2B's null distributions specifically."""
    return list(combinations(items, 2))
