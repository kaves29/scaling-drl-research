"""Loads Angle 2B's already-persisted outputs for Angle 2C - zero training,
zero environment interaction, zero new state-action sampling (see
experiments/angle_2b/matchup_2b.py, which computes and persists exactly the
(s,a) pairs, nabla_a Q, Q(s,a), and actor-parameter gradients Angle 2C
needs, precisely so this module never has to recompute or resample
anything).

This is Angle 2C's counterpart to experiments/angle_2b/checkpoint_io.py.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from experiments.angle_2b.storage import analysis_dir
from experiments.angle_2c.errors import Angle2CDataError

_NULL_SEED_KEY_RE = re.compile(r"^null_seed(\d+)_states$")


@dataclass
class NullPairArrays:
    seed: int
    states: np.ndarray
    actions: np.ndarray
    grad_aq_a_at_a: np.ndarray
    grad_aq_b_at_a: np.ndarray
    q_a_at_a: np.ndarray
    q_b_at_a: np.ndarray


@dataclass
class Angle2BArtifacts:
    environment: str
    seed: int
    matchup_name: str
    run_metadata: Dict[str, Any]
    critic_use_cdq: bool

    # Primary: pi_D held fixed, at D's own (s,a).
    states_d: np.ndarray
    actions_d: np.ndarray
    grad_aq_d_at_d: np.ndarray
    grad_aq_r_at_d: np.ndarray
    q_d_at_d: np.ndarray
    q_r_at_d: np.ndarray
    g_d_given_d: np.ndarray
    g_d_given_r: np.ndarray

    # Secondary: pi_R held fixed, at R's own (s,a).
    states_r: np.ndarray
    actions_r: np.ndarray
    grad_aq_r_at_r: np.ndarray
    grad_aq_d_at_r: np.ndarray
    q_r_at_r: np.ndarray
    q_d_at_r: np.ndarray
    g_r_given_r: np.ndarray
    g_r_given_d: np.ndarray

    # Null A/B pairs, keyed by seed - one per available Angle 2A
    # null-baseline seed (see experiments/angle_2b/null_baseline.py).
    null_pairs: Dict[int, NullPairArrays]


def _require_exists(path: Path, what: str) -> Path:
    if not path.exists():
        raise Angle2CDataError(
            f"Missing {what} at '{path}'. Angle 2C never recomputes a "
            f"missing Angle 2B artifact - run Angle 2B for this "
            f"(environment, seed, matchup) first, or point "
            f"angle_2b_results_root at wherever those results actually live."
        )
    return path


def load_angle_2b_artifacts(
    environment: str,
    seed: int,
    matchup_name: str,
    root: str,
) -> Angle2BArtifacts:
    out_dir = analysis_dir(environment, seed, matchup_name, root=root)

    metadata_path = _require_exists(out_dir / "run_metadata.json", "Angle 2B run_metadata.json")
    with open(metadata_path) as f:
        run_metadata = json.load(f)

    gradients_path = _require_exists(out_dir / "gradients.npz", "Angle 2B gradients.npz")
    with np.load(gradients_path, allow_pickle=False) as npz:
        required = (
            "states_D", "actions_D", "grad_aq_D_at_D", "grad_aq_R_at_D", "q_D_at_D", "q_R_at_D",
            "g_D_given_D", "g_D_given_R",
            "states_R", "actions_R", "grad_aq_R_at_R", "grad_aq_D_at_R", "q_R_at_R", "q_D_at_R",
            "g_R_given_R", "g_R_given_D",
        )
        missing = [k for k in required if k not in npz.files]
        if missing:
            raise Angle2CDataError(
                f"Angle 2B gradients.npz at '{gradients_path}' is missing "
                f"key(s) {missing} that Angle 2C requires. This Angle 2B "
                f"result was likely produced before the Angle 2C extension "
                f"to matchup_2b.py/null_baseline.py - re-run Angle 2B for "
                f"this (environment, seed, matchup) to regenerate it with "
                f"the (s,a)/nabla_a Q/Q fields Angle 2C needs."
            )

        null_seeds = sorted(
            int(m.group(1))
            for key in npz.files
            for m in [_NULL_SEED_KEY_RE.match(key)]
            if m is not None
        )
        if not null_seeds:
            raise Angle2CDataError(
                f"Angle 2B gradients.npz at '{gradients_path}' contains no "
                f"null_seed*_states keys - Angle 2C requires at least one "
                f"persisted null pair to build its own null distributions "
                f"(reusing Angle 2B's null infrastructure directly)."
            )

        null_pairs = {
            ns: NullPairArrays(
                seed=ns,
                states=np.asarray(npz[f"null_seed{ns}_states"]),
                actions=np.asarray(npz[f"null_seed{ns}_actions"]),
                grad_aq_a_at_a=np.asarray(npz[f"null_seed{ns}_grad_aq_a_at_a"]),
                grad_aq_b_at_a=np.asarray(npz[f"null_seed{ns}_grad_aq_b_at_a"]),
                q_a_at_a=np.asarray(npz[f"null_seed{ns}_q_a_at_a"]),
                q_b_at_a=np.asarray(npz[f"null_seed{ns}_q_b_at_a"]),
            )
            for ns in null_seeds
        }

        return Angle2BArtifacts(
            environment=environment,
            seed=seed,
            matchup_name=matchup_name,
            run_metadata=run_metadata,
            critic_use_cdq=bool(run_metadata["critic_use_cdq"]),
            states_d=np.asarray(npz["states_D"]),
            actions_d=np.asarray(npz["actions_D"]),
            grad_aq_d_at_d=np.asarray(npz["grad_aq_D_at_D"]),
            grad_aq_r_at_d=np.asarray(npz["grad_aq_R_at_D"]),
            q_d_at_d=np.asarray(npz["q_D_at_D"]),
            q_r_at_d=np.asarray(npz["q_R_at_D"]),
            g_d_given_d=np.asarray(npz["g_D_given_D"]),
            g_d_given_r=np.asarray(npz["g_D_given_R"]),
            states_r=np.asarray(npz["states_R"]),
            actions_r=np.asarray(npz["actions_R"]),
            grad_aq_r_at_r=np.asarray(npz["grad_aq_R_at_R"]),
            grad_aq_d_at_r=np.asarray(npz["grad_aq_D_at_R"]),
            q_r_at_r=np.asarray(npz["q_R_at_R"]),
            q_d_at_r=np.asarray(npz["q_D_at_R"]),
            g_r_given_r=np.asarray(npz["g_R_given_R"]),
            g_r_given_d=np.asarray(npz["g_R_given_D"]),
            null_pairs=null_pairs,
        )
