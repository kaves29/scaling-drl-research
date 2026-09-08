"""Persistence for the Angle 2A construct-validity prerequisite check (see
experiments/angle_2a/prereq_check.py). Deliberately under its own
results/angle_2a_prereq/ root, never results/angle_2a/, so it can never be
mistaken for or mixed into the main 5-seed comparison's output.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from experiments.angle_2a.probes import Probe
from utils.atomic_io import atomic_write_bytes, atomic_write_text

DEFAULT_OUTPUT_ROOT = "results/angle_2a_prereq"

PROBE_SCALAR_COLUMNS = ["probe_id", "source", "q_d", "mc_return", "mc_return_se", "diagonal_error", "num_rollouts"]


def prereq_seed_dir(environment: str, seed: int, architecture_label: str, root: str = DEFAULT_OUTPUT_ROOT) -> Path:
    return Path(root) / environment / f"seed{seed}" / architecture_label


def _save_phase_probes(out_dir: Path, phase: str, probes: List[Probe]) -> Dict[str, Path]:
    rows = [
        {
            "probe_id": p.probe_id,
            "source": p.source,
            "q_d": p.q_d,
            "mc_return": p.mc_return,
            "mc_return_se": p.mc_return_se,
            "diagonal_error": p.diagonal_error,
            "num_rollouts": len(p.mc_rollout_returns),
        }
        for p in probes
    ]
    probes_df = pd.DataFrame(rows, columns=PROBE_SCALAR_COLUMNS)
    csv_buf = io.StringIO()
    probes_df.to_csv(csv_buf, index=False)
    csv_path = out_dir / f"probes_{phase}.csv"
    atomic_write_text(csv_path, csv_buf.getvalue())

    arrays_buf = io.BytesIO()
    np.savez(
        arrays_buf,
        probe_id=np.array([p.probe_id for p in probes]),
        state=np.stack([p.state for p in probes]) if probes else np.empty((0,)),
        action=np.stack([p.action for p in probes]) if probes else np.empty((0,)),
    )
    arrays_path = out_dir / f"probes_{phase}_arrays.npz"
    atomic_write_bytes(arrays_path, arrays_buf.getvalue())

    return {"csv": csv_path, "arrays": arrays_path}


def save_prereq_seed_result(
    environment: str,
    seed: int,
    architecture_label: str,
    pre_step: int,
    post_step: int,
    pre_probes: List[Probe],
    post_probes: List[Probe],
    e_d_pre: float,
    e_d_pre_se,
    e_d_post: float,
    e_d_post_se,
    pre_checkpoint_dir: Path,
    post_checkpoint_dir: Path,
    root: str = DEFAULT_OUTPUT_ROOT,
) -> Dict[str, Any]:
    """Persists one (architecture, dedicated seed)'s pre/post construct-
    validity check result. No pass/fail field is written - per project
    decision this check is descriptive (report E_D(pre)/E_D(post)/direction
    for human judgment), not an automated gate; see research-methodology.md's
    Angle 2A section."""
    out_dir = prereq_seed_dir(environment, seed, architecture_label, root=root)

    pre_paths = _save_phase_probes(out_dir, "pre", pre_probes)
    post_paths = _save_phase_probes(out_dir, "post", post_probes)

    summary = {
        "environment": environment,
        "seed": seed,
        "architecture": architecture_label,
        "pre_step": pre_step,
        "post_step": post_step,
        "e_d_pre": e_d_pre,
        "e_d_pre_se": e_d_pre_se,
        "e_d_post": e_d_post,
        "e_d_post_se": e_d_post_se,
        "diff": e_d_post - e_d_pre,
        "direction_consistent_with_decline": bool(e_d_post > e_d_pre),
        "num_pre_probes": len(pre_probes),
        "num_post_probes": len(post_probes),
        "pre_checkpoint_dir": str(pre_checkpoint_dir),
        "post_checkpoint_dir": str(post_checkpoint_dir),
    }
    summary_path = out_dir / "summary.json"
    atomic_write_text(summary_path, json.dumps(summary, indent=2, default=str))

    return {
        "summary": summary_path,
        "summary_data": summary,
        "pre_probes": pre_paths,
        "post_probes": post_paths,
    }
