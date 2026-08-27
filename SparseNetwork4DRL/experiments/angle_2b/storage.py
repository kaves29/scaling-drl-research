"""Persistent storage for one Angle 2B analysis's results.

Mirrors experiments/angle_2a/storage.py's format convention: tabular
scalars (null-pair-level distortion values) go in a plain CSV, the raw
flattened gradient vectors (high-dimensional) go in a companion NPZ, and
run-level metadata (primary/secondary distortion metrics, null-comparison
verdicts, provenance) is a JSON sidecar. Independent of WandB - WandB (see
experiments/angle_2_b.py) only gets a run-level summary that references
these paths.
"""

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from experiments.angle_2a.storage import _atomic_write_bytes
from experiments.angle_2b.null_baseline import NullPairResult
from utils.atomic_io import atomic_write_text

DEFAULT_OUTPUT_ROOT = "results/angle_2b"

NULL_PAIR_COLUMNS = ["environment", "seed", "null_matchup_name", "d_dir", "d_mag", "d_grad"]


def analysis_dir(environment: str, seed: int, matchup_name: str, root: str = DEFAULT_OUTPUT_ROOT) -> Path:
    return Path(root) / environment / f"seed{seed}" / matchup_name


def save_angle_2b_result(
    environment: str,
    seed: int,
    matchup_name: str,
    run_metadata: Dict[str, Any],
    null_pairs: List[NullPairResult],
    gradients: Dict[str, np.ndarray],
    root: str = DEFAULT_OUTPUT_ROOT,
) -> Dict[str, Path]:
    out_dir = analysis_dir(environment, seed, matchup_name, root=root)

    metadata = dict(run_metadata)
    metadata.setdefault("environment", environment)
    metadata.setdefault("seed", seed)
    metadata.setdefault("matchup", matchup_name)
    metadata.setdefault("analysis_timestamp", datetime.now(timezone.utc).isoformat())

    metadata_path = out_dir / "run_metadata.json"
    atomic_write_text(metadata_path, json.dumps(metadata, indent=2, default=str))

    rows = [
        {
            "environment": p.environment,
            "seed": p.seed,
            "null_matchup_name": p.null_matchup_name,
            "d_dir": p.d_dir,
            "d_mag": p.d_mag,
            "d_grad": p.d_grad,
        }
        for p in null_pairs
    ]
    null_df = pd.DataFrame(rows, columns=NULL_PAIR_COLUMNS)
    csv_buf = io.StringIO()
    null_df.to_csv(csv_buf, index=False)
    null_csv_path = out_dir / "null_distribution.csv"
    atomic_write_text(null_csv_path, csv_buf.getvalue())

    gradients_buf = io.BytesIO()
    np.savez(gradients_buf, **{k: np.asarray(v) for k, v in gradients.items()})
    gradients_path = out_dir / "gradients.npz"
    _atomic_write_bytes(gradients_path, gradients_buf.getvalue())

    return {
        "metadata": metadata_path,
        "null_distribution_csv": null_csv_path,
        "gradients": gradients_path,
    }
