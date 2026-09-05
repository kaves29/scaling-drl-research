"""Persistent storage for one Angle 2C analysis's results.

Mirrors experiments/angle_2a/storage.py and experiments/angle_2b/storage.py's
format convention: tabular per-null-pair scalars go in a plain CSV, the
high-dimensional per-(s,a) property arrays go in a companion NPZ, and
run-level metadata (aggregated property values, null-comparison verdicts,
co-occurrence/reconstruction results, the onset-timing proxy) is a JSON
sidecar. Independent of WandB - WandB (see experiments/angle_2_c.py) only
gets a run-level summary that references these paths.
"""

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from experiments.angle_2a.storage import _atomic_write_bytes
from utils.atomic_io import atomic_write_text

DEFAULT_OUTPUT_ROOT = "results/angle_2c"

NULL_PAIR_COLUMNS = [
    "environment", "seed", "null_matchup_name",
    "direction_for_null", "magnitude_for_null", "raw_offset", "instability_ratio",
]


def analysis_dir(environment: str, seed: int, matchup_name: str, root: str = DEFAULT_OUTPUT_ROOT) -> Path:
    return Path(root) / environment / f"seed{seed}" / matchup_name


def save_angle_2c_result(
    environment: str,
    seed: int,
    matchup_name: str,
    run_metadata: Dict[str, Any],
    null_pair_rows: List[Dict[str, Any]],
    property_arrays: Dict[str, np.ndarray],
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

    null_df = pd.DataFrame(null_pair_rows, columns=NULL_PAIR_COLUMNS)
    csv_buf = io.StringIO()
    null_df.to_csv(csv_buf, index=False)
    null_csv_path = out_dir / "null_distribution.csv"
    atomic_write_text(null_csv_path, csv_buf.getvalue())

    arrays_buf = io.BytesIO()
    np.savez(arrays_buf, **{k: np.asarray(v) for k, v in property_arrays.items()})
    arrays_path = out_dir / "properties.npz"
    _atomic_write_bytes(arrays_path, arrays_buf.getvalue())

    return {
        "metadata": metadata_path,
        "null_distribution_csv": null_csv_path,
        "properties": arrays_path,
    }
