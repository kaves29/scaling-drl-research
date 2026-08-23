"""Persistent storage for one Angle 2A matchup's run metadata + probe dataset.

Format choice: this repo has no existing parquet dependency (checked: only
pandas/numpy/pickle/CSV are used elsewhere for persistence), so rather than
add a new dependency, probe-level *scalar* fields (source, Q values, MC
return, error, rollout summary stats) go in a plain CSV - consistent with
the rest of the repo's metric logging - while the high-dimensional arrays
(raw state/action vectors, all 15 raw rollout returns per probe) go in a
companion NPZ, keyed by the same `probe_id` order, so nothing high-dimensional
is ever squashed into a CSV cell as a comma-joined string. Run-level metadata
(architectures, onset step, stopping step, config, identifiers, timestamp)
is a small JSON sidecar.

Everything here is independent of WandB: this is the canonical dataset,
per project requirement; WandB (see experiments/angle_2a/matchup.py) only
gets a run-level summary that references these paths.
"""

import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments.angle_2a.probes import Probe
from utils.atomic_io import atomic_write_text

DEFAULT_OUTPUT_ROOT = "results/angle_2a"

PROBE_SCALAR_COLUMNS = [
    "probe_id",
    "source",
    "q_d",
    "q_r",
    "mc_return",
    "diagonal_error",
    "num_rollouts",
    "mc_return_std",
]


def matchup_dir(environment: str, seed: int, matchup_name: str, root: str = DEFAULT_OUTPUT_ROOT) -> Path:
    return Path(root) / environment / f"seed{seed}" / matchup_name


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def save_matchup_result(
    environment: str,
    seed: int,
    matchup_name: str,
    run_metadata: Dict[str, Any],
    probes: List[Probe],
    root: str = DEFAULT_OUTPUT_ROOT,
) -> Dict[str, Path]:
    out_dir = matchup_dir(environment, seed, matchup_name, root=root)

    metadata = dict(run_metadata)
    metadata.setdefault("environment", environment)
    metadata.setdefault("seed", seed)
    metadata.setdefault("matchup", matchup_name)
    metadata.setdefault("num_probes", len(probes))
    metadata.setdefault("analysis_timestamp", datetime.now(timezone.utc).isoformat())

    metadata_path = out_dir / "run_metadata.json"
    atomic_write_text(metadata_path, json.dumps(metadata, indent=2, default=str))

    rows = []
    for p in probes:
        rows.append(
            {
                "probe_id": p.probe_id,
                "source": p.source,
                "q_d": p.q_d,
                "q_r": p.q_r,
                "mc_return": p.mc_return,
                "diagonal_error": p.diagonal_error,
                "num_rollouts": len(p.mc_rollout_returns),
                "mc_return_std": float(np.std(p.mc_rollout_returns)) if p.mc_rollout_returns else None,
            }
        )
    probes_df = pd.DataFrame(rows, columns=PROBE_SCALAR_COLUMNS)
    csv_buf = io.StringIO()
    probes_df.to_csv(csv_buf, index=False)
    csv_path = out_dir / "probes.csv"
    atomic_write_text(csv_path, csv_buf.getvalue())

    max_rollouts = max((len(p.mc_rollout_returns) for p in probes), default=0)
    rollout_returns = np.full((len(probes), max_rollouts), np.nan, dtype=np.float64)
    for i, p in enumerate(probes):
        rollout_returns[i, : len(p.mc_rollout_returns)] = p.mc_rollout_returns

    arrays_buf = io.BytesIO()
    np.savez(
        arrays_buf,
        probe_id=np.array([p.probe_id for p in probes]),
        source=np.array([p.source for p in probes]),
        state=np.stack([p.state for p in probes]) if probes else np.empty((0,)),
        action=np.stack([p.action for p in probes]) if probes else np.empty((0,)),
        rollout_returns=rollout_returns,
    )
    arrays_path = out_dir / "probes_arrays.npz"
    _atomic_write_bytes(arrays_path, arrays_buf.getvalue())

    return {"metadata": metadata_path, "probes_csv": csv_path, "probes_arrays": arrays_path}


def load_matchup_result(
    environment: str,
    seed: int,
    matchup_name: str,
    root: str = DEFAULT_OUTPUT_ROOT,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, np.ndarray]]:
    """Reloads everything save_matchup_result() wrote, without needing to
    rerun training or rollouts. Returns (run_metadata, probes_df, arrays)."""
    out_dir = matchup_dir(environment, seed, matchup_name, root=root)

    with open(out_dir / "run_metadata.json") as f:
        metadata = json.load(f)

    probes_df = pd.read_csv(out_dir / "probes.csv")

    with np.load(out_dir / "probes_arrays.npz", allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}

    return metadata, probes_df, arrays
