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

Frozen-agent snapshots (added for Angle 2B): Angle 2A originally discarded
each matchup's live D/R agent objects once probes/errors were computed (see
matchup.py's `finally: D.close() / R.close()` - envs were closed, but the
JAX actor/critic/temperature params themselves were never checkpointed to
disk anywhere in Angle 1 or Angle 2A). Angle 2B needs exactly those frozen
params (pi_D/Q_D, pi_R/Q_R) with zero additional training, so
save_frozen_agent_snapshot() below persists, per (matchup, role):
  - the agent's full SACAgent.save_checkpoint() state (actor/critic/
    temperature/target_critic/rng), reusable via SACAgent.load_checkpoint()
  - that role's full ProbeCapture states/actions (NOT just the 10 probes
    already sampled into probes_arrays.npz - a fresh, larger, independent
    sample), so Angle 2B can draw its own state batch without being limited
    to Angle 2A's own probe count or re-running training/environment
    interaction to get more.
This is deliberately symmetric across every matchup type (real matchups AND
null-baseline matchups): run_matchup() is shared code, so null-baseline
D/R agents (two independent healthy default-architecture critics) get
snapshotted the same way, which is what lets Angle 2B build its healthy-
critic null distribution from already-existing Angle 2A infrastructure
instead of training anything new.
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


def save_frozen_agent_snapshot(
    environment: str,
    seed: int,
    matchup_name: str,
    role: str,
    agent: Any,
    probe_capture: Any,
    agent_cfg: Dict[str, Any],
    root: str = DEFAULT_OUTPUT_ROOT,
) -> Dict[str, Path]:
    """Persists one role's ("D" or "R") frozen agent + full probe-capture
    data for a matchup, so it can be reloaded later (by Angle 2B or anyone
    else) with zero retraining/environment interaction.

    `agent` may be a raw SACAgent or an ObservationNormalizer-wrapped one
    (see scale_rl.agents.wrappers.normalization) - agent.save_checkpoint()
    handles both correctly (the wrapper override additionally persists
    obs_rms, without which normalized-observation agents could not be
    faithfully reconstructed).

    `probe_capture` is the role's ProbeCapture (see
    experiments/angle_2a/agent_runner.py) as of this matchup's stopping
    step - already correctly frozen/snapshotted for the reference role by
    train_reference_agent_with_snapshots (its live buffer keeps growing
    past this step, but probe_capture does not).

    `agent_cfg` is this role's fully-resolved agent config (as returned by
    experiments.angle_2a.config.build_role_agent_cfg) - persisted verbatim
    so a caller (Angle 2B) can reconstruct an architecturally-identical
    agent shell via scale_rl.agents.create_agent() before calling
    load_checkpoint(), without needing any live Hydra config or a real
    gym/dm_control environment (observation_dim/action_dim are recoverable
    from the saved probe_capture arrays' shapes instead).
    """
    if role not in ("D", "R"):
        raise ValueError(f"role must be 'D' or 'R', got {role!r}")

    out_dir = matchup_dir(environment, seed, matchup_name, root=root)
    checkpoint_dir = out_dir / "checkpoints" / role
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent.save_checkpoint(str(checkpoint_dir))

    n = len(probe_capture)
    if n > 0:
        _idxs, states, actions, _env_states = probe_capture.sample(
            n, np.random.default_rng(seed=0)
        )
    else:
        states = np.empty((0,))
        actions = np.empty((0,))

    probe_capture_buf = io.BytesIO()
    np.savez(probe_capture_buf, states=states, actions=actions)
    probe_capture_path = out_dir / f"probe_capture_{role}.npz"
    _atomic_write_bytes(probe_capture_path, probe_capture_buf.getvalue())

    agent_cfg_path = out_dir / f"agent_cfg_{role}.json"
    atomic_write_text(agent_cfg_path, json.dumps(agent_cfg, indent=2, default=str))

    return {
        "checkpoint_dir": checkpoint_dir,
        "probe_capture": probe_capture_path,
        "agent_cfg": agent_cfg_path,
    }


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
