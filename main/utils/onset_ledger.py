"""Canonical EchoCritic onset-event ledger.

The CSV files written here (`results/ledgers/{exp_name}/architectures/{architecture}/onset_events.csv`)
are the *canonical* source of truth for critic-degradation / pathology-propagation
onset results. WandB run summaries are only ever written *after* the CSV write
succeeds, and only as a secondary mirror for visualization.

Design constraints this module enforces (see project spec):
  - one row per run, keyed uniquely by `run_key` (upsert, never duplicate)
  - atomic writes: a crash or malformed write must never corrupt/replace the
    previous good ledger (see utils.atomic_io.atomic_write_text)
  - schema is validated on every load and every write
  - `exact_run_name` is validated against the live WandB run identity before
    a write is allowed to proceed
  - the CSV must be durably written *before* any WandB summary mirroring
"""

import io
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.atomic_io import atomic_write_text

REQUIRED_COLUMNS: List[str] = [
    "run_key",
    "exact_run_name",
    "wandb_run_id",
    "architecture",
    "environment",
    "seed",
    "critic_degradation_onset_step",
    "critic_degradation_method",
    "propagation_onset_step",
    "propagation_method",
    "propagation_lag",
    "status",
    "detection_notes",
    "analysis_timestamp",
]

VALID_STATUSES = {"success", "no_onset_detected", "needs_manual_review"}
# NOTE: "ambiguous" was previously a valid status but was never actually
# produced by any detection code path and has been intentionally removed
# from the methodology - see analysis/onset_detection.py. Anything that
# can't be confidently classified as success/no_onset_detected uses
# needs_manual_review instead.

DEFAULT_LEDGER_ROOT = "results/ledgers"


class OnsetLedgerError(Exception):
    """Base class for onset-ledger failures."""


class LedgerSchemaError(OnsetLedgerError):
    """Ledger CSV is missing required columns or is unparseable."""


class RunIdentityValidationError(OnsetLedgerError):
    """`exact_run_name` / `wandb_run_id` could not be confirmed against WandB."""


@dataclass
class WandbIdentity:
    """Authoritative identity info used to validate a ledger row before write.

    Prefer passing a live `wandb.run`-like object (anything exposing `.name`
    and `.id`, e.g. the object returned by `wandb.init()`, or a `WandbTrainerLogger`)
    via `run_obj`. If training already finished and no live run object is
    available, pass `entity`/`project` so the WandB public API can be used
    to look the run up by id instead.
    """

    run_obj: Optional[Any] = None
    entity: Optional[str] = None
    project: Optional[str] = None


def _ledger_path(exp_name: str, architecture: str, root: str = DEFAULT_LEDGER_ROOT) -> Path:
    if not exp_name or not architecture:
        raise ValueError("exp_name and architecture must be non-empty strings")
    return Path(root) / exp_name / "architectures" / architecture / "onset_events.csv"


def _empty_ledger_df() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="object") for col in REQUIRED_COLUMNS})


def _validate_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise LedgerSchemaError(
            f"Ledger at '{source}' is missing required column(s): {missing}. "
            f"Required schema: {REQUIRED_COLUMNS}"
        )
    # keep required columns first, preserve any forward-compat extra columns after
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return df[REQUIRED_COLUMNS + extra]


def _load_existing(path: Path) -> pd.DataFrame:
    """Loads and validates an existing ledger CSV.

    Returns an empty, correctly-schema'd DataFrame if the ledger/dir/file
    does not exist yet, or exists but is empty. Raises LedgerSchemaError for
    genuinely malformed content (unparseable CSV or missing columns) rather
    than silently discarding it.
    """
    if not path.exists():
        return _empty_ledger_df()

    if path.stat().st_size == 0:
        warnings.warn(f"Ledger file at '{path}' exists but is empty; treating as new.")
        return _empty_ledger_df()

    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise LedgerSchemaError(
            f"Ledger at '{path}' is malformed and could not be parsed as CSV "
            f"({e}). Refusing to overwrite it automatically; please inspect "
            f"or repair the file manually."
        ) from e

    df = _validate_schema(df, source=str(path))

    if df["run_key"].duplicated().any():
        dupes = df.loc[df["run_key"].duplicated(), "run_key"].unique().tolist()
        warnings.warn(
            f"Ledger at '{path}' contained duplicate run_key value(s) {dupes}; "
            f"keeping the last occurrence of each and dropping earlier ones."
        )
        df = df.drop_duplicates(subset="run_key", keep="last").reset_index(drop=True)

    return df


def _resolve_authoritative_identity(identity: Optional[WandbIdentity]):
    """Returns (exact_run_name, wandb_run_id, verifiable: bool)."""
    if identity is None:
        return None, None, False

    if identity.run_obj is not None:
        name = getattr(identity.run_obj, "name", None)
        run_id = getattr(identity.run_obj, "id", None)
        if name is not None and run_id is not None:
            return name, run_id, True

    if identity.entity and identity.project:
        wandb_run_id_hint = getattr(identity.run_obj, "id", None)
        try:
            import wandb  # local import: keep this module importable w/o wandb installed

            if wandb_run_id_hint is None:
                return None, None, False
            api = wandb.Api()
            api_run = api.run(f"{identity.entity}/{identity.project}/{wandb_run_id_hint}")
            return api_run.name, api_run.id, True
        except Exception as e:  # network / auth / offline-run / not-yet-synced, etc.
            warnings.warn(f"Could not verify WandB run identity via API: {e}")
            return None, None, False

    return None, None, False


def _validate_identity(row: Dict[str, Any], identity: Optional[WandbIdentity]) -> Dict[str, Any]:
    """Validates row['exact_run_name']/['wandb_run_id'] against WandB.

    Returns the (possibly status-downgraded) row. Raises RunIdentityValidationError
    only on a *positive mismatch* (we have authoritative data and it disagrees) —
    that is a real bug in the caller and must not be silently written. If identity
    simply cannot be verified (offline run, WandB unavailable, no run object and
    no API creds), we do not fail the write outright; instead we force
    status='needs_manual_review' and record why, per spec section 9/10.
    """
    auth_name, auth_id, verifiable = _resolve_authoritative_identity(identity)

    if not verifiable:
        row = dict(row)
        note = (
            "WandB run identity could not be verified (no live run object and "
            "no reachable WandB API); recorded exact_run_name/wandb_run_id are "
            "UNVERIFIED."
        )
        row["status"] = "needs_manual_review"
        row["detection_notes"] = (
            f"{row.get('detection_notes', '') or ''} | {note}".strip(" |")
        )
        return row

    mismatches = []
    if row.get("exact_run_name") != auth_name:
        mismatches.append(
            f"exact_run_name mismatch: row has '{row.get('exact_run_name')}', "
            f"WandB reports '{auth_name}'"
        )
    if str(row.get("wandb_run_id")) != str(auth_id):
        mismatches.append(
            f"wandb_run_id mismatch: row has '{row.get('wandb_run_id')}', "
            f"WandB reports '{auth_id}'"
        )
    if mismatches:
        raise RunIdentityValidationError(
            "Refusing to write onset ledger row: " + "; ".join(mismatches)
        )

    return row


def log_onset_event(
    exp_name: str,
    architecture: str,
    row: Dict[str, Any],
    identity: Optional[WandbIdentity] = None,
    root: str = DEFAULT_LEDGER_ROOT,
    mirror_to_wandb: bool = True,
) -> Path:
    """Upserts one onset-event row into the canonical CSV ledger.

    Args:
        exp_name: routing experiment name (e.g. "angle_1"), used for the
            `results/ledgers/{exp_name}/...` path — NOT the free-form
            `cfg.exp_name` analysis label used elsewhere in this repo.
        architecture: architecture id (see scale_rl.common.logger.get_architecture_id).
        row: dict covering (at minimum) all REQUIRED_COLUMNS except
            `analysis_timestamp`, which is stamped here if absent.
        identity: authoritative WandB identity used to validate
            row['exact_run_name'] / row['wandb_run_id']. Strongly recommended;
            see WandbIdentity docstring.
        mirror_to_wandb: if True (default) and a live `identity.run_obj` is
            available, mirror the onset fields into `run.summary` *after* the
            CSV write succeeds. Failures here are warnings, never exceptions —
            the CSV remains canonical regardless of WandB availability.

    Returns:
        Path to the ledger CSV that was written.

    Raises:
        LedgerSchemaError: existing ledger is malformed / missing columns, or
            `row` is missing a required column.
        RunIdentityValidationError: `row`'s claimed WandB identity does not
            match the actual WandB run (a real bug, not a transient issue).
    """
    missing_keys = [c for c in REQUIRED_COLUMNS if c not in row and c != "analysis_timestamp"]
    if missing_keys:
        raise LedgerSchemaError(f"row is missing required key(s): {missing_keys}")

    row = dict(row)
    row.setdefault("analysis_timestamp", datetime.now(timezone.utc).isoformat())

    if row["status"] not in VALID_STATUSES:
        raise LedgerSchemaError(
            f"row['status']='{row['status']}' is not one of {sorted(VALID_STATUSES)}"
        )

    # Identity validation can only ever *downgrade* status (-> needs_manual_review)
    # or raise on a genuine mismatch. It never happens after the CSV write.
    row = _validate_identity(row, identity)

    path = _ledger_path(exp_name, architecture, root=root)
    existing = _load_existing(path)

    new_row_df = pd.DataFrame([{c: row.get(c) for c in existing.columns}])
    updated = pd.concat(
        [existing[existing["run_key"] != row["run_key"]], new_row_df],
        ignore_index=True,
    )
    updated = _validate_schema(updated, source=str(path))

    buf = io.StringIO()
    updated.to_csv(buf, index=False)
    atomic_write_text(path, buf.getvalue())

    if mirror_to_wandb and identity is not None and identity.run_obj is not None:
        try:
            identity.run_obj.summary.update(
                {
                    "onset/critic_degradation_onset_step": row["critic_degradation_onset_step"],
                    "onset/critic_degradation_method": row["critic_degradation_method"],
                    "onset/propagation_onset_step": row["propagation_onset_step"],
                    "onset/propagation_method": row["propagation_method"],
                    "onset/propagation_lag": row["propagation_lag"],
                    "onset/status": row["status"],
                }
            )
        except Exception as e:
            warnings.warn(
                f"Onset ledger CSV write succeeded at '{path}', but mirroring "
                f"to the WandB run summary failed (non-fatal): {e}"
            )

    return path


def load_onset_ledger(
    exp_name: str,
    architecture: Optional[str] = None,
    root: str = DEFAULT_LEDGER_ROOT,
) -> pd.DataFrame:
    """Loads onset-event rows for `exp_name`.

    If `architecture` is given, loads just that architecture's CSV. Otherwise
    loads and concatenates every architecture directory found under
    `{root}/{exp_name}/architectures/`. Missing ledgers/directories return an
    empty, correctly-schema'd DataFrame rather than raising.
    """
    exp_root = Path(root) / exp_name / "architectures"

    if architecture is not None:
        return _load_existing(exp_root / architecture / "onset_events.csv")

    if not exp_root.exists():
        return _empty_ledger_df()

    frames = []
    for arch_dir in sorted(p for p in exp_root.iterdir() if p.is_dir()):
        csv_path = arch_dir / "onset_events.csv"
        if csv_path.exists():
            df = _load_existing(csv_path)
            if "architecture" not in df.columns or df["architecture"].isna().all():
                df["architecture"] = arch_dir.name
            frames.append(df)

    if not frames:
        return _empty_ledger_df()

    return pd.concat(frames, ignore_index=True)
