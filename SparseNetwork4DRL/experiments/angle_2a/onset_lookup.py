"""Seed-specific, architecture-specific Angle 1 onset lookup for Angle 2A.

This is a thin, deterministic query over the existing canonical ledger
(utils/onset_ledger.py) - it does not re-run or approximate onset detection.
Each scaled architecture gets its own independently-looked-up onset; nothing
here ever averages, blends, or substitutes across architectures or seeds.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from experiments.angle_2a.errors import Angle2AOnsetLookupError
from utils.onset_ledger import DEFAULT_LEDGER_ROOT, load_onset_ledger

USABLE_STATUSES = {"success"}


@dataclass(frozen=True)
class OnsetLookupResult:
    onset_step: int
    architecture: str
    environment: str
    seed: int
    run_key: str
    exact_run_name: Optional[str]
    wandb_run_id: Optional[str]
    critic_degradation_method: Optional[str]


def lookup_critic_degradation_onset(
    architecture: str,
    environment: str,
    seed: int,
    source_experiment: str = "angle_1",
    ledger_root: str = DEFAULT_LEDGER_ROOT,
) -> OnsetLookupResult:
    """Returns the exact, single Angle 1 critic-degradation onset for this
    (architecture, environment, seed), or raises Angle2AOnsetLookupError.

    Raises rather than guessing whenever the ledger entry is missing,
    duplicated (should be impossible given run_key uniqueness, but checked
    defensively), or not in a 'success' state - per spec, Angle 2A must never
    invent, average, or borrow another seed's/architecture's onset.
    """
    df = load_onset_ledger(source_experiment, architecture=architecture, root=ledger_root)

    if df.empty:
        raise Angle2AOnsetLookupError(
            f"No Angle 1 onset ledger found for experiment='{source_experiment}', "
            f"architecture='{architecture}' at root='{ledger_root}'. Angle 2A "
            f"requires a completed Angle 1 critic_degradation=true run for "
            f"this exact architecture before it can run."
        )

    matches = df[(df["environment"] == environment) & (df["seed"] == seed)]

    if len(matches) == 0:
        raise Angle2AOnsetLookupError(
            f"No Angle 1 onset ledger entry for experiment='{source_experiment}', "
            f"architecture='{architecture}', environment='{environment}', "
            f"seed={seed}. Run Angle 1 with critic_degradation=true for this "
            f"exact (architecture, environment, seed) first; Angle 2A will not "
            f"substitute another seed's or architecture's onset."
        )

    if len(matches) > 1:
        raise Angle2AOnsetLookupError(
            f"Ambiguous ledger: {len(matches)} rows match experiment="
            f"'{source_experiment}', architecture='{architecture}', "
            f"environment='{environment}', seed={seed}. run_key is supposed "
            f"to be unique per (experiment, architecture, environment, seed); "
            f"this ledger file appears to have been modified outside "
            f"utils.onset_ledger.log_onset_event. Refusing to guess which row "
            f"is authoritative."
        )

    row = matches.iloc[0]
    status = row["status"]

    if status not in USABLE_STATUSES:
        raise Angle2AOnsetLookupError(
            f"Angle 1 onset for architecture='{architecture}', "
            f"environment='{environment}', seed={seed} has status='{status}' "
            f"(requires one of {sorted(USABLE_STATUSES)}). This matchup "
            f"requires manual review of the Angle 1 result before Angle 2A "
            f"can use it; refusing to proceed with an unconfirmed onset."
        )

    onset_step = row["critic_degradation_onset_step"]
    if pd.isna(onset_step):
        raise Angle2AOnsetLookupError(
            f"Ledger row for architecture='{architecture}', "
            f"environment='{environment}', seed={seed} has status='success' "
            f"but critic_degradation_onset_step is missing/NaN. Refusing to "
            f"proceed with an undefined onset step."
        )

    return OnsetLookupResult(
        onset_step=int(onset_step),
        architecture=architecture,
        environment=environment,
        seed=seed,
        run_key=str(row["run_key"]),
        exact_run_name=row.get("exact_run_name"),
        wandb_run_id=row.get("wandb_run_id"),
        critic_degradation_method=row.get("critic_degradation_method"),
    )
