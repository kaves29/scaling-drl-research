"""Persistence for the raw per-run metric time series onset analysis needs.

This is deliberately separate from onset detection (analysis/onset_detection.py)
and from baseline calibration (analysis/baseline_calibration.py): training
writes these files, post-hoc analysis reads them back. Nothing here interprets
the numbers scientifically.

Metrics are recorded once per `logging_per_interaction_step` interval — the
same cadence the rest of this repo's training loop (run.py / experiments/angle_1.py)
already uses for WandB/CSV logging — using `interaction_step` (not `env_step`)
as the primary temporal axis, per project convention. Values are whatever the
existing SAC update step already computed (`train/td_error_var`,
`train/actor_grad_cosine`; see scale_rl/agents/sac/sac_update.py), averaged
over the interval exactly the way the existing WandbTrainerLogger does.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from utils.atomic_io import atomic_write_text

METRICS_ROOT = "results/metrics"

METRIC_COLUMNS = ["interaction_step", "env_step", "td_error_variance", "actor_grad_cosine"]


@dataclass(frozen=True)
class RunIdentity:
    """Identifies a single run for metric storage, baseline calibration, and the ledger.

    `experiment` is the experiment-routing name (e.g. "angle_1"), matching
    the ledger's `exp_name` scoping — not the pre-existing free-form
    `cfg.exp_name` config field used elsewhere in this repo for WandB
    analysis-notebook grouping.
    """

    experiment: str
    architecture: str
    environment: str
    seed: int

    @property
    def run_key(self) -> str:
        return f"{self.experiment}_{self.architecture}_{self.environment}_seed{self.seed}"


def metrics_path(identity: RunIdentity, root: str = METRICS_ROOT) -> Path:
    return (
        Path(root)
        / identity.experiment
        / identity.architecture
        / identity.environment
        / f"{identity.run_key}.csv"
    )


class MetricsRecorder:
    """Collects {interaction_step, env_step, td_error_variance, actor_grad_cosine}
    rows during training and persists them atomically.

    Intentionally does not compute anything: callers pass in whatever the
    training loop's own logging step already produced (e.g. the averaged
    `train/td_error_var` / `train/actor_grad_cosine` for the interval).
    """

    def __init__(self, identity: RunIdentity, root: str = METRICS_ROOT):
        self.identity = identity
        self.path = metrics_path(identity, root=root)
        self._rows = []

    def record(self, interaction_step: int, env_step: int, metrics: Dict[str, float]) -> None:
        self._rows.append(
            {
                "interaction_step": interaction_step,
                "env_step": env_step,
                "td_error_variance": metrics.get("train/td_error_var"),
                "actor_grad_cosine": metrics.get("train/actor_grad_cosine"),
            }
        )

    def load_existing_up_to(self, max_interaction_step: int) -> None:
        """Seeds the in-memory buffer from a prior run for checkpoint-resume continuity."""
        if not self.path.exists():
            return
        df = pd.read_csv(self.path)
        df = df[df["interaction_step"] <= max_interaction_step]
        self._rows = df.to_dict("records")

    def flush(self) -> Path:
        df = pd.DataFrame(self._rows, columns=METRIC_COLUMNS)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        atomic_write_text(self.path, buf.getvalue())
        return self.path


def load_metrics(identity: RunIdentity, root: str = METRICS_ROOT) -> pd.DataFrame:
    path = metrics_path(identity, root=root)
    if not path.exists():
        return pd.DataFrame(columns=METRIC_COLUMNS)
    return pd.read_csv(path)
