"""Scheduling and persistence of the plasticity probes within a run."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import orbax.checkpoint
import pandas as pd

from experiments.exp12.probe import check_steps, critic_optimizer, iqm, probe_config, run_probe, summarize
from experiments.exp12.trigger import bootstrap_interval, f_star, trigger_config, triggered
from utils.atomic_io import atomic_write_text

FRESH_CRITIC_DIR = "fresh_critic"


class RunProbes:
    """Fresh-critic probe before the first update, then paired checks at k/checks of the run.

    Records (one dict per check) live in trainer.extra_state, so they are saved
    and restored with the training state; learning curves go to probes/*.npz.
    """

    def __init__(self, trainer, run_dir: str):
        self.trainer = trainer
        self.cfg = probe_config(trainer.cfg)
        self.seed = int(trainer.cfg.seed)
        self.dir = Path(run_dir) / "probes"
        self.fresh_path = Path(run_dir) / FRESH_CRITIC_DIR
        self.critic_def = trainer._sac_agent.critic.network_def
        self.tx = critic_optimizer(trainer.cfg.agent)
        self.trigger = trigger_config(trainer.cfg)
        checks = check_steps(int(trainer.cfg.num_interaction_steps), self.cfg.checks)
        if checks[0] <= int(trainer.cfg.buffer.min_length):
            raise ValueError(
                f"The first probe check (interaction step {checks[0]}) must come after the fresh-critic "
                f"probe at buffer.min_length={trainer.cfg.buffer.min_length}."
            )
        self.check_index = {step: k for k, step in enumerate(checks, 1)}
        self.fresh = None
        if self.fresh_path.exists():
            self.fresh = orbax.checkpoint.PyTreeCheckpointer().restore(str(self.fresh_path))["params"]

    @property
    def records(self) -> List[Dict]:
        return self.trainer.extra_state.setdefault("probe_records", [])

    def capture_fresh(self, trainer) -> None:
        """Stores the untrained critic and probes it (check 0: the run's initial plasticity score)."""
        if any(r["check_index"] == 0 for r in self.records):
            return
        self.fresh = trainer._sac_agent.critic.params
        orbax.checkpoint.PyTreeCheckpointer().save(str(self.fresh_path), {"params": self.fresh}, force=True)
        score = self._probe(0, {"fresh": (self.critic_def, self.fresh)})["fresh"]["score"]
        self.records.append({
            "check_index": 0,
            "interaction_step": trainer.interaction_step,
            **{f"score_fresh_r{r}": float(v) for r, v in enumerate(score)},
            "score_fresh_iqm": iqm(score) if np.all(np.isfinite(score)) else float("nan"),
        })

    def maybe_check(self, trainer) -> None:
        k = self.check_index.get(trainer.interaction_step)
        if k is None:
            return
        critics = {
            "current": (self.critic_def, trainer._sac_agent.critic.params),
            "fresh": (self.critic_def, self.fresh),
        }
        self.record_check(k, self._probe(k, critics))

    @property
    def f_star(self):
        """First eligible triggering check ({check_index, interaction_step}) or None."""
        return self.trainer.extra_state.get("f_star")

    def record_check(self, k: int, result: Dict) -> Dict:
        s = summarize(result)
        tc = self.trigger
        low, high = bootstrap_interval(s["loss_rounds"], self.seed, k, tc.resamples, tc.confidence)
        row = {
            "check_index": k,
            "interaction_step": self.trainer.interaction_step,
            **{f"score_current_r{r}": float(v) for r, v in enumerate(s["score_current_rounds"])},
            **{f"score_fresh_r{r}": float(v) for r, v in enumerate(s["score_fresh_rounds"])},
            **{f"loss_r{r}": float(v) for r, v in enumerate(s["loss_rounds"])},
            "score_current_iqm": s["score_current_iqm"],
            "score_fresh_iqm": s["score_fresh_iqm"],
            "loss_iqm": s["loss_iqm"],
            "ci_low": low,
            "ci_high": high,
            "triggered": triggered(low, tc.null_threshold) and s["valid"],
            "valid": s["valid"],
        }
        self.records.append(row)
        if self.f_star is None:
            self.trainer.extra_state["f_star"] = f_star(
                self.records, self.cfg.checks, tc.consecutive_checks, tc.eligible_fraction
            )
        return row

    def _probe(self, k: int, critics: Dict) -> Dict:
        t = self.trainer
        result = run_probe(t.agent, t.buffer, self.critic_def, critics, self.tx, self.seed, k, self.cfg)
        self.dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.dir / f"check_{k:02d}.npz",
            **{f"{name}_{key}": value for name, fields in result.items() for key, value in fields.items()},
        )
        return result

    def write_csv(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.dir / "probe_checks.csv", pd.DataFrame(self.records).to_csv(index=False))
