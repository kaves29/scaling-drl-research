"""Scheduling and persistence of the plasticity probes within a run."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import orbax.checkpoint
import pandas as pd

from experiments.exp12 import twin
from experiments.exp12.injection import InjectedSACCritic, injected_optimizer
from experiments.exp12.probe import check_steps, critic_optimizer, iqm, probe_config, run_probe, summarize
from experiments.exp12.trigger import bootstrap_interval, f_star, trigger_config, triggered
from utils.atomic_io import atomic_write_text

FRESH_CRITIC_DIR = "fresh_critic"


class RunProbes:
    """Fresh-critic probe before the first update, then paired checks at k/checks of the run.

    Records (one dict per check) live in trainer.extra_state, so they are saved
    and restored with the training state; learning curves go to probes/*.npz.
    """

    def __init__(self, trainer, run_dir: str, fresh_dir: str = None):
        self.trainer = trainer
        self.cfg = probe_config(trainer.cfg)
        self.seed = int(trainer.cfg.seed)
        self.dir = Path(run_dir) / "probes"
        # Arms read the fresh reference stored by the parent Exp 1 run.
        self.fresh_path = Path(fresh_dir) if fresh_dir is not None else Path(run_dir) / FRESH_CRITIC_DIR
        a = trainer.cfg.agent
        # The original architecture: fresh reference and target generator, also for an injected arm.
        self.critic_def = original_critic_def(trainer.cfg)
        self.tx = critic_optimizer(a)
        self.injected_tx = injected_optimizer(float(a.critic_learning_rate), float(a.critic_weight_decay))
        self.forced_check = trainer.cfg.testing.force_trigger_check
        if self.forced_check is not None and trainer.cfg.run_role != "dev":
            raise ValueError("testing.force_trigger_check is a test-only hook and requires run_role=dev.")
        self.trigger = trigger_config(trainer.cfg)
        if self.forced_check is not None and self.forced_check < self.trigger.consecutive_checks:
            raise ValueError(f"testing.force_trigger_check must be >= trigger.consecutive_checks "
                             f"({self.trigger.consecutive_checks}): f*_run needs that many firing checks")
        checks = check_steps(int(trainer.cfg.num_interaction_steps), self.cfg.checks)
        if checks[0] <= int(trainer.cfg.buffer.min_length):
            raise ValueError(
                f"The first probe check (interaction step {checks[0]}) must come after the fresh-critic "
                f"probe at buffer.min_length={trainer.cfg.buffer.min_length}."
            )
        self.check_every = checks[0]
        self.check_index = {step: k for k, step in enumerate(checks, 1)}
        self.fresh = None
        if self.fresh_path.exists():
            self.fresh = orbax.checkpoint.PyTreeCheckpointer().restore(str(self.fresh_path))["params"]

    def extend_to(self, last_step: int) -> None:
        """Adds checks on the same k/checks grid beyond 100% of the budget (post-fork arms)."""
        k = 1
        while k * self.check_every <= last_step:
            self.check_index[k * self.check_every] = k
            k += 1

    def current_critic(self, trainer):
        critic = trainer._sac_agent.critic
        if isinstance(critic.network_def, InjectedSACCritic):
            return (critic.network_def, critic.params, self.injected_tx)
        return (critic.network_def, critic.params)

    @property
    def records(self) -> List[Dict]:
        return self.trainer.extra_state.setdefault("probe_records", [])

    def capture_fresh(self, trainer) -> None:
        """Stores the untrained critic and probes it (check 0: the run's initial plasticity score)."""
        if any(r["check_index"] == 0 for r in self.records):
            return
        self.fresh = trainer._sac_agent.critic.params
        orbax.checkpoint.PyTreeCheckpointer().save(str(self.fresh_path), {"params": self.fresh}, force=True)
        result = self._probe(0, {"fresh": (self.critic_def, self.fresh)})
        score = result["fresh"]["score"]
        self.records.append({
            "check_index": 0,
            "interaction_step": trainer.interaction_step,
            **{f"score_fresh_r{r}": float(v) for r, v in enumerate(score)},
            "score_fresh_iqm": iqm(score) if np.all(np.isfinite(score)) else float("nan"),
            **per_network_fields(result, fresh_only=True),
        })

    def maybe_check(self, trainer) -> None:
        k = self.check_index.get(trainer.interaction_step)
        if k is None:
            return
        critics = {"current": self.current_critic(trainer), "fresh": (self.critic_def, self.fresh)}
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
            **per_network_fields(result),
        }
        if self.forced_check is not None and self.forced_check - tc.consecutive_checks < k <= self.forced_check:
            row["triggered"] = True  # TEST-ONLY hook (testing.force_trigger_check, run_role=dev only)
            row["forced"] = True
        self.records.append(row)
        if self.f_star is None:
            self.trainer.extra_state["f_star"] = f_star(
                self.records, self.cfg.checks, tc.consecutive_checks, tc.eligible_fraction
            )
        return row

    def _probe(self, k: int, critics: Dict) -> Dict:
        t = self.trainer
        result = run_probe_networks(t.agent, t.buffer, self.critic_def, critics, self.tx, self.injected_tx,
                                    self.seed, k, self.cfg)
        self.dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.dir / f"check_{k:02d}.npz",
            **{f"{name}_{key}": value for name, fields in result.items() for key, value in fields.items()},
        )
        return result

    def write_csv(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.dir / "probe_checks.csv", pd.DataFrame(self.records).to_csv(index=False))


def run_probe_networks(agent, buffer, target_def, critics, tx, injected_tx, seed, k, cfg, shared_offset=None):
    """run_probe, with every twin critic probed as its two networks (amendment (z)); others unchanged."""
    critics, twins = twin.expand(critics, injected_tx)
    result = run_probe(agent, buffer, target_def, critics, tx, seed, k, cfg, shared_offset)
    return twin.combine(result, twins) if twins else result


def per_network_fields(result: Dict, fresh_only: bool = False) -> Dict:
    """Twin critics: each network's paired L = P(fresh_qn) - P(current_qn) per round, and its scores."""
    if "fresh_q1" not in result:
        return {}
    if fresh_only:
        return {f"score_fresh_q{n}_r{r}": float(v) for n in range(1, twin.NUM_QS + 1)
                for r, v in enumerate(result[f"fresh_q{n}"]["score"])}
    out = {}
    for n in range(1, twin.NUM_QS + 1):
        s = summarize(result, f"current_q{n}", f"fresh_q{n}")
        out.update({f"score_current_q{n}_r{r}": float(v) for r, v in enumerate(s["score_current_rounds"])})
        out.update({f"score_fresh_q{n}_r{r}": float(v) for r, v in enumerate(s["score_fresh_rounds"])})
        out.update({f"loss_q{n}_r{r}": float(v) for r, v in enumerate(s["loss_rounds"])})
        out[f"loss_q{n}_iqm"] = s["loss_iqm"]
    return out


def original_critic_def(cfg):
    import jax.numpy as jnp

    from scale_rl.agents.sac.sac_network import SACCritic

    a = cfg.agent
    return SACCritic(a.critic_block_type, int(a.critic_num_blocks), int(a.critic_hidden_dim),
                     jnp.float16 if a.mixed_precision else jnp.float32)
