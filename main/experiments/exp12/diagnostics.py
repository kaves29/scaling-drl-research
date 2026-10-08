"""Exp 1/2 actor diagnostics I1-I4 (Schaul 2022; Tang & Berseth 2024; Bjorck et al. 2022).

All are logged at the existing cadence (logging windows of logging_per_interaction_step
interaction steps) with no new host syncs:
    train/policy_churn, train/churn  existing L2 change of tanh(mean) on the first batch (kept)
    train/policy_kl          I1: per-update KL(pi_t || pi_{t-1}) on a 256-state replay reference
                             batch, window mean. The batch is redrawn at the start of each window
                             from a dedicated stream (seed, window) and normalised with the obs
                             statistics of that moment, so the KL reflects parameter updates only.
    train/actor_gnorm        I2: existing per-update actor gradient norm, window mean (kept)
    train/actor_gnorm_std    I2: population SD of the per-update actor gradient norms in the window
    train/actor_action       I3: existing mean |a| of the actor-update sampled actions (kept)
    train/actor_saturation   I3: fraction of those action components with |a| > 0.99
"""

from typing import Dict, List, Optional

import numpy as np

from experiments.angle_1 import PendingUpdateMetrics

KL_STREAM = 0x4B4C5246  # "KLRF"


class ActorDiagnostics:
    def __init__(self, cfg):
        d = cfg.diagnostics
        self.enabled = bool(d.enabled)
        self.reference_size = int(d.kl_reference_size)
        self.saturation_threshold = float(d.saturation_threshold)
        self.seed = int(cfg.seed)
        self.window_steps = int(cfg.logging_per_interaction_step)
        self.window: Optional[int] = None
        self.kl_ref: Optional[np.ndarray] = None
        self.gnorm: List[float] = []

    def update_kwargs(self, trainer) -> Dict:
        """Extra arguments for agent.update_many at this interaction step."""
        if not self.enabled:
            return {}
        window = (trainer.interaction_step - 1) // self.window_steps
        if window != self.window:
            rng = np.random.default_rng([self.seed, KL_STREAM, window])
            idx = rng.integers(0, trainer.buffer._num_in_buffer, size=self.reference_size)
            obs = trainer.buffer._observations[idx]
            if hasattr(trainer.agent, "_normalize"):
                obs = trainer.agent._normalize(obs)
            self.kl_ref, self.window = np.asarray(obs, np.float32), window
        return {"saturation_threshold": self.saturation_threshold, "kl_ref_observations": self.kl_ref}

    def collect(self, host_info: Dict) -> None:
        if self.enabled and "train/actor_gnorm" in host_info:
            self.gnorm.extend(float(v) for v in np.ravel(host_info["train/actor_gnorm"]))

    def window_metrics(self) -> Dict[str, float]:
        """Closes the logging window."""
        values, self.gnorm = self.gnorm, []
        if not self.enabled or not values:
            return {}
        return {"train/actor_gnorm_std": float(np.std(values))}

    def state(self) -> Dict:
        return {"window": self.window, "kl_ref": self.kl_ref, "gnorm": list(self.gnorm)}

    def load_state(self, state: Dict) -> None:
        self.window, self.kl_ref, self.gnorm = state["window"], state["kl_ref"], list(state["gnorm"])


class DiagnosticPendingUpdateMetrics(PendingUpdateMetrics):
    """PendingUpdateMetrics that also hands the host copy of each flushed update to the diagnostics."""

    def __init__(self, logger, actor_grad_cosine_every: int, diagnostics: ActorDiagnostics):
        super().__init__(logger, actor_grad_cosine_every)
        self._diagnostics = diagnostics

    def flush(self) -> None:
        host_infos = self._materialize()
        for info in host_infos:
            self._diagnostics.collect(info)
        self._replay(host_infos)
