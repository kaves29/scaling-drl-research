"""Shared ordered metric transfer for current Exp1 and Exp2 training."""

import jax

ACTOR_GRAD_COSINE_KEY = "train/actor_grad_cosine"
DONE_MARKER = "DONE"


class PendingUpdateMetrics:
    """Keep update metrics on device and replay them in original update order."""

    def __init__(self, logger, actor_grad_cosine_every: int):
        self._logger = logger
        self._actor_grad_cosine_every = actor_grad_cosine_every
        self._pending = []

    def add(self, first_update_step: int, update_info: dict) -> None:
        self._pending.append((first_update_step, update_info))

    def flush(self) -> None:
        self._replay(self._materialize())

    def _materialize(self):
        return [jax.device_get(info) for _, info in self._pending]

    def _replay(self, host_infos) -> None:
        for (first_update_step, _), info in zip(self._pending, host_infos):
            for i in range(len(info[ACTOR_GRAD_COSINE_KEY])):
                row = {key: float(values[i]) for key, values in info.items()}
                if (first_update_step + i) % self._actor_grad_cosine_every != 0:
                    del row[ACTOR_GRAD_COSINE_KEY]
                self._logger.update_metric(**row)
        self._pending = []
