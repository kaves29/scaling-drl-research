import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from scale_rl.agents.base_agent import AgentWrapper, BaseAgent
from scale_rl.agents.wrappers.utils import RunningMeanStd


class ObservationNormalizer(AgentWrapper):
    """
    This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Observation statistics is updated only on sample_actions with training==True
    """

    def __init__(self, agent: BaseAgent, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            agent (BaseAgent): The agent to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        AgentWrapper.__init__(self, agent)

        self.obs_rms = RunningMeanStd(
            shape=self.agent._observation_space.shape,
            dtype=self.agent._observation_space.dtype,
        )
        self.epsilon = epsilon

    def _normalize(self, observations):
        return (observations - self.obs_rms.mean) / np.sqrt( # returning error because of shape mismatch (258, 17) not compatiable with (4, 17)
            self.obs_rms.var + self.epsilon
        )

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        """
        Defines the sample action function with normalized observation.
        """

        observations = prev_timestep["next_observation"]
        if training:
            self.obs_rms.update(observations)
        prev_timestep["next_observation"] = self._normalize(observations)

        return self.agent.sample_actions(
            interaction_step=interaction_step,
            prev_timestep=prev_timestep,
            training=training,
        )

    def update(self, update_step: int, batch: Dict[str, np.ndarray]):
        batch["observation"] = self._normalize(batch["observation"])
        batch["next_observation"] = self._normalize(batch["next_observation"])
        return self.agent.update(
            update_step=update_step,
            batch=batch,
        )
    def get_metrics(self, update_step: int, batch: Dict[str, np.ndarray]):
        batch["observation"] = self._normalize(batch["observation"])
        batch["next_observation"] = self._normalize(batch["next_observation"])
        return self.agent.get_metrics(
            update_step=update_step,
            batch=batch,
        )

    def get_q_value(self, observations: np.ndarray, actions: np.ndarray):
        """Normalizes observations before delegating, matching update()/get_metrics().

        Without this override, AgentWrapper.__getattr__ would forward
        get_q_value straight to the wrapped SACAgent with *raw* observations,
        silently producing wrong Q-values whenever normalize_observation=true
        (the critic was trained on normalized inputs).
        """
        return self.agent.get_q_value(
            observations=self._normalize(observations),
            actions=actions,
        )

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """Delegates to the wrapped agent, then additionally persists
        obs_rms (mean/var/count).

        Without this override, AgentWrapper.__getattr__ would forward
        save_checkpoint straight to the wrapped SACAgent, which has no
        knowledge of obs_rms - silently dropping the running normalization
        statistics on every save. Since the actor/critic were trained on
        normalized observations, restoring a checkpoint without obs_rms
        would reconstruct an agent whose network parameters no longer match
        the input distribution they were trained on.
        """
        self.agent.save_checkpoint(checkpoint_dir)
        state = {
            "mean": self.obs_rms.mean,
            "var": self.obs_rms.var,
            "count": self.obs_rms.count,
        }
        with open(Path(checkpoint_dir) / "obs_rms.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Delegates to the wrapped agent, then restores obs_rms - see
        save_checkpoint for why this is required for correctness."""
        self.agent.load_checkpoint(checkpoint_dir)
        obs_rms_path = Path(checkpoint_dir) / "obs_rms.pkl"
        if not obs_rms_path.exists():
            raise FileNotFoundError(
                f"No obs_rms checkpoint found at {obs_rms_path}. This "
                f"checkpoint was saved without ObservationNormalizer.save_checkpoint "
                f"(e.g. by an older code path that only called the wrapped "
                f"agent's save_checkpoint directly); refusing to silently "
                f"resume with freshly-initialized (mean=0, var=1) normalization "
                f"statistics, since that would not match the actor/critic's "
                f"actual training distribution."
            )
        with open(obs_rms_path, "rb") as f:
            state = pickle.load(f)
        self.obs_rms.mean = state["mean"]
        self.obs_rms.var = state["var"]
        self.obs_rms.count = state["count"]
