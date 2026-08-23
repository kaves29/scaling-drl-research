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
