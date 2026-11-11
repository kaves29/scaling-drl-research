import gymnasium as gym
import numpy as np


class RepeatAction(gym.Wrapper):
    def __init__(self, env: gym.Env, action_repeat=4):
        super().__init__(env)
        self._action_repeat = action_repeat
        self.last_executed_repeats = 0
        self.raw_step_limit = None

    def step(self, action: np.ndarray):
        total_reward = 0.0
        terminated = False
        truncated = False
        combined_info = {}

        self.last_executed_repeats = 0
        for _ in range(
            min(self._action_repeat, self.raw_step_limit)
            if self.raw_step_limit is not None
            else self._action_repeat
        ):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.last_executed_repeats += 1
            total_reward += float(reward)
            combined_info.update(info)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, combined_info


def repeat_wrapper(vector_env):
    """Find the repeat wrapper for the single-environment Exp12 runner."""
    wrapped = vector_env.envs[0] if hasattr(vector_env, "envs") else vector_env
    while wrapped is not None:
        if isinstance(wrapped, RepeatAction):
            return wrapped
        wrapped = getattr(wrapped, "env", None)
    return None


def executed_raw_steps(vector_env):
    wrapper = repeat_wrapper(vector_env)
    return wrapper.last_executed_repeats if wrapper is not None else 1


def limit_raw_steps(vector_env, remaining):
    if remaining < 1:
        raise ValueError("raw step allowance must be positive")
    wrapper = repeat_wrapper(vector_env)
    if wrapper is not None:
        wrapper.raw_step_limit = int(remaining)
