"""Exp 1/2 environments: the repo's wrapper chain, seeded MyoSuite, exact restore.

Exact restore records the simulator RNG state just before each reset plus the
actions taken since. A freshly built env restores by setting that RNG state,
resetting and replaying the actions. MuJoCo stepping is deterministic, so this
reproduces physics, wrapper counters and per-episode randomised task
parameters without enumerating task-specific fields.
"""

import copy
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RescaleAction, TimeLimit

from scale_rl.envs.dmc import make_dmc_env
from scale_rl.envs.myosuite import MYOSUITE_TASKS_DICT, MyosuiteGymnasiumVersionWrapper
from scale_rl.envs.wrappers import DoNotTerminate, RepeatAction, ScaleReward
from scale_rl.envs.wrappers.vector import SyncVectorEnv

SUPPORTED_ENV_TYPES = ("dmc", "myosuite")


def make_myosuite_env(env_name: str, seed: int) -> gym.Env:
    """Like scale_rl.envs.myosuite.make_myosuite_env, but seeds MyoSuite's RNG."""
    from myosuite.utils import gym as myo_gym

    return MyosuiteGymnasiumVersionWrapper(myo_gym.make(MYOSUITE_TASKS_DICT[env_name], seed=seed))


def get_env_rng_state(env: gym.Env, env_type: str) -> Any:
    base = env.unwrapped
    if env_type == "dmc":
        return base._env.task.random.get_state()
    if env_type == "myosuite":
        return base.np_random.np_random.bit_generator.state
    raise ValueError(f"unsupported env_type {env_type!r}")


def set_env_rng_state(env: gym.Env, env_type: str, state: Any) -> None:
    base = env.unwrapped
    if env_type == "dmc":
        base._env.task.random.set_state(state)
    elif env_type == "myosuite":
        base.np_random.np_random.bit_generator.state = state
    else:
        raise ValueError(f"unsupported env_type {env_type!r}")


class ExactRestore(gym.Wrapper):
    """Records what a fresh env needs to replay itself to the current state."""

    def __init__(self, env: gym.Env, env_type: str):
        super().__init__(env)
        self._env_type = env_type
        self._rng_at_reset = None
        self._actions = []

    def reset(self, **kwargs):
        if kwargs.get("seed") is not None:
            raise ValueError("ExactRestore envs must not be reset with an explicit seed.")
        self._rng_at_reset = copy.deepcopy(get_env_rng_state(self.env, self._env_type))
        self._actions = []
        return self.env.reset(**kwargs)

    def step(self, action):
        self._actions.append(np.array(action, copy=True))
        return self.env.step(action)

    def restore_state(self) -> Dict[str, Any]:
        return {
            "rng_at_reset": copy.deepcopy(self._rng_at_reset),
            "actions": [a.copy() for a in self._actions],
        }

    def restore(self, state: Dict[str, Any]):
        """Replays to the recorded state; returns the last observation."""
        set_env_rng_state(self.env, self._env_type, state["rng_at_reset"])
        obs, _ = self.reset()
        for action in state["actions"]:
            obs, *_ = self.step(action)
        return obs


def _make_one_env(
    env_type, env_name, seed, rescale_action, no_termination, action_repeat, reward_scale, max_episode_steps
) -> gym.Env:
    if env_type == "dmc":
        env = make_dmc_env(env_name, seed)
    elif env_type == "myosuite":
        env = make_myosuite_env(env_name, seed)
    else:
        raise ValueError(f"unsupported env_type {env_type!r}; supported: {SUPPORTED_ENV_TYPES}")
    if rescale_action:
        env = RescaleAction(env, -1.0, 1.0)
    if no_termination:
        env = DoNotTerminate(env)
    env = TimeLimit(env, max_episode_steps)
    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)
    env = ScaleReward(env, reward_scale)
    env = ExactRestore(env, env_type)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    return env


def create_envs(
    env_type: str,
    seed: int,
    env_name: str,
    num_train_envs: int,
    num_eval_envs: int,
    rescale_action: bool,
    no_termination: bool,
    action_repeat: int,
    reward_scale: float,
    max_episode_steps: int,
    **kwargs,
) -> Tuple[SyncVectorEnv, SyncVectorEnv]:
    """scale_rl.envs.create_envs for a single train and eval env, with ExactRestore."""
    if num_train_envs != 1 or num_eval_envs != 1:
        raise ValueError("Exp 1/2 requires num_train_envs == num_eval_envs == 1.")

    def make(scale):
        return SyncVectorEnv([
            lambda: _make_one_env(
                env_type, env_name, seed, rescale_action, no_termination, action_repeat, scale, max_episode_steps
            )
        ])

    return make(reward_scale), make(1.0)


def env_restore_state(vec_env: SyncVectorEnv) -> Dict[str, Any]:
    return {
        "env": vec_env.envs[0].restore_state(),
        "action_space_rng": copy.deepcopy(vec_env.action_space.np_random.bit_generator.state),
    }


def restore_env(vec_env: SyncVectorEnv, state: Dict[str, Any]) -> None:
    vec_env.envs[0].restore(state["env"])
    vec_env.action_space.np_random.bit_generator.state = state["action_space_rng"]
