"""Exp 1/2 environments: the repo's wrapper chain, seeded MyoSuite, exact restore.

Exact restore records the simulator RNG state just before each reset plus the
actions taken since. A freshly built env restores by setting that RNG state,
resetting and replaying the actions. MuJoCo stepping is deterministic, so this
reproduces physics, wrapper counters and per-episode randomised task
parameters without enumerating task-specific fields.
"""

import copy
import importlib.util
import os
import sys
import types
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RescaleAction, TimeLimit

from scale_rl.envs.dmc import make_dmc_env
from scale_rl.envs.myosuite import MYOSUITE_TASKS_DICT, MyosuiteGymnasiumVersionWrapper
from scale_rl.envs.wrappers import DoNotTerminate, RepeatAction, ScaleReward
from scale_rl.envs.wrappers.vector import SyncVectorEnv

SUPPORTED_ENV_TYPES = ("dmc", "myosuite", "humanoid_bench")
HUMANOID_BENCH_TASKS = ("h1-reach-v0", "h1-run-v0")


def make_myosuite_env(env_name: str, seed: int) -> gym.Env:
    """Like scale_rl.envs.myosuite.make_myosuite_env, but seeds MyoSuite's RNG."""
    from myosuite.utils import gym as myo_gym

    return MyosuiteGymnasiumVersionWrapper(myo_gym.make(MYOSUITE_TASKS_DICT[env_name], seed=seed))


def _import_humanoid_bench():
    """Imports HumanoidBench (carlosferrazza/humanoid-bench @ cb11890, installed
    --no-deps) against this repo's pinned mujoco/dm_control.

    Its bundled copy of dm_control's mujoco/index.py carries a size table frozen
    at mujoco 3.1.6 and fails on 3.6.0; the installed dm_control index is the
    same module with a current table, so it is used instead. The torch-only
    policy wrappers (hierarchical tasks, unused by h1-reach / h1-run) are
    replaced by a stub that raises if called.
    """
    if "humanoid_bench" in sys.modules:
        return
    from dm_control.mujoco import index as dm_index

    gl = os.environ.get("MUJOCO_GL")
    if gl not in ("egl", "osmesa") or os.environ.get("PYOPENGL_PLATFORM") != gl:
        raise RuntimeError(
            "HumanoidBench builds an offscreen renderer: export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "
            "on GPU nodes (osmesa for both on CPU-only machines)."
        )
    spec = importlib.util.find_spec("humanoid_bench")
    if spec is None:
        raise ImportError("humanoid_bench is not installed; see scripts/install_humanoid_bench.sh")
    deps = types.ModuleType("humanoid_bench.dmc_deps")
    deps.__path__ = [os.path.join(spec.submodule_search_locations[0], "dmc_deps")]
    deps.dmc_index = dm_index
    sys.modules["humanoid_bench.dmc_deps"] = deps
    sys.modules["humanoid_bench.dmc_deps.dmc_index"] = dm_index

    def unavailable(*args, **kwargs):
        raise RuntimeError("HumanoidBench's torch policy wrappers are not available in this stack.")

    torch_stub = types.ModuleType("humanoid_bench.mjx.flax_to_torch")
    torch_stub.TorchModel = torch_stub.TorchPolicy = unavailable
    sys.modules["humanoid_bench.mjx.flax_to_torch"] = torch_stub
    import humanoid_bench  # noqa: F401  (registers the gym ids)


def make_humanoid_bench_env(env_name: str, seed: int) -> gym.Env:
    if env_name not in HUMANOID_BENCH_TASKS:
        raise ValueError(f"{env_name!r} is not one of the Exp 1/2 HumanoidBench tasks {HUMANOID_BENCH_TASKS}")
    _import_humanoid_bench()
    env = gym.make(env_name)
    env.reset(seed=seed)  # seeds the env's own np_random once, before ExactRestore wraps it
    return env


def get_env_rng_state(env: gym.Env, env_type: str) -> Any:
    base = env.unwrapped
    if env_type == "dmc":
        return base._env.task.random.get_state()
    if env_type == "myosuite":
        return base.np_random.np_random.bit_generator.state
    if env_type == "humanoid_bench":
        # Reach samples its goal from the global np.random at reset.
        return {"np_random": base.np_random.bit_generator.state, "global": np.random.get_state()}
    raise ValueError(f"unsupported env_type {env_type!r}")


def set_env_rng_state(env: gym.Env, env_type: str, state: Any) -> None:
    base = env.unwrapped
    if env_type == "dmc":
        base._env.task.random.set_state(state)
    elif env_type == "myosuite":
        base.np_random.np_random.bit_generator.state = state
    elif env_type == "humanoid_bench":
        base.np_random.bit_generator.state = state["np_random"]
        np.random.set_state(state["global"])
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
        """Replays to the recorded state; returns the last observation. For
        HumanoidBench this also moves the global np.random; callers restore
        the global state afterwards (Exp12Trainer.restore does)."""
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
    elif env_type == "humanoid_bench":
        env = make_humanoid_bench_env(env_name, seed)
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


def create_eval_env(env_type: str, seed: int, env_name: str, rescale_action: bool, no_termination: bool,
                    action_repeat: int, max_episode_steps: int, **kwargs) -> SyncVectorEnv:
    """A standalone evaluation env (reward scale 1), e.g. for the post-fork evaluations."""
    return SyncVectorEnv([
        lambda: _make_one_env(env_type, env_name, seed, rescale_action, no_termination, action_repeat, 1.0,
                              max_episode_steps)
    ])


def env_restore_state(vec_env: SyncVectorEnv) -> Dict[str, Any]:
    return {
        "env": vec_env.envs[0].restore_state(),
        "action_space_rng": copy.deepcopy(vec_env.action_space.np_random.bit_generator.state),
    }


def restore_env(vec_env: SyncVectorEnv, state: Dict[str, Any]) -> None:
    vec_env.envs[0].restore(state["env"])
    vec_env.action_space.np_random.bit_generator.state = state["action_space_rng"]
