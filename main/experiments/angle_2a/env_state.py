"""Exact environment-state capture/restore for Angle 2A's Monte Carlo rollouts.

Angle 2A's protocol requires resetting the environment to an exact previously
visited state `s` and forcing action `a`. This repo's replay buffer only
stores flattened *observations* (see scale_rl/buffers/numpy_buffer.py), which
are not guaranteed to be sufficient to reconstruct the underlying simulator
state exactly (many dm_control tasks drop globally-invariant coordinates from
the observation). Re-deriving state from the observation would therefore be
an approximation, not an exact reproduction.

Instead, this module captures the dm_control physics state directly, at
transition-collection time, via `physics.get_state()` / `physics.set_state()`
(see dm_control.mujoco.Physics) - the same mechanism dm_control itself uses
for state save/restore - plus the outer `TimeLimit` wrapper's elapsed-step
counter, so truncation semantics on resume match what they would have been
at the original moment (not a stale/reset counter).

Known, documented limitation (see Angle 2A deliverables notes): this captures
*dynamical* state (qpos/qvel/act) only. A small number of dm_control tasks
additionally randomize static, non-dynamical model parameters once per
episode (e.g. a target body position written to `physics.named.model`, not
`physics.data`); those are not captured/restored here. This implementation
does not silently claim exactness for such tasks - it should be re-verified
against the specific dm_control task's `initialize_episode` before treating
cross-episode probe rollouts as bit-exact for that task.

Only env_type='dmc' is supported. This is enforced explicitly (loud failure)
rather than silently attempting an inexact fallback for other env types.
"""

from typing import Any, Dict, Optional

from gymnasium.wrappers import TimeLimit

from experiments.angle_2a.errors import Angle2AEnvironmentError


def assert_dmc_env_type(env_type: str) -> None:
    if env_type != "dmc":
        raise Angle2AEnvironmentError(
            f"Angle 2A's exact-state Monte Carlo rollout only supports "
            f"env_type='dmc' (dm_control); got env_type='{env_type}'. "
            f"Extending this to other env types requires implementing "
            f"get/set-state support for that simulator; see "
            f"experiments/angle_2a/env_state.py."
        )


def _find_wrapper(env, cls) -> Optional[Any]:
    e = env
    while e is not None:
        if isinstance(e, cls):
            return e
        e = getattr(e, "env", None)
    return None


def _get_dmc_physics(env):
    base = getattr(env, "unwrapped", env)
    physics = getattr(base, "physics", None)
    if physics is None:
        inner = getattr(base, "_env", None)
        physics = getattr(inner, "physics", None)
    if physics is None:
        raise Angle2AEnvironmentError(
            "Could not locate dm_control physics on this environment "
            "(checked env.unwrapped.physics and env.unwrapped._env.physics). "
            "Angle 2A's exact Monte Carlo rollout requires direct physics "
            "access; only env_type='dmc' is supported."
        )
    return physics


def capture_env_state(env) -> Dict[str, Any]:
    """Captures everything needed to exactly resume `env` from its current instant."""
    physics = _get_dmc_physics(env)
    time_limit = _find_wrapper(env, TimeLimit)
    return {
        "physics_state": physics.get_state().copy(),
        "elapsed_steps": time_limit._elapsed_steps if time_limit is not None else None,
    }


def restore_env_state(env, captured: Dict[str, Any]) -> None:
    """Restores `env` to exactly the instant `captured` was taken from."""
    physics = _get_dmc_physics(env)
    physics.set_state(captured["physics_state"])
    physics.forward()

    if captured["elapsed_steps"] is not None:
        time_limit = _find_wrapper(env, TimeLimit)
        if time_limit is not None:
            time_limit._elapsed_steps = captured["elapsed_steps"]
