"""Exact environment-state capture/restore for Angle 2A's Monte Carlo rollouts.

Angle 2A's protocol requires resetting the environment to an exact previously
visited state `s` and forcing action `a`. This repo's replay buffer only
stores flattened *observations* (see scale_rl/buffers/numpy_buffer.py), which
are not guaranteed to be sufficient to reconstruct the underlying simulator
state exactly (many dm_control/MyoSuite tasks drop globally-invariant
coordinates from the observation). Re-deriving state from the observation
would therefore be an approximation, not an exact reproduction.

Instead, this module captures the underlying MuJoCo physics state directly at
transition-collection time, plus the outer `TimeLimit` wrapper's elapsed-step
counter, so truncation semantics on resume match what they would have been at
the original moment (not a stale/reset counter). Two env types are supported,
each via its own physics-access path (dm_control's `Physics` object for
env_type='dmc', direct `mj_data`/`mj_model` access for env_type='myosuite') -
dispatched by the `env_type` field this module stores in every captured dict,
so callers never need to track or pass it separately at restore time.

Known, documented limitation (see Angle 2A deliverables notes): this captures
*dynamical* state (qpos/qvel/act, plus mocap pose where present) only. A
small number of tasks additionally randomize static, non-dynamical model
parameters once per episode (e.g. a target site/body position); those are not
captured/restored here. This is safe for how Angle 2A actually uses this
module - restore always happens without an intervening env.reset() (mid-
episode, or via train_agent_to_step's own resume path), so such static
per-episode parameters never change between a capture and its later restore -
but this implementation does not silently claim exactness for a use pattern
that included a reset in between.

MyoSuite-specific notes (confirmed empirically, 2026-09-07):

1. MyoSuite's own `get_env_state()`/`set_env_state()` methods are NOT used
   here, even though they look like the obvious equivalent of dm_control's
   `physics.get_state()`/`set_state()`. `set_env_state()` internally calls
   `mujoco.mj_step()`, which silently advances physics by one full
   integration step instead of freezing state where it was captured
   (confirmed directly: qpos/qvel/act/time all shift by exactly one
   timestep's worth of dynamics after a "restore"). This module instead
   assigns the same state fields directly and calls `mujoco.mj_forward()`
   (recompute derived quantities only, never integrate), mirroring
   dm_control's own `physics.forward()` after `set_state()`.

2. State is read from/written to `env.robot.mj_data`/`env.robot.mj_model`,
   NOT `env.mj_data`/`env.mj_model` - the two are DIFFERENT objects
   (confirmed: `env.robot.mj_data is env.mj_data` is False). The Robot
   interface (myosuite/robot/robot.py) performs the actual physics
   stepping on its own `mj_data`; `env.mj_data` is a passive mirror that
   only gets overwritten to match it as a side effect of `get_obs()`
   (via `Robot.sensor2sim`). Restoring `env.mj_data` alone (as an initial,
   incorrect implementation of this module did) restores something no
   physics step ever reads, so the true simulation state - and therefore
   every subsequent observation/rollout - silently continues from wherever
   it already was, not from the captured instant. `env.mj_data` self-
   corrects on the next `get_obs()` call, so it does not need restoring
   separately once `env.robot.mj_data` is correct.

3. `qacc_warmstart` and `ctrl` (the solver's warm-start hint and the last-
   applied control signal) must also be restored, for the same
   iterative-solver-path and rate-limit reasons as dm_control's own
   `qacc_warmstart` case below - confirmed by direct empirical round-trip
   testing, not assumed.

4. `steps` (a plain Python int some MyoSuite tasks maintain on the env
   itself, outside any mj_data/mj_model field - confirmed present on
   myoLegWalk-v0, used there to compute a gait "phase_var" observation
   feature; see myosuite/envs/myo/myobase/walk_v0.py) is captured/restored
   defensively (only if the attribute exists), since not every MyoSuite
   task has it and this module has no reliable way to enumerate every
   possible task-specific non-physics attribute a future task might add.
   Verified bit-exact for 5 of the 6 MyoSuite environments in this study,
   including all 4 core (myo-elbow-pose-random, myo-reach, myo-key-turn,
   myo-leg-walk) plus held-out myo-pen-twirl. myo-baoding-p1 (the other
   held-out env) has its own analogous task-specific counter
   (self.counter, indexing a precomputed goal trajectory - see
   myosuite/envs/myo/myochallenge/baoding_v1.py) that this module does not
   capture/restore, producing a small (~1e-3 magnitude) residual
   divergence - diagnosed but deliberately not fixed, since Angle 2A never
   trains/evaluates on held-out environments (Angle-3-only; see
   validate_myosuite_core4) and generically enumerating every possible
   task-specific counter name is not a tractable strategy. Would need
   fixing (adding self.counter alongside self.steps, or a more general
   mechanism) before this module could be trusted for Angle 3's eventual
   use of the held-out set, if that use ever needs exact-state rollouts.

Only env_type in {'dmc', 'myosuite'} is supported. Anything else is an
explicit, loud failure rather than a silent inexact fallback.
"""

from typing import Any, Dict, Optional

import mujoco

from gymnasium.wrappers import TimeLimit

from experiments.angle_2a.errors import Angle2AEnvironmentError

SUPPORTED_ENV_TYPES = ("dmc", "myosuite")


def assert_supported_env_type(env_type: str) -> None:
    if env_type not in SUPPORTED_ENV_TYPES:
        raise Angle2AEnvironmentError(
            f"Angle 2A's exact-state Monte Carlo rollout only supports "
            f"env_type in {SUPPORTED_ENV_TYPES}; got env_type='{env_type}'. "
            f"Extending this to another env type requires implementing "
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
            "access."
        )
    return physics


def _capture_dmc_state(env) -> Dict[str, Any]:
    """physics.get_state() (qpos/qvel/act) alone is insufficient for
    bit-exact restore in contact-heavy tasks: MuJoCo's iterative solver warm-
    starts from data.qacc_warmstart, which get_state()/set_state() never
    touch. Confirmed empirically (2026-09-07): restoring without it produces
    ~1e-12-magnitude divergences in dog-run/dog-trot/humanoid-run/quadruped-
    run/manipulator-bring_ball after a handful of steps (solver converging
    along a slightly different path, not a wrong physical state); including
    it makes the divergence exactly 0.0."""
    physics = _get_dmc_physics(env)
    return {
        "physics_state": physics.get_state().copy(),
        "qacc_warmstart": physics.data.qacc_warmstart.copy(),
    }


def _restore_dmc_state(env, captured: Dict[str, Any]) -> None:
    physics = _get_dmc_physics(env)
    physics.set_state(captured["physics_state"])
    physics.data.qacc_warmstart[:] = captured["qacc_warmstart"]
    physics.forward()


def _get_myosuite_robot(env):
    """Returns env's Robot interface, whose OWN mj_data/mj_model (not
    env.mj_data/mj_model - see module docstring, note 2) is where physics is
    actually stepped."""
    base = getattr(env, "unwrapped", env)
    robot = getattr(base, "robot", None)
    if robot is None or not hasattr(robot, "mj_data") or not hasattr(robot, "mj_model"):
        raise Angle2AEnvironmentError(
            "Could not locate MyoSuite's robot.mj_data/robot.mj_model on "
            "this environment (checked env.unwrapped.robot.mj_data/"
            "mj_model). Angle 2A's exact Monte Carlo rollout requires "
            "direct physics access."
        )
    return base, robot


def _capture_myosuite_state(env) -> Dict[str, Any]:
    """site_pos/body_pos (present in MyoSuite's own get_env_state(), which
    this does not call - see module docstring) are deliberately omitted -
    see module docstring's "known limitation" note above."""
    base, robot = _get_myosuite_robot(env)
    d, m = robot.mj_data, robot.mj_model
    return {
        "time": float(d.time),
        "qpos": d.qpos.ravel().copy(),
        "qvel": d.qvel.ravel().copy(),
        "act": d.act.ravel().copy() if m.na > 0 else None,
        "ctrl": d.ctrl.ravel().copy(),
        "qacc_warmstart": d.qacc_warmstart.ravel().copy(),
        "mocap_pos": d.mocap_pos.copy() if m.nmocap > 0 else None,
        "mocap_quat": d.mocap_quat.copy() if m.nmocap > 0 else None,
        "steps": getattr(base, "steps", None),
    }


def _restore_myosuite_state(env, captured: Dict[str, Any]) -> None:
    """Deliberately does not call MyoSuite's own set_env_state() - see module
    docstring for why (it internally calls mujoco.mj_step(), which advances
    physics by one step instead of freezing it). mj_forward is the correct
    call here, exactly mirroring _restore_dmc_state's use of
    physics.forward() instead of physics.step(). Only robot.mj_data needs
    writing (not env.mj_data too - see module docstring, note 2)."""
    base, robot = _get_myosuite_robot(env)
    d, m = robot.mj_data, robot.mj_model
    d.time = captured["time"]
    d.qpos[:] = captured["qpos"]
    d.qvel[:] = captured["qvel"]
    if m.na > 0:
        d.act[:] = captured["act"]
    d.ctrl[:] = captured["ctrl"]
    d.qacc_warmstart[:] = captured["qacc_warmstart"]
    if m.nmocap > 0:
        d.mocap_pos[:] = captured["mocap_pos"]
        d.mocap_quat[:] = captured["mocap_quat"]
    mujoco.mj_forward(m, d)
    if captured["steps"] is not None:
        base.steps = captured["steps"]


def capture_env_state(env, env_type: str) -> Dict[str, Any]:
    """Captures everything needed to exactly resume `env` from its current
    instant. `env_type` is stored in the returned dict so restore_env_state
    can dispatch without the caller needing to track it separately."""
    assert_supported_env_type(env_type)
    time_limit = _find_wrapper(env, TimeLimit)
    state = _capture_dmc_state(env) if env_type == "dmc" else _capture_myosuite_state(env)
    state["env_type"] = env_type
    state["elapsed_steps"] = time_limit._elapsed_steps if time_limit is not None else None
    return state


def restore_env_state(env, captured: Dict[str, Any]) -> None:
    """Restores `env` to exactly the instant `captured` was taken from."""
    env_type = captured["env_type"]
    if env_type == "dmc":
        _restore_dmc_state(env, captured)
    else:
        _restore_myosuite_state(env, captured)

    if captured["elapsed_steps"] is not None:
        time_limit = _find_wrapper(env, TimeLimit)
        if time_limit is not None:
            time_limit._elapsed_steps = captured["elapsed_steps"]
