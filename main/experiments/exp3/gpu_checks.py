"""Measurements for the Exp3 GPU validation harness. No tolerance is chosen here.

Oracle comparisons and TF32 sensitivity are REPORTED as maximum absolute/relative
differences; whether a difference is acceptable is an owner decision. Exactness
(bitwise) checks are separate unittest suites run by scripts/exp3_gpu_validation.py.
"""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from experiments.exp3.artifacts import core
from experiments.exp3.guidance import measure_chunk, optimizer_step
from experiments.exp3.target_policy import fit_target, targets
from scale_rl.agents import create_agent
from scale_rl.agents.sac.sac_update import update_actor

PRECISIONS = ("highest", "tensorfloat32")


def difference(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError("compared measurements have different shapes")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {"max_abs": None, "max_rel": None, "nonfinite": True, "bitwise_equal": False}
    d = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.divide(d, scale, out=np.zeros_like(d), where=scale > 0)
    return {"max_abs": float(d.max(initial=0.0)), "max_rel": float(rel.max(initial=0.0)),
            "nonfinite": False, "bitwise_equal": bool(np.array_equal(a, b))}


def make_agent(cfg_agent, obs_dim, act_dim):
    obs = gym.spaces.Box(-np.inf, np.inf, (1, obs_dim), dtype=np.float32)
    act = gym.spaces.Box(-1, 1, (1, act_dim), dtype=np.float32)
    return core(create_agent(obs, act, cfg_agent))


def synthetic_inputs(seed, states, obs_dim, act_dim):
    """Synthetic normalized states/transitions: engineering inputs, not a scientific panel."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((states, obs_dim)).astype(np.float32)
    return {
        "observation": obs,
        "action": rng.uniform(-1, 1, (states, act_dim)).astype(np.float32),
        "reward": rng.standard_normal(states).astype(np.float32),
        "terminated": np.zeros(states, np.float32),
        "truncated": np.zeros(states, np.float32),
        "next_observation": rng.standard_normal((states, obs_dim)).astype(np.float32),
    }


def oracle_measurements(u, i, batch, key, twin):
    """Alternative computation graphs for the same quantity, at the current precision."""
    obs = jnp.asarray(batch["observation"])
    alpha = u.temperature()
    values, grads = measure_chunk(u.actor, u.critic, i.critic, obs, key, alpha, twin)

    def row_loss(p, row):
        d = u.actor.apply({"params": p}, observations=obs)
        a = d.sample(seed=key)  # the same batched draw as measure_chunk
        return (alpha * d.log_prob(a) - _q(u.critic, obs, a, twin))[row]

    # Alternative graph: gradient of one state's loss vs that row of the per-state Jacobian.
    first = difference(values["sac_parameter_gradient_u"][0],
                       ravel_pytree(jax.grad(row_loss)(u.actor.params, 0))[0])
    ordinary = update_actor(key, u.actor, u.critic, u.temperature, {"observation": obs}, twin)[0]
    via_gradient = optimizer_step(u.actor, grads["u"])
    actor_update = difference(ravel_pytree(via_gradient.params)[0], ravel_pytree(ordinary.params)[0])
    noise = jax.random.normal(key, (len(obs), batch["action"].shape[-1]))
    jb = {k: jnp.asarray(v) for k, v in batch.items()}
    target, info = targets(u.actor, u._target_critic, jb, noise, float(alpha), 0.99, 1, twin)
    dist = u.actor(observations=jb["next_observation"])
    q = _q(u._target_critic, jb["next_observation"], info["action"], twin)
    unsaturated = np.asarray((jnp.abs(info["action"]) < 0.99).all(-1))
    production_lp = dist.log_prob(info["action"])
    production_target = jb["reward"] + 0.99 * (1 - jb["terminated"]) * (q - float(alpha) * production_lp)
    fitted, _ = fit_target(u.critic, jb, target)
    return {
        "per_state_jacobian_row_vs_row_grad": first,
        "panel_gradient_update_vs_update_actor": actor_update,
        "target_vs_production_formula_unsaturated": difference(
            np.asarray(target)[unsaturated], np.asarray(production_target)[unsaturated]),
        "unsaturated_rows": int(unsaturated.sum()),
        "critic_fit_finite": bool(np.isfinite(ravel_pytree(fitted.params)[0]).all()),
    }


def _q(critic, obs, actions, twin):
    q = critic(observations=obs, actions=actions)
    return jnp.minimum(q[0], q[1]).reshape(-1) if twin else q.reshape(-1)


def measurement_fields(u, i, obs, key, twin):
    values, _ = measure_chunk(u.actor, u.critic, i.critic, jnp.asarray(obs), key, u.temperature(), twin)
    return {k: np.asarray(v) for k, v in values.items()}


def tf32_sensitivity(u, i, obs, key, twin):
    """Same inputs at 'highest' and 'tensorfloat32': every Pilot 1 measurement field."""
    with jax.default_matmul_precision("highest"):
        high = measurement_fields(u, i, obs, key, twin)
    with jax.default_matmul_precision("tensorfloat32"):
        tf32 = measurement_fields(u, i, obs, key, twin)
    report = {}
    for k in sorted(high):
        if high[k].dtype == bool:
            report[k] = {"changed": int((high[k] != tf32[k]).sum())}
        elif k.endswith("_cosine"):
            defined = np.isfinite(high[k]) & np.isfinite(tf32[k])
            report[k] = {**difference(high[k][defined], tf32[k][defined]),
                         "sign_flips": int((np.sign(high[k][defined]) != np.sign(tf32[k][defined])).sum())}
        else:
            report[k] = difference(high[k], tf32[k])
    return report


def repeat_bitwise(u, i, obs, key, twin):
    """Two identical measurement calls must be bitwise identical (same program, same inputs)."""
    a = measurement_fields(u, i, obs, key, twin)
    b = measurement_fields(u, i, obs, key, twin)
    return all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)
