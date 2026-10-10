"""Matched within-actor critic guidance and isolated action-gradient interventions."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from experiments.exp3.artifacts import tree_equal


def q_values(critic, observations, actions, twin):
    q = critic(observations=observations, actions=actions)
    return jnp.minimum(q[0], q[1]).reshape(-1) if twin else q.reshape(-1)


def policy(actor, params, observations, key):
    dist = actor.apply({"params": params}, observations=observations)
    actions = dist.sample(seed=key)
    return actions, dist.log_prob(actions)


def flatten_per_state(tree, size):
    return jnp.concatenate(
        [x.reshape(size, -1) for x in jax.tree_util.tree_leaves(tree)], axis=1
    )


def signal_summary(a, b):
    """No epsilon/threshold chosen: direction is undefined at an exactly zero norm."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    valid = (na != 0) & (nb != 0)
    cos = np.full_like(na, np.nan)
    np.divide((a * b).sum(-1), na * nb, out=cos, where=valid)
    return {"norm_u": na, "norm_i": nb, "cosine": cos, "direction_defined": valid}


def measure_chunk(
    actor, u, i, observations, key, alpha, twin, critic_observations=None
):
    """Per-state gradients use the SAME batched draw as the measured actions/losses."""
    obs = jnp.asarray(observations)
    actions, logs = policy(actor, actor.params, obs, key)
    out = {
        "observation": obs,
        "action": actions,
        "log_prob": logs,
        "alpha": jnp.asarray(alpha),
    }
    grads = {}
    for label, critic in (("u", u), ("i", i)):
        critic_obs = obs if critic_observations is None else critic_observations[label]
        out[f"critic_observation_{label}"] = critic_obs
        out[f"q_{label}"] = critic(observations=critic_obs, actions=actions)
        out[f"dq_da_{label}"] = jax.grad(
            lambda a: q_values(critic, critic_obs, a, twin).sum()
        )(actions)

        def critic_losses(params):
            act, _ = policy(actor, params, obs, key)
            return -q_values(critic, critic_obs, act, twin)

        def full_losses(params):
            act, lp = policy(actor, params, obs, key)
            return jnp.asarray(alpha) * lp - q_values(critic, critic_obs, act, twin)

        cg = jax.jacrev(critic_losses)(actor.params)
        fg = jax.jacrev(full_losses)(actor.params)
        out[f"critic_parameter_gradient_{label}"] = flatten_per_state(cg, len(obs))
        out[f"sac_parameter_gradient_{label}"] = flatten_per_state(fg, len(obs))
        grads[label] = jax.tree_util.tree_map(lambda g: g.mean(axis=0), fg)
    # Undefined direction cosines are allowed below; model/loss/gradient NaNs are not.
    if any(not np.isfinite(np.asarray(v)).all() for v in out.values()):
        raise ValueError("nonfinite action/value/log-probability/gradient measurement")
    for family in ("dq_da", "critic_parameter_gradient", "sac_parameter_gradient"):
        for k, v in signal_summary(out[f"{family}_u"], out[f"{family}_i"]).items():
            out[f"{family}_{k}"] = v
    return out, grads


def substitution(u, i, mode, zero_policy):
    if mode in ("sanity", "full"):
        return u if mode == "sanity" else i
    if mode not in ("direction", "magnitude") or zero_policy not in (
        "error",
        "keep_baseline",
    ):
        raise ValueError("unapproved intervention/zero-signal choice")
    nu, ni = jnp.linalg.norm(u, axis=-1, keepdims=True), jnp.linalg.norm(
        i, axis=-1, keepdims=True
    )
    valid = (nu != 0) & (ni != 0)
    if zero_policy == "error" and not bool(jnp.all(valid)):
        raise ValueError("direction/magnitude undefined for zero action gradient")
    # The nonzero branch is the protocol's exact ratio; no numerical epsilon added.
    candidate = (
        i * (nu / jnp.where(ni != 0, ni, 1))
        if mode == "direction"
        else u * (ni / jnp.where(nu != 0, nu, 1))
    )
    return jnp.where(valid, candidate, u)


def intervention_gradient(
    actor,
    u,
    i,
    obs,
    key,
    alpha,
    twin,
    mode,
    zero_policy,
    critic_observations=None,
    space="action",
):
    if space == "parameter_per_state" and mode not in ("sanity", "full"):
        values, _ = measure_chunk(
            actor, u, i, obs, key, alpha, twin, critic_observations
        )
        cu, ci = (
            values["critic_parameter_gradient_u"],
            values["critic_parameter_gradient_i"],
        )
        entropy = values["sac_parameter_gradient_u"] - cu
        chosen = substitution(cu, ci, mode, zero_policy)
        _, unravel = ravel_pytree(actor.params)
        return unravel(jnp.mean(entropy + chosen, axis=0))
    if space not in ("action", "parameter_per_state"):
        raise ValueError("unapproved intervention space")
    uobs = obs if critic_observations is None else critic_observations["u"]
    iobs = obs if critic_observations is None else critic_observations["i"]
    if mode in ("sanity", "full"):
        critic = u if mode == "sanity" else i

        def ordinary_loss(p):
            actions, logs = policy(actor, p, obs, key)
            return (
                alpha * logs
                - q_values(critic, uobs if mode == "sanity" else iobs, actions, twin)
            ).mean()

        return jax.grad(ordinary_loss)(actor.params)
    actions, _ = policy(actor, actor.params, obs, key)
    du = jax.grad(lambda a: q_values(u, uobs, a, twin).sum())(actions)
    di = jax.grad(lambda a: q_values(i, iobs, a, twin).sum())(actions)
    chosen = jax.lax.stop_gradient(substitution(du, di, mode, zero_policy))

    def loss(p):
        a, lp = policy(actor, p, obs, key)
        return (alpha * lp - (a * chosen).sum(-1)).mean()

    return jax.grad(loss)(actor.params)


def optimizer_step(actor, gradient):
    """Use the existing Trainer optimizer/masking/update semantics from matched state."""
    if actor.dynamic_scale is not None:
        raise ValueError(
            "Exp3 pilots require existing FP32 artifacts, not mixed-precision fixtures"
        )
    constant = jax.tree_util.tree_map(jax.lax.stop_gradient, gradient)

    def loss(p):
        value = sum(
            jnp.vdot(x, y)
            for x, y in zip(
                jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(constant)
            )
        )
        return value, {}

    return actor.apply_gradient(loss)[0]


def assert_fork_equivalence(u, i):
    if not tree_equal(u._actor, i._actor) or not tree_equal(
        u._temperature, i._temperature
    ):
        raise ValueError("fork actors/optimizers/alpha differ")
    # Network structures can differ; Q/dQ Check1 is the independent function assertion.


def updated_policy_arrays(before, after, observations):
    # Existing actor function diagnostics use full FP32, ordinary update stays TF32.
    with jax.default_matmul_precision("highest"):
        old = before(observations=observations).distribution
        new = after(observations=observations).distribution
    delta, _ = ravel_pytree(
        jax.tree_util.tree_map(lambda a, b: a - b, after.params, before.params)
    )
    return {
        "mean_before": old.mean(),
        "mean_after": new.mean(),
        "std_before": old.stddev(),
        "std_after": new.stddev(),
        "deterministic_action_delta": jnp.tanh(new.mean()) - jnp.tanh(old.mean()),
        "parameter_delta": delta,
    }
