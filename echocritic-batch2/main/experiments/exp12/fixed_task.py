"""Retrospective fixed-task probes. Does not replace the operational detector."""

import functools
import hashlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.exp12.probe import ORIGINAL_FIT_STEPS, _chunked


class FitState(NamedTuple):
    params: object
    optimizer: object


def extended_indices(key, steps, batch_size, pool_size):
    """Preserves the frozen first 2k draws; extends the second stream in blocks."""
    if steps < 2000 or steps % ORIGINAL_FIT_STEPS:
        raise ValueError("diagnostic horizons must be >=2000 and multiples of1000")
    blocks = [jax.random.randint(key, (1000, batch_size), 0, pool_size)]
    blocks.extend(
        jax.random.randint(jax.random.fold_in(key, i), (1000, batch_size), 0, pool_size)
        for i in range(1, steps // 1000)
    )
    return jnp.concatenate(blocks)


@functools.partial(jax.jit, static_argnames=("network_def", "chunk"))
def predictions(network_def, params, obs, act, chunk):
    return _chunked(
        lambda o, a: network_def.apply({"params": params}, o, a).reshape(-1),
        obs,
        act,
        chunk,
    )


@functools.partial(jax.jit, static_argnames=("network_def", "chunk"))
def pool_metrics(network_def, params, obs, act, targets, chunk):
    q = predictions(network_def, params, obs, act, chunk)
    residual = q - targets
    return {
        "mse": jnp.mean(residual**2),
        "mean_error": jnp.mean(residual),
        "centered_error_variance": jnp.var(residual),
        "q_mean": jnp.mean(q),
        "q_variance": jnp.var(q),
    }


@functools.partial(jax.jit, static_argnames=("network_def", "tx"))
def fit_segment(network_def, tx, state, obs, act, targets, indices):
    """Exact frozen scalar loss/AdamW update; state persists across segments."""

    def loss_fn(params, idx):
        q = network_def.apply({"params": params}, obs[idx], act[idx]).reshape(-1)
        return jnp.mean((q - targets[idx]) ** 2)

    def step(carry, idx):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, idx)
        updates, opt_state = tx.update(grads, opt_state, params)
        return FitState(optax.apply_updates(params, updates), opt_state), loss

    return jax.lax.scan(step, state, indices)


@functools.partial(jax.jit, static_argnames=("network_def",))
def initial_gradient(network_def, params, obs, act, targets):
    """One full minibatch diagnostic; it never updates parameters."""

    def loss(p):
        q = network_def.apply({"params": p}, obs, act).reshape(-1)
        return jnp.mean((q - targets) ** 2)

    value, grad = jax.value_and_grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grad)
    return value, jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))


@functools.partial(jax.jit, static_argnames=("network_def", "tx", "chunk"))
def first_update(network_def, tx, state, obs, act, targets, indices, chunk):
    """Measure one real AdamW update on copies, including its full-pool effect."""
    loss, grad_norm = initial_gradient(
        network_def, state.params, obs[indices], act[indices], targets[indices]
    )
    before = predictions(network_def, state.params, obs, act, chunk)
    after_state, _ = fit_segment(
        network_def, tx, state, obs, act, targets, indices[None, :]
    )
    after = predictions(network_def, after_state.params, obs, act, chunk)
    delta = after - before
    changes = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda a, b: a - b, after_state.params, state.params)
    )
    parameter_squared_change = sum(jnp.sum(x * x) for x in changes)
    return {
        "minibatch_mse_before": loss,
        "gradient_l2": grad_norm,
        "parameter_change_l2": jnp.sqrt(parameter_squared_change),
        "parameter_change_rms": jnp.sqrt(
            parameter_squared_change / sum(x.size for x in changes)
        ),
        "prediction_change_rms": jnp.sqrt(jnp.mean(delta**2)),
        "prediction_change_mean": jnp.mean(delta),
        "full_pool_mse_before": jnp.mean((before - targets) ** 2),
        "full_pool_mse_after": jnp.mean((after - targets) ** 2),
    }


def curve_summary(losses):
    """Descriptive pre-update minibatch curves; no scientific pass threshold."""
    x = np.asarray(losses)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("loss curve must be finite and nonempty")
    early = x[: min(100, len(x))].astype(np.float64)
    first = float(x[0])
    # A zero initial loss has no defined ratio; do not add an arbitrary epsilon.
    return {
        "updates": len(x),
        "initial_minibatch_mse": first,
        "early_mean_mse": float(early.mean()),
        "early_peak_mse": float(early.max()),
        "early_mean_over_initial": float(early.mean() / first) if first else None,
        "early_peak_over_initial": float(early.max() / first) if first else None,
        "final_minibatch_mse": float(x[-1]),
    }


def equivalence_report(
    reference_losses, reference_mse, segmented_losses, segmented_mse
):
    """Retain the existing GPU guard, reporting rather than hiding divergence."""
    a, b = np.asarray(reference_losses), np.asarray(segmented_losses)
    if (
        a.ndim != 1
        or a.shape != b.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.isfinite([reference_mse, segmented_mse]).all()
    ):
        raise ValueError("equivalence requires complete finite matching curves")
    final_error = float(abs(float(segmented_mse) - float(reference_mse)))
    guard = max(1e-4, 0.005 * abs(float(reference_mse)))
    return {
        "curve_max_abs_error": float(np.max(np.abs(a - b))),
        "curve_rmse": float(np.sqrt(np.mean((a.astype(np.float64) - b) ** 2))),
        "final_abs_error": final_error,
        "final_abs_guard": guard,
        "bitwise_curve_equal": bool(np.array_equal(a, b)),
        "passed_existing_final_guard": final_error <= guard,
        "qualification": "engineering guard only; no historical task identity proof",
    }


def tree_sha256(tree):
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        a = np.asarray(value)
        digest.update(str(path).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.dtype.str.encode())
        digest.update(a.tobytes())
    return digest.hexdigest()
