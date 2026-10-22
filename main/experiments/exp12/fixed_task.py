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


def tree_sha256(tree):
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        a = np.asarray(value)
        digest.update(str(path).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.dtype.str.encode())
        digest.update(a.tobytes())
    return digest.hexdigest()
