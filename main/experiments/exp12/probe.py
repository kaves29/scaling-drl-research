"""Plasticity probe (Lyle et al. 2023, Sec. 3.1), adapted to the SAC critic.

One round fits a copy of each critic to g(x) = a + sin(1e5 * f(x; w0)) on a
fixed pool of replay (s, a) pairs, where f(.; w0) is a freshly initialised
critic of the same architecture and a is the fitted critic's own mean prediction
on the pool. Both critics share the pool, w0 and the minibatch order. The
optimizer is the critic's training optimizer (AdamW) with fresh state.
P = b - final pool MSE, b = Var(targets) (Lyle eq. 5). Plasticity loss
L = P(fresh) - P(current), positive when plasticity is lost.

Probes never touch training state: copies are functional (JAX), and the pool
and keys come from dedicated streams that do not consume the training RNGs.
"""

import functools
from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import trim_mean

PROBE_STREAM = 0x50524F42  # "PROB"


@dataclass(frozen=True)
class ProbeConfig:
    rounds: int
    steps: int
    checks: int
    pool_size: int
    batch_size: int
    target_scale: float
    eval_chunk: int


def iqm(values) -> float:
    """Interquartile mean, as rliable's metrics.aggregate_iqm."""
    return float(trim_mean(np.asarray(values, dtype=np.float64), proportiontocut=0.25, axis=None))


def check_steps(num_interaction_steps: int, checks: int):
    """Interaction steps of the k/checks checks, k = 1..checks."""
    if num_interaction_steps % checks:
        raise ValueError(f"{num_interaction_steps} interaction steps are not divisible into {checks} checks.")
    return [k * num_interaction_steps // checks for k in range(1, checks + 1)]


def _chunked(fn, obs, act, chunk):
    n = obs.shape[0]
    out = jax.lax.map(
        lambda xs: fn(xs[0], xs[1]),
        (obs.reshape(n // chunk, chunk, -1), act.reshape(n // chunk, chunk, -1)),
    )
    return out.reshape(-1)


@functools.partial(jax.jit, static_argnames=("network_def", "chunk", "scale"))
def _base_targets(network_def, key, obs, act, chunk, scale):
    omega = network_def.init(key, observations=obs[:1], actions=act[:1])["params"]
    f = _chunked(lambda o, a: network_def.apply({"params": omega}, o, a).reshape(-1), obs, act, chunk)
    return jnp.sin(scale * f)


@functools.partial(jax.jit, static_argnames=("network_def", "chunk"))
def _mean_prediction(network_def, params, obs, act, chunk):
    return jnp.mean(_chunked(lambda o, a: network_def.apply({"params": params}, o, a).reshape(-1), obs, act, chunk))


@functools.partial(jax.jit, static_argnames=("network_def", "tx", "chunk"))
def _fit(network_def, tx, params, obs, act, base_targets, idx, chunk, offset=None):
    """Returns (per-step minibatch losses, final pool MSE, offset a). a defaults to the critic's own mean prediction."""

    def predict(p):
        return _chunked(lambda o, a: network_def.apply({"params": p}, o, a).reshape(-1), obs, act, chunk)

    if offset is None:
        offset = jnp.mean(predict(params))
    targets = offset + base_targets

    def loss_fn(p, batch_idx):
        q = network_def.apply({"params": p}, obs[batch_idx], act[batch_idx]).reshape(-1)
        return jnp.mean((q - targets[batch_idx]) ** 2)

    def step(carry, batch_idx):
        p, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(p, batch_idx)
        updates, opt_state = tx.update(grads, opt_state, p)
        return (optax.apply_updates(p, updates), opt_state), loss

    (fitted, _), losses = jax.lax.scan(step, (params, tx.init(params)), idx)
    final = jnp.mean((predict(fitted) - targets) ** 2)
    return losses, final, offset


def sample_pool(agent, buffer, rng: np.random.Generator, pool_size: int):
    """Uniform (s, a) pool from the filled replay buffer, obs normalised as the critic sees them."""
    idx = rng.integers(0, buffer._num_in_buffer, size=pool_size)
    obs = buffer._observations[idx]
    if hasattr(agent, "_normalize"):
        obs = agent._normalize(obs)
    return jnp.asarray(obs), jnp.asarray(buffer._actions[idx])


def _unpack(critic, tx):
    return critic if len(critic) == 3 else (*critic, tx)


def probe_round(target_def, critics: Dict[str, tuple], tx, obs, act, key, cfg: ProbeConfig,
                shared_offset: Optional[str] = None) -> Dict[str, Dict]:
    """critics: name -> (network_def, params) or (network_def, params, optimizer); the default
    optimizer is `tx`. All share the pool, w0 and the minibatch order. An injected critic is
    probed with its own optimizer, so its frozen parts stay frozen (decision A8).

    shared_offset (one-time sensitivity check, amendment (c)): None gives each critic its own
    offset; a critic name gives all critics that critic's mean prediction; "mean" gives all
    critics the average of their mean predictions."""
    target_key, idx_key = jax.random.split(key)
    base = _base_targets(target_def, target_key, obs, act, cfg.eval_chunk, cfg.target_scale)
    b = jnp.var(base)
    idx = jax.random.randint(idx_key, (cfg.steps, cfg.batch_size), 0, cfg.pool_size)
    common = None
    if shared_offset is not None:
        means = {n: _mean_prediction(_unpack(c, tx)[0], c[1], obs, act, cfg.eval_chunk) for n, c in critics.items()}
        common = jnp.mean(jnp.stack(list(means.values()))) if shared_offset == "mean" else means[shared_offset]
    out = {}
    for name, critic in critics.items():
        network_def, params, critic_tx = _unpack(critic, tx)
        shared = {} if common is None else {"offset": common}
        losses, final, offset = _fit(network_def, critic_tx, params, obs, act, base, idx, cfg.eval_chunk, **shared)
        out[name] = {"losses": losses, "final_loss": final, "offset": offset, "b": b, "score": b - final}
    return out


def probe_key(seed: int, check_index: int, round_index: int):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), PROBE_STREAM)
    return jax.random.fold_in(jax.random.fold_in(key, check_index), round_index)


def probe_rng(seed: int, check_index: int, round_index: int) -> np.random.Generator:
    return np.random.default_rng([seed, PROBE_STREAM, check_index, round_index])


def run_probe(agent, buffer, target_def, critics: Dict[str, tuple], tx, seed: int, check_index: int,
              cfg: ProbeConfig, shared_offset: Optional[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """All rounds of one check. Transfers to host once, after the last round."""
    rounds = []
    for r in range(cfg.rounds):
        obs, act = sample_pool(agent, buffer, probe_rng(seed, check_index, r), cfg.pool_size)
        rounds.append(probe_round(target_def, critics, tx, obs, act, probe_key(seed, check_index, r), cfg,
                                  shared_offset))
    rounds = jax.device_get(rounds)
    return {
        name: {k: np.stack([rd[name][k] for rd in rounds]) for k in rounds[0][name]}
        for name in critics
    }


def paired_loss(result, current="current", fresh="fresh"):
    """Amendment (z): subtract each paired network's FP32 scores before averaging.

    Single-critic subtraction retains its original arithmetic. Twin score means
    remain useful for reporting P, but are not the operands of longitudinal L.
    """
    if f"{fresh}_q1" in result:
        return np.mean([result[f"{fresh}_q{q}"]["score"] -
                        result[f"{current}_q{q}"]["score"] for q in (1, 2)], axis=0)
    return result[fresh]["score"] - result[current]["score"]


def summarize(result: Dict[str, Dict[str, np.ndarray]], current: str = "current", fresh: str = "fresh") -> Dict:
    """Per-round paired plasticity loss L_r = P_r(fresh) - P_r(current) and its IQM."""
    p_cur, p_fresh = result[current]["score"], result[fresh]["score"]
    loss = paired_loss(result, current, fresh)
    valid = bool(np.all(np.isfinite(loss)))
    return {
        "score_current_rounds": p_cur,
        "score_fresh_rounds": p_fresh,
        "loss_rounds": loss,
        "score_current_iqm": iqm(p_cur) if valid else float("nan"),
        "score_fresh_iqm": iqm(p_fresh) if valid else float("nan"),
        "loss_iqm": iqm(loss) if valid else float("nan"),
        "valid": valid,
    }


def probe_config(cfg) -> ProbeConfig:
    p = cfg.probe
    if int(p.pool_size) % int(p.eval_chunk):
        raise ValueError("probe.pool_size must be a multiple of probe.eval_chunk")
    return ProbeConfig(
        rounds=int(p.rounds), steps=int(p.steps), checks=int(p.checks), pool_size=int(p.pool_size),
        batch_size=int(p.batch_size), target_scale=float(p.target_scale), eval_chunk=int(p.eval_chunk),
    )


def critic_optimizer(agent_cfg) -> optax.GradientTransformation:
    """The critic's training optimizer, with fresh state per fit."""
    return optax.adamw(learning_rate=agent_cfg.critic_learning_rate, weight_decay=agent_cfg.critic_weight_decay)
