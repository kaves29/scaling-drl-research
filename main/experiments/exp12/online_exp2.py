"""Opt-in, source-compatible primitives for the online Exp2 prototype.

The reference agent owns collection, replay and its own update. This module only
updates shadow critics and disposable copies of the reference actor.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from scale_rl.agents.sac.sac_update import update_target_network
from scale_rl.networks.utils import tree_norm


@dataclass(frozen=True)
class Shadow:
    critic: object
    target: object
    fresh_params: object
    updates: int = 0


def _finite_batch(batch):
    required = ("observation", "action", "reward", "terminated", "next_observation")
    if not all(name in batch for name in required):
        raise ValueError("shadow batch is missing a SAC field")
    if np.ndim(batch["reward"]) == 0:
        raise ValueError("invalid shadow batch reward shape")
    size = np.shape(batch["reward"])[0]
    for name in required:
        value = np.asarray(batch[name])
        if (value.ndim == 0 or value.shape[0] != size or
                not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value))):
            raise ValueError(f"invalid shadow batch field: {name}")
    if (np.shape(batch["reward"]) != (size,) or np.shape(batch["terminated"]) != (size,)
            or np.ndim(batch["observation"]) != 2 or np.ndim(batch["action"]) != 2
            or np.shape(batch["next_observation"]) != np.shape(batch["observation"])):
        raise ValueError("invalid shadow batch shapes")
    if np.any((np.asarray(batch["terminated"]) != 0) & (np.asarray(batch["terminated"]) != 1)):
        raise ValueError("terminated must be a binary bootstrap mask")


def paired_shadow_update(actor, temperature, shadows, batch, key, gamma, n_step,
                         target_tau, critic_use_cdq=False):
    """One paired update; caller supplies the reference-normalized batch.

    The single sampled next action/log probability is shared. Each shadow's
    target critic supplies its own bootstrapped Q. No reference state is written.
    """
    _finite_batch(batch)
    batch = {name: jnp.asarray(value) for name, value in batch.items()}
    next_dist = actor(observations=batch["next_observation"])
    next_actions = next_dist.sample(seed=key)
    next_log_probs = next_dist.log_prob(next_actions)
    mask = 1 - batch["terminated"]
    updated, info = {}, {}
    for name, shadow in shadows.items():
        if critic_use_cdq:
            q1, q2 = shadow.target(observations=batch["next_observation"], actions=next_actions)
            next_q = jnp.minimum(q1, q2).reshape(-1)
        else:
            next_q = shadow.target(observations=batch["next_observation"], actions=next_actions).reshape(-1)
        target = batch["reward"] + (gamma ** n_step) * mask * next_q
        target -= (gamma ** n_step) * mask * temperature() * next_log_probs
        target = jax.lax.stop_gradient(target)

        def loss_fn(params):
            predicted = shadow.critic.apply(
                variables={"params": params}, observations=batch["observation"],
                actions=batch["action"])
            if critic_use_cdq:
                loss = ((predicted[0].reshape(-1) - target) ** 2 +
                        (predicted[1].reshape(-1) - target) ** 2).mean()
            else:
                loss = ((predicted.reshape(-1) - target) ** 2).mean()
            return loss, {"loss": loss, "q_target_mean": target.mean()}

        critic, metrics = shadow.critic.apply_gradient(loss_fn)
        target_critic, _ = update_target_network(critic, shadow.target, target_tau)
        updated[name] = Shadow(critic, target_critic, shadow.fresh_params, shadow.updates + 1)
        info[name] = metrics
    return updated, {"next_actions": next_actions, "next_log_probs": next_log_probs,
                     "critics": info}


def paired_actor_intervention(actor, temperature, shadows, observations, key,
                              critic_use_cdq=False):
    """Apply one source SAC actor step to two copies of actor and optimizer state."""
    observations = jnp.asarray(observations)
    if not bool(jnp.all(jnp.isfinite(observations))):
        raise ValueError("nonfinite actor observations")
    clones, records = {}, {}
    for name, shadow in shadows.items():
        def loss_fn(params):
            dist = actor.apply(variables={"params": params}, observations=observations)
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            predicted = shadow.critic(observations=observations, actions=actions)
            q = jnp.minimum(*predicted) if critic_use_cdq else predicted
            loss = (log_probs * temperature() - q.reshape(-1)).mean()
            return loss, {"loss": loss, "entropy": -log_probs.mean()}

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor.params)
        clone, _ = actor.apply_gradient(loss_fn)
        delta = jax.tree.map(lambda new, old: new - old, clone.params, actor.params)
        clones[name] = clone
        records[name] = {**metrics, "gradient": grads, "gradient_norm": tree_norm(grads),
                         "parameter_delta": delta, "delta_norm": tree_norm(delta)}
    return clones, records


def gradient_cosine(left, right):
    a, _ = ravel_pytree(left)
    b, _ = ravel_pytree(right)
    na, nb = jnp.linalg.norm(a), jnp.linalg.norm(b)
    if float(na) == 0 or float(nb) == 0:
        return None
    return float(jnp.dot(a, b) / (na * nb))
