"""Descriptive train/test Bellman residuals; not ground-truth value error."""
import jax.numpy as jnp
import numpy as np


def td_residuals(actor, temperature, shadows, batch, key, gamma, n_step, twin):
    obs, nxt, actions = (jnp.asarray(batch[k]) for k in ('observation', 'next_observation', 'action'))
    dist = actor(observations=nxt)
    next_actions = dist.sample(seed=key)
    logp = dist.log_prob(next_actions).reshape(-1)
    mask = 1.0 - jnp.asarray(batch['terminated']).reshape(-1)
    reward = jnp.asarray(batch['reward']).reshape(-1)
    residuals = {}
    for name, shadow in shadows.items():
        next_q = shadow.target(observations=nxt, actions=next_actions)
        pred = shadow.critic(observations=obs, actions=actions)
        if twin:
            next_q = jnp.minimum(*next_q)
            pred = jnp.minimum(*pred)
        target = reward + gamma ** n_step * mask * (next_q.reshape(-1) - temperature() * logp)
        residuals[name] = np.asarray(pred.reshape(-1) - target)
        if not np.all(np.isfinite(residuals[name])):
            raise ValueError('nonfinite Bellman residual')
    return residuals
