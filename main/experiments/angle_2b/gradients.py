"""Frozen, single-step counterfactual actor-gradient computation for Angle 2B.

The loss formula here is copied verbatim from scale_rl/agents/sac/sac_update.py's
update_actor() (actor_loss = (log_probs * temperature() - q).mean()) - Angle
2B must never redefine the actor's objective (see
.claude/rules/jax-rl-safety.md and CLAUDE.md's "do not change an algorithm
... because another approach seems better"). The only difference from
update_actor is that this never calls actor.apply_gradient() (which would
also update actor/optimizer state) - it only ever computes jax.grad() of the
identical loss and returns the raw gradient pytree, since Angle 2B measures
the gradient a frozen actor WOULD receive, without ever applying it (no
continued training anywhere in Angle 2B - see research-methodology.md's
Angle 2B scope boundary).
"""

import functools
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from scale_rl.networks.trainer import PRNGKey, Trainer

_COS_EPS = 1e-8  # matches sac_update.py's compute_actor_gradient_cosine convention


def _actor_loss(
    actor: Trainer,
    actor_params,
    critic: Trainer,
    temperature: Trainer,
    observations: jnp.ndarray,
    key: PRNGKey,
    critic_use_cdq: bool,
) -> jnp.ndarray:
    dist = actor.apply(variables={"params": actor_params}, observations=observations)
    actions = dist.sample(seed=key)
    log_probs = dist.log_prob(actions)

    if critic_use_cdq:
        q1, q2 = critic(observations=observations, actions=actions)
        q = jnp.minimum(q1, q2).reshape(-1)
    else:
        q = critic(observations=observations, actions=actions).reshape(-1)

    return (log_probs * temperature() - q).mean()


@functools.partial(jax.jit, static_argnames=("critic_use_cdq",))
def compute_counterfactual_actor_gradients(
    key: PRNGKey,
    actor: Trainer,
    critic_same: Trainer,
    critic_swap: Trainer,
    temperature: Trainer,
    observations: jnp.ndarray,
    critic_use_cdq: bool,
) -> Tuple[Dict, Dict]:
    """Holds `actor` (params + architecture) and `temperature` completely
    fixed; computes the actor's loss gradient twice, swapping only which
    critic supplies the Q-term. The SAME `key` is reused for both calls, so
    the sampled actions/log_probs (and thus the entropy term) are bit-
    identical between the two - only the critic's Q-term differs, exactly
    matching the Angle 2B spec ("Only the critic changes between these two;
    pi_D's parameters, the state batch, and the entropy term must be
    identical in both").

    This is the one primitive every Angle 2B analysis is built from:
      primary:   actor=pi_D, critic_same=Q_D, critic_swap=Q_R, temperature=D's own
      secondary: actor=pi_R, critic_same=Q_R, critic_swap=Q_D, temperature=R's own
      null:      actor=pi_A, critic_same=Q_A, critic_swap=Q_B, temperature=A's own

    Returns (grad_same, grad_swap): two gradient pytrees, same structure as
    actor.params (so they're directly comparable via compute_distortion_metrics).
    """
    grad_fn = jax.grad(_actor_loss, argnums=1)
    grad_same = grad_fn(actor, actor.params, critic_same, temperature, observations, key, critic_use_cdq)
    grad_swap = grad_fn(actor, actor.params, critic_swap, temperature, observations, key, critic_use_cdq)
    return grad_same, grad_swap


def _flatten(grad_pytree) -> jnp.ndarray:
    leaves, _ = jax.tree_util.tree_flatten(grad_pytree)
    return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])


@jax.jit
def compute_distortion_metrics(grad_same, grad_swap) -> Dict[str, jnp.ndarray]:
    """D_dir = 1 - cos(g_same, g_swap); D_mag = log(||g_same|| / ||g_swap||);
    D_grad = ||g_same - g_swap||. `grad_same` plays g_{D|D}/g_{R|R}/g_{A|A};
    `grad_swap` plays g_{D|R}/g_{R|D}/g_{A|B}, depending on which analysis
    called compute_counterfactual_actor_gradients."""
    v_same = _flatten(grad_same)
    v_swap = _flatten(grad_swap)

    norm_same = jnp.linalg.norm(v_same)
    norm_swap = jnp.linalg.norm(v_swap)

    cos_sim = jnp.dot(v_same, v_swap) / (norm_same * norm_swap + _COS_EPS)
    d_dir = 1.0 - cos_sim
    d_mag = jnp.log((norm_same + _COS_EPS) / (norm_swap + _COS_EPS))
    d_grad = jnp.linalg.norm(v_same - v_swap)

    return {"d_dir": d_dir, "d_mag": d_mag, "d_grad": d_grad}
