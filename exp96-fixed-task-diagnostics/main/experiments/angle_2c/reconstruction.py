"""Angle 2C's co-occurrence reconstruction test.

When more than one candidate property (direction, magnitude, instability)
diverges meaningfully from its null simultaneously, research-methodology.md
requires: "construct synthetic gradients combining one critic's direction
with the other's magnitude (and vice versa), run each synthetic version
through the same frozen-actor procedure, and check which reconstruction
better reproduces the real agent's actual observed gradient."

Per explicit decision (see End-of-Task Summary), "the real agent's actual
observed gradient" here means g_{D|R} - the real, fully-swapped, DISTORTED
counterfactual gradient Angle 2B already computed - not g_{D|D} (the
undistorted real signal). The reconstructions are built by mixing
properties FROM Q_D and Q_R, so the natural thing they should be tested
against is the actual distortion that occurred when Q_R was really swapped
in, not the undistorted baseline.

Implementation: "run each synthetic version through the same frozen-actor
procedure used in Angle 2B" is implemented via the chain rule, WITHOUT
touching experiments/angle_2b/gradients.py's _actor_loss or
compute_counterfactual_actor_gradients at all (per CLAUDE.md: do not change
an algorithm because another approach seems better - and there is no need
to, since Angle 2B's actor_loss = (log_probs * temperature() - Q).mean()
decomposes cleanly):

    d(actor_loss)/d(theta) = d(entropy_term)/d(theta) - d(Q_term)/d(theta)
    d(Q_term)/d(theta) = d(a(theta))/d(theta)^T . nabla_a Q(s, a(theta))

The entropy term never touches the critic, so it is computed for real via a
normal jax.grad. The Q-term's parameter-space contribution is obtained via
jax.vjp of the actor's (reparameterized, differentiable) action-sampling
function, pulling back a SUBSTITUTE nabla_a Q (the synthetic direction+
magnitude combination) instead of a real critic's gradient - mathematically
exactly what would happen if a real critic with that exact nabla_a Q existed,
without needing to construct a fake Flax critic module.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from scale_rl.networks.trainer import PRNGKey, Trainer


def synthesize_action_gradient(
    direction_source: jnp.ndarray,
    magnitude_source: jnp.ndarray,
) -> jnp.ndarray:
    """Combines one nabla_a Q's DIRECTION with another's MAGNITUDE, per
    (s,a) pair: synthetic = normalize(direction_source) * ||magnitude_source||.
    """
    norm_dir = jnp.linalg.norm(direction_source, axis=-1, keepdims=True)
    norm_mag = jnp.linalg.norm(magnitude_source, axis=-1, keepdims=True)
    unit_direction = direction_source / (norm_dir + 1e-8)
    return unit_direction * norm_mag


def _entropy_term_grad(actor: Trainer, temperature: Trainer, observations: jnp.ndarray, key: PRNGKey):
    def entropy_loss(actor_params):
        dist = actor.apply(variables={"params": actor_params}, observations=observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        return (log_probs * temperature()).mean()

    return jax.grad(entropy_loss)(actor.params)


@jax.jit
def synthesize_actor_gradient(
    actor: Trainer,
    temperature: Trainer,
    observations: jnp.ndarray,
    key: PRNGKey,
    synthetic_grad_aq: jnp.ndarray,
) -> Dict:
    """Reconstructs the actor-parameter gradient pi_D would have received
    from a hypothetical critic whose action-gradient at these (s,a) pairs is
    EXACTLY `synthetic_grad_aq`, holding the actor and its sampled actions
    fixed - see module docstring for the chain-rule derivation.

    `observations`/`key` must be the SAME ones used for the real primary
    analysis (batch_d, the analysis key) so the sampled actions this
    reconstruction implicitly uses are bit-identical to g_{D|D}/g_{D|R}'s -
    see experiments/angle_2b/gradients.py's sample_actor_actions.
    """
    def sample_fn(actor_params):
        dist = actor.apply(variables={"params": actor_params}, observations=observations)
        return dist.sample(seed=key)

    _actions, vjp_fn = jax.vjp(sample_fn, actor.params)
    batch_size = observations.shape[0]
    # actor_loss's Q-term is Q.mean() = (1/N) sum_i Q_i; VJP's cotangent
    # convention (v^T . J) means passing synthetic_grad_aq/N as the
    # cotangent yields exactly d(mean(Q))/d(theta) with nabla_a Q replaced
    # by the synthetic substitute.
    (q_term_grad,) = vjp_fn(synthetic_grad_aq / batch_size)
    entropy_grad = _entropy_term_grad(actor, temperature, observations, key)

    # actor_loss = entropy_term - Q_term  =>  grad = entropy_grad - q_term_grad
    return jax.tree_util.tree_map(lambda e, q: e - q, entropy_grad, q_term_grad)


def _flatten(grad_pytree) -> jnp.ndarray:
    leaves, _ = jax.tree_util.tree_flatten(grad_pytree)
    return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])


def reconstruction_similarity(synthetic_grad_pytree, target_flat: jnp.ndarray) -> Dict[str, float]:
    """How closely a synthesized actor-parameter gradient reproduces the
    real, fully-observed target gradient (g_{D|R} - see module docstring for
    why that target, not g_{D|D}). Same distortion-metric shapes as
    experiments/angle_2b/gradients.py.compute_distortion_metrics (cosine
    distance + relative norm), computed directly here rather than reusing
    that function, since it also flattens its OWN two pytree inputs -
    `target_flat` here is already flat (loaded from Angle 2B's persisted
    gradients.npz, see loader.py), so reusing it as-is would require
    re-wrapping it into a fake pytree for no benefit.
    """
    synthetic_flat = _flatten(synthetic_grad_pytree)
    norm_synth = jnp.linalg.norm(synthetic_flat)
    norm_target = jnp.linalg.norm(target_flat)
    cos_sim = jnp.dot(synthetic_flat, target_flat) / (norm_synth * norm_target + 1e-8)
    return {
        "cosine_similarity_to_target": float(cos_sim),
        "l2_distance_to_target": float(jnp.linalg.norm(synthetic_flat - target_flat)),
    }


def run_reconstruction_test(
    actor: Trainer,
    temperature: Trainer,
    observations: jnp.ndarray,
    key: PRNGKey,
    grad_aq_d_at_d: jnp.ndarray,
    grad_aq_r_at_d: jnp.ndarray,
    target_g_d_given_r: jnp.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Builds both reconstructions (D-direction+R-magnitude, and
    R-direction+D-magnitude), synthesizes each one's actor-parameter
    gradient, and scores both against g_{D|R}. Whichever scores a higher
    cosine_similarity_to_target (a closer reproduction) indicates that
    property is doing more of the causal work in the real observed
    distortion - see research-methodology.md's Angle 2C co-occurrence
    section.
    """
    synthetic_d_direction_r_magnitude = synthesize_action_gradient(
        direction_source=grad_aq_d_at_d, magnitude_source=grad_aq_r_at_d,
    )
    synthetic_r_direction_d_magnitude = synthesize_action_gradient(
        direction_source=grad_aq_r_at_d, magnitude_source=grad_aq_d_at_d,
    )

    grad_d_dir_r_mag = synthesize_actor_gradient(actor, temperature, observations, key, synthetic_d_direction_r_magnitude)
    grad_r_dir_d_mag = synthesize_actor_gradient(actor, temperature, observations, key, synthetic_r_direction_d_magnitude)

    return {
        "d_direction_r_magnitude": reconstruction_similarity(grad_d_dir_r_mag, target_g_d_given_r),
        "r_direction_d_magnitude": reconstruction_similarity(grad_r_dir_d_mag, target_g_d_given_r),
    }
