"""Frozen actor assignment, shared fixed evaluator and fixed protocol-derived alpha."""

import jax
import jax.numpy as jnp

from experiments.exp3.guidance import q_values


def targets(
    actor,
    evaluator,
    batch,
    noise,
    alpha,
    gamma,
    n_step,
    twin,
    actor_observations=None,
    evaluator_observations=None,
):
    """Shared BASE Gaussian noise; SAC tanh/Jacobian log-prob from existing distribution."""
    actor_obs = (
        batch["next_observation"] if actor_observations is None else actor_observations
    )
    eval_obs = (
        batch["next_observation"]
        if evaluator_observations is None
        else evaluator_observations
    )
    dist = actor(observations=actor_obs)
    base = dist.distribution
    latent = base.mean() + base.stddev() * noise
    actions = dist.bijector.forward(latent)
    # Retain the actual preimage, as dist.sample/log_prob does in ordinary SAC.
    # An inverse tanh of a rounded +/-1 action would introduce artificial NaNs.
    log_probs = base.log_prob(latent) - dist.bijector.forward_log_det_jacobian(
        latent, event_ndims=1
    )
    q = q_values(evaluator, eval_obs, actions, twin)
    discount = (gamma**n_step) * (1 - batch["terminated"])
    target = batch["reward"] + discount * q
    target -= discount * alpha * log_probs
    return jax.lax.stop_gradient(target), {
        "action": actions,
        "log_prob": log_probs,
        "q": q,
    }


@jax.jit
def fit_target(critic, batch, target):
    """Frozen-target regression through the existing critic Trainer/optimizer."""

    def loss(params):
        pred = critic.apply(
            {"params": params},
            observations=batch["observation"],
            actions=batch["action"],
        )
        errors = (
            (pred.reshape(-1) - target) ** 2
            if pred.ndim == 2
            else (
                (pred[0].reshape(-1) - target) ** 2
                + (pred[1].reshape(-1) - target) ** 2
            )
        )
        return errors.mean(), {"loss": errors.mean()}

    return critic.apply_gradient(loss)


class TargetPair:
    def __init__(
        self, critic, actor_u, actor_i, evaluator, alpha, gamma, n_step, twin, key
    ):
        self.critics = {"u": critic, "i": critic}
        self.actors = {"u": actor_u, "i": actor_i}
        self.evaluator = evaluator
        self.alpha, self.gamma, self.n_step, self.twin = alpha, gamma, n_step, twin
        self.key, self.update_step = key, 0

    def update(self, batch, actor_observations=None, evaluator_observations=None):
        self.key, draw = jax.random.split(self.key)
        noise = jax.random.normal(
            draw,
            (
                len(batch["reward"]),
                self.actors["u"](observations=batch["next_observation"])
                .distribution.mean()
                .shape[-1],
            ),
        )
        results = {}
        for label in ("u", "i"):
            target, info = targets(
                self.actors[label],
                self.evaluator,
                batch,
                noise,
                self.alpha,
                self.gamma,
                self.n_step,
                self.twin,
                None if actor_observations is None else actor_observations[label],
                evaluator_observations,
            )
            if not bool(jnp.isfinite(target).all()):
                raise ValueError("nonfinite frozen-policy SAC targets")
            self.critics[label], fit = fit_target(self.critics[label], batch, target)
            results[label] = {"target": target, **info, **fit}
        self.update_step += 1
        return {
            "noise": noise,
            "u": results["u"],
            "i": results["i"],
            "target_difference": results["i"]["target"] - results["u"]["target"],
        }
