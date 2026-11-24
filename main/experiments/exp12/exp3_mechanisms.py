"""Actor update interventions for the Experiment 3 mechanism pilots."""

import math

from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jnp
import optax

from scale_rl.agents.sac.sac_network import SACActor
from scale_rl.networks.utils import tree_norm


def critic_values(critic, observations, actions, *, critic_use_cdq):
    """Match the SAC actor's frozen critic objective."""
    values = critic(observations=observations, actions=actions)
    if critic_use_cdq:
        values = jnp.minimum(values[0], values[1])
    return values.reshape(-1)


def _actor_terms(actor, params, critic, observations, key, critic_use_cdq):
    dist = actor.apply({"params": params}, observations=observations)
    actions = dist.sample(seed=key)
    log_prob = dist.log_prob(actions)
    q = critic_values(critic, observations, actions, critic_use_cdq=critic_use_cdq)
    return q, log_prob, actions


def propose_actor_update(
    actor,
    critic,
    temperature,
    observations,
    key,
    *,
    critic_use_cdq,
    saturation_penalty=0.0,
    saturation_threshold=0.99,
    entropy_weight=None,
):
    """Make a one-step actor proposal without mutating the original trainer."""
    alpha = temperature() if entropy_weight is None else entropy_weight

    def loss(params):
        q, log_prob, actions = _actor_terms(
            actor, params, critic, observations, key, critic_use_cdq
        )
        objective = (log_prob * alpha - q).mean()
        if saturation_penalty:
            excess = jax.nn.relu(jnp.abs(actions) - saturation_threshold)
            objective = objective + saturation_penalty * jnp.mean(excess**2)
        return objective, {
            "actor/loss": objective,
            "actor/entropy": -log_prob.mean(),
            "actor/saturation": jnp.mean(jnp.abs(actions) > saturation_threshold),
            "actor/temperature": alpha,
            "actor/q": q.mean(),
        }

    return actor.apply_gradient(loss)


def actor_gradient_components(actor, critic, observations, key, *, critic_use_cdq):
    """Return the -Q gradient and the log-probability gradient at alpha = 1."""

    def terms(params):
        q, log_prob, _ = _actor_terms(
            actor, params, critic, observations, key, critic_use_cdq
        )
        return -q.mean(), log_prob.mean()

    _, backward = jax.vjp(terms, actor.params)
    return backward((1.0, 0.0))[0], backward((0.0, 1.0))[0]


def gradient_balance(critic_grad, entropy_grad, temperature):
    """Report pre-optimizer component norms and their cosine alignment."""
    q_norm, h_norm = tree_norm(critic_grad), tree_norm(entropy_grad)
    inner_product = sum(
        jnp.vdot(x, y)
        for x, y in zip(jax.tree.leaves(critic_grad), jax.tree.leaves(entropy_grad))
    )
    return {
        "critic_norm": q_norm,
        "entropy_norm": h_norm,
        "weighted_entropy_norm": temperature * h_norm,
        "entropy_to_critic_norm": jnp.where(
            q_norm > 0, temperature * h_norm / q_norm, jnp.nan
        ),
        "cosine": jnp.where(
            q_norm * h_norm > 0, inner_product / (q_norm * h_norm), jnp.nan
        ),
    }


def matched_entropy_weight(reference_critic_grad, treatment_critic_grad, temperature):
    """Match the weighted entropy/Q norm ratio at the common starting actor."""
    ref, target, alpha = map(
        float,
        (
            tree_norm(reference_critic_grad),
            tree_norm(treatment_critic_grad),
            temperature,
        ),
    )
    if any(not math.isfinite(x) or x <= 0 for x in (ref, target, alpha)):
        raise ValueError(
            "entropy matching requires positive finite critic norms and alpha"
        )
    matched = alpha * target / ref
    if not math.isfinite(matched):
        raise ValueError("matched entropy weight is not finite")
    return matched


@jax.jit
def scale_actor_displacement(actor, proposed_actor, fraction):
    """Scale a proposal for evaluation, retaining the starting optimizer state."""
    return actor.replace(
        params=jax.tree.map(
            lambda start, end: start + fraction * (end - start),
            actor.params,
            proposed_actor.params,
        )
    )


@jax.jit
def actor_policy_kl(actor, candidate, observations):
    """Mean KL(start || candidate) on held-out states, invariant to the tanh bijector."""
    start = actor(observations=observations).distribution
    end = candidate(observations=observations).distribution
    return jnp.mean(start.kl_divergence(end))


def reset_actor_optimizer(actor):
    """Use a fresh optimizer state for both actor width conditions."""
    if actor.tx is None:
        raise ValueError("actor has no optimizer")
    if actor.dynamic_scale is not None:
        raise ValueError("actor width comparison requires fixed precision")
    return actor.replace(opt_state=actor.tx.init(actor.params), update_step=0)


def widen_actor(actor, *, factor=2):
    """Replicate SimBa features while preserving the initial action distribution.

    Equal replication preserves residual LayerNorm statistics. Split the
    predictor's outgoing weights unequally across replicas so the actor can
    learn different features instead of staying symmetric under its optimizer.
    """
    net = actor.network_def
    if (
        not isinstance(net, SACActor)
        or net.block_type not in {"mlp", "residual"}
        or actor.sparse
        or actor.network_mask is not None
        or type(factor) is not int
        or factor < 2
    ):
        raise ValueError(
            "widening requires a dense SimBa SAC actor and integer factor >= 2"
        )

    params = unfreeze(actor.params)
    encoder = params["encoder"]

    def output_repeat(layer):
        layer["kernel"] = jnp.repeat(layer["kernel"], factor, axis=1)
        layer["bias"] = jnp.repeat(layer["bias"], factor)

    def input_repeat(layer):
        layer["kernel"] = jnp.repeat(layer["kernel"] / factor, factor, axis=0)

    def repeat_norm(layer):
        for key in ("scale", "bias"):
            layer[key] = jnp.repeat(layer[key], factor)

    if net.block_type == "mlp":
        block = encoder["MLPBlock_0"]
        output_repeat(block["Dense_0"])
        input_repeat(block["Dense_1"])
        output_repeat(block["Dense_1"])
    else:
        output_repeat(encoder["Dense_0"])
        for index in range(net.num_blocks):
            block = encoder[f"ResidualBlock_{index}"]
            repeat_norm(block["LayerNorm_0"])
            for name in ("Dense_0", "Dense_1"):
                input_repeat(block[name])
                output_repeat(block[name])
        repeat_norm(encoder["LayerNorm_0"])

    for layer in params["predictor"].values():
        if not isinstance(layer, dict) or "kernel" not in layer:
            raise ValueError("unsupported SAC actor predictor layout")
        kernel = layer["kernel"]
        split = jnp.full((factor,), 1 / factor, dtype=kernel.dtype)
        split = split.at[0].add(0.5 / factor).at[1].add(-0.5 / factor)
        layer["kernel"] = (kernel[:, None, :] * split[None, :, None]).reshape(
            net.hidden_dim * factor, net.action_dim
        )
    if isinstance(actor.params, FrozenDict):
        params = freeze(params)
    wider_net = net.clone(hidden_dim=net.hidden_dim * factor)
    return reset_actor_optimizer(actor.replace(network_def=wider_net, params=params))
