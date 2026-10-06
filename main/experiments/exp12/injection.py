"""Plasticity injection (Nikishin et al. 2023) for the SimBa SAC critic.

Head = last m residual blocks + post-LayerNorm + output layer; the trunk (input
projection + the first D - m blocks) stays trainable. After injection the critic
computes

    Q(s, a) = old(z) + (new(z) - copy(z)),   z = trunk(s, a)

where `old` is the frozen original head and `new` / `copy` are two identical
freshly initialised heads (SimBa initialisers), `new` trainable and `copy` frozen.
Grouping the difference first makes new(z) - copy(z) exactly 0 at injection, so
predictions and dQ/da are unchanged bit for bit. Frozen heads have their
parameter gradients stopped, but gradients still flow through them into the
trunk. The trainable parameter count is unchanged; the total grows by two heads.

Optimizer: the trunk keeps its existing AdamW state (moments and step count), the
new head gets a newly created AdamW state with its own step count, and frozen
parameters receive exactly zero updates (no weight decay). The target critic is
injected the same way: its own trunk and head become its trunk and frozen `old`,
and its `new` / `copy` take the online critic's new-head values, so its
prediction is unchanged too. Polyak averaging then runs over the whole tree.
"""

from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax.lax import convert_element_type

from scale_rl.networks.critics import LinearCritic
from scale_rl.networks.layers import ResidualBlock
from scale_rl.networks.trainer import Trainer
from scale_rl.networks.utils import orthogonal_init

M_LABELS = ("last", "half", "all")
# Default for the unit tests; runs read checks.check1_tolerance_eps from the config (proposed, pending approval).
CHECK1_TOLERANCE_EPS = 64
INJECTION_STREAM = 0x494E4A43  # "INJC"


def head_blocks(label: str, num_blocks: int) -> int:
    """m for the Methodology's candidates: last block, half of the residual blocks, all of them."""
    if label == "last":
        return 1
    if label == "half":
        return max(1, num_blocks // 2)
    if label == "all":
        return num_blocks
    raise ValueError(f"injection m must be one of {M_LABELS}, got {label!r}")


class _Trunk(nn.Module):
    num_blocks: int
    hidden_dim: int
    dtype: Any

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal_init(1), dtype=self.dtype)(x)
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.hidden_dim, dtype=self.dtype)(x)
        return x


class _Head(nn.Module):
    num_blocks: int
    hidden_dim: int
    dtype: Any

    @nn.compact
    def __call__(self, z):
        for _ in range(self.num_blocks):
            z = ResidualBlock(self.hidden_dim, dtype=self.dtype)(z)
        z = nn.LayerNorm(dtype=self.dtype)(z)
        return LinearCritic()(z)


class InjectedSACCritic(nn.Module):
    num_blocks: int
    hidden_dim: int
    head_blocks: int
    dtype: Any

    def setup(self):
        self.trunk = _Trunk(self.num_blocks - self.head_blocks, self.hidden_dim, self.dtype)
        self.old = _Head(self.head_blocks, self.hidden_dim, self.dtype)
        self.new = _Head(self.head_blocks, self.hidden_dim, self.dtype)
        self.copy = _Head(self.head_blocks, self.hidden_dim, self.dtype)

    def apply(self, variables, *args, **kwargs):
        """Frozen heads: their parameters get no gradient, but inputs still do."""
        params = dict(variables["params"])
        params["old"] = jax.lax.stop_gradient(params["old"])
        params["copy"] = jax.lax.stop_gradient(params["copy"])
        return super().apply({**variables, "params": params}, *args, **kwargs)

    def __call__(self, observations, actions):
        inputs = convert_element_type(jnp.concatenate((observations, actions), axis=1), self.dtype)
        z = self.trunk(inputs)
        return self.old(z) + (self.new(z) - self.copy(z))


def split_params(params, num_blocks: int, m: int) -> Tuple[Dict, Dict]:
    """Original SACCritic params -> (trunk params, head params) in the injected naming."""
    enc = params["encoder"]
    k = num_blocks - m
    trunk = {"Dense_0": enc["Dense_0"], **{f"ResidualBlock_{i}": enc[f"ResidualBlock_{i}"] for i in range(k)}}
    head = {
        **{f"ResidualBlock_{i}": enc[f"ResidualBlock_{k + i}"] for i in range(m)},
        "LayerNorm_0": enc["LayerNorm_0"],
        "LinearCritic_0": params["predictor"],
    }
    return trunk, head


def injected_optimizer(learning_rate: float, weight_decay: float) -> optax.GradientTransformation:
    """AdamW on `trunk` and on `new`, each with its own state; zero updates for `old` and `copy`."""
    adamw = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    def init(params):
        return {"trunk": adamw.init(params["trunk"]), "new": adamw.init(params["new"])}

    def update(grads, state, params):
        trunk_updates, trunk_state = adamw.update(grads["trunk"], state["trunk"], params["trunk"])
        new_updates, new_state = adamw.update(grads["new"], state["new"], params["new"])
        zeros = lambda tree: jax.tree_util.tree_map(jnp.zeros_like, tree)
        updates = {"trunk": trunk_updates, "new": new_updates, "old": zeros(params["old"]),
                   "copy": zeros(params["copy"])}
        return updates, {"trunk": trunk_state, "new": new_state}

    return optax.GradientTransformation(init, update)


def _carry_trunk_state(old_state, trunk_params, num_blocks, m, learning_rate, weight_decay):
    """The existing AdamW state restricted to the trunk parameters (step count kept)."""
    adam_state, *rest = old_state
    trunk_mu, _ = split_params(adam_state.mu, num_blocks, m)
    trunk_nu, _ = split_params(adam_state.nu, num_blocks, m)
    template = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay).init(trunk_params)
    carried = template[0]._replace(count=adam_state.count, mu=trunk_mu, nu=trunk_nu)
    return (carried, *rest)


def injection_key(seed: int):
    return jax.random.fold_in(jax.random.PRNGKey(seed), INJECTION_STREAM)


def inject(critic: Trainer, target_critic: Trainer, m_label: str, key, learning_rate: float,
           weight_decay: float) -> Tuple[Trainer, Trainer]:
    """Returns (injected critic, injected target critic)."""
    original = critic.network_def
    num_blocks, hidden = original.num_blocks, original.hidden_dim
    m = head_blocks(m_label, num_blocks)
    network_def = InjectedSACCritic(num_blocks, hidden, m, original.dtype)
    new_head = _Head(m, hidden, original.dtype).init(key, jnp.zeros((1, hidden), original.dtype))["params"]
    new_head = jax.tree_util.tree_map(jnp.asarray, new_head)
    copy_head = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), new_head)

    trunk, old = split_params(critic.params, num_blocks, m)
    params = {"trunk": trunk, "old": old, "new": new_head, "copy": copy_head}
    target_trunk, target_old = split_params(target_critic.params, num_blocks, m)
    target_params = {"trunk": target_trunk, "old": target_old,
                     "new": jax.tree_util.tree_map(jnp.array, new_head),
                     "copy": jax.tree_util.tree_map(jnp.array, new_head)}

    tx = injected_optimizer(learning_rate, weight_decay)
    opt_state = {
        "trunk": _carry_trunk_state(critic.opt_state, trunk, num_blocks, m, learning_rate, weight_decay),
        "new": optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay).init(new_head),
    }
    injected = Trainer(network_def=network_def, params=params, tx=tx, opt_state=opt_state,
                       update_step=critic.update_step, dynamic_scale=None, sparse=False, network_mask=None)
    injected_target = Trainer(network_def=network_def, params=target_params, tx=None, opt_state=None,
                              update_step=target_critic.update_step, dynamic_scale=None, sparse=False,
                              network_mask=None)
    return injected, injected_target


def count_params(tree) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(tree)))


def trainable_count(injected_params) -> int:
    return count_params(injected_params["trunk"]) + count_params(injected_params["new"])


INJECTED_TWIN = "VmapInjectedSACCritic_0"


class InjectedClippedDoubleCritic(nn.Module):
    """A twin critic after injection (amendment (z)): InjectedSACCritic vmapped as SACClippedDoubleCritic."""

    num_blocks: int
    hidden_dim: int
    head_blocks: int
    dtype: Any
    num_qs: int = 2

    def apply(self, variables, *args, **kwargs):
        """Frozen heads of both networks: their parameters get no gradient, but inputs still do."""
        params = dict(variables["params"])
        inner = dict(params[INJECTED_TWIN])
        inner["old"] = jax.lax.stop_gradient(inner["old"])
        inner["copy"] = jax.lax.stop_gradient(inner["copy"])
        params[INJECTED_TWIN] = inner
        return super().apply({**variables, "params": params}, *args, **kwargs)

    @nn.compact
    def __call__(self, observations, actions):
        vmapped = nn.vmap(InjectedSACCritic, variable_axes={"params": 0}, split_rngs={"params": True},
                          in_axes=0, out_axes=0, axis_size=self.num_qs)
        tile = lambda x: jnp.broadcast_to(x, (self.num_qs, *x.shape))
        return vmapped(self.num_blocks, self.hidden_dim, self.head_blocks, self.dtype)(tile(observations),
                                                                                       tile(actions))


def injected_twin_optimizer(learning_rate: float, weight_decay: float) -> optax.GradientTransformation:
    """injected_optimizer on both networks at once; AdamW is elementwise, so each network is updated as alone."""
    inner = injected_optimizer(learning_rate, weight_decay)

    def update(grads, state, params):
        updates, state = inner.update(grads[INJECTED_TWIN], state, params[INJECTED_TWIN])
        return {INJECTED_TWIN: updates}, state

    return optax.GradientTransformation(lambda params: inner.init(params[INJECTED_TWIN]), update)


def inject_twin(critic: Trainer, target_critic: Trainer, m_label: str, key, learning_rate: float,
                weight_decay: float) -> Tuple[Trainer, Trainer]:
    """inject() for a twin critic: the same head (last m blocks, post-LayerNorm, output layer) in both networks
    and both target networks. Each network's new head is initialised from its own key, as the twin's networks
    are; its frozen copy and the target's new and copy take the same values."""
    from experiments.exp12.twin import TWIN

    original = critic.network_def
    num_blocks, hidden, n = original.num_blocks, original.hidden_dim, original.num_qs
    m = head_blocks(m_label, num_blocks)
    network_def = InjectedClippedDoubleCritic(num_blocks, hidden, m, original.dtype, n)
    heads = [_Head(m, hidden, original.dtype).init(k, jnp.zeros((1, hidden), original.dtype))["params"]
             for k in jax.random.split(key, n)]
    twin_new = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *heads)
    twin_copy = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), twin_new)

    trunk, old = split_params(critic.params[TWIN], num_blocks, m)
    params = {INJECTED_TWIN: {"trunk": trunk, "old": old, "new": twin_new, "copy": twin_copy}}
    target_trunk, target_old = split_params(target_critic.params[TWIN], num_blocks, m)
    target_params = {INJECTED_TWIN: {"trunk": target_trunk, "old": target_old,
                                     "new": jax.tree_util.tree_map(jnp.array, twin_new),
                                     "copy": jax.tree_util.tree_map(jnp.array, twin_new)}}

    adam_state, *rest = critic.opt_state
    per_network = (adam_state._replace(mu=adam_state.mu[TWIN], nu=adam_state.nu[TWIN]), *rest)
    twin_opt_state = {
        "trunk": _carry_trunk_state(per_network, trunk, num_blocks, m, learning_rate, weight_decay),
        "new": optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay).init(twin_new),
    }
    injected = Trainer(network_def=network_def, params=params, tx=injected_twin_optimizer(learning_rate, weight_decay),
                       opt_state=twin_opt_state, update_step=critic.update_step, dynamic_scale=None, sparse=False,
                       network_mask=None)
    injected_target = Trainer(network_def=network_def, params=target_params, tx=None, opt_state=None,
                              update_step=target_critic.update_step, dynamic_scale=None, sparse=False,
                              network_mask=None)
    return injected, injected_target
