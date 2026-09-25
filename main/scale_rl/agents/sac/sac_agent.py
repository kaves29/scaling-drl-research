import functools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import orbax.checkpoint

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import dynamic_scale

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.sac.sac_network import (
    SACActor,
    SACClippedDoubleCritic,
    SACCritic,
    SACTemperature,
)
from scale_rl.agents.sac.sac_update import (
    update_actor,
    update_critic,
    update_target_network,
    update_temperature,
    get_actor_with_metrics,
    get_critic_with_metrics,
    compute_actor_gradient_cosine,
)
from scale_rl.buffers.base_buffer import Batch
from scale_rl.networks.trainer import PRNGKey, Trainer
from scale_rl.agents.sparse import get_sparsities_erdos_renyi,get_var_shape_dict,create_random_mask
from scale_rl.networks.metrics import print_num_parameters,flatten_dict,format_params_str
"""
The @dataclass decorator must have `frozen=True` to ensure the instance is immutable,
allowing it to be treated as a static variable in JAX.
"""


@dataclass(frozen=True)
class SACConfig:
    seed: int
    num_train_envs: int
    max_episode_steps: int
    normalize_observation: bool

    actor_block_type: str
    actor_num_blocks: int
    actor_hidden_dim: int
    actor_learning_rate: float
    actor_weight_decay: float

    critic_block_type: str
    critic_num_blocks: int
    critic_hidden_dim: int
    critic_learning_rate: float
    critic_weight_decay: float
    critic_use_cdq: bool

    temp_target_entropy: float
    temp_target_entropy_coef: float
    temp_initial_value: float
    temp_learning_rate: float
    temp_weight_decay: float

    target_tau: float
    gamma: float
    n_step: int

    mixed_precision: bool

    actor_sparsity: float
    critic_sparsity: float

    # Cadence for the expensive actor_grad_cosine diagnostic (2026-09-24,
    # throughput Step 1) - default has a default (unlike every field above)
    # so existing configs/tests that construct SACConfig without it (e.g.
    # tests/test_sac_agent_checkpoint.py's _make_cfg()) keep working
    # unchanged. See SACAgent.update() for how it's used.
    actor_grad_cosine_every: int = 30


# @functools.partial(
#     jax.jit,
#     static_argnames=(
#         "observation_dim",
#         "action_dim",
#         "cfg",
#     ),
# )
def _init_sac_networks(
    observation_dim: int,
    action_dim: int,
    cfg: SACConfig,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer]:
    fake_observations = jnp.zeros((1, observation_dim))
    fake_actions = jnp.zeros((1, action_dim))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
    compute_dtype = jnp.float16 if cfg.mixed_precision else jnp.float32

    if cfg.actor_sparsity > 0.0:
        actor_sparse = True
    else:
       actor_sparse = False

    # When initializing the network in the flax.nn.Module class, rng_key should be passed as rngs.
    actor_network_def=SACActor(
            block_type=cfg.actor_block_type,
            num_blocks=cfg.actor_num_blocks,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            dtype=compute_dtype,
        )
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        _fake_actor_variables = actor_network_def.init(rng, fake_observations)
    fake_actor_params = jax.device_put(_fake_actor_variables['params'])  # A FrozenDict of parameters
    if actor_sparse:
        actor_network_parm_dict = get_var_shape_dict(fake_actor_params)
        actor_network_sparse = get_sparsities_erdos_renyi(actor_network_parm_dict, default_sparsity=float(cfg.actor_sparsity))
        mask_rng = jax.random.PRNGKey(cfg.seed)
        actor_network_mask = create_random_mask(fake_actor_params, actor_network_sparse, mask_rng)
    else:
        actor_network_mask = None

    actor = Trainer.create(
        network_def=actor_network_def,
        network_inputs={"rngs": actor_key, "observations": fake_observations},
        tx=optax.adamw(
            learning_rate=cfg.actor_learning_rate,
            weight_decay=cfg.actor_weight_decay,
        ),
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
        sparse = actor_sparse,
        network_mask = actor_network_mask
    )


    if cfg.critic_use_cdq:
        critic_network_def = SACClippedDoubleCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype,
        )
    else:
        critic_network_def = SACCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype,
        )
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        _fake_critic_variables = critic_network_def.init(rng, observations=fake_observations, actions=fake_actions)
    fake_critic_params = jax.device_put(_fake_critic_variables['params'])  # A FrozenDict of parameters
    if cfg.critic_sparsity > 0.0:
        critic_sparse = True
    else:
       critic_sparse = False
    if critic_sparse:
        critic_network_parm_dict = get_var_shape_dict(fake_critic_params)
        critic_network_sparse = get_sparsities_erdos_renyi(critic_network_parm_dict, default_sparsity=float(cfg.critic_sparsity))
        mask_rng = jax.random.PRNGKey(cfg.seed)
        critic_network_mask = create_random_mask(fake_critic_params, critic_network_sparse, mask_rng)
    else:
        critic_network_mask = None

    critic = Trainer.create(
        network_def=critic_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=optax.adamw(
            learning_rate=cfg.critic_learning_rate,
            weight_decay=cfg.critic_weight_decay,
        ),
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
        sparse = critic_sparse,
        network_mask = critic_network_mask,
    )

    # we set target critic's parameters identical to critic by using same rng.
    target_network_def = critic_network_def
    target_critic = Trainer.create(
        network_def=target_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=None,
        sparse = critic_sparse,
        network_mask = critic_network_mask,
    )

    temperature = Trainer.create(
        network_def=SACTemperature(cfg.temp_initial_value),
        network_inputs={
            "rngs": temp_key,
        },
        tx=optax.adamw(
            learning_rate=cfg.temp_learning_rate,
            weight_decay=cfg.temp_weight_decay,
        ),
        sparse=False,
        network_mask=None,
    )

    actor_loss_buffer = []
    actor_entropy_buffer = []
    churn_buffer = []
    churn_ref_batch = None
    return rng, actor, critic, target_critic, temperature, actor_loss_buffer, actor_entropy_buffer, churn_buffer, churn_ref_batch


@jax.jit
def _sample_sac_actions(
    rng: PRNGKey,
    actor: Trainer,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    dist = actor(observations=observations, temperature=temperature)
    actions = dist.sample(seed=key)

    return rng, actions


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma",
        "n_step",
        "critic_use_cdq",
        "target_tau",
        "temp_target_entropy",
    ),
)
def _update_sac_networks(
    rng: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    temperature: Trainer,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    target_tau: float,
    temp_target_entropy: float,
    churn_ref_batch: dict,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer, Dict[str, float]]:
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    def get_deterministic_actions(actor_params, actor, observations):
        dist = actor.apply(variables={"params": actor_params}, observations=observations)
        pre_squash_mean = dist.distribution.mean()
        return jnp.tanh(pre_squash_mean)

    a_prev = get_deterministic_actions(actor.params, actor, churn_ref_batch["observation"])
    new_actor, actor_info = update_actor(
        key=actor_key,
        actor=actor,
        critic=critic,
        temperature=temperature,
        batch=batch,
        critic_use_cdq=critic_use_cdq,
    )
    a_curr = get_deterministic_actions(new_actor.params, new_actor, churn_ref_batch["observation"])
    churn = jnp.mean(jnp.linalg.norm(a_curr - a_prev, axis=-1))

    new_temperature, temperature_info = update_temperature(
        temperature=temperature,
        entropy=actor_info["train/entropy"],
        target_entropy=temp_target_entropy,
    )

    new_critic, critic_info = update_critic(
        key=critic_key,
        actor=new_actor,
        critic=critic,
        target_critic=target_critic,
        temperature=new_temperature,
        batch=batch,
        gamma=gamma,
        n_step=n_step,
        critic_use_cdq=critic_use_cdq,
    )

    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_tau=target_tau,
    )

    # actor_grad_cosine (2026-09-24, throughput Step 1): no longer computed
    # here. It's expensive (256-sample vmap'd gradient through the full
    # critic) and was previously computed unconditionally on every call -
    # i.e. 5x per interaction step. compute_actor_gradient_cosine is now
    # its own separately-jitted function (see sac_update.py), called
    # conditionally from SACAgent.update() every actor_grad_cosine_every
    # steps instead, using this same actor_key's sibling split there (see
    # update()'s own docstring note) - it never fed back into new_actor/
    # new_critic/new_target_critic/new_temperature, so removing it from
    # this jitted graph changes nothing about the actual training update.

    info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
        "train/policy_churn": churn,
    }

    return (rng, new_actor, new_critic, new_target_critic, new_temperature, info)


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma",
        "n_step",
        "critic_use_cdq",
        "target_tau",
        "temp_target_entropy",
        "num_updates",
    ),
)
def _scan_update_sac_networks(
    rng: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    temperature: Trainer,
    batch_sequence: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    target_tau: float,
    temp_target_entropy: float,
    churn_ref_batch: dict,
    num_updates: int,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer, Dict[str, float]]:
    """Throughput Step 3 (2026-09-24): fuses `num_updates` calls to
    _update_sac_networks into a single jax.lax.scan, so the caller
    (SACAgent.update_scanned()) does one host->device transfer and one
    compiled dispatch for the whole updates_per_interaction_step batch,
    instead of updates_per_interaction_step separate ones.

    `batch_sequence` is a pytree whose leaves each have a leading axis of
    length num_updates (e.g. observation: [num_updates, batch_size,
    obs_dim]) - lax.scan slices along that axis automatically, feeding one
    per-update batch to _update_sac_networks each iteration, threading
    (rng, actor, critic, target_critic, temperature) through as the scan
    carry exactly as SACAgent.update()'s Python loop did before. Does NOT
    change what any individual update computes - only that all
    num_updates of them are dispatched to the device as one compiled
    program rather than num_updates separate ones.

    churn_ref_batch is intentionally NOT scanned-over (same fixed batch,
    closed over each iteration) - it's a fixed diagnostic reference batch
    (see SACAgent.__init__), not part of the per-update training data.
    """

    def scan_body(carry, batch_i):
        rng, actor, critic, target_critic, temperature = carry
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_temperature,
            info,
        ) = _update_sac_networks(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temperature=temperature,
            batch=batch_i,
            gamma=gamma,
            n_step=n_step,
            critic_use_cdq=critic_use_cdq,
            target_tau=target_tau,
            temp_target_entropy=temp_target_entropy,
            churn_ref_batch=churn_ref_batch,
        )
        new_carry = (new_rng, new_actor, new_critic, new_target_critic, new_temperature)
        return new_carry, info

    init_carry = (rng, actor, critic, target_critic, temperature)
    final_carry, stacked_info = jax.lax.scan(
        scan_body, init_carry, batch_sequence, length=num_updates,
    )
    final_rng, final_actor, final_critic, final_target_critic, final_temperature = final_carry
    return final_rng, final_actor, final_critic, final_target_critic, final_temperature, stacked_info


class SACAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: SACConfig,
    ):
        """
        An agent that randomly selects actions without training.
        Useful for collecting baseline results and for debugging purposes.
        """

        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]

        cfg["temp_target_entropy"] = cfg["temp_target_entropy_coef"] * self._action_dim

        super(SACAgent, self).__init__(
            observation_space,
            action_space,
            cfg,
        )

        # map dictionary to dataclass
        self._cfg = SACConfig(**cfg)

        self._init_network()

    def _init_network(self):
        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            self.actor_loss_buffer,
            self.actor_entropy_buffer,
            self.churn_buffer,
            self.churn_ref_batch
        ) = _init_sac_networks(self._observation_dim, self._action_dim, self._cfg)

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        if training:
            temperature = 1.0
        else:
            temperature = 0.0

        # current timestep observation is "next" observations from the previous timestep
        observations = jnp.asarray(prev_timestep["next_observation"])

        self._rng, actions = _sample_sac_actions(
            self._rng, self._actor, observations, temperature
        )
        actions = np.array(actions)

        return actions
    
    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)

        if self.churn_ref_batch is None:
            self.churn_ref_batch = {k: jnp.array(v) for k, v in batch.items()}

        # Captured before reassignment below (Step 1): compute_actor_gradient_cosine
        # uses the PRE-update actor/critic/temperature - exactly what the
        # old inline call inside _update_sac_networks received.
        pre_update_actor = self._actor
        pre_update_critic = self._critic
        pre_update_temperature = self._temperature

        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            update_info,
        ) = _update_sac_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            critic_use_cdq=self._cfg.critic_use_cdq,
            target_tau=self._cfg.target_tau,
            temp_target_entropy=self._cfg.temp_target_entropy,
            churn_ref_batch=self.churn_ref_batch,
        )

        # Step 1 (throughput, 2026-09-24): actor_grad_cosine is expensive
        # (256-sample vmap'd gradient through the full critic) and is only
        # needed as an infrequent diagnostic. Only split self._rng
        # (consuming one extra key) on the steps where it's actually
        # computed, so every other step's rng trajectory is unchanged from
        # before this change.
        if update_step % self._cfg.actor_grad_cosine_every == 0:
            self._rng, grad_cosine_key = jax.random.split(self._rng)
            update_info["train/actor_grad_cosine"] = compute_actor_gradient_cosine(
                key=grad_cosine_key,
                actor=pre_update_actor,
                critic=pre_update_critic,
                temperature=pre_update_temperature,
                batch=batch,
                critic_use_cdq=self._cfg.critic_use_cdq,
            )

        self.actor_entropy_buffer.append(update_info["train/entropy"])
        self.churn_buffer.append(update_info["train/policy_churn"])
        self.actor_loss_buffer.append(update_info["train/actor_loss"])

        # Step 2 (throughput, 2026-09-24): float() removed here - it forced
        # a device->host sync on every one of updates_per_interaction_step
        # calls (blocking async dispatch), most of which exist only to feed
        # a long-window running average that isn't read until the next
        # logging boundary. update_info now stays device arrays; the
        # caller's WandbTrainerLogger.update_metric()/log_metric() (see
        # scale_rl/common/logger.py) materialize exactly once per logging
        # window instead. This changes WHEN materialization happens, not
        # what is computed.
        return update_info

    def update_scanned(
        self, update_step_start: int, batch_sequence: Dict[str, np.ndarray], num_updates: int
    ) -> List[Dict]:
        """Throughput Step 3 (2026-09-24): fused equivalent of calling
        update(update_step_start + i, {k: v[i] for k, v in
        batch_sequence.items()}) num_updates times in a Python loop - one
        host->device transfer and one compiled jax.lax.scan dispatch
        instead of num_updates separate ones. See _scan_update_sac_networks
        (module level, above) for the actual scan.

        `batch_sequence[key]` must have leading shape [num_updates, ...]
        (the caller stacks num_updates individually-buffer.sample()'d
        batches before calling this - see experiments/angle_1.py).

        Returns a list of num_updates per-update info dicts, in the same
        order/content update() would have returned them individually, so
        callers don't need any special-casing (each is passed to
        logger.update_metric() exactly as before).

        actor_grad_cosine cadence (Step 1, combined with this fusion):
        checked once, against update_step_start only, using the
        PRE-scan actor/critic/temperature and the FIRST batch in the
        sequence. This is exactly equivalent to checking every individual
        update_step_start+i inside the scan (which would need
        jax.lax.cond + a NaN-sentinel to keep the scan's per-iteration
        output pytree shape static, adding real complexity) precisely
        because actor_grad_cosine_every (default 30) is always configured
        as a whole multiple of updates_per_interaction_step (locked at 5 -
        see research-methodology.md) - so at most one of
        [update_step_start, ..., update_step_start+num_updates-1] can ever
        satisfy `% actor_grad_cosine_every == 0`, and when one does, it is
        always update_step_start itself (0 mod K implies the next K-1
        consecutive integers are not, and num_updates <= K for every
        config this study uses). If a future config ever set
        actor_grad_cosine_every to something NOT a whole multiple of
        updates_per_interaction_step, this would silently under-sample
        relative to the naive per-update check - not this study's
        configuration, but worth knowing if either value is ever changed
        independently.
        """
        for key, value in batch_sequence.items():
            batch_sequence[key] = jnp.asarray(value)

        if self.churn_ref_batch is None:
            self.churn_ref_batch = {k: jnp.array(v[0]) for k, v in batch_sequence.items()}

        pre_update_actor = self._actor
        pre_update_critic = self._critic
        pre_update_temperature = self._temperature

        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            stacked_info,
        ) = _scan_update_sac_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch_sequence=batch_sequence,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            critic_use_cdq=self._cfg.critic_use_cdq,
            target_tau=self._cfg.target_tau,
            temp_target_entropy=self._cfg.temp_target_entropy,
            churn_ref_batch=self.churn_ref_batch,
            num_updates=num_updates,
        )

        self.actor_entropy_buffer.append(stacked_info["train/entropy"][-1])
        self.churn_buffer.append(stacked_info["train/policy_churn"][-1])
        self.actor_loss_buffer.append(stacked_info["train/actor_loss"][-1])

        grad_cosine = None
        if update_step_start % self._cfg.actor_grad_cosine_every == 0:
            self._rng, grad_cosine_key = jax.random.split(self._rng)
            first_batch = {k: v[0] for k, v in batch_sequence.items()}
            grad_cosine = compute_actor_gradient_cosine(
                key=grad_cosine_key,
                actor=pre_update_actor,
                critic=pre_update_critic,
                temperature=pre_update_temperature,
                batch=first_batch,
                critic_use_cdq=self._cfg.critic_use_cdq,
            )

        per_update_infos = []
        for i in range(num_updates):
            info_i = {k: v[i] for k, v in stacked_info.items()}
            if grad_cosine is not None and i == 0:
                info_i["train/actor_grad_cosine"] = grad_cosine
            per_update_infos.append(info_i)

        return per_update_infos

    def flush_actor_loss_var(self):
        if len(self.actor_loss_buffer) == 0:
            return None
        losses = jnp.stack(self.actor_loss_buffer)
        var = float(jnp.var(losses))
        self.actor_loss_buffer = []
        return var

    def flush_policy_churn(self):
        if len(self.churn_buffer) == 0:
            return None
        val = float(jnp.mean(jnp.stack(self.churn_buffer)))
        self.churn_buffer = []
        return val

    def mean_entropy(self):
        if len(self.actor_entropy_buffer) == 0:
            return None
        entropy = jnp.stack(self.actor_entropy_buffer)
        mean = float(jnp.mean(entropy))
        self.actor_entropy_buffer = []
        return mean
    
    def get_metrics(self, update_step: int, batch: Dict[str, np.ndarray]):
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)
        actor_metrics_info = get_actor_with_metrics(actor=self._actor,batch=batch)
        critic_metrics_info = get_critic_with_metrics(key=self._rng,actor=self._actor,critic=self._critic,batch=batch,critic_use_cdq=self._cfg.critic_use_cdq)
        actor_loss_var = self.flush_actor_loss_var()
        actor_mean_entropy = self.mean_entropy()
        policy_churn = self.flush_policy_churn()
        combined_metric_info = {**actor_metrics_info, 
                                **critic_metrics_info,
                                "train/actor_loss_var": actor_loss_var,
                                "train/entropy": actor_mean_entropy,
                                "train/churn": policy_churn
                                }
        return combined_metric_info

    def get_q_value(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Raw critic Q-value estimate for (observation, action) pairs.

        Uses the same critic-invocation convention as the actor loss
        (min(q1, q2) under clipped double-Q; see update_actor in
        sac_update.py) so this is consistent with how the critic is already
        used everywhere else in this codebase, not a new estimator.

        Callers must pass already-preprocessed observations (e.g. already
        normalized if this agent is wrapped by ObservationNormalizer) since
        this method operates on the raw SACAgent only. Prefer calling
        `agent.get_q_value(...)` on the (possibly wrapped) agent object
        returned by create_agent so normalization happens automatically.
        """
        observations = jnp.asarray(observations)
        actions = jnp.asarray(actions)

        if self._cfg.critic_use_cdq:
            q1, q2 = self._critic(observations=observations, actions=actions)
            q = jnp.minimum(q1, q2)
        else:
            q = self._critic(observations=observations, actions=actions)

        return np.array(q).reshape(-1)

    @property
    def actor(self) -> Trainer:
        """Read-only access to the frozen actor Trainer (params + network_def),
        for callers that need the raw JAX pytree directly (e.g. Angle 2B's
        counterfactual actor-gradient measurement) rather than going through
        sample_actions/update."""
        return self._actor

    @property
    def critic(self) -> Trainer:
        """Read-only access to the frozen critic Trainer. See `actor`."""
        return self._critic

    @property
    def temperature(self) -> Trainer:
        """Read-only access to the frozen temperature (entropy coefficient)
        Trainer. See `actor`."""
        return self._temperature

    def get_num_parameters(self):
        actor_num_params = print_num_parameters(flatten_dict(self._actor.params),network_type='actor_simba')
        critic_num_params = print_num_parameters(flatten_dict(self._critic.params),network_type='critic_simba')
        total_params = actor_num_params + critic_num_params

        num_str = format_params_str(total_params)
        num_str_actor = format_params_str(actor_num_params)
        num_str_critic = format_params_str(critic_num_params)
        print(f" total params:",num_str)

        return  total_params, num_str, num_str_actor, num_str_critic
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Saves the agent's JAX PyTree state (networks, optimizers, PRNG key, churn batch) using Orbax."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, "agent_ckpt")

        state = {
            "rng": self._rng,
            "actor": self._actor,
            "critic": self._critic,
            "target_critic": self._target_critic,
            "temperature": self._temperature,
            "churn_ref_batch": self.churn_ref_batch,
        }

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpointer.save(ckpt_path, state, force=True)

    def load_checkpoint(self, checkpoint_dir: str):
        """Restores the agent's JAX PyTree state back into their exact Flax TrainState structures."""
        ckpt_path = os.path.join(checkpoint_dir, "agent_ckpt")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        # churn_ref_batch excluded: PyTreeRestore uses the target's shape as
        # ground truth per leaf, and self.churn_ref_batch is always None
        # here - a None leaf in `item` silently discards a real saved value.
        target_state = {
            "rng": self._rng,
            "actor": self._actor,
            "critic": self._critic,
            "target_critic": self._target_critic,
            "temperature": self._temperature,
        }
        restored = checkpointer.restore(ckpt_path, args=orbax.checkpoint.args.PyTreeRestore(item=target_state))
        self._rng = restored["rng"]
        self._actor = restored["actor"]
        self._critic = restored["critic"]
        self._target_critic = restored["target_critic"]
        self._temperature = restored["temperature"]

        # Unconstrained restore (no `item=`) recovers its real on-disk shape.
        churn_restored = checkpointer.restore(ckpt_path)
        self.churn_ref_batch = churn_restored["churn_ref_batch"]
    