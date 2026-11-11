"""Primitives for isolated, branch-local online actor diagnostics.

These helpers are not an experiment runner and do not authorize a pilot.
"""
import copy
import math

import numpy as np
import jax
import jax.numpy as jnp

from scale_rl.agents.sac.sac_update import update_actor
from scale_rl.networks.utils import tree_norm


def copy_reference_replay(reference):
    """Copy reference transitions without joining its unfinished episode to a branch."""
    branch = copy.deepcopy(reference)
    branch._n_step_transitions.clear()
    return branch


def learning_branch_step(actor_key, critic_key, actor, critic, target, temperature,
                         batch, gamma, n_step, critic_use_cdq, target_tau, target_entropy):
    """Use the source SAC update for an isolated, learning actor/critic branch."""
    from scale_rl.agents.sac.sac_agent import _sac_update
    return _sac_update(actor_key, critic_key, actor, critic, target, temperature,
                       batch, gamma, n_step, critic_use_cdq, target_tau,
                       target_entropy, batch)


def reference_roles(regime):
    """Return (reference, small shadow, large shadow) block/width pairs."""
    if regime == "reference_small":
        return (2, 512), (2, 512), (4, 1536)
    if regime == "reference_large":
        return (4, 1536), (2, 512), (4, 1536)
    raise ValueError(f"unknown reference regime: {regime}")


def diagnostic_actor_step(actor, critic, temperature, observations, key, critic_use_cdq):
    """Perform the existing SAC actor update against a frozen critic."""
    observations = jnp.asarray(observations)
    if observations.ndim < 2 or not bool(jnp.all(jnp.isfinite(observations))):
        raise ValueError("actor observations must be a finite batch")
    updated, info = update_actor(key, actor, critic, temperature,
                                 {"observation": observations}, critic_use_cdq)
    delta = jax.tree.map(lambda new, old: new - old, updated.params, actor.params)
    info["parameter_update_norm"] = tree_norm(delta)
    if not bool(jnp.isfinite(info["parameter_update_norm"])):
        raise ValueError("nonfinite actor update")
    return updated, info


class ObservationBuffer:
    """Retain raw branch observations and sample uniformly with replacement."""

    def __init__(self, seed):
        self._observations = []
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self._observations)

    def append(self, observation):
        value = np.asarray(observation)
        if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
            raise ValueError("observation must be finite and numeric")
        if self._observations and value.shape != self._observations[0].shape:
            raise ValueError("inconsistent observation shape")
        self._observations.append(value.copy())

    def sample(self, batch_size):
        if not self._observations or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("sampling requires observations and a positive batch size")
        indices = self._rng.integers(0, len(self), size=int(batch_size))
        return np.stack([self._observations[i] for i in indices]), indices

    def state_dict(self):
        return {"observations": [x.copy() for x in self._observations],
                "rng": copy.deepcopy(self._rng.bit_generator.state)}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(0)
        for observation in state["observations"]:
            obj.append(observation)
        obj._rng.bit_generator.state = copy.deepcopy(state["rng"])
        return obj


class ActorUpdateSchedule:
    """Match Exp12Trainer's fractional learner-update accumulator."""

    def __init__(self, updates_per_interaction, carry=0.):
        self.rate = float(updates_per_interaction)
        self.carry = float(carry)
        if not math.isfinite(self.rate) or self.rate < 0 or not 0 <= self.carry < 1:
            raise ValueError("invalid update rate or carry")

    def advance(self):
        self.carry += self.rate
        count = 0
        while self.carry >= 1:
            self.carry -= 1
            count += 1
        return count

    def state_dict(self):
        return {"updates_per_interaction": self.rate, "carry": self.carry}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["updates_per_interaction"], state["carry"])


class DiagnosticActorBranch:
    """Actor and sampler state for one isolated diagnostic branch.

    The caller owns environment interaction and supplies pre-action observations.
    State snapshots here are in-memory; durable environment resume is not yet wired.
    """

    def __init__(self, actor, critic, temperature, seed, batch_size,
                 updates_per_interaction, critic_use_cdq, carry=0.):
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.actor = actor
        self.critic = critic
        self.temperature = temperature
        self.buffer = ObservationBuffer(seed)
        self.schedule = ActorUpdateSchedule(updates_per_interaction, carry)
        self.key = jax.random.PRNGKey(seed)
        self.collection_key = jax.random.fold_in(self.key, 1)
        self.batch_size = int(batch_size)
        self.critic_use_cdq = bool(critic_use_cdq)
        self.interactions = 0
        self.optimizer_updates = 0

    def act(self, raw_observation, normalize=None):
        """Sample from this branch's current actor without touching update RNG."""
        observation = np.asarray(raw_observation)
        if normalize is not None:
            observation = normalize(observation)
        self.collection_key, key = jax.random.split(self.collection_key)
        action = np.asarray(self.actor(
            observations=jnp.asarray(observation), temperature=1.).sample(seed=key))
        if not np.all(np.isfinite(action)):
            raise ValueError("nonfinite diagnostic action")
        return action

    def observe_and_update(self, raw_observation, normalize=None):
        """Record one branch observation and execute its scheduled actor updates."""
        self.buffer.append(raw_observation)
        self.interactions += 1
        indices_log, update_log = [], []
        for _ in range(self.schedule.advance()):
            batch, indices = self.buffer.sample(self.batch_size)
            if normalize is not None:
                batch = normalize(batch)
            self.key, action_key = jax.random.split(self.key)
            self.actor, info = diagnostic_actor_step(
                self.actor, self.critic, self.temperature, batch, action_key,
                self.critic_use_cdq)
            self.optimizer_updates += 1
            indices_log.append(indices.tolist())
            update_log.append({name: float(info[name]) for name in
                               ("train/actor_gnorm", "parameter_update_norm")})
        return {"sample_indices": indices_log, "updates": update_log,
                "interactions": self.interactions,
                "optimizer_updates": self.optimizer_updates}

    def state_dict(self):
        return {"actor": self.actor, "buffer": self.buffer.state_dict(),
                "schedule": self.schedule.state_dict(), "key": np.asarray(self.key).copy(),
                "collection_key": np.asarray(self.collection_key).copy(),
                "batch_size": self.batch_size, "critic_use_cdq": self.critic_use_cdq,
                "interactions": self.interactions, "optimizer_updates": self.optimizer_updates}

    @classmethod
    def from_state_dict(cls, state, critic, temperature):
        schedule = ActorUpdateSchedule.from_state_dict(state["schedule"])
        obj = cls(state["actor"], critic, temperature, 0, state["batch_size"],
                  schedule.rate, state["critic_use_cdq"], schedule.carry)
        obj.buffer = ObservationBuffer.from_state_dict(state["buffer"])
        obj.key = jnp.asarray(state["key"])
        obj.collection_key = jnp.asarray(state["collection_key"])
        obj.interactions = int(state["interactions"])
        obj.optimizer_updates = int(state["optimizer_updates"])
        return obj


def paired_returns(baseline, small, large):
    """Keep matched episode returns and baseline-relative contrasts separate."""
    arrays = [np.asarray(values, dtype=float) for values in (baseline, small, large)]
    if (any(a.ndim != 1 or not a.size or not np.all(np.isfinite(a)) for a in arrays)
            or len({len(a) for a in arrays}) != 1):
        raise ValueError("paired episode returns must be finite and equally sized")
    return [{"baseline": float(b), "small": float(s), "large": float(l),
             "delta_small": float(s - b), "delta_large": float(l - b),
             "advantage_large": float(l - s)} for b, s, l in zip(*arrays)]


def regime_contrasts(small_reference_checkpoints, large_reference_checkpoints):
    """Equal-checkpoint means within regime, then equal weight across regimes.

    Inputs are checkpoint-level means for one environment, seed, and horizon;
    callers must not pool episodes or mix environments in these inputs.
    """
    if (not isinstance(small_reference_checkpoints, dict)
            or not isinstance(large_reference_checkpoints, dict)
            or not small_reference_checkpoints
            or small_reference_checkpoints.keys() != large_reference_checkpoints.keys()):
        raise ValueError("reference regimes must have matching checkpoint coverage")
    arrays = [np.asarray(list(x.values()), dtype=float) for x in
              (small_reference_checkpoints, large_reference_checkpoints)]
    if any(a.ndim != 1 or not a.size or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("checkpoint means must be finite nonempty vectors")
    small, large = (float(np.mean(a)) for a in arrays)
    return {"small_reference": small, "large_reference": large,
            "sensitivity": large - small, "equal_weight_mean": (small + large) / 2}


def combine_pilot_analyses(small_reference, large_reference):
    """Compare matched independent runs without pooling episodes as replicas."""
    if (small_reference.get("reference_regime") != "reference_small"
            or large_reference.get("reference_regime") != "reference_large"
            or small_reference.get("environment") != large_reference.get("environment")
            or small_reference.get("training_seed") != large_reference.get("training_seed")
            or small_reference["horizons"].keys() != large_reference["horizons"].keys()):
        raise ValueError("pilot analyses must match environment, seed, and horizons")
    return {int(horizon): regime_contrasts(
        small_reference["horizons"][horizon]["checkpoint_means"],
        large_reference["horizons"][horizon]["checkpoint_means"])
        for horizon in small_reference["horizons"]}
