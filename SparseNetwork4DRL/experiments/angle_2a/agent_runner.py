"""Trains one fully independent SAC agent up to an exact interaction step,
capturing enough per-transition environment state to later sample probes and
run exact-state Monte Carlo rollouts from its own replay buffer.

This deliberately duplicates (rather than imports/calls) the shape of
experiments/angle_1.py's training loop, because Angle 2A's protocol has hard
constraints Angle 1 doesn't: exactly one participant's transitions may ever
enter its own buffer (no shared buffers/actors/critics/optimizer state/RNG
across agents - see module docstring in experiments/angle_2_a.py), and each
agent must additionally record raw environment state per transition, which
Angle 1 has no reason to do. Reusing scale_rl.agents.create_agent,
scale_rl.buffers.create_buffer, and scale_rl.envs.create_envs exactly as
Angle 1 does keeps everything else consistent with the rest of the
repository.
"""

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.config import RoleArchitecture, build_role_agent_cfg
from experiments.angle_2a.env_state import assert_dmc_env_type, capture_env_state
from experiments.angle_2a.errors import Angle2AConfigError
from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from scale_rl.envs import create_envs


def derive_rng_seed(base_seed: int, context: str) -> int:
    """Deterministic, process-independent seed derived from (base_seed, context).

    Deliberately uses hashlib rather than Python's built-in hash(): hash() of
    a str (or anything containing one) is randomized per-process by default
    (PYTHONHASHSEED), which would silently break reproducibility across
    separate invocations of the same command even with the same base_seed.

    Used to give each independently-trained Angle 2A agent (D_5x768,
    D_7x1024, the shared R_2x512 trajectory, and each null-baseline agent)
    its own deterministic global NumPy/Python random stream for replay-buffer
    sampling (scale_rl.buffers.numpy_buffer.NpyUniformBuffer.sample() draws
    from the global np.random state, not a locally-seeded Generator) -
    without making that stream depend on which agent happened to train
    first in the process. Actor/critic initialization and environment task
    randomization deliberately continue to use `base_seed` directly
    (unchanged) - that is the intentional "same nominal seed" convention for
    controlled architecture comparisons; only the downstream, execution-
    order-vulnerable global RNG consumption is what this addresses.
    """
    digest = hashlib.sha256(f"{base_seed}:{context}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def seed_global_rng_for_agent(base_seed: int, context: str) -> int:
    """Resets numpy's and Python's global RNG state to a value deterministically
    derived from (base_seed, context), and returns that derived seed. Call this
    once per independently-trained agent, right before its own training loop
    starts, so its buffer-sampling sequence is reproducible and independent of
    training order."""
    agent_seed = derive_rng_seed(base_seed, context)
    np.random.seed(agent_seed)
    random.seed(agent_seed)
    return agent_seed


class ProbeCapture:
    """Self-contained, index-aligned record of (observation, action,
    env_state) for every transition an agent has collected, independent of
    (and never sharing memory with) the agent's own NpyUniformBuffer.

    A dedicated structure - rather than reaching into the SAC replay buffer's
    private arrays - is used because the replay buffer has no concept of raw
    environment state; keeping the two aligned via manual indexing into the
    buffer's internals would be fragile and would couple Angle 2A to Angle
    1's buffer implementation details.
    """

    def __init__(self, capacity: int, observation_shape, action_shape):
        self.capacity = capacity
        self._observations = np.empty((capacity,) + tuple(observation_shape), dtype=np.float32)
        self._actions = np.empty((capacity,) + tuple(action_shape), dtype=np.float32)
        self._env_states: List[Optional[Dict[str, Any]]] = [None] * capacity
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def add(self, idx: int, observation: np.ndarray, action: np.ndarray, env_state: Dict[str, Any]) -> None:
        slot = idx % self.capacity
        self._observations[slot] = observation
        self._actions[slot] = action
        self._env_states[slot] = env_state
        self._count = min(self._count + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator):
        """Samples `n` *distinct* transitions. Raises rather than sampling
        with replacement if fewer than `n` transitions were ever collected -
        the protocol calls for exactly n probes, not n draws."""
        if self._count < n:
            raise ValueError(
                f"Only {self._count} transitions were collected before the "
                f"stopping step, but {n} distinct probes were requested. "
                f"Refusing to sample with replacement (that would silently "
                f"weaken the experimental protocol)."
            )
        idxs = rng.choice(self._count, size=n, replace=False)
        return idxs, self._observations[idxs].copy(), self._actions[idxs].copy(), [self._env_states[i] for i in idxs]

@dataclass
class TrainedAgentHandle:
    role: str  # "D" or "R"
    architecture_label: str
    architecture: RoleArchitecture
    agent: Any
    buffer: Any
    train_env: Any
    eval_env: Any
    single_env: Any  # the one underlying (non-vectorized) env instance
    stop_step: int
    probe_capture: ProbeCapture

    def close(self) -> None:
        self.train_env.close()
        self.eval_env.close()


def _run_training_loop(
    agent,
    buffer,
    train_env,
    single_env,
    probe_capture: "ProbeCapture",
    base_cfg,
    stop_step: int,
) -> Iterator[int]:
    """Advances training one interaction_step at a time, yielding the
    interaction_step number immediately after it has been fully processed
    (transition collected, probe captured, any due agent updates applied).

    A generator so train_agent_to_step() can drain it fully; kept as a
    generator (rather than a plain loop) so a future caller needing to
    observe intermediate interaction_step values doesn't require
    reimplementing this stepping logic.
    """
    observations, _ = train_env.reset()
    timestep = None
    update_step = 0
    update_counter = 0

    for interaction_step in range(1, stop_step + 1):
        env_state = capture_env_state(single_env)

        if timestep is not None:
            actions = agent.sample_actions(interaction_step, prev_timestep=timestep, training=True)
        else:
            actions = train_env.action_space.sample()

        probe_capture.add(interaction_step - 1, observations[0], actions[0], env_state)

        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(actions)
        next_buffer_observations = next_observations.copy()
        for env_idx in range(int(base_cfg.env.num_train_envs)):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_observation"][env_idx]

        timestep = {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": next_buffer_observations,
        }
        buffer.add(timestep)
        timestep["next_observation"] = next_observations
        observations = next_observations

        if buffer.can_sample():
            update_counter += base_cfg.updates_per_interaction_step
            while update_counter >= 1:
                batch = buffer.sample()
                agent.update(update_step, batch)
                update_counter -= 1
                update_step += 1

        yield interaction_step


def _check_single_env_dmc(base_cfg) -> None:
    assert_dmc_env_type(base_cfg.env.env_type)
    if int(base_cfg.env.num_train_envs) != 1:
        raise Angle2AConfigError(
            f"Angle 2A requires env.num_train_envs == 1 (got "
            f"{base_cfg.env.num_train_envs}) so that the single underlying "
            f"dm_control environment instance can be captured/restored "
            f"exactly for Monte Carlo rollouts. This is a deliberate, "
            f"documented scope limitation, not an oversight."
        )


def _build_agent_and_env(architecture: RoleArchitecture, base_cfg):
    train_env, eval_env = create_envs(**base_cfg.env)
    observation_space = train_env.observation_space
    action_space = train_env.action_space

    buffer = create_buffer(
        observation_space=observation_space,
        action_space=action_space,
        **OmegaConf.to_container(base_cfg.buffer, resolve=True, throw_on_missing=True),
    )
    buffer.reset()

    agent_cfg_dict = build_role_agent_cfg(base_cfg.agent, architecture)
    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        cfg=OmegaConf.create(agent_cfg_dict),
    )

    single_env = train_env.envs[0]
    return train_env, eval_env, single_env, buffer, agent, observation_space, action_space


def train_agent_to_step(
    role: str,
    architecture: RoleArchitecture,
    architecture_label: str,
    base_cfg,
    stop_step: int,
    seed_context: str,
) -> TrainedAgentHandle:
    """Trains one brand-new agent (own actor, critic, optimizer state,
    replay buffer, environment, RNG) from scratch to exactly `stop_step`
    interaction steps, recording ProbeCapture data along the way.

    `base_cfg` provides everything EXCEPT the critic architecture (env,
    buffer, agent hyperparameters, seed): the same `cfg.seed` is used for
    every role in a matchup by design (matching Angle 1's convention of
    holding the seed fixed to make an architecture comparison meaningful) -
    it does not mean any state is shared; each call here constructs entirely
    separate objects.

    `seed_context` must be a string that's unique to this specific agent
    within the whole experiment (e.g. "matchup_1:D:D5W768") - it's combined
    with `base_cfg.seed` to derive this agent's own deterministic global RNG
    stream for replay-buffer sampling (see seed_global_rng_for_agent), so
    that stream doesn't depend on which agent happens to train first in the
    process.
    """
    _check_single_env_dmc(base_cfg)

    train_env, eval_env, single_env, buffer, agent, observation_space, action_space = _build_agent_and_env(
        architecture, base_cfg
    )

    probe_capacity = min(int(base_cfg.buffer.max_length), stop_step)
    probe_capture = ProbeCapture(
        capacity=probe_capacity,
        observation_shape=observation_space.shape[-1:],
        action_shape=action_space.shape[-1:],
    )

    seed_global_rng_for_agent(int(base_cfg.seed), seed_context)
    for _ in _run_training_loop(agent, buffer, train_env, single_env, probe_capture, base_cfg, stop_step):
        pass

    return TrainedAgentHandle(
        role=role,
        architecture_label=architecture_label,
        architecture=architecture,
        agent=agent,
        buffer=buffer,
        train_env=train_env,
        eval_env=eval_env,
        single_env=single_env,
        stop_step=stop_step,
        probe_capture=probe_capture,
    )
