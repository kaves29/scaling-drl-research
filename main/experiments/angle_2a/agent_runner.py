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
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a.config import RoleArchitecture, build_role_agent_cfg
from experiments.angle_2a.env_state import assert_supported_env_type, capture_env_state, restore_env_state
from experiments.angle_2a.errors import Angle2AConfigError
from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from scale_rl.envs import create_envs
from scale_rl.envs.dmc import validate_dmc_not_heldout
from scale_rl.envs.myosuite import validate_myosuite_core4


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

    def save(self, checkpoint_dir: str) -> None:
        """Mirrors scale_rl.buffers.base_buffer.BaseBuffer.save's plain-pickle
        approach - _env_states is a list of dicts (dm_control or MyoSuite
        physics arrays, dispatched by env_type - see env_state.py), not a
        single array, so this can't reuse that function's numpy-array-
        specific trimming logic directly, but the underlying idea (persist
        everything needed to resume exactly) is the same."""
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "probe_capture_state.pkl", "wb") as f:
            pickle.dump(
                {
                    "capacity": self.capacity,
                    "observations": self._observations,
                    "actions": self._actions,
                    "env_states": self._env_states,
                    "count": self._count,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, checkpoint_dir: str) -> None:
        path = Path(checkpoint_dir) / "probe_capture_state.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No ProbeCapture checkpoint found at {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state["capacity"] != self.capacity:
            raise ValueError(
                f"Checkpointed ProbeCapture capacity ({state['capacity']}) does "
                f"not match this instance's capacity ({self.capacity}) - "
                f"refusing to load a mismatched checkpoint."
            )
        self._observations[:] = state["observations"]
        self._actions[:] = state["actions"]
        self._env_states = state["env_states"]
        self._count = state["count"]


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


def _save_training_checkpoint(
    checkpoint_dir: str,
    agent,
    buffer,
    probe_capture: "ProbeCapture",
    interaction_step: int,
    update_step: int,
    update_counter: int,
    observations: np.ndarray,
    env_state: Dict[str, Any],
) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    agent.save_checkpoint(str(path))
    buffer.save(str(path))
    probe_capture.save(str(path))
    with open(path / "meta.pkl", "wb") as f:
        pickle.dump(
            {
                "interaction_step": interaction_step,
                "update_step": update_step,
                "update_counter": update_counter,
                "observations": observations,
                "env_state": env_state,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_training_checkpoint_meta(checkpoint_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """Returns the persisted meta dict if a training checkpoint exists at
    `checkpoint_dir`, else None. Doesn't touch agent/buffer/probe_capture -
    train_agent_to_step calls their own .load_checkpoint()/.load() once it
    knows (from this) that a resume is happening."""
    if not checkpoint_dir:
        return None
    meta_path = Path(checkpoint_dir) / "meta.pkl"
    if not meta_path.exists():
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def run_training_loop(
    agent,
    buffer,
    train_env,
    single_env,
    probe_capture: "ProbeCapture",
    base_cfg,
    stop_step: int,
    start_step: int = 1,
    resumed_update_step: int = 0,
    resumed_update_counter: int = 0,
    resumed_observations: Optional[np.ndarray] = None,
    resumed_env_state: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
) -> Iterator[int]:
    """Advances training one interaction_step at a time, yielding the
    interaction_step number immediately after it has been fully processed.
    A generator so a caller can observe/act on intermediate steps (see
    prereq_check.py's pre/post checkpointing) rather than only draining it
    fully to stop_step (train_agent_to_step's own usage).

    start_step > 1 (with resumed_observations/resumed_env_state given) means
    a resume: the environment's physics state is restored to exactly where
    training left off (see env_state.py) instead of train_env.reset(), and
    `timestep` is reconstructed from the persisted observation so the first
    post-resume action uses the trained policy, not a fresh warm-up random
    action - agent/buffer/probe_capture are assumed already loaded by the
    caller (train_agent_to_step) before this generator starts.

    checkpoint_dir + checkpoint_interval (both required together) save a
    full, resumable snapshot (agent, buffer, probe_capture, and this
    function's own step/update counters + current observation/env state)
    every `checkpoint_interval` interaction steps.
    """
    if resumed_observations is not None:
        restore_env_state(single_env, resumed_env_state)
        observations = resumed_observations
        timestep = {"next_observation": resumed_observations}
    else:
        observations, _ = train_env.reset()
        timestep = None
    update_step = resumed_update_step
    update_counter = resumed_update_counter

    for interaction_step in range(start_step, stop_step + 1):
        env_state = capture_env_state(single_env, base_cfg.env.env_type)

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

        if checkpoint_dir and checkpoint_interval and interaction_step % checkpoint_interval == 0:
            _save_training_checkpoint(
                checkpoint_dir, agent, buffer, probe_capture,
                interaction_step, update_step, update_counter,
                observations, capture_env_state(single_env, base_cfg.env.env_type),
            )

        yield interaction_step


def check_single_env_type(base_cfg) -> None:
    """Angle 2A supports env_type in {"dmc", "myosuite"} (see env_state.py).
    validate_dmc_not_heldout/validate_myosuite_core4 are no-ops for the other
    env_type, so calling both unconditionally is correct regardless of which
    type this config uses."""
    env_type = base_cfg.env.env_type
    assert_supported_env_type(env_type)
    validate_dmc_not_heldout(env_type, base_cfg.env.env_name)
    validate_myosuite_core4(env_type, base_cfg.env.env_name)
    if int(base_cfg.env.num_train_envs) != 1:
        raise Angle2AConfigError(
            f"Angle 2A requires env.num_train_envs == 1 (got "
            f"{base_cfg.env.num_train_envs}) so that the single underlying "
            f"environment instance can be captured/restored exactly for "
            f"Monte Carlo rollouts. This is a deliberate, documented scope "
            f"limitation, not an oversight."
        )


def build_agent_and_env(architecture: RoleArchitecture, base_cfg):
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
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
) -> TrainedAgentHandle:
    """Trains one brand-new agent (own actor, critic, optimizer state,
    replay buffer, environment, RNG) from scratch to exactly `stop_step`
    interaction steps, recording ProbeCapture data along the way - or
    resumes one from `checkpoint_dir` if a checkpoint already exists there
    (see load_training_checkpoint_meta/run_training_loop).

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
    process. Re-derived identically on resume (matching experiments/angle_1.py's
    own precedent of always reseeding unconditionally near the top of run(),
    resume or not) - some loss of bit-exact reproducibility of the exact
    post-resume buffer-sampling sequence is an accepted, pre-existing
    limitation shared with Angle 1, not something new here.

    checkpoint_dir + checkpoint_interval (both required together to actually
    checkpoint) are passed straight through to run_training_loop.
    """
    check_single_env_type(base_cfg)

    train_env, eval_env, single_env, buffer, agent, observation_space, action_space = build_agent_and_env(
        architecture, base_cfg
    )

    probe_capacity = min(int(base_cfg.buffer.max_length), stop_step)
    probe_capture = ProbeCapture(
        capacity=probe_capacity,
        observation_shape=observation_space.shape[-1:],
        action_shape=action_space.shape[-1:],
    )

    seed_global_rng_for_agent(int(base_cfg.seed), seed_context)

    meta = load_training_checkpoint_meta(checkpoint_dir)
    if meta is not None:
        agent.load_checkpoint(checkpoint_dir)
        buffer.load(checkpoint_dir)
        probe_capture.load(checkpoint_dir)
        start_step = meta["interaction_step"] + 1
        print(f"[angle_2a] resuming {role} ({architecture_label}, {seed_context}) from interaction_step {start_step}")
        loop = run_training_loop(
            agent, buffer, train_env, single_env, probe_capture, base_cfg, stop_step,
            start_step=start_step,
            resumed_update_step=meta["update_step"],
            resumed_update_counter=meta["update_counter"],
            resumed_observations=meta["observations"],
            resumed_env_state=meta["env_state"],
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
        )
    else:
        loop = run_training_loop(
            agent, buffer, train_env, single_env, probe_capture, base_cfg, stop_step,
            checkpoint_dir=checkpoint_dir, checkpoint_interval=checkpoint_interval,
        )
    for _ in loop:
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
