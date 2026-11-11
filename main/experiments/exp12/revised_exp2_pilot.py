"""Revised online Exp2 pilot: four isolated policy branches per checkpoint.

Each checkpoint runs the established plasticity probe and then, sequentially,
four actor branches that cross the small/large shadow critic with a frozen or
learning critic. Every branch starts from an independent copy of the reference
actor/temperature state and runs 10,000 actual raw environment steps, counted
after action repeat and early termination.
"""

import copy
import hashlib
import json
import math
import pickle
import random

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import SyncVectorEnv
from omegaconf import OmegaConf
from scale_rl.envs.wrappers.repeat_action import RepeatAction

from experiments.exp12.envs import _make_one_env, env_restore_state, restore_env
from experiments.exp12.online_exp2_diagnostic import (
    ActorUpdateSchedule, ObservationBuffer, copy_reference_replay,
    diagnostic_actor_step, learning_branch_step,
)
from experiments.exp12.online_exp2_pilot import (
    _executed_repeats, _hash, _reference_state, _seed, _success,
    frozen_normalizer,
)
from utils.atomic_io import atomic_write_bytes, atomic_write_text

CONDITIONS = ("small_frozen", "large_frozen", "small_learning", "large_learning")
ARCHITECTURES = {"small": (2, 512), "large": (4, 1536)}
UPDATE_FIELDS = (
    "train/actor_loss", "train/entropy", "train/actor_gnorm", "train/actor_action",
    "train/actor_pnorm", "train/critic_loss", "train/critic_gnorm", "train/q1_mean",
    "train/q2_mean", "train/td_error_var", "train/policy_churn", "train/temperature",
)
REPLAY_FIELDS = ("observations", "actions", "rewards", "terminateds", "truncateds",
                 "next_observations")


def _condition(name):
    architecture, mode = name.split("_", 1)
    if architecture not in ARCHITECTURES or mode not in ("frozen", "learning"):
        raise ValueError(f"unknown revised pilot condition: {name}")
    return architecture, mode == "learning"


def _float_info(info):
    result = {}
    for name in UPDATE_FIELDS:
        if name in info:
            value = float(info[name])
            if not math.isfinite(value):
                raise ValueError(f"nonfinite branch metric {name}")
            result[name] = value
    return result


def _branch_env(cfg, seed):
    e = OmegaConf.to_container(cfg.env, resolve=True)
    return SyncVectorEnv([lambda: _make_one_env(
        e["env_type"], e["env_name"], seed, e["rescale_action"],
        e["no_termination"], e["action_repeat"], e["reward_scale"],
        e["max_episode_steps"])])


def _repeat_wrapper(env):
    wrapped = env.envs[0]
    while wrapped is not None:
        if isinstance(wrapped, RepeatAction):
            return wrapped
        wrapped = getattr(wrapped, "env", None)
    raise ValueError("branch environment has no RepeatAction wrapper")


def _scoped_repeat(env, repeats):
    """Temporarily cap action repeat so the final raw-step does not overshoot."""
    wrapper = _repeat_wrapper(env)
    previous = wrapper._action_repeat
    wrapper._action_repeat = int(repeats)
    return lambda: setattr(wrapper, "_action_repeat", previous)


def _episode_seed(cfg, step, condition_index, episode_index):
    return _seed("revised-exp2-branch", cfg.seed, step, condition_index, episode_index)


def _sample_replay(replay, rng, batch_size):
    n = int(replay._num_in_buffer)
    if n < 1:
        raise ValueError("branch replay is empty")
    indices = rng.integers(0, n, size=int(batch_size))
    return {name: np.array(getattr(replay, f"_{name}")[indices])
            for name in REPLAY_FIELDS}, indices


def _sac_batch(batch):
    return {"observation": batch["observations"], "action": batch["actions"],
            "reward": batch["rewards"], "terminated": batch["terminateds"],
            "truncated": batch["truncateds"],
            "next_observation": batch["next_observations"]}


class RevisedBranch:
    """Actor and (optionally) critic state for one isolated policy branch."""

    def __init__(self, name, actor, critic, target, temperature, replay, normalizer,
                 seed, batch_size, updates_per_interaction, carry, critic_use_cdq,
                 gamma, n_step, target_tau, target_entropy):
        self.name = name
        self.learning = _condition(name)[1]
        self.actor = actor
        self.critic = critic
        self.target = target
        self.temperature = temperature
        self.replay = replay
        self.buffer = None if self.learning else ObservationBuffer(seed)
        self.appended = []
        self.normalize = normalizer
        self.schedule = ActorUpdateSchedule(updates_per_interaction, carry)
        self.key = jax.random.PRNGKey(seed)
        self.collection_key = jax.random.fold_in(self.key, 1)
        self.action_rng = np.random.default_rng(seed)
        self.batch_size = int(batch_size)
        self.critic_use_cdq = bool(critic_use_cdq)
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.target_tau = float(target_tau)
        self.target_entropy = float(target_entropy)
        self.interactions = 0
        self.raw_steps = 0
        self.optimizer_updates = 0

    def act(self, raw_observation):
        observation = self.normalize(np.asarray(raw_observation))
        self.collection_key, key = jax.random.split(self.collection_key)
        action = np.asarray(self.actor(
            observations=jnp.asarray(observation), temperature=1.).sample(seed=key))
        if not np.all(np.isfinite(action)):
            raise ValueError("nonfinite branch action")
        return action

    def observe_and_update(self, timestep, observation):
        """Record branch experience, then execute its scheduled learner updates."""
        if self.learning:
            self.replay.add(timestep)
            self.appended.append({key: np.array(value) for key, value in timestep.items()})
        else:
            self.buffer.append(observation)
        self.interactions += 1
        metrics, indices = [], []
        for _ in range(self.schedule.advance()):
            if self.learning:
                batch, sampled = _sample_replay(self.replay, self.action_rng, self.batch_size)
                for key in ("observations", "next_observations"):
                    batch[key] = self.normalize(batch[key])
                self.key, actor_key, critic_key = jax.random.split(self.key, 3)
                self.actor, self.critic, self.target, self.temperature, info = learning_branch_step(
                    actor_key, critic_key, self.actor, self.critic, self.target,
                    self.temperature, _sac_batch(batch), self.gamma, self.n_step,
                    self.critic_use_cdq, self.target_tau, self.target_entropy)
            else:
                batch, sampled = self.buffer.sample(self.batch_size)
                batch = self.normalize(batch)
                self.key, actor_key, q_key = jax.random.split(self.key, 3)
                self.actor, info = diagnostic_actor_step(
                    self.actor, self.critic, self.temperature, batch, actor_key,
                    self.critic_use_cdq)
                actions = self.actor(observations=jnp.asarray(batch),
                                     temperature=1.).sample(seed=q_key)
                q = self.critic(observations=jnp.asarray(batch), actions=actions)
                info["train/q1_mean"] = jnp.mean(jnp.minimum(*q) if self.critic_use_cdq else q)
            self.optimizer_updates += 1
            indices.append(sampled.tolist())
            metrics.append(_float_info(info))
        return metrics, indices

    def state_dict(self):
        state = {
            "name": self.name, "learning": self.learning,
            "actor": flax.serialization.to_bytes(self.actor),
            "critic": flax.serialization.to_bytes(self.critic),
            "target": None if self.target is None else flax.serialization.to_bytes(self.target),
            "temperature": flax.serialization.to_bytes(self.temperature),
            "schedule": self.schedule.state_dict(),
            "key": np.asarray(self.key).copy(),
            "collection_key": np.asarray(self.collection_key).copy(),
            "action_rng": copy.deepcopy(self.action_rng.bit_generator.state),
            "batch_size": self.batch_size,
            "critic_use_cdq": self.critic_use_cdq,
            "gamma": self.gamma, "n_step": self.n_step,
            "target_tau": self.target_tau, "target_entropy": self.target_entropy,
            "interactions": self.interactions, "raw_steps": self.raw_steps,
            "optimizer_updates": self.optimizer_updates,
        }
        if self.learning:
            state["appended"] = [{key: np.array(value) for key, value in item.items()}
                                 for item in self.appended]
        else:
            state["buffer"] = self.buffer.state_dict()
        return state

    @classmethod
    def from_state_dict(cls, state, actor_template, critic_template, target_template,
                        temperature_template, reference_replay, normalizer):
        obj = cls.__new__(cls)
        obj.name = state["name"]
        obj.learning = bool(state["learning"])
        obj.actor = flax.serialization.from_bytes(actor_template, state["actor"])
        # A learning branch's critic/target/temperature move away from the
        # reference; restoring them from the reference would break bit-exact resume.
        obj.critic = flax.serialization.from_bytes(critic_template, state["critic"])
        obj.target = None if state["target"] is None else flax.serialization.from_bytes(
            target_template, state["target"])
        obj.temperature = flax.serialization.from_bytes(
            temperature_template, state["temperature"])
        obj.buffer = (None if obj.learning
                      else ObservationBuffer.from_state_dict(state["buffer"]))
        obj.appended = []
        if obj.learning:
            obj.replay = copy_reference_replay(reference_replay)
            for item in state["appended"]:
                obj.replay.add(item)
                obj.appended.append({key: np.array(value) for key, value in item.items()})
        else:
            obj.replay = None
        obj.normalize = normalizer
        obj.schedule = ActorUpdateSchedule.from_state_dict(state["schedule"])
        obj.key = jnp.asarray(state["key"])
        obj.collection_key = jnp.asarray(state["collection_key"])
        obj.action_rng = np.random.default_rng(0)
        obj.action_rng.bit_generator.state = copy.deepcopy(state["action_rng"])
        obj.batch_size = int(state["batch_size"])
        obj.critic_use_cdq = bool(state["critic_use_cdq"])
        obj.gamma = float(state["gamma"])
        obj.n_step = int(state["n_step"])
        obj.target_tau = float(state["target_tau"])
        obj.target_entropy = float(state["target_entropy"])
        obj.interactions = int(state["interactions"])
        obj.raw_steps = int(state["raw_steps"])
        obj.optimizer_updates = int(state["optimizer_updates"])
        return obj


def _new_branch(runner, name, step, condition_index, normalizer, evaluation_seeds=None):
    architecture, learning = _condition(name)
    reference = runner.trainer
    agent = reference._sac_agent
    shadow = runner.shadows[architecture]
    learner = runner.cfg.online_exp2
    replay = copy_reference_replay(reference.buffer) if learning else None
    branch = RevisedBranch(
        name, copy.deepcopy(agent.actor), copy.deepcopy(shadow.critic),
        copy.deepcopy(shadow.target) if learning else shadow.target,
        copy.deepcopy(agent.temperature), replay, normalizer,
        _episode_seed(runner.cfg, step, condition_index, 0),
        int(reference.cfg.buffer.sample_batch_size),
        float(runner.cfg.updates_per_interaction_step), float(reference.update_counter),
        bool(reference.cfg.agent.critic_use_cdq), float(reference.cfg.agent.gamma),
        int(reference.cfg.agent.n_step), float(reference.cfg.agent.target_tau),
        float(agent._cfg.temp_target_entropy))
    return branch


def _wandb_logger(enabled, cfg):
    if not enabled:
        return None
    try:
        import wandb
        return wandb if wandb.run is not None else None
    except Exception:
        return None


def _run_branch(runner, name, step, condition_index, progress, directory, normalize,
                wandb_module):
    cfg = runner.cfg
    raw_budget = int(cfg.online_exp2.diagnostic_raw_steps)
    save_every = int(cfg.online_exp2.diagnostic_save_every)
    global_base = int(cfg.action_repeat) * int(cfg.num_train_envs) * step
    state = progress["branches"].get(name)
    original_np, original_py = np.random.get_state(), random.getstate()
    env = None
    try:
        if state is None:
            branch = _new_branch(runner, name, step, condition_index, normalize)
            seed = _episode_seed(cfg, step, condition_index, 0)
            np.random.seed(seed)
            random.seed(seed)
            env = _branch_env(cfg, seed)
            observation, _ = env.reset()
            state = {"branch": branch.state_dict(),
                     "observation": np.array(observation, copy=True),
                     "environment_seed": seed, "environment_seeds": [seed],
                     "episode_index": 0, "episode_return": 0.,
                     "episode_length": 0, "episode_raw_steps": 0,
                     "episodes": [], "windows": [], "policy_change": {},
                     "env": env_restore_state(env),
                     "numpy_rng": np.random.get_state(),
                     "python_rng": random.getstate()}
            progress["branches"][name] = state
            progress["in_progress"] = name
            _save(progress, directory, runner)
        else:
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            branch = RevisedBranch.from_state_dict(
                state["branch"], runner.trainer._sac_agent.actor,
                runner.shadows[_condition(name)[0]].critic,
                runner.shadows[_condition(name)[0]].target,
                runner.trainer._sac_agent.temperature,
                runner.trainer.buffer, normalize)
            env = _branch_env(cfg, state["environment_seed"])
            restore_env(env, state["env"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])

        while branch.raw_steps < raw_budget:
            runner._check_resource_cap()
            remaining = raw_budget - branch.raw_steps
            stop_repeat = None
            if remaining < int(cfg.env.action_repeat):
                stop_repeat = _scoped_repeat(env, remaining)
            try:
                observation = state["observation"]
                pre_action = np.array(observation, copy=True)
                action = branch.act(pre_action)
                next_observation, reward, term, trunc, info = env.step(action)
            finally:
                if stop_repeat is not None:
                    stop_repeat()
            executed = min(int(_executed_repeats(env)), remaining)
            if executed < 1:
                raise ValueError("branch environment executed no raw steps")
            ended = bool(term[0]) or bool(trunc[0])
            final = info.get("final_observation")
            final_observation = None if final is None else final[0]
            if final_observation is None:
                final_observation = next_observation
            timestep = {"observation": np.array([pre_action[0]]),
                        "action": np.array([action[0]]),
                        "reward": np.array([reward[0]], np.float32),
                        "terminated": np.array([term[0]], np.float32),
                        "truncated": np.array([trunc[0]], np.float32),
                        "next_observation": np.array([final_observation])}
            branch.raw_steps += executed
            state["episode_raw_steps"] += executed
            state["episode_return"] += float(reward[0])
            state["episode_length"] += 1
            updates, indices = branch.observe_and_update(timestep, pre_action[0])
            if updates:
                state["windows"].append({"raw_steps": branch.raw_steps,
                                         "interactions": branch.interactions,
                                         "updates": updates,
                                         "sample_indices": indices})

            if branch.raw_steps >= 5000:
                _mark(wandb_module, name, branch, 5000, global_base)
            if branch.raw_steps >= 10000:
                _mark(wandb_module, name, branch, 10000, global_base)

            if ended:
                episode = {"return": state["episode_return"],
                           "length": state["episode_length"],
                           "raw_steps": state["episode_raw_steps"],
                           "terminated": bool(term[0]), "truncated": bool(trunc[0]),
                           "success": _success(info)}
                state["episodes"].append(episode)
                _log_branch_return(wandb_module, name, branch, global_base, episode)
                state["episode_return"] = 0.
                state["episode_length"] = 0
                state["episode_raw_steps"] = 0
                state["episode_index"] += 1
                env.close()
                seed = _episode_seed(cfg, step, condition_index, state["episode_index"])
                state["environment_seed"] = seed
                state["environment_seeds"].append(seed)
                np.random.seed(seed)
                random.seed(seed)
                env = _branch_env(cfg, seed)
                next_observation, _ = env.reset()
            state.update({"observation": np.array(next_observation, copy=True),
                          "env": env_restore_state(env),
                          "numpy_rng": np.random.get_state(),
                          "python_rng": random.getstate()})
            if (branch.raw_steps % save_every == 0
                    or branch.raw_steps == raw_budget):
                state["branch"] = branch.state_dict()
                _save(progress, directory, runner)
        if branch.raw_steps != raw_budget:
            raise ValueError(f"branch {name} ended at {branch.raw_steps} raw steps")
        _log_branch_summary(wandb_module, name, branch, global_base, state)
        return branch, state
    finally:
        if env is not None:
            env.close()
        np.random.set_state(original_np)
        random.setstate(original_py)


def _mark(wandb_module, name, branch, mark, global_base):
    if wandb_module is None or getattr(branch, f"_marked_{mark}", False):
        return
    setattr(branch, f"_marked_{mark}", True)
    wandb_module.log({f"pilot/{name}/global_raw_step": global_base + branch.raw_steps,
                      f"pilot/{name}/branch_raw_step": branch.raw_steps,
                      f"pilot/{name}/marker_{mark}": 1.})


def _log_branch_return(wandb_module, name, branch, global_base, episode):
    if wandb_module is None:
        return
    wandb_module.log({f"pilot/{name}/global_raw_step": global_base + branch.raw_steps,
                      f"pilot/{name}/branch_raw_step": branch.raw_steps,
                      f"pilot/{name}/episode_return": episode["return"],
                      f"pilot/{name}/episode_raw_steps": episode["raw_steps"],
                      f"pilot/{name}/episode_success": episode["success"]})


def _log_branch_summary(wandb_module, name, branch, global_base, state):
    if wandb_module is None:
        return
    returns = [episode["return"] for episode in state["episodes"]]
    payload = {f"pilot/{name}/global_raw_step": global_base + branch.raw_steps,
               f"pilot/{name}/branch_raw_step": branch.raw_steps,
               f"pilot/{name}/episodes": len(returns),
               f"pilot/{name}/mean_episode_return": float(np.mean(returns)) if returns else 0.,
               f"pilot/{name}/optimizer_updates": branch.optimizer_updates,
               f"pilot/{name}/interactions": branch.interactions}
    wandb_module.log(payload)


def _save(progress, directory, runner):
    progress["reference_state"] = _reference_state(runner.trainer)
    serializable = dict(progress)
    serializable["branches"] = {}
    for name, state in progress["branches"].items():
        serializable["branches"][name] = dict(state)
    atomic_write_bytes(directory / "progress.pkl",
                       pickle.dumps(serializable, protocol=pickle.HIGHEST_PROTOCOL))


def _load(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def _branch_summary(name, branch, state):
    return {"complete": True, "name": name,
            "interactions": branch.interactions, "raw_steps": branch.raw_steps,
            "optimizer_updates": branch.optimizer_updates,
            "episodes": list(state["episodes"]),
            "environment_seeds": list(state["environment_seeds"]),
            "window_count": len(state["windows"]),
            "actor_hash": _hash(branch.actor.params),
            "critic_hash": _hash(branch.critic.params),
            "temperature_hash": _hash(branch.temperature)}


def revised_pilot_checkpoint(runner, trainer, wandb_enabled):
    """Run the probe and four isolated branches at one reference checkpoint."""
    cfg = runner.cfg
    step = int(trainer.interaction_step)
    directory = runner.root / "diagnostics" / f"step_{step:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "progress.pkl"
    identity = {"protocol": "revised_v1", "config_sha256": runner.config_hash,
                "source_sha256": runner.source_hash, "source_commit": runner.source_commit,
                "checkpoint": step,
                "reference_regime": cfg.online_exp2.reference_regime,
                "training_seed": int(cfg.seed),
                "raw_budget": int(cfg.online_exp2.diagnostic_raw_steps)}
    trainer.extra_state["online_exp2_pending_checkpoint"] = step
    runner.save()
    runner._check_resource_cap()
    normalize, normalizer_hash = frozen_normalizer(trainer.agent)
    if path.exists():
        progress = _load(path)
        if progress["identity"] != identity:
            raise ValueError("revised pilot resume identity mismatch")
        if progress["normalizer_hash"] != normalizer_hash:
            raise ValueError("revised pilot normalizer changed during resume")
    else:
        progress = {"identity": identity, "branches": {}, "in_progress": None,
                    "normalizer_hash": normalizer_hash,
                    "actor_hash": _hash(trainer._sac_agent.actor.params),
                    "optimizer_hash": _hash(trainer._sac_agent.actor.opt_state),
                    "temperature_hash": _hash(trainer._sac_agent.temperature),
                    "shadow_hashes": {name: _hash(shadow.critic.params)
                                      for name, shadow in runner.shadows.items()},
                    "shadow_target_hashes": {name: _hash(shadow.target.params)
                                             for name, shadow in runner.shadows.items()},
                    "reference_state": _reference_state(trainer)}
        _save(progress, directory, runner)
    reference_temperature = copy.deepcopy(trainer._sac_agent.temperature)
    wandb_module = _wandb_logger(wandb_enabled, cfg)
    if wandb_module is not None:
        for name in CONDITIONS:
            wandb_module.define_metric(f"pilot/{name}/*",
                                       step_metric=f"pilot/{name}/global_raw_step")
    original_np, original_py = np.random.get_state(), random.getstate()
    try:
        if step in cfg.online_exp2.probe_checkpoints:
            runner._check_resource_cap()
            runner._probe(step)
        for index, name in enumerate(CONDITIONS):
            if progress["branches"].get(name, {}).get("complete"):
                continue
            runner._check_resource_cap()
            global_base = step * 100000 + index * 100000
            branch, state = _run_branch(
                runner, name, step, index, progress, directory, normalize, wandb_module)
            progress["branches"][name] = _branch_summary(name, branch, state)
            progress["in_progress"] = None
            _save(progress, directory, runner)
    except Exception as exc:
        atomic_write_text(directory / "failure.json", json.dumps({
            "type": type(exc).__name__, "message": str(exc), "checkpoint": step,
            "condition": progress.get("in_progress"),
            "reference_regime": cfg.online_exp2.reference_regime}, sort_keys=True))
        raise
    finally:
        np.random.set_state(original_np)
        random.setstate(original_py)
    if (_hash(trainer._sac_agent.actor.params) != progress["actor_hash"]
            or _hash(trainer._sac_agent.actor.opt_state) != progress["optimizer_hash"]
            or _hash(trainer._sac_agent.temperature) != progress["temperature_hash"]
            or {name: _hash(shadow.critic.params) for name, shadow in runner.shadows.items()}
            != progress["shadow_hashes"]
            or {name: _hash(shadow.target.params) for name, shadow in runner.shadows.items()}
            != progress["shadow_target_hashes"]
            or frozen_normalizer(trainer.agent)[1] != progress["normalizer_hash"]
            or _reference_state(trainer) != progress["reference_state"]):
        raise ValueError("revised pilot modified reference or shadow state")
    for name in CONDITIONS:
        state = progress["branches"][name]
        if state["raw_steps"] != int(cfg.online_exp2.diagnostic_raw_steps):
            raise ValueError(f"incomplete revised branch: {name}")
    runner.rows.append({"type": "revised_pilot_checkpoint", **identity,
                        "normalizer_hash": progress["normalizer_hash"],
                        "actor_hash": progress["actor_hash"],
                        "optimizer_hash": progress["optimizer_hash"],
                        "temperature_hash": progress["temperature_hash"],
                        "shadow_hashes": progress["shadow_hashes"],
                        "shadow_target_hashes": progress["shadow_target_hashes"],
                        "shadow_architectures": {name: f"D{b}W{w}" for name,
                                                 (b, w) in ARCHITECTURES.items()},
                        "branches": {name: {key: value for key, value in
                                            progress["branches"][name].items()
                                            if key not in ("episodes", "environment_seeds")}
                                     for name in CONDITIONS},
                        "branch_episodes": {name: progress["branches"][name]["episodes"]
                                            for name in CONDITIONS},
                        "branch_environment_seeds": {
                            name: progress["branches"][name]["environment_seeds"]
                            for name in CONDITIONS},
                        "progress_artifact": str(path)})
    trainer.extra_state.pop("online_exp2_pending_checkpoint", None)
    runner.save()


def revised_analysis(runner):
    """Per-run, per-checkpoint branch summaries; no episode pooling across checkpoints."""
    cfg = runner.cfg
    checkpoints = [int(step) for step in cfg.online_exp2.checkpoint_steps]
    conditions = {}
    for name in CONDITIONS:
        returns = {step: [] for step in checkpoints}
        for row in runner.rows:
            if row.get("type") != "revised_pilot_checkpoint":
                continue
            for episode in row["branch_episodes"][name]:
                returns[int(row["checkpoint"])].append(episode["return"])
        missing = [step for step, values in returns.items() if not values]
        if missing:
            raise ValueError(f"missing branch episodes for {name}: {missing}")
        checkpoint_means = {step: float(np.mean(values)) for step, values in returns.items()}
        conditions[name] = {"checkpoint_means": checkpoint_means,
                            "mean_checkpoint_means": float(np.mean(
                                list(checkpoint_means.values())))}
    return {"protocol": "revised_v1",
            "reference_regime": cfg.online_exp2.reference_regime,
            "environment": cfg.env_name, "training_seed": int(cfg.seed),
            "raw_steps_per_branch": int(cfg.online_exp2.diagnostic_raw_steps),
            "checkpoints": checkpoints, "conditions": conditions,
            "independent_units": 1}


def reference_raw_steps(cfg, interaction_step):
    return int(interaction_step) * int(cfg.action_repeat) * int(cfg.num_train_envs)
