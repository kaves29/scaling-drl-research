"""Checkpoint actor adaptation for the opt-in online Experiment 2 pilot."""

import hashlib
import json
import pickle
import random

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import SyncVectorEnv
from omegaconf import OmegaConf
from scale_rl.envs.wrappers.repeat_action import RepeatAction

from experiments.exp12.envs import _make_one_env, create_eval_env, env_restore_state, restore_env
from experiments.exp12.online_exp2_diagnostic import DiagnosticActorBranch
from utils.atomic_io import atomic_write_bytes, atomic_write_text


def _seed(*parts):
    data = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "little")


def _hash(value):
    return hashlib.sha256(flax.serialization.to_bytes(value)).hexdigest()


def frozen_normalizer(agent):
    if not hasattr(agent, "obs_rms"):
        return None, "none"
    mean = np.array(agent.obs_rms.mean, copy=True)
    var = np.array(agent.obs_rms.var, copy=True)
    epsilon = float(agent.epsilon)
    count = np.asarray(agent.obs_rms.count)
    identifier = hashlib.sha256(
        mean.tobytes() + var.tobytes() + count.tobytes() + str(epsilon).encode()).hexdigest()

    def normalize(value):
        result = (np.asarray(value) - mean) / np.sqrt(var + epsilon)
        if not np.all(np.isfinite(result)):
            raise ValueError("nonfinite normalized observation")
        return result

    return normalize, identifier


def _branch_env(cfg, seed):
    e = OmegaConf.to_container(cfg.env, resolve=True)
    return SyncVectorEnv([lambda: _make_one_env(
        e["env_type"], e["env_name"], seed, e["rescale_action"],
        e["no_termination"], e["action_repeat"], e["reward_scale"],
        e["max_episode_steps"])])


def _executed_repeats(env):
    wrapped = env.envs[0]
    while wrapped is not None:
        if isinstance(wrapped, RepeatAction):
            return wrapped.last_executed_repeats
        wrapped = getattr(wrapped, "env", None)
    return 1


def _success(info):
    if "final_info" in info and info["final_info"][0] is not None:
        return float(info["final_info"][0].get("success", 0.))
    if "success" in info:
        return float(info["success"][0])
    return 0.


def _evaluate(cfg, actor, normalize, seed):
    e = OmegaConf.to_container(cfg.env, resolve=True)
    e["seed"] = seed
    saved_np, saved_py = np.random.get_state(), random.getstate()
    env = None
    try:
        np.random.seed(seed)
        random.seed(seed)
        env = create_eval_env(**e)
        observation, _ = env.reset()
        key = jax.random.PRNGKey(seed)
        total, length, raw_steps, success = 0., 0, 0, 0.
        terminated = truncated = False
        while not (terminated or truncated):
            model_observation = normalize(observation) if normalize else observation
            key, action_key = jax.random.split(key)
            mode = 0. if cfg.online_exp2.evaluation_mode == "deterministic" else 1.
            action = np.asarray(actor(
                observations=jnp.asarray(model_observation), temperature=mode).sample(seed=action_key))
            if not np.all(np.isfinite(action)):
                raise ValueError("nonfinite evaluation action")
            observation, reward, term, trunc, info = env.step(action)
            total += float(reward[0])
            length += 1
            raw_steps += _executed_repeats(env)
            terminated, truncated = bool(term[0]), bool(trunc[0])
            success = max(success, _success(info))
        if not np.isfinite(total):
            raise ValueError("nonfinite evaluation return")
        return {"return": total, "length": length, "raw_env_steps": raw_steps,
                "success": success,
                "terminated": terminated, "truncated": truncated, "seed": seed}
    finally:
        if env is not None:
            env.close()
        np.random.set_state(saved_np)
        random.setstate(saved_py)


def _policy_change(reference_actor, branch_actor, observation, normalize, seed):
    model_observation = normalize(observation) if normalize else observation
    key = jax.random.PRNGKey(seed)
    actions = [np.asarray(actor(
        observations=jnp.asarray(model_observation), temperature=0.).sample(seed=key))
        for actor in (reference_actor, branch_actor)]
    if not all(np.all(np.isfinite(action)) for action in actions):
        raise ValueError("nonfinite policy change action")
    return float(np.linalg.norm(actions[1] - actions[0]))


def _save_progress(path, progress):
    serializable = dict(progress)
    serializable["branches"] = {}
    for name, original in progress["branches"].items():
        state = dict(original)
        branch = dict(state["branch"])
        branch["actor"] = flax.serialization.to_bytes(branch["actor"])
        state["branch"] = branch
        serializable["branches"][name] = state
    atomic_write_bytes(path, pickle.dumps(serializable, protocol=pickle.HIGHEST_PROTOCOL))


def _load_progress(path, actor_template):
    progress = pickle.loads(path.read_bytes())
    for state in progress["branches"].values():
        branch = state["branch"]
        actor = flax.serialization.from_bytes(actor_template, branch["actor"])
        actor = actor.replace(update_step=jnp.asarray(
            actor.update_step, dtype=jnp.asarray(actor_template.update_step).dtype))
        branch["actor"] = actor
    return progress


def _episode_seeds(cfg, step):
    return [_seed("exp2-eval", cfg.seed, step, i)
            for i in range(int(cfg.online_exp2.evaluation_episodes))]


def _reference_state(trainer):
    agent = trainer._sac_agent
    return {"actor": _hash(agent.actor), "critic": _hash(agent.critic),
            "target": _hash(agent._target_critic), "temperature": _hash(agent.temperature),
            "rng": _hash(agent._rng), "replay_length": int(trainer.buffer._num_in_buffer),
            "train_env": hashlib.sha256(pickle.dumps(env_restore_state(trainer.train_env))).hexdigest(),
            "eval_env": hashlib.sha256(pickle.dumps(env_restore_state(trainer.eval_env))).hexdigest()}


def _run_branch(runner, name, step, progress, path, normalize, temperature, episode_seeds):
    cfg = runner.cfg
    seed = _seed("exp2-branch", cfg.seed, step, 0)
    saved_np, saved_py = np.random.get_state(), random.getstate()
    branch_state = progress["branches"].get(name)
    if branch_state is None:
        np.random.set_state(progress["main_numpy_rng"])
        random.setstate(progress["main_python_rng"])
    else:
        seed = branch_state["environment_seed"]
        np.random.set_state(branch_state["numpy_rng"])
        random.setstate(branch_state["python_rng"])
    env = None
    try:
        if branch_state is None:
            np.random.seed(seed)
            random.seed(seed)
        env = _branch_env(cfg, seed)
        if branch_state is None:
            observation, _ = env.reset()
            branch = DiagnosticActorBranch(
                runner.trainer._sac_agent.actor, runner.shadows[name].critic,
                temperature, seed, int(cfg.buffer.sample_batch_size),
                float(cfg.updates_per_interaction_step), bool(cfg.agent.critic_use_cdq),
                float(runner.trainer.update_counter))
            branch_state = {"branch": branch.state_dict(), "observation": observation,
                            "environment_seed": seed, "environment_seeds": [seed],
                            "episode_index": 0,
                            "initial_actor_hash": _hash(branch.actor.params),
                            "initial_optimizer_hash": _hash(branch.actor.opt_state),
                            "env": env_restore_state(env), "numpy_rng": np.random.get_state(),
                            "python_rng": random.getstate(), "episode_return": 0.,
                            "episode_length": 0, "episode_raw_steps": 0,
                            "raw_env_steps": 0, "episodes": [], "measurements": {},
                            "initial_observation": np.array(observation, copy=True),
                            "update_records": [], "policy_change": {}}
            progress["branches"][name] = branch_state
            _save_progress(path, progress)
        else:
            restore_env(env, branch_state["env"])
            np.random.set_state(branch_state["numpy_rng"])
            random.setstate(branch_state["python_rng"])
            observation = branch_state["observation"]
            branch = DiagnosticActorBranch.from_state_dict(
                branch_state["branch"], runner.shadows[name].critic, temperature)

        for _ in range(branch.interactions, int(cfg.online_exp2.diagnostic_horizon)):
            runner._check_resource_cap()
            pre_action = np.array(observation, copy=True)
            action = branch.act(pre_action, normalize)
            observation, reward, term, trunc, info = env.step(action)
            executed_repeats = _executed_repeats(env)
            branch_state["raw_env_steps"] += executed_repeats
            branch_state["episode_raw_steps"] += executed_repeats
            branch_state["episode_return"] += float(reward[0])
            branch_state["episode_length"] += 1
            ended = bool(term[0]) or bool(trunc[0])
            if ended:
                branch_state["episodes"].append({
                    "return": branch_state["episode_return"],
                    "length": branch_state["episode_length"],
                    "raw_env_steps": branch_state["episode_raw_steps"],
                    "terminated": bool(term[0]), "truncated": bool(trunc[0]),
                    "success": _success(info)})
                branch_state["episode_return"] = 0.
                branch_state["episode_length"] = 0
                branch_state["episode_raw_steps"] = 0
            update = branch.observe_and_update(pre_action[0], normalize)
            if ended:
                env.close()
                branch_state["episode_index"] += 1
                next_seed = _seed("exp2-branch", cfg.seed, step, branch_state["episode_index"])
                branch_state["environment_seed"] = next_seed
                branch_state["environment_seeds"].append(next_seed)
                np.random.seed(next_seed)
                random.seed(next_seed)
                env = _branch_env(cfg, next_seed)
                observation, _ = env.reset()
            if update["updates"]:
                branch_state["update_records"].append({"interaction": branch.interactions,
                                                        "samples": update["sample_indices"],
                                                        "updates": update["updates"]})
            horizon = branch.interactions
            if horizon in cfg.online_exp2.measurement_interactions:
                branch_state["measurements"][horizon] = [
                    _evaluate(cfg, branch.actor, normalize, evaluation_seed)
                    for evaluation_seed in episode_seeds]
                branch_state["policy_change"][horizon] = _policy_change(
                    runner.trainer._sac_agent.actor, branch.actor,
                    branch_state["initial_observation"], normalize,
                    _seed("exp2-action-change", cfg.seed, step))
            if (horizon % int(cfg.online_exp2.diagnostic_save_every) == 0
                    or horizon in cfg.online_exp2.measurement_interactions):
                branch_state.update({"branch": branch.state_dict(), "observation": observation,
                                     "env": env_restore_state(env),
                                     "numpy_rng": np.random.get_state(),
                                     "python_rng": random.getstate()})
                _save_progress(path, progress)
        return branch_state
    finally:
        if env is not None:
            env.close()
        np.random.set_state(saved_np)
        random.setstate(saved_py)


def pilot_checkpoint(runner, trainer):
    """Run or resume two isolated diagnostic branches at one reference checkpoint."""
    cfg = runner.cfg
    step = trainer.interaction_step
    directory = runner.root / "diagnostics" / f"step_{step:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "progress.pkl"
    identity = {"config_sha256": runner.config_hash, "source_sha256": runner.source_hash,
                "source_commit": runner.source_commit, "checkpoint": step,
                "reference_regime": cfg.online_exp2.reference_regime, "training_seed": int(cfg.seed)}
    if path.exists():
        progress = _load_progress(path, trainer._sac_agent.actor)
        if progress["identity"] != identity:
            raise ValueError("diagnostic resume identity mismatch")
    else:
        trainer.extra_state["online_exp2_pending_checkpoint"] = step
        runner.save()
        runner._check_resource_cap()
        normalize, normalizer_hash = frozen_normalizer(trainer.agent)
        actor = trainer._sac_agent.actor
        seeds = _episode_seeds(cfg, step)
        baseline = []
        for seed in seeds:
            runner._check_resource_cap()
            baseline.append(_evaluate(cfg, actor, normalize, seed))
        progress = {"identity": identity, "main_numpy_rng": np.random.get_state(),
                    "main_python_rng": random.getstate(), "branches": {},
                    "evaluation_seeds": seeds,
                    "baseline": baseline,
                    "actor_hash": _hash(actor.params), "optimizer_hash": _hash(actor.opt_state),
                    "temperature_hash": _hash(trainer._sac_agent.temperature),
                    "normalizer_hash": normalizer_hash,
                    "shadow_hashes": {name: _hash(shadow.critic.params)
                                      for name, shadow in runner.shadows.items()},
                    "shadow_target_hashes": {name: _hash(shadow.target.params)
                                             for name, shadow in runner.shadows.items()},
                    "reference_state": _reference_state(trainer)}
        _save_progress(path, progress)
    normalize, normalizer_hash = frozen_normalizer(trainer.agent)
    if (normalizer_hash != progress["normalizer_hash"]
            or _hash(trainer._sac_agent.actor.params) != progress["actor_hash"]
            or _hash(trainer._sac_agent.actor.opt_state) != progress["optimizer_hash"]
            or _hash(trainer._sac_agent.temperature) != progress["temperature_hash"]
            or {name: _hash(shadow.critic.params) for name, shadow in runner.shadows.items()}
            != progress["shadow_hashes"]
            or {name: _hash(shadow.target.params) for name, shadow in runner.shadows.items()}
            != progress["shadow_target_hashes"]
            or _reference_state(trainer) != progress["reference_state"]):
        raise ValueError("diagnostic checkpoint states changed during resume")
    original_np, original_py = np.random.get_state(), random.getstate()
    try:
        for name in ("small", "large"):
            runner._check_resource_cap()
            if (name not in progress["branches"] or
                    progress["branches"][name]["branch"]["interactions"] <
                    int(cfg.online_exp2.diagnostic_horizon)):
                _run_branch(runner, name, step, progress, path, normalize,
                            trainer._sac_agent.temperature, progress["evaluation_seeds"])
    except Exception as exc:
        atomic_write_text(directory / "failure.json", json.dumps({
            "type": type(exc).__name__, "message": str(exc), "checkpoint": step,
            "reference_regime": cfg.online_exp2.reference_regime}, sort_keys=True))
        raise
    finally:
        np.random.set_state(original_np)
        random.setstate(original_py)
    if (_reference_state(trainer) != progress["reference_state"]
            or {name: _hash(shadow.critic.params) for name, shadow in runner.shadows.items()}
            != progress["shadow_hashes"]
            or {name: _hash(shadow.target.params) for name, shadow in runner.shadows.items()}
            != progress["shadow_target_hashes"]
            or frozen_normalizer(trainer.agent)[1] != progress["normalizer_hash"]):
        raise ValueError("diagnostic modified frozen or reference state")
    for name in ("small", "large"):
        branch = progress["branches"][name]
        if (branch["initial_actor_hash"] != progress["actor_hash"]
                or branch["initial_optimizer_hash"] != progress["optimizer_hash"]):
            raise ValueError("diagnostic branch did not start from reference actor state")
        if branch["branch"]["interactions"] != int(cfg.online_exp2.diagnostic_horizon):
            raise ValueError("incomplete diagnostic branch")
        if branch["branch"]["optimizer_updates"] != progress["branches"]["small"]["branch"]["optimizer_updates"]:
            raise ValueError("diagnostic actor update counts diverged")
    for horizon in cfg.online_exp2.measurement_interactions:
        if not horizon:
            continue
        for episode, seed in enumerate(progress["evaluation_seeds"]):
            base = progress["baseline"][episode]
            small = progress["branches"]["small"]["measurements"][horizon][episode]
            large = progress["branches"]["large"]["measurements"][horizon][episode]
            runner.rows.append({"type": "pilot_evaluation", "reference_regime": cfg.online_exp2.reference_regime,
                                "training_seed": int(cfg.seed), "interaction_step": step,
                                "diagnostic_interactions": int(horizon), "episode": episode,
                                "seed": seed, "baseline": base, "small": small, "large": large,
                                "delta_small": small["return"] - base["return"],
                                "delta_large": large["return"] - base["return"],
                                "advantage_large": large["return"] - small["return"]})
    runner.rows.append({"type": "pilot_diagnostic", **identity,
                        "actor_hash": progress["actor_hash"],
                        "actor_optimizer_hash": progress["optimizer_hash"],
                        "temperature_hash": progress["temperature_hash"],
                        "normalizer_hash": progress["normalizer_hash"],
                        "shadow_hashes": progress["shadow_hashes"],
                        "shadow_target_hashes": progress["shadow_target_hashes"],
                        "shadow_architectures": {"small": "D2W512", "large": "D4W1536"},
                        "evaluation_seeds": progress["evaluation_seeds"],
                        "branch_environment_seeds": {
                            name: progress["branches"][name]["environment_seeds"]
                            for name in ("small", "large")},
                        "baseline": progress["baseline"],
                        "progress_artifact": str(path),
                        "branch_counters": {name: {
                            "interactions": progress["branches"][name]["branch"]["interactions"],
                            "optimizer_updates": progress["branches"][name]["branch"]["optimizer_updates"],
                            "raw_env_steps": progress["branches"][name]["raw_env_steps"],
                            "updates_per_interaction": progress["branches"][name]["branch"]["optimizer_updates"] /
                                                       int(cfg.online_exp2.diagnostic_horizon)}
                            for name in ("small", "large")}})
    runner.rows[-1]["policy_change"] = {
        name: progress["branches"][name]["policy_change"] for name in ("small", "large")}
    if step in runner.probes:
        runner._check_resource_cap()
        runner._probe(step)
        runner._check_resource_cap()
    trainer.extra_state.pop("online_exp2_pending_checkpoint", None)
    runner.save()
