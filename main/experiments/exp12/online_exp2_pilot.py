"""Sequential, restartable raw-step policy branches for the revised Exp2 pilot."""

import copy
import hashlib
import json
import pickle
import random
from types import SimpleNamespace

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import SyncVectorEnv
from omegaconf import OmegaConf

from experiments.exp12.diagnostics import ActorDiagnostics
from experiments.exp12.envs import _make_one_env, env_restore_state, restore_env
from experiments.exp12.online_exp2_diagnostic import (
    ActorUpdateSchedule,
    DiagnosticActorBranch,
    copy_reference_replay,
)
from scale_rl.envs.wrappers.repeat_action import executed_raw_steps, limit_raw_steps
from utils.atomic_io import atomic_write_bytes, atomic_write_text

_executed_repeats = executed_raw_steps  # compatibility for existing primitive callers

CONDITIONS = tuple(
    (architecture, mode)
    for mode in ("frozen", "learning")
    for architecture in ("small", "large")
)
MODELS = ("actor", "critic", "_target_critic", "temperature")


def _seed(*parts):
    return int.from_bytes(
        hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little"
    )


def _hash(value):
    def canonical(item):
        if isinstance(item, dict):
            return {key: canonical(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [canonical(value) for value in item]
        if hasattr(item, "shape"):
            item = np.asarray(item)
            return item.item() if item.shape == () else item
        return item

    return hashlib.sha256(
        flax.serialization.msgpack_serialize(
            canonical(flax.serialization.to_state_dict(value))
        )
    ).hexdigest()


def frozen_normalizer(agent):
    if not hasattr(agent, "obs_rms"):
        return None, "none"
    mean, var = np.array(agent.obs_rms.mean), np.array(agent.obs_rms.var)
    epsilon = float(agent.epsilon)
    identifier = hashlib.sha256(
        mean.tobytes()
        + var.tobytes()
        + np.asarray(agent.obs_rms.count).tobytes()
        + str(epsilon).encode()
    ).hexdigest()
    return lambda value: (np.asarray(value) - mean) / np.sqrt(var + epsilon), identifier


def _branch_env(cfg, seed):
    e = OmegaConf.to_container(cfg.env, resolve=True)
    return SyncVectorEnv(
        [
            lambda: _make_one_env(
                e["env_type"],
                e["env_name"],
                seed,
                e["rescale_action"],
                e["no_termination"],
                e["action_repeat"],
                e["reward_scale"],
                e["max_episode_steps"],
            )
        ]
    )


def _success(info):
    if "final_info" in info and info["final_info"][0] is not None:
        return float(info["final_info"][0].get("success", 0.0))
    return float(info.get("success", [0.0])[0])


def _replay_hash(buffer):
    digest = hashlib.sha256()
    for name in (
        "_observations",
        "_actions",
        "_rewards",
        "_terminateds",
        "_truncateds",
        "_next_observations",
    ):
        digest.update(memoryview(getattr(buffer, name)[: len(buffer)]).cast("B"))
    digest.update(
        pickle.dumps(
            (
                buffer._num_in_buffer,
                buffer._current_idx,
                list(buffer._n_step_transitions),
            )
        )
    )
    return digest.hexdigest()


def _reference_state(trainer):
    agent = trainer._sac_agent
    result = {name: _hash(getattr(agent, name)) for name in MODELS}
    result.update(
        rng=_hash(agent._rng),
        normalizer=frozen_normalizer(trainer.agent)[1],
        raw_env_steps=trainer.raw_env_steps,
        replay=_replay_hash(trainer.buffer),
        train_env=hashlib.sha256(
            pickle.dumps(env_restore_state(trainer.train_env))
        ).hexdigest(),
        eval_env=hashlib.sha256(
            pickle.dumps(env_restore_state(trainer.eval_env))
        ).hexdigest(),
    )
    return result


def _write(path, state):
    atomic_write_bytes(path, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


def _clone_agent(trainer, shadow):
    """Models are immutable; copy mutable host windows and observation statistics."""
    sac = copy.copy(trainer._sac_agent)
    sac._critic, sac._target_critic = shadow.critic, shadow.target
    for name in ("actor_loss_buffer", "actor_entropy_buffer", "churn_buffer"):
        setattr(sac, name, [])
    sac.churn_ref_batch = copy.deepcopy(sac.churn_ref_batch)
    if hasattr(trainer.agent, "obs_rms"):
        agent = copy.copy(trainer.agent)
        agent.agent = sac
        agent.obs_rms = copy.deepcopy(trainer.agent.obs_rms)
        return agent, sac
    return sac, sac


def _model_state(sac):
    return {name: flax.serialization.to_bytes(getattr(sac, name)) for name in MODELS}


def _restore_models(sac, state):
    for name in MODELS:
        template = getattr(sac, name)
        model = flax.serialization.from_bytes(template, state[name])
        model = model.replace(
            update_step=jnp.asarray(
                model.update_step, dtype=jnp.asarray(template.update_step).dtype
            )
        )
        setattr(sac, "_" + name if not name.startswith("_") else name, model)


def _emit(runner, row):
    """Use condition-specific axes; reference and branch steps never share an axis."""
    runner.rows.append(row)
    import wandb

    if wandb.run is None:
        return
    prefix = f"branch/{row['condition']}/{row['checkpoint']}"
    axis = prefix + "/raw_env_steps"
    wandb.define_metric(prefix + "/*", step_metric=axis)
    payload = {axis: row["branch_raw_env_steps"]}
    payload.update(
        {
            prefix + "/" + key: value
            for key, value in row.items()
            if isinstance(value, (float, int)) and key != "checkpoint"
        }
    )
    payload.update(
        {prefix + "/" + key: value for key, value in row.get("metrics", {}).items()}
    )
    # No explicit W&B global step: reference steps may be smaller than branch steps.
    wandb.log(payload)


def _run_branch(runner, architecture, mode, identity, directory):
    cfg, trainer = runner.cfg, runner.trainer
    condition = architecture + "_" + mode
    path = directory / (condition + ".pkl")
    seed = _seed("exp2-branch", cfg.seed, identity["checkpoint"], 0)
    agent, sac = _clone_agent(trainer, runner.shadows[architecture])
    learning = mode == "learning"
    normalizer, _ = frozen_normalizer(trainer.agent)
    schedule = ActorUpdateSchedule(
        cfg.updates_per_interaction_step, trainer.update_counter
    )
    diagnostics = ActorDiagnostics(cfg)
    frozen = DiagnosticActorBranch(
        trainer._sac_agent.actor,
        runner.shadows[architecture].critic,
        trainer._sac_agent.temperature,
        seed,
        int(cfg.buffer.sample_batch_size),
        cfg.updates_per_interaction_step,
        cfg.agent.critic_use_cdq,
        trainer.update_counter,
    )
    env = None
    try:
        np.random.seed(seed)
        random.seed(seed)
        env = _branch_env(cfg, seed)
        if path.exists():
            state = pickle.loads(path.read_bytes())
            if state["identity"] != identity or state["condition"] != condition:
                raise ValueError("branch resume identity mismatch")
            if state["complete"]:
                return state["summary"]
            observation = state["observation"]
            restore_env(env, state["env"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            _restore_models(sac, state["models"])
            sac._rng = jnp.asarray(state["rng"])
            sac.churn_ref_batch = state["churn_ref_batch"]
            if hasattr(agent, "obs_rms"):
                agent.obs_rms = state["obs_rms"]
            replay = state["replay"]
            schedule = ActorUpdateSchedule.from_state_dict(state["schedule"])
            diagnostics.load_state(state["diagnostics"])
            if not learning:
                restored = dict(state["frozen"])
                restored["actor"] = sac.actor
                frozen = DiagnosticActorBranch.from_state_dict(
                    restored, sac.critic, sac.temperature
                )
        else:
            observation, _ = env.reset()
            replay = (
                copy_reference_replay(
                    trainer.buffer, int(cfg.online_exp2.branch_raw_steps)
                )
                if learning
                else None
            )
            state = {
                "identity": identity,
                "condition": condition,
                "complete": False,
                "interactions": 0,
                "updates": 0,
                "raw_env_steps": 0,
                "initial_observation": np.array(observation, copy=True),
                "window_metrics": {},
                "episode_return": 0.0,
                "episode_length": 0,
                "episode_raw_steps": 0,
                "episodes": [],
                "records": [],
                "markers": {},
                "initial_models": {name: _hash(getattr(sac, name)) for name in MODELS},
                "initial_actor_hash": _hash(sac.actor.params),
                "initial_optimizer_hash": _hash(sac.actor.opt_state),
            }

        def save(complete=False):
            if not learning:
                sac._actor = frozen.actor
            frozen_state = frozen.state_dict() if not learning else None
            if frozen_state is not None:
                frozen_state.pop("actor")
            state.update(
                models=_model_state(sac),
                rng=np.asarray(sac._rng),
                churn_ref_batch=sac.churn_ref_batch,
                obs_rms=(
                    copy.deepcopy(agent.obs_rms) if hasattr(agent, "obs_rms") else None
                ),
                replay=replay,
                frozen=frozen_state,
                schedule=schedule.state_dict(),
                diagnostics=diagnostics.state(),
                observation=observation,
                env=env_restore_state(env),
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
                complete=complete,
            )
            _write(path, state)

        save()
        horizon = int(cfg.online_exp2.branch_raw_steps)
        markers = list(map(int, cfg.online_exp2.measurement_raw_steps))
        while state["raw_env_steps"] < horizon:
            runner._check_resource_cap()
            previous = np.array(observation, copy=True)
            previous_raw_steps = state["raw_env_steps"]
            boundary = horizon
            limit_raw_steps(env, boundary - state["raw_env_steps"])
            if learning:
                action = agent.sample_actions(
                    state["interactions"] + 1,
                    {"next_observation": previous.copy()},
                    training=True,
                )
            else:
                action = frozen.act(previous, normalizer)
            if not np.all(np.isfinite(action)):
                raise ValueError("nonfinite branch action")
            observation, reward, term, trunc, info = env.step(action)
            if not np.all(np.isfinite(reward)) or not np.all(np.isfinite(observation)):
                raise ValueError("nonfinite branch transition")
            raw = executed_raw_steps(env)
            if raw < 1 or state["raw_env_steps"] + raw > boundary:
                raise ValueError("invalid executed raw-step count")
            state["raw_env_steps"] += raw
            state["interactions"] += 1
            state["episode_raw_steps"] += raw
            state["episode_length"] += 1
            state["episode_return"] += float(reward[0])
            ended = bool(term[0] or trunc[0])
            metrics = {}
            if learning:
                next_buffer_observation = np.array(observation, copy=True)
                if ended and "final_observation" in info:
                    next_buffer_observation[0] = info["final_observation"][0]
                replay.add(
                    {
                        "observation": previous,
                        "action": action,
                        "reward": reward,
                        "terminated": term,
                        "truncated": trunc,
                        "next_observation": next_buffer_observation,
                    }
                )
                count = schedule.advance() if replay.can_sample() else 0
                if count:
                    batches = [replay.sample() for _ in range(count)]
                    batches = {
                        key: np.stack([batch[key] for batch in batches])
                        for key in batches[0]
                    }
                    local = SimpleNamespace(
                        interaction_step=state["interactions"],
                        buffer=replay,
                        agent=agent,
                    )
                    update = agent.update_many(
                        trainer.update_step + state["updates"],
                        batches,
                        int(cfg.actor_grad_cosine_every),
                        **diagnostics.update_kwargs(local),
                    )
                    host = jax.device_get(update)
                    for key, value in host.items():
                        if key != "train/actor_grad_cosine" and not np.all(
                            np.isfinite(value)
                        ):
                            raise ValueError(
                                "nonfinite source SAC branch metric: " + key
                            )
                    diagnostics.collect(host)
                    metrics = {
                        key: float(np.nanmean(value))
                        for key, value in host.items()
                        if np.any(np.isfinite(value))
                    }
                    state["updates"] += count
            else:
                # Reuse the established fixed-window KL sampling and saturation definition.
                observations = frozen.buffer._observations + [previous[0]]
                local_buffer = SimpleNamespace(
                    _num_in_buffer=len(observations),
                    _observations=np.asarray(observations),
                )
                fixed_agent = (
                    SimpleNamespace(_normalize=normalizer) if normalizer else object()
                )
                local = SimpleNamespace(
                    interaction_step=state["interactions"],
                    buffer=local_buffer,
                    agent=fixed_agent,
                )
                frozen.diagnostics = (
                    diagnostics.update_kwargs(local) if diagnostics.enabled else None
                )
                frozen.cosine_every = int(cfg.actor_grad_cosine_every)
                update = frozen.observe_and_update(previous[0], normalizer)
                state["updates"] = frozen.optimizer_updates
                if update["updates"]:
                    for entry in update["updates"]:
                        diagnostics.collect(entry)
                    metrics = {
                        key: float(
                            np.mean(
                                [
                                    record[key]
                                    for record in update["updates"]
                                    if key in record
                                ]
                            )
                        )
                        for key in set().union(
                            *(record.keys() for record in update["updates"])
                        )
                    }
            for key, value in metrics.items():
                state["window_metrics"].setdefault(key, []).append(value)
            if not learning and update["updates"]:
                for entry in update["updates"]:
                    sac.actor_loss_buffer.append(entry["train/actor_loss"])
                    sac.actor_entropy_buffer.append(entry["train/entropy"])
                    sac.churn_buffer.append(entry.get("train/policy_churn", 0.0))
            if ended:
                episode = {
                    "return": state["episode_return"],
                    "length": state["episode_length"],
                    "raw_env_steps": state["episode_raw_steps"],
                    "end_raw_env_steps": state["raw_env_steps"],
                    "success": _success(info),
                    "terminated": bool(term[0]),
                    "truncated": bool(trunc[0]),
                }
                state["episodes"].append(episode)
                state["episode_return"] = 0.0
                state["episode_length"] = state["episode_raw_steps"] = 0
                record = {
                    "type": "pilot_training_episode",
                    **identity,
                    "architecture": architecture,
                    "condition": condition,
                    "branch_raw_env_steps": state["raw_env_steps"],
                    **episode,
                }
                state["records"].append(record)
            if ended and "final_observation" not in info:
                observation, _ = env.reset()
            cadence = (
                state["interactions"] % int(cfg.online_exp2.diagnostic_save_every) == 0
            )
            crossed_markers = [
                marker
                for marker in markers
                if previous_raw_steps < marker <= state["raw_env_steps"]
            ]
            marked = bool(crossed_markers)
            if cadence or marked:
                metrics = {
                    key: float(np.mean(values))
                    for key, values in state["window_metrics"].items()
                }
                state["window_metrics"] = {}
                if not learning and len(frozen.buffer):
                    from scale_rl.agents.sac.sac_update import (
                        get_actor_with_metrics,
                        get_critic_with_metrics,
                    )

                    # Dedicated diagnostic sampling never advances the training sampler.
                    rng = np.random.default_rng(
                        [
                            int(cfg.seed),
                            identity["checkpoint"],
                            state["raw_env_steps"],
                            0xD1A6,
                        ]
                    )
                    obs = np.asarray(frozen.buffer._observations)[
                        rng.integers(
                            0,
                            len(frozen.buffer),
                            size=int(cfg.buffer.sample_batch_size),
                        )
                    ]
                    obs = normalizer(obs) if normalizer else obs
                    diagnostic_key = jax.random.PRNGKey(seed)
                    acts = frozen.actor(observations=jnp.asarray(obs)).sample(
                        seed=diagnostic_key
                    )
                    batch = {"observation": jnp.asarray(obs), "action": acts}
                    metrics.update(
                        {
                            key: float(value)
                            for key, value in {
                                **get_actor_with_metrics(frozen.actor, batch),
                                **get_critic_with_metrics(
                                    diagnostic_key,
                                    frozen.actor,
                                    frozen.critic,
                                    batch,
                                    frozen.critic_use_cdq,
                                ),
                            }.items()
                            if value is not None
                            and np.ndim(value) == 0
                            and np.isfinite(value)
                        }
                    )
                    metrics.update(diagnostics.window_metrics())
                    for key, value in (
                        ("train/actor_loss_var", sac.flush_actor_loss_var()),
                        ("train/entropy", sac.mean_entropy()),
                        ("train/churn", sac.flush_policy_churn()),
                    ):
                        if value is not None:
                            metrics[key] = value
                if learning and replay.can_sample():
                    # get_metrics flushes the source host windows, retaining the source definitions.
                    metrics.update(
                        {
                            key: float(value)
                            for key, value in agent.get_metrics(
                                state["updates"], replay.sample()
                            ).items()
                            if value is not None
                            and np.ndim(value) == 0
                            and np.isfinite(value)
                        }
                    )
                    metrics.update(diagnostics.window_metrics())
                if marked:
                    obs = state["initial_observation"]
                    obs = normalizer(obs) if normalizer else obs
                    actors = (
                        trainer._sac_agent.actor,
                        sac.actor if learning else frozen.actor,
                    )
                    actions = [
                        np.asarray(
                            actor(
                                observations=jnp.asarray(obs), temperature=0.0
                            ).sample(seed=jax.random.PRNGKey(seed))
                        )
                        for actor in actors
                    ]
                    metrics["policy_change"] = float(
                        np.linalg.norm(actions[1] - actions[0])
                    )
                    for marker in crossed_markers:
                        state["markers"][marker] = {
                            "observed_raw_env_steps": state["raw_env_steps"],
                            "completed_episodes": len(state["episodes"]),
                            "partial_episode_return": state["episode_return"],
                        }
                record = {
                    "type": "pilot_branch_metrics",
                    **identity,
                    "architecture": architecture,
                    "condition": condition,
                    "branch_raw_env_steps": state["raw_env_steps"],
                    "interactions": state["interactions"],
                    "updates": state["updates"],
                    "metrics": metrics,
                }
                state["records"].append(record)
                # Bound host lists between checkpoints, including after resumes.
                for name in (
                    "actor_loss_buffer",
                    "actor_entropy_buffer",
                    "churn_buffer",
                ):
                    setattr(sac, name, [])
                save()
        if not learning:
            sac._actor = frozen.actor
            if (
                _hash(sac.critic) != state["initial_models"]["critic"]
                or _hash(sac._target_critic)
                != state["initial_models"]["_target_critic"]
                or _hash(sac.temperature) != state["initial_models"]["temperature"]
            ):
                raise ValueError("frozen branch changed critic, target or temperature")
        state["summary"] = {
            "condition": condition,
            "architecture": architecture,
            "raw_env_steps": state["raw_env_steps"],
            "interactions": state["interactions"],
            "optimizer_updates": state["updates"],
            "episodes": state["episodes"],
            "markers": state["markers"],
            "records": state["records"],
            "initial_actor_hash": state["initial_actor_hash"],
            "initial_optimizer_hash": state["initial_optimizer_hash"],
            "final_models": {name: _hash(getattr(sac, name)) for name in MODELS},
            "artifact": str(path),
            "normalizer_semantics": (
                "branch-local-updating" if learning else "checkpoint-frozen"
            ),
            "temperature_semantics": (
                "source-SAC-learned" if learning else "checkpoint-frozen"
            ),
        }
        # Completed branches resume from their compact receipt. Only an active
        # branch needs replay/model tensors; retaining all 60 copies exhausts Kaggle disk.
        state["summary"]["final_replay_hash"] = (
            _replay_hash(replay) if learning else None
        )
        state["summary"]["final_replay_length"] = len(replay) if learning else 0
        state["complete"] = True
        for key in (
            "replay",
            "models",
            "frozen",
            "churn_ref_batch",
            "obs_rms",
            "observation",
            "env",
        ):
            state.pop(key, None)
        _write(path, state)
        return state["summary"]
    finally:
        if env is not None:
            env.close()


def _plot_training_curves(runner, progress, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 4))
    for condition, summary in progress["branches"].items():
        episodes = summary["episodes"]
        axis.plot(
            [episode["end_raw_env_steps"] for episode in episodes],
            [episode["return"] for episode in episodes],
            marker=".",
            label=condition,
        )
    for marker in runner.cfg.online_exp2.measurement_raw_steps:
        if marker:
            axis.axvline(marker, color="gray", linestyle="--", alpha=0.6)
    identity = progress["identity"]
    axis.set(
        xlabel="Branch raw environment steps",
        ylabel="Training episode return",
        title=f"{identity['environment']} / {identity['reference_regime']} / seed {identity['training_seed']} / checkpoint {identity['checkpoint']}",
    )
    axis.legend()
    fig.tight_layout()
    path = directory / "training_returns.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    runner.rows.append(
        {
            "type": "pilot_training_plot",
            **identity,
            "artifact": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
    import wandb

    if wandb.run is not None:
        wandb.log(
            {
                f"branch/training_returns/{identity['checkpoint']}": wandb.Image(
                    str(path)
                )
            }
        )


def pilot_checkpoint(runner, trainer):
    """Pause reference training, then run one isolated branch at a time."""
    cfg = runner.cfg
    checkpoint = trainer.raw_env_steps
    directory = runner.root / "diagnostics" / f"raw_{checkpoint:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "progress.pkl"
    identity = {
        "config_sha256": runner.config_hash,
        "source_sha256": runner.source_hash,
        "source_commit": runner.source_commit,
        "checkpoint": checkpoint,
        "reference_interaction_step": trainer.interaction_step,
        "reference_regime": cfg.online_exp2.reference_regime,
        "environment": cfg.env_name,
        "training_seed": int(cfg.seed),
        "run_id": trainer.logger.run_id or runner.root.name,
    }
    original_np, original_py = np.random.get_state(), random.getstate()
    before = _reference_state(trainer)
    shadows = {
        name: (_hash(shadow.critic), _hash(shadow.target))
        for name, shadow in runner.shadows.items()
    }
    try:
        if path.exists():
            progress = pickle.loads(path.read_bytes())
            if (
                progress["identity"] != identity
                or progress["reference_state"] != before
            ):
                raise ValueError(
                    "diagnostic resume identity or reference state mismatch: "
                    + str(
                        {
                            key: (value, before.get(key))
                            for key, value in progress["reference_state"].items()
                            if value != before.get(key)
                        }
                    )
                    + " identity="
                    + str(
                        {
                            key: (value, identity.get(key))
                            for key, value in progress["identity"].items()
                            if value != identity.get(key)
                        }
                    )
                )
        else:
            trainer.extra_state["online_exp2_pending_checkpoint"] = checkpoint
            runner.save()
            progress = {
                "identity": identity,
                "reference_state": before,
                "branches": {},
                "probe_done": False,
            }
            _write(path, progress)
        for architecture, mode in CONDITIONS:
            condition = architecture + "_" + mode
            if condition not in progress["branches"]:
                np.random.set_state(original_np)
                random.setstate(original_py)
                progress["branches"][condition] = _run_branch(
                    runner, architecture, mode, identity, directory
                )
                _write(path, progress)
        np.random.set_state(original_np)
        random.setstate(original_py)
        if checkpoint in runner.probes and not progress["probe_done"]:
            runner._check_resource_cap()
            start_rows = len(runner.rows)
            runner._probe(checkpoint)
            progress["probe_rows"] = runner.rows[start_rows:]
            progress["probe_done"] = True
            _write(path, progress)
        for row in progress.get("probe_rows", []):
            if row not in runner.rows:
                runner.rows.append(row)
        if _reference_state(trainer) != before or shadows != {
            name: (_hash(shadow.critic), _hash(shadow.target))
            for name, shadow in runner.shadows.items()
        }:
            raise ValueError("diagnostic modified reference or shadow state")
        for condition, summary in progress["branches"].items():
            if summary["raw_env_steps"] != int(cfg.online_exp2.branch_raw_steps):
                raise ValueError("incomplete raw-step branch")
            for record in summary["records"]:
                _emit(runner, record)
        runner.rows.append(
            {
                "type": "pilot_diagnostic",
                **identity,
                "progress_artifact": str(path),
                "branch_counters": {
                    name: {
                        key: summary[key]
                        for key in (
                            "interactions",
                            "optimizer_updates",
                            "raw_env_steps",
                        )
                    }
                    for name, summary in progress["branches"].items()
                },
            }
        )
        _plot_training_curves(runner, progress, directory)
        trainer.extra_state.pop("online_exp2_pending_checkpoint", None)
        runner.save()
    except Exception as exc:
        atomic_write_text(
            directory / "failure.json",
            json.dumps(
                {"type": type(exc).__name__, "message": str(exc), **identity},
                sort_keys=True,
            ),
        )
        raise
    finally:
        np.random.set_state(original_np)
        random.setstate(original_py)
