"""Exp 1/2 SAC training loop with exact, complete state save/restore.

The per-step body mirrors experiments/angle_1.py's loop, in the same RNG
consumption order, except that (as in SimBa's released run.py) actions are
uniform random until the replay buffer holds min_length transitions. With
min_length=1 the two loops coincide exactly, which the parity test uses. Differences are only in what is persisted: a save captures
everything needed for a bit-exact continuation (agent incl. optimizer state and
JAX key, obs_rms, replay buffer, global numpy/python RNG, both envs mid-episode,
the vector action-space RNG, loop counters, and partially-filled logging
windows).
"""

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import omegaconf
import pandas as pd

from experiments.angle_1 import PendingUpdateMetrics
from experiments.exp12.envs import create_envs, env_restore_state, restore_env
from experiments.exp12.state import (
    commit_state_dir,
    load_buffer,
    load_meta,
    new_state_dir,
    save_buffer,
    save_meta,
)
from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from scale_rl.common import WandbTrainerLogger
from scale_rl.common.logger import AverageMeter
from utils.atomic_io import atomic_write_text
from utils.paths import require_absolute

# SACAgent's per-window host buffers, flushed into metrics at each logging step.
AGENT_WINDOW_BUFFERS = ("actor_loss_buffer", "actor_entropy_buffer", "churn_buffer")


def evaluate_episodes(agent, env, num_episodes: int):
    """scale_rl.evaluation.evaluate for one env, also returning per-episode values."""
    returns, lengths, successes = [], [], []
    for _ in range(num_episodes):
        episode_return, length, success, done = 0.0, 0, 0.0, False
        observations, _ = env.reset()
        prev_timestep = {"next_observation": observations}
        while not done:
            actions = agent.sample_actions(interaction_step=0, prev_timestep=prev_timestep, training=False)
            observations, rewards, terminateds, truncateds, infos = env.step(actions)
            prev_timestep = {"next_observation": observations}
            episode_return += rewards[0]
            length += 1
            if "success" in infos:
                success += float(infos["success"][0])
            elif "final_info" in infos and "success" in infos["final_info"][0]:
                success += float(infos["final_info"][0]["success"])
            done = bool(terminateds[0] or truncateds[0])
        returns.append(episode_return)
        lengths.append(length)
        successes.append(float(bool(success)))
    eval_info = {
        "eval/avg_return": np.mean(returns),
        "eval/avg_length": np.mean(lengths),
        "eval/avg_success": np.mean(successes),
    }
    return eval_info, returns, lengths, successes


class Exp12Trainer:
    def __init__(self, cfg, run_dir: str):
        self.cfg = cfg
        self.run_dir = Path(require_absolute(run_dir, "run_dir"))
        self.train_env, self.eval_env = create_envs(**cfg.env)
        observation_space = self.train_env.observation_space
        action_space = self.train_env.action_space
        self.buffer = create_buffer(observation_space=observation_space, action_space=action_space, **cfg.buffer)
        self.buffer.reset()
        self.agent = create_agent(observation_space=observation_space, action_space=action_space, cfg=cfg.agent)

        _, total, actor, critic = self.agent.get_num_parameters()
        omegaconf.OmegaConf.set_struct(cfg, False)
        cfg.update({"num_params": total, "actor_num_params": actor, "critic_num_params": critic})
        omegaconf.OmegaConf.set_struct(cfg, True)

        self.interaction_step = 0
        self.update_step = 0
        self.update_counter = 0
        self.observations = None
        self.timestep = None
        self.metrics_rows: List[Dict] = []
        self.eval_rows: List[Dict] = []
        self.logger = None
        self.pending = None
        # Small per-experiment state (e.g. probe records) persisted with every save.
        self.extra_state: Dict = {}
        self.run_name = (
            f"{cfg.env_name}_CD{cfg.agent.critic_num_blocks}_CW{cfg.agent.critic_hidden_dim}"
            f"_AD{cfg.agent.actor_num_blocks}_AW{cfg.agent.actor_hidden_dim}_seed{cfg.seed}"
        )

    def _attach_logger(self, run_id: Optional[str] = None) -> None:
        self.logger = WandbTrainerLogger(self.cfg, run_id=run_id)
        self.pending = PendingUpdateMetrics(self.logger, int(self.cfg.actor_grad_cosine_every))

    def start(self) -> None:
        """Fresh start: step-0 evaluation, then the first train-env reset."""
        self._attach_logger()
        self._evaluate(0, self.cfg.num_eval_episodes)
        self.logger.log_metric(step=0)
        snapshot = {"interaction_step": 0, "env_step": 0}
        snapshot.update(self.logger.average_meter_dict.averages())
        self.metrics_rows.append(snapshot)
        self.logger.reset()
        self.observations, _ = self.train_env.reset()

    def _evaluate(self, interaction_step: int, num_episodes: int) -> None:
        eval_info, returns, lengths, successes = evaluate_episodes(self.agent, self.eval_env, num_episodes)
        self.logger.update_metric(**eval_info)
        for i, (ret, length, success) in enumerate(zip(returns, lengths, successes)):
            self.eval_rows.append({
                "interaction_step": interaction_step,
                "env_step": interaction_step * self.cfg.action_repeat,
                "episode": i,
                "return": float(ret),
                "length": int(length),
                "success": float(success),
            })

    def train(
        self,
        last_step: int,
        after_step: Optional[Callable[["Exp12Trainer"], None]] = None,
        before_first_update: Optional[Callable[["Exp12Trainer"], None]] = None,
    ) -> None:
        cfg = self.cfg
        for interaction_step in range(self.interaction_step + 1, int(last_step) + 1):
            self.interaction_step = interaction_step
            # SimBa's run.py: uniform random actions until the buffer can be sampled;
            # the agent still sees each observation so obs_rms keeps updating.
            if self.timestep:
                actions = self.agent.sample_actions(interaction_step, prev_timestep=self.timestep, training=True)
            if not self.buffer.can_sample():
                actions = self.train_env.action_space.sample()
            next_observations, rewards, terminateds, truncateds, env_infos = self.train_env.step(actions)
            next_buffer_observations = next_observations.copy()
            for env_idx in range(cfg.num_train_envs):
                if terminateds[env_idx] or truncateds[env_idx]:
                    next_buffer_observations[env_idx] = env_infos["final_observation"][env_idx]
            timestep = {
                "observation": self.observations,
                "action": actions,
                "reward": rewards,
                "terminated": terminateds,
                "truncated": truncateds,
                "next_observation": next_buffer_observations,
            }
            self.buffer.add(timestep)
            timestep["next_observation"] = next_observations
            self.timestep = timestep
            self.observations = next_observations

            if self.buffer.can_sample():
                if self.update_step == 0 and before_first_update is not None:
                    before_first_update(self)
                self.update_counter += cfg.updates_per_interaction_step
                num_updates = 0
                while self.update_counter >= 1:
                    self.update_counter -= 1
                    num_updates += 1
                if num_updates:
                    batches = [self.buffer.sample() for _ in range(num_updates)]
                    batches = {key: np.stack([b[key] for b in batches]) for key in batches[0]}
                    update_info = self.agent.update_many(self.update_step, batches, int(cfg.actor_grad_cosine_every))
                    self.pending.add(self.update_step, update_info)
                    self.update_step += num_updates

            if interaction_step % cfg.logging_per_interaction_step == 0:
                self.pending.flush()
                metrics_info = self.agent.get_metrics(self.update_step, self.buffer.sample())
                self.logger.update_metric(**metrics_info)

            if interaction_step % cfg.evaluation_per_interaction_step == 0:
                self.pending.flush()
                final = interaction_step + cfg.evaluation_per_interaction_step > cfg.num_interaction_steps
                self._evaluate(interaction_step, cfg.num_eval_episodes * (10 if final else 1))

            if interaction_step % cfg.logging_per_interaction_step == 0:
                env_step = interaction_step * cfg.action_repeat * cfg.num_train_envs
                snapshot = {"env_step": env_step}
                snapshot.update(self.logger.average_meter_dict.averages())
                snapshot.update(self.logger.media_dict)
                self.metrics_rows.append(snapshot)
                self.logger.log_metric(step=env_step)
                self.logger.reset()

            if after_step is not None:
                after_step(self)

    def save(self, root, keep_previous: bool = False) -> Path:
        """Writes the complete state under root and repoints root/LATEST to it."""
        self.pending.flush()
        path = new_state_dir(root, self.interaction_step)
        self.agent.save_checkpoint(str(path))
        save_buffer(self.buffer, path)
        meters = {
            name: (m.val, m.avg, m.sum, m.count) for name, m in self.logger.average_meter_dict.meters.items()
        }
        save_meta(path, {
            "interaction_step": self.interaction_step,
            "update_step": self.update_step,
            "update_counter": self.update_counter,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "train_env": env_restore_state(self.train_env),
            "eval_env": env_restore_state(self.eval_env),
            "observations": self.observations,
            "timestep": self.timestep,
            "meters": meters,
            "media": dict(self.logger.media_dict),
            "metrics_rows": self.metrics_rows,
            "eval_rows": self.eval_rows,
            "wandb_run_id": self.logger.run_id,
            "extra_state": self.extra_state,
            "agent_window_buffers": {
                name: [np.asarray(v) for v in getattr(self._sac_agent, name)] for name in AGENT_WINDOW_BUFFERS
            },
        })
        commit_state_dir(root, path, keep_previous=keep_previous)
        self.write_logs()
        return path

    def restore(self, state_dir, new_wandb_run: bool = False) -> None:
        """Restores a state written by save(); envs first, since replaying
        them must not perturb the global RNG states restored after."""
        state_dir = Path(state_dir)
        meta = load_meta(state_dir)
        injected = meta["extra_state"].get("injection")
        if injected is not None and "injection" not in self.extra_state:
            self.inject(injected["m"], injected["seed"])  # rebuild the structure the weights belong to
        self.agent.load_checkpoint(str(state_dir))
        load_buffer(self.buffer, state_dir)
        restore_env(self.train_env, meta["train_env"])
        restore_env(self.eval_env, meta["eval_env"])
        np.random.set_state(meta["numpy_rng_state"])
        random.setstate(meta["python_rng_state"])
        self.interaction_step = meta["interaction_step"]
        self.update_step = meta["update_step"]
        self.update_counter = meta["update_counter"]
        self.observations = meta["observations"]
        self.timestep = meta["timestep"]
        self.metrics_rows = meta["metrics_rows"]
        self.eval_rows = meta["eval_rows"]
        self.extra_state = meta["extra_state"]
        self._attach_logger(None if new_wandb_run else meta["wandb_run_id"])
        for name, (val, avg, total, count) in meta["meters"].items():
            meter = AverageMeter()
            meter.val, meter.avg, meter.sum, meter.count = val, avg, total, count
            self.logger.average_meter_dict.meters[name] = meter
        self.logger.media_dict = dict(meta["media"])
        for name, values in meta["agent_window_buffers"].items():
            setattr(self._sac_agent, name, [jnp.asarray(v) for v in values])

    def inject(self, m_label: str, seed: int) -> None:
        """Plasticity injection into the online and target critic (experiments/exp12/injection.py)."""
        from experiments.exp12.injection import inject, injection_key

        a = self._sac_agent
        a._critic, a._target_critic = inject(
            a._critic, a._target_critic, m_label, injection_key(seed),
            float(self.cfg.agent.critic_learning_rate), float(self.cfg.agent.critic_weight_decay),
        )
        self.extra_state["injection"] = {"m": m_label, "seed": int(seed)}

    @property
    def _sac_agent(self):
        return getattr(self.agent, "agent", self.agent)

    def write_logs(self) -> None:
        logs = self.run_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        atomic_write_text(logs / f"{self.run_name}.csv", pd.DataFrame(self.metrics_rows).to_csv(index=False))
        atomic_write_text(logs / f"{self.run_name}_eval_episodes.csv", pd.DataFrame(self.eval_rows).to_csv(index=False))

    def close(self) -> None:
        self.train_env.close()
        self.eval_env.close()
