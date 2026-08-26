"""Angle 1: the original run.py training pipeline, now behind the experiment
registry, with optional critic-degradation / pathology-propagation onset
tracking bolted on as an opt-in extra.

Everything up to and including the training loop is unchanged from the
pre-refactor run.py (same config loading, checkpointing, WandB init, env/seed
handling). The only additions, and they only activate when
`critic_degradation` and/or `pathology_prop` are true in the resolved config:

  1. a MetricsRecorder appends {interaction_step, env_step, td_error_variance,
     actor_grad_cosine} at the same cadence the existing logger already logs
     at (`logging_per_interaction_step`), and flushes to
     `results/metrics/...` once training finishes.
  2. after training (never during it), analysis/pipeline.py reads those
     persisted metrics back, runs onset detection against the calibrated
     baseline, and upserts a row into the canonical CSV ledger
     (utils/onset_ledger.py), mirroring onto the WandB run summary last.

`train/td_error_var` and `train/actor_grad_cosine` are already computed
unconditionally by the existing SAC update step (see
scale_rl/agents/sac/sac_update.py); the flags below control whether that
already-computed data is *persisted for onset analysis*, not whether it's
computed. Making the computation itself conditional would mean touching the
jitted `_update_sac_networks` function, which is out of scope for a minimal,
low-risk integration (see README-equivalent notes in the project deliverables).
"""

import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'glfw')
if sys.platform == 'darwin':
    # Apple Metal only exists on macOS; on Linux (Kaggle/Colab) leave
    # JAX_PLATFORMS untouched so JAX auto-detects CUDA/CPU normally.
    os.environ.setdefault('JAX_PLATFORMS', 'METAL,cpu')

import pickle
import random
from pathlib import Path

import hydra
import jax
import numpy as np
import omegaconf
import pandas as pd
import tqdm
import wandb
from dotmap import DotMap

from analysis.metrics_store import MetricsRecorder, RunIdentity
from analysis.pipeline import run_post_hoc_onset_analysis
from experiments.registry import register_experiment
from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from scale_rl.common import WandbTrainerLogger
from scale_rl.common.logger import get_architecture_id
from scale_rl.envs import create_envs
from scale_rl.evaluation import evaluate
from utils.onset_ledger import WandbIdentity

jax.config.update("jax_enable_x64", False)


@register_experiment("angle_1")
def run(args: dict) -> None:
    ###############################
    # configs
    ###############################
    args = DotMap(args)
    experiment_name = args.experiment or "angle_1"
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    # hydra.initialize() resolves a relative config_path relative to the
    # *calling module's* file location, not the process CWD - which broke
    # once training moved out of run.py (repo root) into this submodule.
    # initialize_config_dir() takes an absolute path instead, preserving the
    # original "--config_path is relative to wherever you invoke run.py
    # from" contract.
    hydra.initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_path))
    cfg = hydra.compose(config_name=config_name, overrides=overrides)

    def eval_resolver(s: str):
        return eval(s)

    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver, replace=True)
    omegaconf.OmegaConf.resolve(cfg)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    critic_degradation_enabled = bool(cfg.get("critic_degradation", False))
    pathology_prop_enabled = bool(cfg.get("pathology_prop", False))
    onset_cfg = (
        omegaconf.OmegaConf.to_container(cfg.onset_detection, resolve=True)
        if "onset_detection" in cfg
        else {}
    )

    #############################
    # envs
    #############################
    train_env, eval_env = create_envs(**cfg.env)
    observation_space = train_env.observation_space
    action_space = train_env.action_space

    #############################
    # buffer
    #############################
    buffer = create_buffer(
        observation_space=observation_space,
        action_space=action_space,
        **cfg.buffer
    )
    buffer.reset()

    #############################
    # agent
    #############################
    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        cfg=cfg.agent,
    )

    checkpoint_dir = args.checkpoint_dir
    start_step = 1
    resumed_update_step = 0
    resumed_update_counter = 0

    if checkpoint_dir and (Path(checkpoint_dir) / "meta.pkl").exists():
        agent.load_checkpoint(checkpoint_dir)
        buffer.load(checkpoint_dir)
        with open(Path(checkpoint_dir) / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        start_step = meta["interaction_step"] + 1
        resumed_update_step = meta["update_step"]
        resumed_update_counter = meta["update_counter"]
        print(f"Resumed from interaction_step {start_step}")
    #############################
    # train
    #############################
    os.environ["WANDB_MODE"] = "online"
    # Keep the metrics CSV cache next to the checkpoint when one is
    # configured, so it rides along with whatever persistence mechanism
    # already covers checkpoint_dir (e.g. a Kaggle Dataset push) instead of
    # living in a throwaway, machine-specific location. Falls back to a
    # repo-relative ./logs when running without checkpointing.
    if checkpoint_dir:
        LOGS_DIR = str(Path(checkpoint_dir) / "logs")
    else:
        LOGS_DIR = str(Path(__file__).resolve().parents[1] / "logs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    run_name = f"{cfg.env_name}_CD{cfg.agent.critic_num_blocks}_CW{cfg.agent.critic_hidden_dim}_AD{cfg.agent.actor_num_blocks}_AW{cfg.agent.actor_hidden_dim}_seed{cfg.seed}"

    csv_path = os.path.join(LOGS_DIR, f"{run_name}.csv")
    if checkpoint_dir and os.path.exists(csv_path):
        raw_cache = pd.read_csv(csv_path).to_dict("records")
        resumed_env_step = (start_step - 1) * cfg.action_repeat * cfg.num_train_envs
        local_metrics_cache = [
            row for row in raw_cache if row.get("env_step", 0) <= resumed_env_step
        ]
    else:
        local_metrics_cache = []

    _total_params, params_str_total, params_str_actor, params_str_critic= agent.get_num_parameters()
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.update({'num_params':params_str_total})
    cfg.update({'actor_num_params':params_str_actor})
    cfg.update({'critic_num_params':params_str_critic})
    omegaconf.OmegaConf.set_struct(cfg, True)

    logger = WandbTrainerLogger(cfg)

    #############################
    # onset tracking (opt-in)
    #############################
    architecture = get_architecture_id(cfg)
    run_identity = RunIdentity(
        experiment=experiment_name,
        architecture=architecture,
        environment=cfg.env_name,
        seed=cfg.seed,
    )
    metrics_recorder = None
    if critic_degradation_enabled or pathology_prop_enabled:
        metrics_recorder = MetricsRecorder(run_identity)
        if checkpoint_dir:
            metrics_recorder.load_existing_up_to(start_step - 1)

    # initial evaluation
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
    logger.update_metric(**eval_info)
    logger.log_metric(step=0)
    step_snapshot = {"interaction_step": 0, "env_step": 0}
    step_snapshot.update(logger.average_meter_dict.averages())
    local_metrics_cache.append(step_snapshot)
    logger.reset()

    # start training
    update_step = resumed_update_step
    update_counter = resumed_update_counter
    observations, env_infos = train_env.reset()
    timestep = None
    checkpoint_start_step = int(args.checkpoint_start_frac * cfg.num_interaction_steps)

    for interaction_step in tqdm.tqdm(
        range(start_step, int(cfg.num_interaction_steps + 1)), smoothing=0.1
    ):
        if timestep:
            actions = agent.sample_actions(
                interaction_step, prev_timestep=timestep, training=True
            )
        elif not buffer.can_sample() is False:
            actions = train_env.action_space.sample()
        else:
            actions = train_env.action_space.sample()
        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(
            actions
        )
        next_buffer_observations = next_observations.copy()
        for env_idx in range(cfg.num_train_envs):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_observation"][
                    env_idx
                ]
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
            update_counter += cfg.updates_per_interaction_step
            while update_counter >= 1:
                batch = buffer.sample()
                update_info = agent.update(update_step, batch)
                logger.update_metric(**update_info)
                update_counter -= 1
                update_step += 1

        # log metrics
        if interaction_step % cfg.logging_per_interaction_step == 0:
            log_metrics_batch = buffer.sample()
            metrics_info = agent.get_metrics(update_step, log_metrics_batch)
            logger.update_metric(**metrics_info)

        # checkpoint model save
        if (
            checkpoint_dir
            and interaction_step >= checkpoint_start_step
            and interaction_step % args.checkpoint_interval == 0
        ):
            agent.save_checkpoint(checkpoint_dir)
            buffer.save(checkpoint_dir)
            with open(Path(checkpoint_dir) / "meta.pkl", "wb") as f:
                pickle.dump({
                    "interaction_step": interaction_step,
                    "update_step": update_step,
                    "update_counter": update_counter,
                }, f)
            pd.DataFrame(local_metrics_cache).to_csv(csv_path, index=False)
            if metrics_recorder is not None:
                metrics_recorder.flush()

        # evaluation
        if interaction_step % cfg.evaluation_per_interaction_step == 0:
            if interaction_step + cfg.evaluation_per_interaction_step > cfg.num_interaction_steps:
                eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes*10)
                logger.update_metric(**eval_info)
            else:
                eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
                logger.update_metric(**eval_info)

        # Unified Logging and CSV Caching Step
        if interaction_step % cfg.logging_per_interaction_step == 0:
            env_step = interaction_step * cfg.action_repeat * cfg.num_train_envs

            step_snapshot = {"env_step": env_step}
            step_snapshot.update(logger.average_meter_dict.averages())
            step_snapshot.update(logger.media_dict)
            local_metrics_cache.append(step_snapshot)

            if metrics_recorder is not None:
                metrics_recorder.record(interaction_step, env_step, step_snapshot)

            logger.log_metric(step=env_step)
            logger.reset()

    df = pd.DataFrame(local_metrics_cache)
    df.to_csv(csv_path, index=False)

    if metrics_recorder is not None:
        metrics_recorder.flush()

    train_env.close()
    eval_env.close()

    #############################
    # post-hoc onset analysis (never during training)
    #############################
    if critic_degradation_enabled or pathology_prop_enabled:
        baseline_seeds = onset_cfg.get("baseline_seeds", [])
        default_architecture = onset_cfg.get("default_architecture")
        baseline_architecture = onset_cfg.get("baseline_architecture") or default_architecture
        baseline_experiment = onset_cfg.get("baseline_experiment") or experiment_name

        if baseline_architecture is None:
            raise ValueError(
                "Neither onset_detection.baseline_architecture nor "
                "onset_detection.default_architecture is set; cannot "
                "determine which architecture's 5 seeds to calibrate the "
                "baseline from. Set onset_detection.default_architecture in "
                "your config (the canonical default-SimBa architecture id), "
                "or set onset_detection.baseline_architecture explicitly."
            )
        if baseline_architecture == architecture and architecture != default_architecture:
            raise ValueError(
                f"Refusing to calibrate architecture '{architecture}' "
                f"against itself as its own baseline: this run's "
                f"architecture does not match the configured "
                f"onset_detection.default_architecture "
                f"('{default_architecture}'), so baseline_architecture="
                f"'{baseline_architecture}' would be a self-baseline "
                f"comparison rather than a comparison against the default "
                f"SimBa critic. Set onset_detection.baseline_architecture "
                f"explicitly to the actual default architecture for this "
                f"scaled run."
            )

        baseline_identities = [
            RunIdentity(
                experiment=baseline_experiment,
                architecture=baseline_architecture,
                environment=cfg.env_name,
                seed=s,
            )
            for s in baseline_seeds
        ]
        wandb_identity = WandbIdentity(run_obj=wandb.run)
        try:
            ledger_path = run_post_hoc_onset_analysis(
                run_identity=run_identity,
                critic_degradation_enabled=critic_degradation_enabled,
                pathology_prop_enabled=pathology_prop_enabled,
                baseline_identities=baseline_identities,
                onset_cfg=onset_cfg,
                logging_per_interaction_step=int(cfg.logging_per_interaction_step),
                wandb_identity=wandb_identity,
            )
            print(f"[onset-ledger] onset analysis complete -> {ledger_path}")
        except Exception:
            print(
                "[onset-ledger] Post-hoc onset analysis/ledger write FAILED. "
                "Training itself completed successfully and all checkpoints/"
                "metrics were already saved before this point. Re-raising so "
                "the failure is not silently swallowed; rerun analysis offline "
                "via analysis.pipeline.run_post_hoc_onset_analysis once fixed."
            )
            raise
