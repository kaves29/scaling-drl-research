import os
os.environ['PYOPENGL_PLATFORM'] = 'glfw'
os.environ['JAX_PLATFORMS'] = 'mps,cpu'
import argparse
import random
import hydra
import pandas as pd
import numpy as np
import omegaconf
import tqdm
from dotmap import DotMap
from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from scale_rl.common import WandbTrainerLogger
from scale_rl.envs import create_envs
from scale_rl.evaluation import evaluate, record_video
import jax
import pickle
from pathlib import Path
jax.config.update("jax_enable_x64", False)

def run(args):
    ###############################
    # configs
    ###############################
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    def eval_resolver(s: str):
        return eval(s)
    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver)
    omegaconf.OmegaConf.resolve(cfg)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

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
    os.environ["WANDB_MODE"] = "offline"
    LOGS_DIR = "/Users/shouryakaveti/VS_Projects/sparse-ppo-drl-research/SparseNetwork4DRL/logs"
    os.makedirs(LOGS_DIR, exist_ok=True)
    run_name = f"{cfg.env_name}_CD{cfg.agent.critic_num_blocks}_CW{cfg.agent.critic_hidden_dim}_AD{cfg.agent.actor_num_blocks}_AW{cfg.agent.actor_hidden_dim}_seed{cfg.seed}"
    
    csv_path = os.path.join(LOGS_DIR, f"{run_name}.csv")
    if checkpoint_dir and os.path.exists(csv_path):
        local_metrics_cache = pd.read_csv(csv_path).to_dict("records")
    else:
        local_metrics_cache = []

    _total_params, params_str_total, params_str_actor, params_str_critic= agent.get_num_parameters()
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.update({'num_params':params_str_total})
    cfg.update({'actor_num_params':params_str_actor})
    cfg.update({'critic_num_params':params_str_critic})
    omegaconf.OmegaConf.set_struct(cfg, True)
    
    logger = WandbTrainerLogger(cfg)
    
    # initial evaluation
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
    logger.update_metric(**eval_info)
    logger.log_metric(step=0)
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
        if buffer.can_sample() is False:
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
            
            logger.log_metric(step=env_step)
            logger.reset()
            
    df = pd.DataFrame(local_metrics_cache)
    df.to_csv(csv_path, index=False)
    
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=100_000)
    parser.add_argument("--checkpoint_start_frac", type=float, default=0.5)
    args = parser.parse_args()
    print("RUNNING SUCCESSFULLY, STANDBY!")
    run(vars(args))
