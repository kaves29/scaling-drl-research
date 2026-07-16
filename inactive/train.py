from sparseppo import *
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import yaml
import wandb
from wandb.integration.sb3 import WandbCallback
import os

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

LOG_DIR = f"logs/validation_actor"
MODEL_DIR = f"models/validation_actor"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

wandb.tensorboard.patch(root_logdir="logs/")

def main():
    for env_name, env_cfg in config["environments"].items():
        for seed in range(config['experiment']['num_seeds']):
            set_seed(seed)

            run = wandb.init(
                project="sparseppo_drl_project",
                group="Validation_Actor_Criterion",
                name=f"{config['algorithm']['actor_saliency']}_seed{seed}_{env_name}",
                sync_tensorboard=True, 
                save_code=True,
                config=config
            )

            env = make_vec_env(env_name, n_envs=config['num_envs'], seed=seed)
            model = PPO("MlpPolicy", 
                        env,
                        verbose=0,
                        tensorboard_log=f"{LOG_DIR}/{config['algorithm']['actor_saliency']}_seed{seed}_{env_name}"
                        )
            
            try:

                model.learn(total_timesteps=env_cfg["total_timesteps"], callback=WandbCallback())
                model.save(f"{MODEL_DIR}/{config['algorithm']['actor_saliency']}_seed{seed}_{env_name}")
                wandb.log({
                    "seed": seed,
                    "env": env_name,
                    "criterion_type": config['algorithm']['actor_saliency'],
                    "final_reward": model.ep_info_buffer.mean() if len(model.ep_info_buffer) > 0 else 0
                })
                
            except Exception as e:

                print(f"CRITICAL: Code failure executing {config['algorithm']['actor_saliency']}_seed{seed}_{env_name}")
                print(f"Error: {e}")

            finally:

                env.close()
                run.finish()

            

if __name__ == "__main__":
    main()