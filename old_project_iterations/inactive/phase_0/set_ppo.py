import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import yaml

from set_utils import *
from pathology_computation_utils import *

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

@dataclass
class Args:
    exp_name: str = "phase_0_validation"
    """the name of this experiment"""
    seed: int = 9
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "sparse-ppo-drl-research"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `models/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = config["phase_0"]["exp"]["num_envs"]
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    zeta: float = 0.3
    """hyperparameter to compute how many weights are pruned and grow every pruning cycle"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

args = tyro.cli(Args)

def make_env(env_id, idx, capture_video, run_name, gamma):
    capture_video = False
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, func=lambda obs: np.clip(obs, -10, 10), observation_space=None)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.actor_topology_hist = {"previous_mask": torch.tensor([]), "current_mask": torch.tensor([])}
        self.critic_topology_hist = {"previous_mask": torch.tensor([]), "current_mask": torch.tensor([])} 
        self.activations = {}

        self.critic[0].register_forward_hook(self.get_hook('critic_h1'))
        self.critic[2].register_forward_hook(self.get_hook('critic_h2'))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_hook(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
if __name__ == "__main__":
    for env_idx in range(len(config["phase_0"]["envs"]["env_list"])):
        env_name = config["phase_0"]["envs"]["env_list"][env_idx]
        print(f"STARTING ENV {env_name}")

        for sparsity_level_idx in range(len(config["phase_0"]["sparsity"]["dense_allocation"])):
            sparsity_level = config["phase_0"]["sparsity"]["dense_allocation"][sparsity_level_idx]
            print(f"STARTING PRUNING LEVEL {sparsity_level} IN ENV {env_name}")
            
            for env_seed in range(config["phase_0"]["exp"]["num_seeds"]):
                args.env_id = config["phase_0"]["envs"][env_name]["env_id"]
                args.total_timesteps = config["phase_0"]["envs"][env_name]["total_timesteps"]
                args.batch_size = int(args.num_envs * args.num_steps)
                args.minibatch_size = int(args.batch_size // args.num_minibatches)
                args.num_iterations = args.total_timesteps // args.batch_size
                args.compute_iteration = max(1, int(args.num_iterations // 25))

                run_name = f"SET__{args.env_id}__{args.exp_name}__{sparsity_level}"
                if args.track:
                    import wandb

                    wandb.init(
                        project=args.wandb_project_name,
                        entity=args.wandb_entity,
                        sync_tensorboard=True,
                        config=vars(args),
                        name=run_name,
                        monitor_gym=True,
                        save_code=True,
                        group="phase_0/set",
                        job_type=env_name,
                        resume="allow"
                    )
                writer = SummaryWriter(f"runs/{run_name}")
                writer.add_text(
                    "hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
                )

                # TRY NOT TO MODIFY: seeding
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.backends.cudnn.deterministic = args.torch_deterministic

                device = torch.device("cpu")

                # env setup
                envs = gym.vector.SyncVectorEnv(
                    [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
                )
                assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

                agent = Agent(envs).to(device)
                optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
                
                sparse_engine = SparseManager(agent, sparsity_level)
                sparse_engine.apply_mask(agent)

                # ALGO Logic: Storage setup
                obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
                actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
                logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
                rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
                dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
                values = torch.zeros((args.num_steps, args.num_envs)).to(device)

                # TRY NOT TO MODIFY: start the game
                global_step = 0
                start_time = time.time()
                next_obs, _ = envs.reset(seed=args.seed)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)

                actor_layer_masks = [torch.ones_like(p).bool().flatten() for name, p in agent.named_parameters() 
                     if "actor_mean" in name and name.endswith(".weight")]
                critic_layer_masks = [torch.ones_like(p).bool().flatten() for name, p in agent.named_parameters() 
                                    if "critic" in name and name.endswith(".weight")]
                
                initial_stacked_actor = torch.cat(actor_layer_masks)
                initial_stacked_critic = torch.cat(critic_layer_masks)
                agent.actor_topology_hist["previous_mask"] = initial_stacked_actor
                agent.actor_topology_hist["current_mask"] = initial_stacked_actor
                agent.critic_topology_hist["previous_mask"] = initial_stacked_critic
                agent.critic_topology_hist["current_mask"] = initial_stacked_critic

                for iteration in range(1, args.num_iterations + 1):

                    for step in range(0, args.num_steps):
                        global_step += args.num_envs
                        obs[step] = next_obs
                        dones[step] = next_done

                        # ALGO LOGIC: action logic
                        with torch.no_grad():
                            action, logprob, _, value = agent.get_action_and_value(next_obs)
                            values[step] = value.flatten()
                        actions[step] = action
                        logprobs[step] = logprob

                        # TRY NOT TO MODIFY: execute the game and log data.
                        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                        next_done = np.logical_or(terminations, truncations)
                        rewards[step] = torch.tensor(reward).to(device).view(-1)
                        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                        if "final_info" in infos:
                            for info in infos["final_info"]:
                                if info and "episode" in info:
                                    r = info['episode']['r'].item() if hasattr(info['episode']['r'], 'item') else info['episode']['r']
                                    l = info['episode']['l'].item() if hasattr(info['episode']['l'], 'item') else info['episode']['l']
                                    print(f"Success! global_step={global_step}, episodic_return={float(r)}")
                                    writer.add_scalar(f"phase_0/set/episodic_return/{env_name}", float(r), global_step)
                                    writer.add_scalar(f"phase_0/set/episodic_length/{env_name}", float(l), global_step)
                        
                        elif isinstance(infos, dict) and "episode" in infos:
                            # Find which environments in the batch just finished
                            for env_idx in range(len(infos["episode"]["r"])):
                                if infos["_episode"][env_idx]: 
                                    r = infos["episode"]["r"][env_idx]
                                    l = infos["episode"]["l"][env_idx]
                                    print(f"--> SUCCESS (FALLBACK)! global_step={global_step}, episodic_return={float(r)}")
                                    writer.add_scalar(f"phase_0/set/episodic_return/{env_name}", float(r), global_step)
                                    writer.add_scalar(f"phase_0/set/episodic_length/{env_name}", float(l), global_step)

                    # SET Prune & Regrowth
                    sparse_engine.evolve(agent, args.zeta)

                    # bootstrap value if not done
                    with torch.no_grad():
                        next_value = agent.get_value(next_obs).reshape(1, -1)
                        advantages = torch.zeros_like(rewards).to(device)
                        lastgaelam = 0

                        for t in reversed(range(args.num_steps)):
                            if t == args.num_steps - 1:
                                nextnonterminal = 1.0 - next_done
                                nextvalues = next_value
                            else:
                                nextnonterminal = 1.0 - dones[t + 1]
                                nextvalues = values[t + 1]
                            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        returns = advantages + values

                    # flatten the batch
                    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                    b_logprobs = logprobs.reshape(-1)
                    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                    b_advantages = advantages.reshape(-1)
                    b_returns = returns.reshape(-1)
                    b_values = values.reshape(-1)

                    # Optimizing the policy and value network
                    b_inds = np.arange(args.batch_size)
                    clipfracs = []

                    actor_grad_history = {name: [] for name, param in agent.named_parameters() if param.requires_grad and "actor_mean" in name and name.endswith(".weight")}
                    critic_grad_history = {name: [] for name, param in agent.named_parameters() if param.requires_grad and "critic" in name and name.endswith(".weight")}

                    for epoch in range(args.update_epochs):
                        np.random.shuffle(b_inds)
                        for start in range(0, args.batch_size, args.minibatch_size):
                            end = start + args.minibatch_size
                            mb_inds = b_inds[start:end]

                            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()

                            if iteration % args.compute_iteration == 0 and epoch == 0 and start == 0:
                                rank_h1 = compute_critic_effective_rank(agent.activations['critic_h1'])
                                rank_h2 = compute_critic_effective_rank(agent.activations['critic_h2'])
                                writer.add_scalar(f"phase_0/set/critic_h1_rank/{env_name}", rank_h1.item(), global_step)
                                writer.add_scalar(f"phase_0/set/critic_h2_rank/{env_name}", rank_h2.item(), global_step)

                            with torch.no_grad():
                                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                            mb_advantages = b_advantages[mb_inds]
                            if args.norm_adv:
                                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                            # Policy loss
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            # Value loss
                            newvalue = newvalue.view(-1)
                            if args.clip_vloss:
                                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                                v_clipped = b_values[mb_inds] + torch.clamp(
                                    newvalue - b_values[mb_inds],
                                    -args.clip_coef,
                                    args.clip_coef,
                                )
                                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                                v_loss = 0.5 * v_loss_max.mean()
                            else:
                                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                            entropy_loss = entropy.mean()
                            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                            if iteration % args.compute_iteration == 0:
                                for name, param in agent.named_parameters():
                                    if param.grad is not None and "actor_mean" in name and name.endswith(".weight"):
                                        actor_grad_history[name].append(param.grad.detach().clone())
                                    if param.grad is not None and "critic" in name and name.endswith(".weight"):
                                        critic_grad_history[name].append(param.grad.detach().clone())

                            optimizer.step()

                            sparse_engine.apply_mask(agent)

                        if args.target_kl is not None and approx_kl > args.target_kl:
                            break
                    if iteration % args.compute_iteration == 0:
                        # Compute and log gradient variance across all minibatches collected
                        actor_grad_var, critic_grad_var = compute_grad_variance(actor_grad_history, critic_grad_history)
                        writer.add_scalar(f"phase_0/set/actor_gradient_variance/{env_name}", actor_grad_var, global_step)
                        writer.add_scalar(f"phase_0/set/critic_gradient_variance/{env_name}", critic_grad_var, global_step)

                        # Extract current masks to check for pruning updates
                        curr_actor_layers = []
                        curr_critic_layers = []
                        for name, param in agent.named_parameters():
                            if param.requires_grad:
                                mask = (param.data != 0)
                                if "actor_mean" in name and name.endswith(".weight"):
                                    curr_actor_layers.append(mask.flatten())
                                elif "critic" in name and name.endswith(".weight"):
                                    curr_critic_layers.append(mask.flatten())
                        if curr_actor_layers:
                            latest_actor_stacked = torch.cat(curr_actor_layers).bool()
                            latest_critic_stacked = torch.cat(curr_critic_layers).bool()

                            has_pruned = not torch.equal(latest_actor_stacked, agent.actor_topology_hist["current_mask"])
                            if has_pruned:
                                update_topology_history(agent, sparse_engine.masks)
                                actor_jaccard, critic_jaccard = compute_mask_jaccard(agent.actor_topology_hist, agent.critic_topology_hist)
                                writer.add_scalar(f"phase_0/set/actor_jaccard/{env_name}", actor_jaccard, global_step)
                                writer.add_scalar(f"phase_0/set/critic_jaccard/{env_name}", critic_jaccard, global_step)
                        
                        layer_norms = compute_layer_gradient_norms(agent)
                        for layer_name, norm in layer_norms.items():
                            writer.add_scalar(f"phase_0/set/layer_grad_norm/{layer_name}/{env_name}", norm, global_step)

                        dead_neurons = compute_dead_neurons(agent)
                        for layer_name, dead_pct in dead_neurons.items():
                            writer.add_scalar(f"phase_0/set/dead_neuron_pct/{layer_name}/{env_name}", dead_pct, global_step)

                    current_entropy = compute_action_entropy(agent)
                    writer.add_scalar(f"phase_0/set/actor_entropy/{env_name}", current_entropy.item(), global_step)
                    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                    # TRY NOT TO MODIFY: record rewards for plotting purposes
                    writer.add_scalar(f"charts/set/learning_rate/{env_name}", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar(f"losses/set/value_loss/{env_name}", v_loss.item(), global_step)
                    writer.add_scalar(f"losses/set/policy_loss/{env_name}", pg_loss.item(), global_step)
                    writer.add_scalar(f"losses/set/entropy/{env_name}", entropy_loss.item(), global_step)
                    writer.add_scalar(f"losses/set/old_approx_kl/{env_name}", old_approx_kl.item(), global_step)
                    writer.add_scalar(f"losses/set/approx_kl/{env_name}", approx_kl.item(), global_step)
                    writer.add_scalar(f"losses/set/clipfrac/{env_name}", np.mean(clipfracs), global_step)
                    writer.add_scalar(f"losses/set/explained_variance/{env_name}", explained_var, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(f"charts/set/SPS/{env_name}", int(global_step / (time.time() - start_time)), global_step)

                if args.save_model:
                    model_path = f"models/phase_0/set/{env_name}/{run_name}.pt"
                    model_dir = os.path.dirname(model_path)
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(agent.state_dict(), model_path)
                    print(f"model saved to {model_path}")

                envs.close()
                writer.close()