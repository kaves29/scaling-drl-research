import torch
import math

def compute_action_entropy(agent):
    """measure diversity of the actions the actor makes"""
    action_stdev = torch.exp(agent.actor_logstd)
    entropy_tensor = 0.5 * torch.log(2 * math.pi * math.e * (action_stdev**2))

    return entropy_tensor.sum(dim=1).mean()

def compute_critic_effective_rank(hidden_layer_outputs):
    """measure the linear complexity of critic network via SVD"""
    _, S, _ = torch.linalg.svd(hidden_layer_outputs)
    normalized_S = S / S.sum()
    entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-9))

    return torch.exp(entropy)

def compute_grad_variance(actor_grad_history, critic_grad_history):
    """measure training instability caused by sparsity levels for both actor and critic networks"""
    actor_layer_variances = []
    critic_layer_variances = []

    # Compute actor layer variances across minibatches
    for _, grads in actor_grad_history.items():
        stacked_grads = torch.stack(grads)
        var = torch.var(stacked_grads, dim=0, correction=0)
        actor_layer_variances.append(var.mean())

    # Compute critic layer variances across minibatches
    for _, grads in critic_grad_history.items():
        stacked_grads = torch.stack(grads)
        var = torch.var(stacked_grads, dim=0, correction=0)
        critic_layer_variances.append(var.mean())

    return torch.stack(actor_layer_variances).mean(), torch.stack(critic_layer_variances).mean()

def compute_mask_jaccard(actor_topology_history, critic_topology_history):
    """measure how much topology changed from the previous pruning iteration for both actor and critic networks"""
    # Computing Jaccard for Actor
    actor_old_mask = actor_topology_history["previous_mask"]
    actor_new_mask = actor_topology_history["current_mask"]

    actor_intersection = (actor_old_mask & actor_new_mask).sum()
    actor_union = (actor_old_mask | actor_new_mask).sum()

    # Computing Jaccard for Critic
    critic_old_mask = critic_topology_history["previous_mask"]
    critic_new_mask = critic_topology_history["current_mask"]

    critic_intersection = (critic_old_mask & critic_new_mask).sum()
    critic_union = (critic_old_mask | critic_new_mask).sum()

    actor_jacc = (1 - (actor_intersection / actor_union)) if actor_union > 0 else torch.tensor(0.0)
    critic_jacc = (1 - (critic_intersection / critic_union)) if critic_union > 0 else torch.tensor(0.0)

    return actor_jacc, critic_jacc

def update_topology_history(agent, pruning_engine):
    """helper to update topoly history dictionaries for jaccard calculations"""
    actor_layer_masks = []
    critic_layer_masks = []

    if isinstance(pruning_engine, dict):
        for name, agent_param in agent.named_parameters():
            for param_name, mask in pruning_engine.items():
                if param_name == name:
                    if "actor_mean" in name and name.endswith(".weight"):
                            if hasattr(mask, 'to_dense'):
                                actor_layer_masks.append(mask.to_dense().flatten())
                            else:
                                actor_layer_masks.append(mask.flatten())
                    if "critic" in name and name.endswith(".weight"):
                        if hasattr(mask, 'to_dense'):
                            critic_layer_masks.append(mask.to_dense().flatten())
                        else:
                            critic_layer_masks.append(mask.flatten())
                    break
    else:
        for hook in pruning_engine.backward_hook_objects:
            if hook is None:
                continue
                
            param = hook.param
            mask = hook.mask
            for name, agent_param in agent.named_parameters():
                if agent_param.data_ptr() == param.data_ptr():
                    if "actor_mean" in name and name.endswith(".weight"):
                        if hasattr(mask, 'to_dense'):
                            actor_layer_masks.append(mask.to_dense().flatten())
                        else:
                            actor_layer_masks.append(mask.flatten())
                    if "critic" in name and name.endswith(".weight"):
                        if hasattr(mask, 'to_dense'):
                            critic_layer_masks.append(mask.to_dense().flatten())
                        else:
                            critic_layer_masks.append(mask.flatten())
                    break

    new_prev_actor_mask = agent.actor_topology_hist["current_mask"].clone()
    new_prev_critic_mask = agent.critic_topology_hist["current_mask"].clone()
    agent.actor_topology_hist["previous_mask"] = new_prev_actor_mask
    agent.critic_topology_hist["previous_mask"] = new_prev_critic_mask
    agent.actor_topology_hist["current_mask"] = torch.cat(actor_layer_masks)
    agent.critic_topology_hist["current_mask"] = torch.cat(critic_layer_masks)

def compute_layer_gradient_norms(agent):
    """measure if gradients are flowing through each layer"""
    layer_norms = {}
    for name, param in agent.named_parameters():
        if param.grad is not None and name.endswith(".weight"):
            layer_norms[name] = param.grad.norm().item()
    return layer_norms

def compute_dead_neurons(agent):
    """measure % of neurons with zero activations"""
    dead_percentages = {}
    for name, output in agent.activations.items():
        dead_pct = (output == 0).float().mean().item()
        dead_percentages[name] = dead_pct
    return dead_percentages