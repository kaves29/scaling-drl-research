import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from stable_baselines3 import PPO
import copy
import yaml

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

class SparsePPOCriterion:

    def __init__(self, config, seed):
        self.config = config
        self.masks = {}
        self.saved_gradients = {}
        self.saliency_scores = {}
        self.device = torch.device("cpu")
        self.seed = seed
    
    def initialize_module_sparsity(self, agent, target_modules, sparsity_ratio):
        for name, param in agent.named_parameters():
            if any(module_key in name for module_key in target_modules) and name.endswith(".weight"):
                random_matrix = torch.rand_like(param, device=self.device)

                num_active = int(param.numel() * (1.0 - sparsity_ratio))
                threshold = torch.kthvalue(random_matrix.flatten(), param.numel() - num_active + 1).values
                self.masks[name] = (random_matrix >= threshold).float()

                self.saliency_scores[name] = torch.zeros_like(param, device=self.device)
                self.saved_gradients[name] = torch.zeros_like(param, device=self.device)

                with torch.no_grad():
                    param.mul_(self.masks[name])


    def actor_saliency(self, agent, config):
        """
        Compute importance scores for actor weights.
        INPUT: gradient_history (dict of per-weight gradients over time), config
        OUTPUT: importance score tensor
        """

        named_params = dict(agent.named_parameters())

        for name in self.masks.keys():
            if name not in named_params:
                continue

            weights = named_params[name]
            gradients = self.saved_gradients[name]
            mask = self.masks[name]

            abs_w = torch.abs(weights)
            abs_g = torch.abs(gradients)

            relu_safeguard = (weights < 0) & (abs_w > abs_g)

            dom_score_numerator = abs_w - abs_g
            dom_score_denominator = abs_w + abs_g + config["dst"]["actor_eplison"]
            xor_boost = 1 + torch.pow(dom_score_numerator / dom_score_denominator, 2)
            other_weights_branch = (abs_w * abs_g) * xor_boost

            raw_scores = torch.where(relu_safeguard, (abs_w * abs_g), other_weights_branch)
            self.saliency_scores[name] = raw_scores * mask
            

    def critic_saliency(self, gradient_history, config):
        """
        Compute importance scores for critic weights using median + variance.
        INPUT: gradient_history (dict of per-weight gradients over time), config
        OUTPUT: importance score tensor
        """
        pass
    
    def update_sparsity_masks(self, agent, config):
        self.actor_saliency(agent, config)
        all_actor_active_scores = []

        for name in self.masks.keys():
            actor_scores = self.saliency_scores[name]
            actor_mask = self.masks[name]

            active_actor_scores_in_layer = actor_scores[actor_mask==1]
            all_actor_active_scores.append(active_actor_scores_in_layer)

        global_active_actor_pool = torch.cat(all_actor_active_scores)

        num_actor_to_mask = int(len(global_active_actor_pool) * config["dst"]["regrow_ratio"])
        global_actor_mask_threshold = torch.kthvalue(global_active_actor_pool, 
                                                    num_actor_to_mask).values

        all_actor_inactive_scores = []

        for name in self.masks.keys():
            actor_gradients = self.saved_gradients[name]
            actor_mask = self.masks[name]

            inactive_actor_grads_in_layer = torch.abs(actor_gradients)[actor_mask == 0.0]
            all_actor_inactive_scores.append(inactive_actor_grads_in_layer)

        global_actor_inactive_pool = torch.cat(all_actor_inactive_scores)
        num_actor_to_regrow = num_actor_to_mask

        global_actor_regrow_threshold = torch.kthvalue(global_actor_inactive_pool, 
                                                        len(global_actor_inactive_pool) - num_actor_to_regrow + 1).values
        

        with torch.no_grad():
            for name in self.masks.keys():
                actor_scores = self.saliency_scores[name]
                actor_abs_grads = torch.abs(self.saved_gradients[name])
                actor_mask = self.masks[name]
                
                # Global Actor Mask Condition
                actor_mask_condition = (actor_scores <= global_actor_mask_threshold) & (actor_mask == 1.0)
                actor_mask[actor_mask_condition] = 0.0
                
                # Global Actor Regrow Condition
                actor_regrow_condition = (actor_abs_grads >= global_actor_regrow_threshold) & (actor_mask == 0.0)
                actor_mask[actor_regrow_condition] = 1.0
                
                # Update Agent's Actor Weights
                named_params = dict(agent.named_parameters())
                named_params[name].data *= actor_mask
            


    def count_dormant_neurons(self, model, threshold=1e-4):
        """
        Count neurons with near-zero activations in actor/critic.
        INPUT: SB3 model, threshold for "dead"
        OUTPUT: dict with dormant counts for actor and critic
        """
        pass

    def calculate_sparsity(self, agent, module):
        total_weights = 0
        total_inactive_weights = 0

        for name, param in agent.named_parameters():
            if module in name and name.endswith(".weight"):
                total_weights += self.masks[name].numel()
                total_inactive_weights += (self.masks[name] == 0).sum().item()

        return total_inactive_weights / total_weights