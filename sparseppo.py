import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from stable_baselines3 import PPO
import time

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

class SparsePPOCriterion:

    def __init__(self, config, seed):
        self.config = config
        self.gradient_history = {
            "actor": defaultdict(lambda: deque(maxlen=config["dst"]["gradient_history_window"])),
            "critic": defaultdict(lambda: deque(maxlen=config["dst"]["gradient_history_window"]))
        }
        self.masks = {"actor": {}, "critic": {}}
        self.param_shapes = {}
        self.seed = seed

    def actor_saliency(self, gradient_history, config):
        """
        Compute importance scores for actor weights.
        INPUT: gradient_history (dict of per-weight gradients over time), config
        OUTPUT: importance score tensor
        """

        actor_importance_scores = {}
        for param_name, gradients in gradient_history.items():
            history = torch.stack(list(gradients))
            grad_median = torch.median(torch.abs(history), dim=0).values
            grad_var = torch.var(history, dim=0)

            actor_importance_scores[param_name] = grad_median / (1 + grad_var)
        
        return actor_importance_scores

    def critic_saliency(self, gradient_history, config):
        """
        Compute importance scores for critic weights using median + variance.
        INPUT: gradient_history (dict of per-weight gradients over time), config
        OUTPUT: importance score tensor
        """
        pass


    def mask_by_saliency(self, actor_importance_scores, config): # add back critic_importance_scores
        """
        Create mask: keep top (1 - sparsity_ratio)% of weights by importance.
        INPUT: importance scores, sparsity ratio (e.g., 0.8 = prune 80%)
        OUTPUT: boolean mask (True = keep, False = prune)
        """
        """actor_scores_array = np.array(list(actor_importance_scores.values()))
        actor_importance_score_cutoff = np.nanpercentile(actor_scores_array, (1 - config['dst']['sparsity_ratio']) * 100)
        actor_mask = {}

        for param_name, score in actor_importance_scores.items():
            mask_value = 1 if score > actor_importance_score_cutoff else 0
            actor_mask[param_name] = np.full(self.param_shapes[param_name], mask_value)

        """"""critic_scores_array = np.array(list(critic_importance_scores.values()))
        critic_importance_score_cutoff = np.nanpercentile(critic_scores_array, (1 - config['dst']['sparsity_ratio']) * 100)
        critic_mask = {}

        for param_name, score in critic_importance_scores.items():
            mask_value = 1 if score > critic_importance_score_cutoff else 0
            critic_mask[param_name] = np.full(self.param_shapes[param_name], mask_value)""""""

        return actor_mask # remember to add critic_mask back""" 

        actor_prune_ratio = config["dst"]["actor_prune_ratio"]

        all_actor_scores = torch.cat(
            [   
                actor_score.flatten()
                for actor_score in actor_importance_scores.values()
            ]
        )

        actor_threshold = torch.quantile(all_actor_scores, actor_prune_ratio)

        actor_mask = {}
        for param_name, importance_scores in actor_importance_scores.items():
            if "bias" in param_name or "logstd" in param_name:
                continue

            mask = (importance_scores >= actor_threshold).float()

            actor_mask[param_name] = mask
        
        return actor_mask
    

    def count_dormant_neurons(self, model, threshold=1e-4):
        """
        Count neurons with near-zero activations in actor/critic.
        INPUT: SB3 model, threshold for "dead"
        OUTPUT: dict with dormant counts for actor and critic
        """
        pass