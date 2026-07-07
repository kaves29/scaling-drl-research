import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from stable_baselines3 import PPO
import time

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

class SparsePPOCriterion:

    def __init__(self, config, seed):
        self.config = config
        self.gradient_history = {}
        self.masks = {"actor": {}, "critic": ""}
        self.seed = seed

    def actor_saliency(self, gradient_history, config):
        """
        Compute importance scores for actor weights.
        INPUT: gradient_history (dict of per-weight gradients over time), config
        OUTPUT: importance score tensor
        """

        importance_scores = {}
        for param_name, gradients in gradient_history.items():
            grad_median = np.median(gradients)
            grad_std = np.std(gradients)

            importance_scores[param_name] = grad_median / (1 + grad_std)
        
        return importance_scores

    def critic_saliency(self, gradient_history, config):
        """
        Compute importance scores for critic weights using median + variance.
        INPUT: gradient_history (dict of per-weight gradients over time), config
        OUTPUT: importance score tensor
        """
        pass


    def mask_by_saliency(self, importance_scores, config):
        """
        Create mask: keep top (1 - sparsity_ratio)% of weights by importance.
        INPUT: importance scores, sparsity ratio (e.g., 0.8 = prune 80%)
        OUTPUT: boolean mask (True = keep, False = prune)
        """
        scores_array = np.array(list(importance_scores.values()))
        importance_score_cutoff = np.nanpercentile(scores_array, (1 - config['dst']['sparsity_ratio']) * 100)
        mask = {}

        for param_name, score in importance_scores.items():
            mask[param_name] = 1 if score > importance_score_cutoff else 0

        return mask
    

    def count_dormant_neurons(self, model, threshold=1e-4):
        """
        Count neurons with near-zero activations in actor/critic.
        INPUT: SB3 model, threshold for "dead"
        OUTPUT: dict with dormant counts for actor and critic
        """
        pass