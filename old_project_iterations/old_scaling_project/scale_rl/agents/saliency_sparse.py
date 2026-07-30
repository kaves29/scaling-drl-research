# based on jaxpruner: https://github.com/google-research/jaxpruner
from typing import Callable, Mapping, Optional, Union, Tuple
import logging
import chex
import flax
import jax
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
FilterFnType = Callable[[Tuple[str], chex.Array], bool]
CustomSparsityMapType = Mapping[Tuple[str], float]
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import freeze, unfreeze
KERNEL_FILTER_FN = lambda key, param: key[-1] == 'kernel'
NOT_DIM_ONE_FILTER_FN = lambda key, param: param.ndim > 1

def get_var_shape_dict(params):
    # Flatten the nested parameter dictionary
    flat_params = flatten_dict(params, sep='/')
    var_shape_dict = {}
    for param_path, param_value in flat_params.items():
        # param_path might be a tuple like ('Dense_0', 'kernel') -> convert it to a string
        if isinstance(param_path, tuple):
            var_name = '/'.join(param_path)
        else:
            var_name = param_path
        var_shape_dict[var_name] = param_value.shape
    return var_shape_dict

def get_prunable_keys(params):
    """Dynamically find all prunable weight matrices in params"""
    prunable = []
    if 'encoder' in params:
        encoder = params['encoder']
        # First linear layer before residual blocks
        if 'Dense_0' in encoder and 'kernel' in encoder['Dense_0']:
            prunable.append(('encoder', 'Dense_0', 'kernel'))
        # Residual blocks
        for key in encoder:
            if key.startswith('ResidualBlock_'):
                block = encoder[key]
                # Get linear layers 1 & 2 in each residual block
                if 'Dense_0' in block and 'kernel' in block['Dense_0']:
                    prunable.append(('encoder', key, 'Dense_0', 'kernel'))
                if 'Dense_1' in block and 'kernel' in block['Dense_1']:
                    prunable.append(('encoder', key, 'Dense_1', 'kernel'))
    if 'predictor' in params:
        predictor = params['predictor']
        if 'Dense_0' in predictor and 'kernel' in predictor['Dense_0']:
            prunable.append(('predictor', 'Dense_0', 'kernel'))
        # handles NormalTanhPolicy (actor) which may have different structure
        for key in predictor:
            if key.startswith('Dense_') and key != 'Dense_0':
                if 'kernel' in predictor[key]:
                    prunable.append(('predictor', key, 'kernel'))
    return prunable

def r_fn(abs_params_pytree, module_name, module_def, cfg):
    """obs_ones = jnp.ones((1, cfg.obs_dims), dtype=jnp.float64)
    action_ones = jnp.ones((1, cfg.action_dims), dtype=jnp.float64)"""
    rng = jax.random.PRNGKey(cfg.seed)
    obs_ones = jax.random.normal(rng, shape=(1, cfg.obs_dims), dtype=jnp.float64) * 0.1
    action_ones = jax.random.normal(rng, shape=(1, cfg.action_dims), dtype=jnp.float64) * 0.1
    encoder_out = module_def.apply(
        {'params': abs_params_pytree},
        obs_ones if module_name == "actor" else jnp.concatenate((obs_ones, action_ones), axis=1),
        method=module_def.encode
    )
    print(f"{module_name} post-encoder output (first 10):", encoder_out.flatten()[:10])
    print(f"{module_name} post-encoder all zero?:", jnp.all(encoder_out == 0))

    if module_name == "actor":
        dist = module_def.apply({'params': abs_params_pytree}, observations=obs_ones)
        pre_squash_mean = dist.distribution.mean()
        return jnp.sum(pre_squash_mean)
    else:
        q = module_def.apply({'params': abs_params_pytree}, observations=obs_ones, actions=action_ones)
        return jnp.sum(q)

def get_saliency_scores(prunable_keys, abs_params_pytree, module_name, module_def, cfg):
    grads = (jax.grad(r_fn, argnums=0)(abs_params_pytree, module_name, module_def, cfg)) 
    all_saliency_scores = {}
    for prunable_tuple in prunable_keys:
        g = grads
        p = abs_params_pytree
        for key in prunable_tuple:
            g = g[key]
            p = p[key]
        all_saliency_scores[prunable_tuple] = (g * p).flatten()
    flat_scores = jnp.concatenate([v for v in all_saliency_scores.values()])
    print(module_name, "min abs score:", jnp.min(jnp.abs(flat_scores)),
          "num exactly zero:", jnp.sum(flat_scores == 0), "out of", flat_scores.size)
    for prunable_tuple in prunable_keys:
        s = all_saliency_scores[prunable_tuple]
        print(f"  {prunable_tuple}: {jnp.sum(s == 0)}/{s.size} zero")

    return all_saliency_scores

def apply_mask_to_params(params, masks):
    """Apply final mask to prunable weights in params"""
    for prunable_object in masks.keys():
        mask = masks[prunable_object]
        value = params
        for key in prunable_object:
            value = value[key]
        weight = value
        mask_reshaped = mask.reshape(weight.shape)
        masked_weight = weight * mask_reshaped
        params = _set_nested_value(params, prunable_object, masked_weight)
    
    return params

def _set_nested_value(d, keys, value):
    """Set value in nested dict"""
    if len(keys) == 1:
        return {**d, keys[0]: value}
    key = keys[0]
    return {**d, key: _set_nested_value(d[key], keys[1:], value)}

def calculate_sparsity(prunable_keys, params, module_name, cfg):
    total_prunable_weights = 0
    pruned_weights = 0
    for prunable_object in prunable_keys:
        value = params
        for key in prunable_object:
            value = value[key]
        total_prunable_weights += value.size
        pruned_weights += jnp.sum(value == 0)
    achieved = pruned_weights / total_prunable_weights
    target = getattr(cfg, f"{module_name}_sparsity")
    result = f"{module_name} Sparsity: {achieved:.4f} | {module_name} Target: {target}"
    with open("sparsity_log.txt", "a") as f:
        f.write(result + "\n")

    return result



def create_saliency_mask(params, module_name, module_def, cfg):
    params_unfrozen = unfreeze(params)
    prunable_weight_keys = get_prunable_keys(params_unfrozen)
    abs_params_pytree = jax.tree_util.tree_map(lambda x: jnp.abs(x).astype(jnp.float64), params)
    compression_ratio = 1 / (1 - getattr(cfg, f"{module_name}_sparsity"))
    new_masks = {}
    for prune_iteration in range(cfg.num_pruning_iterations):
        saliency_scores = get_saliency_scores(prunable_weight_keys, abs_params_pytree, module_name, module_def, cfg)
        all_scores_flattened = jnp.concatenate([v.flatten() for v in saliency_scores.values()])
        prune_threshold = jnp.percentile(
            all_scores_flattened,
            100 * (1 - compression_ratio ** (-(prune_iteration + 1) / cfg.num_pruning_iterations)) # changed from 100 * (1 - compression_ratio ** ((-prune_iteration) / cfg.num_pruning_iterations))
        )
        for prunable_object, scores in saliency_scores.items():
            new_masks[prunable_object] = (scores > prune_threshold).astype(jnp.int32)
        for prunable_object, mask in new_masks.items():
            value = abs_params_pytree
            for key in prunable_object:
                value = value[key]
            pruned_value = value * mask.reshape(value.shape)
            abs_params_pytree = _set_nested_value(abs_params_pytree, prunable_object, pruned_value)
    flat_params = flatten_dict(params_unfrozen) 
    full_mask_flat = {}
    for key_tuple, value in flat_params.items():
        if key_tuple in new_masks:
            full_mask_flat[key_tuple] = new_masks[key_tuple].reshape(value.shape)
        else:
            full_mask_flat[key_tuple] = jnp.ones_like(value)
    full_mask = unflatten_dict(full_mask_flat)
    print(calculate_sparsity(prunable_weight_keys, full_mask, module_name, cfg))

    return freeze(full_mask)


        
