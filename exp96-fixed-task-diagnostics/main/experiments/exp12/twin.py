"""Twin critics (clipped double Q, HumanoidBench only; amendment (z)) in the Exp 1/2 machinery.

A twin critic is SimBa's vmapped pair: every parameter has a leading axis of 2, network k is slice k and
is an ordinary SACCritic (or InjectedSACCritic after injection). The probe treats each network as its own
critic, against its own fresh copy, on the shared pool, targets and minibatches; the per-round
plasticity loss is the mean of the two networks' L, and both are recorded.
"""

from typing import Dict, List, Tuple

import jax
import numpy as np

TWIN = "VmapSACCritic_0"
INJECTED_TWIN = "VmapInjectedSACCritic_0"
NUM_QS = 2


def twin_key(params):
    for key in (TWIN, INJECTED_TWIN):
        if key in params:
            return key
    return None


def is_twin(params) -> bool:
    return twin_key(params) is not None


def network_params(params, k: int):
    return jax.tree_util.tree_map(lambda x: x[k], params[twin_key(params)])


def network_def(twin_def):
    """The single-network module of a twin network_def (or a single one, unchanged)."""
    from scale_rl.agents.sac.sac_network import SACClippedDoubleCritic, SACCritic

    from experiments.exp12.injection import InjectedClippedDoubleCritic, InjectedSACCritic

    if isinstance(twin_def, SACClippedDoubleCritic):
        return SACCritic(twin_def.block_type, twin_def.num_blocks, twin_def.hidden_dim, twin_def.dtype)
    if isinstance(twin_def, InjectedClippedDoubleCritic):
        return InjectedSACCritic(twin_def.num_blocks, twin_def.hidden_dim, twin_def.head_blocks, twin_def.dtype)
    return twin_def


def expand(critics: Dict[str, tuple], injected_tx) -> Tuple[Dict[str, tuple], List[str]]:
    """name -> (def, params[, tx]) with twin params becomes name_q1, name_q2 -> one network each."""
    out, twins = {}, []
    for name, critic in critics.items():
        if not is_twin(critic[1]):
            out[name] = critic
            continue
        twins.append(name)
        single = network_def(critic[0])
        injected = twin_key(critic[1]) == INJECTED_TWIN
        for k in range(NUM_QS):
            view = (single, network_params(critic[1], k))
            out[f"{name}_q{k + 1}"] = (*view, injected_tx) if injected else view
    return out, twins


def combine(result: Dict[str, Dict], names: List[str]) -> Dict[str, Dict]:
    """Adds, for each twin name, the per-round mean of its networks' score and final loss (the pool and
    targets are shared, so b is common). L = P(fresh) - P(current) of these means is the mean of the two L."""
    out = dict(result)
    for name in names:
        parts = [result[f"{name}_q{k + 1}"] for k in range(NUM_QS)]
        out[name] = {"score": np.mean([p["score"] for p in parts], axis=0),
                     "final_loss": np.mean([p["final_loss"] for p in parts], axis=0), "b": parts[0]["b"]}
    return out
