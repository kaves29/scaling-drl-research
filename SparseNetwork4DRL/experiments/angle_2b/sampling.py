"""Deterministic state-batch construction for Angle 2B's frozen
counterfactual gradient measurements.

Per explicit direction: 60 states total per analysis - 30 sampled from one
agent's probe-capture states, 30 from the other's - reused unchanged across
every gradient computation within that analysis (both g_same and g_swap see
the identical batch, satisfying "identical state-action inputs ... compute
two counterfactual actor gradients"). Only states matter functionally here:
the actor's loss samples its OWN action from pi given each state (see
gradients.py's _actor_loss, mirroring update_actor in sac_update.py) rather
than reusing whatever action was originally recorded in the replay buffer -
that is how the real actor-update gradient is computed everywhere else in
this codebase, and Angle 2B must not redefine it.
"""

import numpy as np

from experiments.angle_2a.agent_runner import derive_rng_seed
from experiments.angle_2b.errors import Angle2BConfigError

NUM_STATES_PER_SOURCE = 30


def sample_state_batch(
    states_a: np.ndarray,
    states_b: np.ndarray,
    seed: int,
    context: str,
    num_states_per_source: int = NUM_STATES_PER_SOURCE,
) -> np.ndarray:
    """Samples `num_states_per_source` distinct states from each of
    states_a/states_b (without replacement within each source) and
    concatenates them into one (2*num_states_per_source, obs_dim) batch.

    `context` must be unique per (environment, seed, matchup_name, analysis
    role) so the derived RNG stream (see
    experiments.angle_2a.agent_runner.derive_rng_seed, reused here for the
    same reproducibility guarantee - a process-randomized hash() would
    silently break reproducibility across separate invocations) doesn't
    collide between, e.g., a matchup's own batch and a null-baseline pair's
    batch drawn under the same seed.
    """
    for name, arr in (("states_a", states_a), ("states_b", states_b)):
        if arr.shape[0] < num_states_per_source:
            raise Angle2BConfigError(
                f"Only {arr.shape[0]} states available in {name} (context="
                f"'{context}'), but {num_states_per_source} were requested. "
                f"Refusing to sample with replacement (that would silently "
                f"weaken the experimental protocol)."
            )

    rng = np.random.default_rng(seed=derive_rng_seed(seed, context))
    idx_a = rng.choice(states_a.shape[0], size=num_states_per_source, replace=False)
    idx_b = rng.choice(states_b.shape[0], size=num_states_per_source, replace=False)

    return np.concatenate([states_a[idx_a], states_b[idx_b]], axis=0).astype(np.float32)
