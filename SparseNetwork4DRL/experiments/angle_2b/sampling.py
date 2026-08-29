"""Deterministic state-batch construction for Angle 2B's frozen
counterfactual gradient measurements.

Own-buffer-only sourcing (hard requirement): every analysis's state batch is
drawn EXCLUSIVELY from its held-fixed actor's own probe-capture buffer -
e.g. the primary analysis (pi_D held fixed) draws only from D's buffer,
never R's; secondary (pi_R held fixed) draws only from R's, never D's; a
null pair (pi_A held fixed) draws only from A's, never B's. Sourcing from
the OTHER agent's buffer would evaluate the held-fixed actor on states/
actions it never actually visited, reintroducing an out-of-distribution
confound - this is why there is exactly one `states` array per call, not
two. Consequently primary and secondary (and each null pair) are measured
on DIFFERENT underlying states from each other; only the distortion metrics,
not the raw scenes, are compared side by side.

The sampled batch is reused unchanged for both gradient calls within one
analysis (both g_same and g_swap see the identical batch, satisfying
"identical state-action inputs ... compute two counterfactual actor
gradients"). Only states matter functionally here: the actor's loss samples
its OWN action from pi given each state (see gradients.py's _actor_loss,
mirroring update_actor in sac_update.py) rather than reusing whatever action
was originally recorded in the replay buffer - that is how the real
actor-update gradient is computed everywhere else in this codebase, and
Angle 2B must not redefine it.
"""

import numpy as np

from experiments.angle_2a.agent_runner import derive_rng_seed
from experiments.angle_2b.errors import Angle2BConfigError

NUM_STATES_PER_SOURCE = 40


def sample_state_batch(
    states: np.ndarray,
    seed: int,
    context: str,
    num_states_per_source: int = NUM_STATES_PER_SOURCE,
) -> np.ndarray:
    """Samples `num_states_per_source` distinct states, without replacement,
    from `states` - the held-fixed actor's OWN probe-capture buffer for this
    analysis, and nothing else (see module docstring).

    `context` must be unique per (environment, seed, matchup_name, analysis
    role) so the derived RNG stream (see
    experiments.angle_2a.agent_runner.derive_rng_seed, reused here for the
    same reproducibility guarantee - a process-randomized hash() would
    silently break reproducibility across separate invocations) doesn't
    collide between, e.g., a matchup's primary batch and its secondary
    batch, or two different null-baseline pairs drawn under the same seed.
    """
    if states.shape[0] < num_states_per_source:
        raise Angle2BConfigError(
            f"Only {states.shape[0]} states available (context='{context}'), "
            f"but {num_states_per_source} were requested. Refusing to sample "
            f"with replacement (that would silently weaken the experimental "
            f"protocol)."
        )

    rng = np.random.default_rng(seed=derive_rng_seed(seed, context))
    idx = rng.choice(states.shape[0], size=num_states_per_source, replace=False)
    return states[idx].astype(np.float32)
