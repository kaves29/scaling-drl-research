"""Local-instability perturbation measurement for Angle 2C.

For each (s,a) pair, perturbs a by K small Gaussian offsets, computes
nabla_a Q at each perturbed point for both critics, and compares the
variance of these nearby gradients between the two critics - elevated
variance in a critic indicates a locally jagged landscape near the actor's
actual operating point (see research-methodology.md's Angle 2C section).

nabla_a Q at a PERTURBED action was never computed by Angle 2B (it only
computed the gradient at the actual sampled action - see
experiments/angle_2b/matchup_2b.py) - this module therefore evaluates the
already-frozen critics from Angle 2A's checkpoints directly (see
experiments/angle_2b/checkpoint_io.py, reused unchanged) at NEW action
inputs. This is new computation, but not new training, new environment
interaction, or new state-action SAMPLING: the states are exactly Angle 2B's
already-fixed (s,a) pairs, only the action coordinate is locally perturbed
by a small, explicitly-seeded amount to probe the existing frozen critic's
local curvature - the same category of "extra forward/backward pass through
an already-trained network" as Angle 2A's own Monte Carlo rollouts.

K=20, sigma=0.01 (relative to the tanh-squashed [-1,1] action range): K=20
gives a stable per-(s,a) variance estimate without being expensive (20x the
critic forward+backward passes already being done elsewhere in Angle 2C);
sigma=0.01 is small enough to stay "local" (the spec requires "small random
offsets") while still resolving real curvature - two orders of magnitude
below the action range's own scale, avoiding crossing into a qualitatively
different action. Perturbed actions are clipped back into [-1, 1] so the
critic is never evaluated on actions the environment could not actually
produce (the actor's tanh output range).
"""

from typing import Callable

import numpy as np


def perturb_actions(
    actions: np.ndarray,
    num_perturbations: int,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Returns shape (num_perturbations, N, action_dim): K independently
    perturbed copies of `actions` (shape (N, action_dim)), each offset by
    iid N(0, sigma^2) noise per element, clipped to [-1, 1].

    Deterministic given `seed` - the SAME K perturbations (identical
    offsets) must be reused across both critics being compared at a given
    (s,a), so any measured difference reflects the critics' landscapes, not
    which random perturbations happened to be drawn for each.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=(num_perturbations,) + actions.shape).astype(actions.dtype)
    perturbed = actions[None, :, :] + noise
    return np.clip(perturbed, -1.0, 1.0)


def local_instability_variance(
    compute_grad_aq: Callable[[np.ndarray], np.ndarray],
    perturbed_actions: np.ndarray,
) -> np.ndarray:
    """Per-(s,a) local-instability score for ONE critic: the mean squared
    deviation of nabla_a Q across K perturbed actions from their own mean
    (the trace of the empirical covariance of the K perturbed gradients) -
    a standard, non-negative multivariate variance scalar per (s,a) pair.

    `compute_grad_aq(actions_kxNxD) -> nabla_a Q, shape (K, N, action_dim)`
    must evaluate the SAME fixed states this critic's real (s,a) batch used
    (states are captured via closure by the caller - see properties.py) at
    each of the K perturbed action sets.

    `perturbed_actions`: shape (K, N, action_dim), from perturb_actions() -
    pass the SAME array to both critics being compared.

    Returns shape (N,): one variance value per (s,a) pair.
    """
    grads = compute_grad_aq(perturbed_actions)  # (K, N, action_dim)
    mean_grad = grads.mean(axis=0, keepdims=True)  # (1, N, action_dim)
    sq_dev = np.sum((grads - mean_grad) ** 2, axis=-1)  # (K, N)
    return sq_dev.mean(axis=0)  # (N,)
