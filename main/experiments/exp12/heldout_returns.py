"""Reward-only reference-policy returns for held-out replay diagnostics.

SAC Q includes entropy and bootstraps time limits, so these finite-episode
returns are a diagnostic, not ground-truth soft Q targets.
"""

import numpy as np


def discounted_episode_returns(rewards, terminated, truncated, gamma):
    """Return discounted reward-to-go and masks for complete/censored episodes."""
    reward = np.asarray(rewards, dtype=np.float64)
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    if reward.ndim != 1 or term.shape != reward.shape or trunc.shape != reward.shape:
        raise ValueError("incompatible transition arrays")
    if not np.all(np.isfinite(reward)) or not 0 <= gamma <= 1:
        raise ValueError("nonfinite reward or invalid discount")
    values = np.empty(len(reward), dtype=np.float64)
    complete = np.zeros(len(reward), dtype=bool)
    censored = np.zeros(len(reward), dtype=bool)
    running = 0.0
    status = 0
    for index in range(len(reward) - 1, -1, -1):
        if term[index] or trunc[index]:
            running = 0.0
            status = 1 if term[index] else 2
        running = reward[index] + gamma * running
        values[index] = running
        complete[index] = status == 1
        censored[index] = status != 1
    return values, complete, censored
