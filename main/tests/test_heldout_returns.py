import numpy as np
import pytest

from experiments.exp12.heldout_returns import discounted_episode_returns


def test_complete_truncated_and_partial_episodes():
    values, complete, censored = discounted_episode_returns(
        [1, 2, 3, 4, 5], [False, True, False, False, False],
        [False, False, False, True, False], 0.5
    )
    np.testing.assert_allclose(values, [2, 2, 5, 4, 5])
    np.testing.assert_array_equal(complete, [True, True, False, False, False])
    np.testing.assert_array_equal(censored, [False, False, True, True, True])


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        discounted_episode_returns([1], [False, True], [False], 0.5)
    with pytest.raises(ValueError):
        discounted_episode_returns([float('nan')], [True], [False], 0.5)
