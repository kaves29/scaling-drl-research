from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from experiments.exp12.heldout_td import td_residuals


def test_td_residual_uses_termination_and_shared_next_actions():
    class Dist:
        def sample(self, seed): return jnp.ones((2, 1))
        def log_prob(self, actions): return jnp.zeros(2)
    actor = lambda **kwargs: Dist()
    critic = lambda observations, actions: jnp.ones((2, 1)) * 3
    target = lambda observations, actions: jnp.ones((2, 1)) * 4
    shadows = {name: SimpleNamespace(critic=critic, target=target) for name in ('small', 'large')}
    batch = {'observation': np.zeros((2, 1)), 'next_observation': np.zeros((2, 1)),
             'action': np.zeros((2, 1)), 'terminated': np.array([0, 1]), 'reward': np.ones(2)}
    result = td_residuals(actor, lambda: 1., shadows, batch, jax.random.PRNGKey(0), .5, 1, False)
    np.testing.assert_allclose(result['small'], [0., 2.])
    np.testing.assert_array_equal(result['small'], result['large'])
