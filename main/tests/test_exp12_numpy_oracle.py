"""Independent NumPy oracle arithmetic checks; no production numerical budget."""

import unittest

import numpy as np

from tests.exp12_numpy_oracle import _dense, _norm, single, twin


class NumpyOracleTest(unittest.TestCase):
    def test_dense_and_layernorm_analytic_jacobian(self):
        rng = np.random.default_rng(104)
        x = rng.normal(size=(7, 4))
        jac = rng.normal(size=(7, 4, 2))
        params = {"scale": rng.normal(size=4), "bias": rng.normal(size=4)}
        _, got = _norm(x, jac, params)
        h = 1e-6
        for a in range(2):
            plus, _ = _norm(x + h * jac[:, :, a], jac, params)
            minus, _ = _norm(x - h * jac[:, :, a], jac, params)
            np.testing.assert_allclose(
                got[:, :, a], (plus - minus) / (2 * h), rtol=1e-7, atol=1e-8
            )
        dense = {"kernel": rng.normal(size=(4, 3)), "bias": rng.normal(size=3)}
        _, got = _dense(x, jac, dense)
        for a in range(2):
            np.testing.assert_array_equal(got[:, :, a], jac[:, :, a] @ dense["kernel"])

    def test_min_selection_and_tie_derivative(self):
        params = {
            "encoder": {
                "Dense_0": {"kernel": np.eye(3), "bias": np.zeros(3)},
                "LayerNorm_0": {"scale": np.ones(3), "bias": np.zeros(3)},
            },
            "predictor": {
                "Dense_0": {
                    "kernel": np.array([[1.0], [0.0], [-1.0]]),
                    "bias": np.zeros(1),
                }
            },
        }

        def stack(tree):
            return {
                k: stack(x) if isinstance(x, dict) else np.stack([x, x])
                for k, x in tree.items()
            }

        twin_params = {"VmapSACCritic_0": stack(params)}
        obs = np.array([[1.0, 2.0], [2.0, 1.0]])
        act = np.array([[0.3], [0.2]])
        q, g, j = twin(twin_params, obs, act)
        np.testing.assert_array_equal(q[0], q[1])
        np.testing.assert_array_equal(g, j.mean(0))
        h = 1e-6
        qp, _, _ = twin(twin_params, obs, act + h)
        qm, _, _ = twin(twin_params, obs, act - h)
        np.testing.assert_allclose(
            g[:, 0], (qp.min(0) - qm.min(0)) / (2 * h), rtol=1e-7, atol=1e-8
        )
