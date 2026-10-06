"""Independent float64 NumPy SimBa forward and analytic action Jacobian."""

from collections.abc import Mapping

import numpy as np


def _float64(tree, index):
    return {
        k: (
            _float64(v, index)
            if isinstance(v, Mapping)
            else np.asarray(v[index], np.float64)
        )
        for k, v in tree.items()
    }


def _dense(x, jac, params):
    return x @ params["kernel"] + params["bias"], np.einsum(
        "nha,hk->nka", jac, params["kernel"]
    )


def _norm(x, jac, params):
    mean = x.mean(-1, keepdims=True)
    centered = x - mean
    variance = np.maximum((x * x).mean(-1, keepdims=True) - mean * mean, 0.0)
    inv = 1 / np.sqrt(variance + 1e-6)
    dcentered = jac - jac.mean(1, keepdims=True)
    dvariance = 2 * np.mean(centered[:, :, None] * jac, axis=1, keepdims=True)
    derivative = (
        dcentered * inv[:, :, None]
        - 0.5 * centered[:, :, None] * inv[:, :, None] ** 3 * dvariance
    )
    return (
        centered * inv * params["scale"] + params["bias"],
        derivative * params["scale"][None, :, None],
    )


def single(params, observation, action):
    x = np.concatenate((observation, action), axis=1).astype(np.float64)
    jac = np.zeros((*x.shape, action.shape[1]), np.float64)
    jac[:, observation.shape[1] :, :] = np.eye(action.shape[1])
    x, jac = _dense(x, jac, params["encoder"]["Dense_0"])
    blocks = sorted(
        (k for k in params["encoder"] if k.startswith("ResidualBlock_")),
        key=lambda k: int(k.rsplit("_", 1)[1]),
    )
    for key in blocks:
        block = params["encoder"][key]
        skip, dskip = x, jac
        x, jac = _norm(x, jac, block["LayerNorm_0"])
        x, jac = _dense(x, jac, block["Dense_0"])
        jac *= (x > 0)[:, :, None]
        x = np.maximum(x, 0)
        x, jac = _dense(x, jac, block["Dense_1"])
        x, jac = x + skip, jac + dskip
    x, jac = _norm(x, jac, params["encoder"]["LayerNorm_0"])
    x, jac = _dense(x, jac, params["predictor"]["Dense_0"])
    return x[:, 0], jac[:, 0, :]


def twin(params, observation, action):
    outputs = [
        single(_float64(params["VmapSACCritic_0"], k), observation, action)
        for k in (0, 1)
    ]
    q = np.stack([x[0] for x in outputs])
    jac = np.stack([x[1] for x in outputs])
    tied = q[0] == q[1]
    gradient = jac[np.argmin(q, axis=0), np.arange(len(action)), :]
    gradient[tied] = jac[:, tied].mean(0)
    return q, gradient, jac
