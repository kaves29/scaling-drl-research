"""CPU synthetic fixed-parameter compiler comparison, never a CUDA acceptance test."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("EXP12_JAX_CACHE_DIR", "off")

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from experiments.exp12.probe import _chunked
from scale_rl.agents.sac.sac_network import SACCritic


def main():
    """Report numerical sensitivity without fitting, calibration, or policy changes.

    D4W32 and the synthetic 64-row pool are explanatory CPU fixtures, not the
    production architecture/pool. Fixed parameters isolate forward arithmetic
    from initialization differences. No observed difference is a new gate.
    """
    assert jax.default_backend() == "cpu"
    rng = np.random.default_rng(22740708)
    obs = jnp.asarray(rng.normal(size=(64, 9)), jnp.float32)
    act = jnp.asarray(rng.uniform(-1, 1, (64, 3)), jnp.float32)
    net = SACCritic("simba", 4, 32, jnp.float32)
    params = net.init(jax.random.PRNGKey(990), obs[:1], act[:1])["params"]

    def forward(chunk):
        return _chunked(
            lambda o, a: net.apply({"params": params}, o, a), obs, act, chunk
        )

    with jax.disable_jit():
        reference_q = np.asarray(forward(8))
        reference_target = np.asarray(
            jnp.sin(jnp.float32(1e5) * jnp.asarray(reference_q))
        )
    rows = []
    for chunk in (8, 16, 64):
        q, target = jax.jit(
            lambda: (forward(chunk), jnp.sin(jnp.float32(1e5) * forward(chunk)))
        )()
        q, target = np.asarray(q), np.asarray(target)
        rows.append(
            {
                "chunk": chunk,
                "max_abs_Q_difference_fixed_params_vs_eager": float(
                    np.max(np.abs(q - reference_q))
                ),
                "max_abs_target_difference_vs_eager": float(
                    np.max(np.abs(target - reference_target))
                ),
                "target_values_differing_of64": int(np.sum(target != reference_target)),
            }
        )
    print(
        json.dumps(
            {
                "label": "CPU D4W32 synthetic pool, fixed parameters; not A100 or D4W1536 qualification",
                "backend": jax.default_backend(),
                "rows": rows,
                "one_Q_ULP_at1_times_1e5": float(
                    (np.nextafter(np.float32(1), np.float32(2)) - np.float32(1)) * 1e5
                ),
                "pairing_note": "Both critics within a production round share one already-generated target; this comparison does not establish broken pairing or CUDA error.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
