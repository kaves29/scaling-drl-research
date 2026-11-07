#!/usr/bin/env python3
"""Characterize test-only numerical errors; does not set any pass/fail criterion."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import jax
import jax.numpy as jnp
import numpy as np

from experiments.exp12.fork import panel_q_and_grad
from exp12_numpy_oracle import twin
from test_exp12_diagnostics import _agent


def errors(got, reference):
    difference = np.abs(np.asarray(got, np.float64) - reference)
    scale = float(np.abs(reference).max())
    relative = difference / np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    return dict(
        max_abs=float(difference.max()),
        scale=scale,
        max_normalized=float(difference.max() / scale),
        max_eps_units=float(difference.max() / scale / np.finfo(np.float32).eps),
        abs_quantiles=np.quantile(difference, [0.5, 0.9, 0.99, 1]).tolist(),
        max_element_relative=float(relative.max()),
    )


def characterize():
    cases = []
    for blocks, width in ((2, 8), (1, 16), (4, 32)):
        for seed in range(5):
            agent = _agent(
                seed=seed,
                extra=[
                    "agent.critic_use_cdq=true",
                    f"agent.critic_num_blocks={blocks}",
                    f"agent.critic_hidden_dim={width}",
                ],
            )
            agent = getattr(agent, "agent", agent)
            rng = np.random.default_rng(seed)
            panel = {
                "observation": rng.normal(size=(256, 6)).astype(np.float32),
                "action": rng.uniform(-1, 1, size=(256, 3)).astype(np.float32),
            }
            obs, act = map(jnp.asarray, (panel["observation"], panel["action"]))
            q_fn = lambda a: agent.critic.network_def.apply(
                {"params": agent.critic.params}, obs, a
            )
            with jax.default_matmul_precision("highest"):
                eager_q = np.asarray(q_fn(act))
                eager_g = np.asarray(
                    jax.grad(lambda a: jnp.minimum(*q_fn(a)).sum())(act)
                )
                actual = panel_q_and_grad(agent.critic, panel)
                first = np.asarray(jax.jit(jax.grad(lambda a: q_fn(a)[0].sum()))(act))
                second = np.asarray(jax.jit(jax.grad(lambda a: q_fn(a)[1].sum()))(act))
                maximum = np.asarray(
                    jax.jit(jax.grad(lambda a: jnp.maximum(*q_fn(a)).sum()))(act)
                )
                mean = np.asarray(
                    jax.jit(jax.grad(lambda a: q_fn(a).mean(0).sum()))(act)
                )
            q64, g64, _ = twin(
                agent.critic.params, panel["observation"], panel["action"]
            )
            finite = []
            h = 1e-6
            for axis in range(3):
                plus = panel["action"].astype(np.float64)
                minus = plus.copy()
                plus[:, axis] += h
                minus[:, axis] -= h
                pq, _, _ = twin(agent.critic.params, panel["observation"], plus)
                mq, _, _ = twin(agent.critic.params, panel["observation"], minus)
                finite.append((pq.min(0) - mq.min(0)) / (2 * h))
            gradients = dict(
                eager=errors(eager_g, g64), jit=errors(actual["dq_da"], g64)
            )
            values = dict(
                eager=errors(eager_q.reshape(2, -1), q64),
                jit=errors(actual["q"].reshape(2, -1), q64),
            )
            values["per_network"] = {
                f"q{k+1}": {
                    "eager": errors(eager_q.reshape(2, -1)[k], q64[k]),
                    "jit": errors(actual["q"].reshape(2, -1)[k], q64[k]),
                }
                for k in (0, 1)
            }
            pair_error = {
                "q": errors(
                    actual["q"].reshape(2, -1),
                    eager_q.reshape(2, -1).astype(np.float64),
                ),
                "gradient": errors(actual["dq_da"], eager_g.astype(np.float64)),
            }
            mutations = {
                "first_Q_gradient": errors(first, g64),
                "second_Q_gradient": errors(second, g64),
                "maximum_gradient": errors(maximum, g64),
                "mean_gradient": errors(mean, g64),
                "zero_gradient": errors(np.zeros_like(g64), g64),
                "swapped_Q_order": errors(actual["q"].reshape(2, -1)[::-1], q64),
                "duplicate_Q1": errors(np.stack([actual["q"][:256]] * 2), q64),
            }
            cases.append(
                dict(
                    blocks=blocks,
                    width=width,
                    seed=seed,
                    q=values,
                    eager_vs_jit=pair_error,
                    gradient=gradients,
                    finite_difference=errors(np.stack(finite, axis=1), g64),
                    mutations=mutations,
                    min_Q_gap=float(np.abs(q64[0] - q64[1]).min()),
                    min_selection_differences=int(
                        np.sum(
                            np.argmin(actual["q"].reshape(2, -1), axis=0)
                            != np.argmin(q64, axis=0)
                        )
                    ),
                )
            )
            print(
                f"characterized D{blocks}W{width} seed {seed}",
                file=sys.stderr,
                flush=True,
            )
            jax.clear_caches()
    maximum = max(
        c[field][mode]["max_eps_units"]
        for c in cases
        for field in ("q", "gradient")
        for mode in ("eager", "jit")
    )
    maximum = max(
        maximum,
        max(
            c["q"]["per_network"][q][mode]["max_eps_units"]
            for c in cases
            for q in ("q1", "q2")
            for mode in ("eager", "jit")
        ),
    )
    proposed = 2 ** math.ceil(math.log2(4 * maximum))
    return dict(
        scope="CPU characterization of 15 tiny panels; no adopted criterion or CUDA evidence",
        jax=jax.__version__,
        device=str(jax.devices()[0]),
        cases=cases,
        max_legitimate_eps_units=maximum,
        proposed_cpu_eps_units=proposed,
        derivation="Next power of two above four times maximum observed independent-reference normalized error",
        minimum_mutation_margin=min(
            m["max_eps_units"] / proposed
            for c in cases
            for m in c["mutations"].values()
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = characterize()
    args.out.write_text(json.dumps(result, indent=2))
