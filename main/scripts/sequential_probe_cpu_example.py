"""Tiny synthetic CPU demonstration, NOT campaign evidence or qualification."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
import numpy as np

from diagnostics.sequential_probe import (
    Settings,
    EVEN_MILESTONES,
    ALL_TARGETS,
    create_bank,
    create_reference,
    evaluate,
)
from scale_rl.agents.sac.sac_network import SACCritic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--even-milestones", action="store_true")
    args = parser.parse_args()
    root = Path(args.out).resolve()
    repository = Path(__file__).resolve().parents[2]
    if root == repository or repository in root.parents:
        raise ValueError("example output must be outside repository")
    root.mkdir(parents=True, exist_ok=False)
    assert all(d.platform == "cpu" for d in jax.devices())
    obs = np.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0], [2.0, -1.0]], np.float32)
    act = np.asarray([[0.0], [1.0], [2.0], [-1.0]], np.float32)
    net = SACCritic(
        block_type="residual", num_blocks=1, hidden_dim=4, dtype=jnp.float32
    )
    fresh = net.init(
        jax.random.PRNGKey(11),
        observations=jnp.asarray(obs[:1]),
        actions=jnp.asarray(act[:1]),
    )["params"]
    create_bank(
        root / "bank",
        observations=obs,
        actions=act,
        num_filled=4,
        warmup=4,
        rms=None,
        environment="SYNTHETIC-CPU-ONLY",
        seed=7,
        teacher=net,
        teacher_architecture="tiny-residual-W4",
        pool_size=4,
        teacher_chunk=2,
        source="synthetic-CPU-example",
    )
    create_reference(
        root / "reference",
        root / "bank",
        net,
        fresh,
        architecture="tiny-residual-W4",
        sac_updates=0,
    )
    for checkpoint, current in [
        (0, fresh),
        (1, jax.tree_util.tree_map(lambda p: p + jnp.float32(0.01), fresh)),
    ]:
        evaluate(
            enabled=True,
            bank=root / "bank",
            reference=root / "reference",
            output=root / f"checkpoint_{checkpoint}",
            network=net,
            current=current,
            fresh=fresh,
            checkpoint=checkpoint,
            settings=Settings(batch_size=2, chunk=2),
            schedule=EVEN_MILESTONES if args.even_milestones else ALL_TARGETS,
        )
    print("Synthetic CPU artifacts:", root)
    print("No SAC training, GPU execution, scientific qualification or launch.")


if __name__ == "__main__":
    main()
