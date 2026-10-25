"""Synthetic tiny-network diagnostic output; no training, updates or probe fit."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from scale_rl.agents.sac.sac_network import SACCritic
from scale_rl.diagnostics.critic_metrics import CriticMeasurements
from scale_rl.networks.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError("this synthetic example is CPU only")
    batch = {
        "observation": np.arange(24, dtype=np.float32).reshape(8, 3) / 8,
        "action": np.zeros((8, 1), np.float32),
    }
    network = SACCritic("residual", 1, 4, jnp.float32)
    params = network.init(jax.random.PRNGKey(3), batch["observation"], batch["action"])[
        "params"
    ]
    tx = optax.adamw(1e-4)
    critic = Trainer(
        network_def=network, params=params, tx=tx, opt_state=tx.init(params)
    )
    observer = CriticMeasurements(args.out, freeze_first=True, compute_rank=True)
    for synthetic_label in (1, 2):
        record = observer.record(
            critic, batch, interaction_step=synthetic_label, optimizer_step=0
        )
    summary = {
        "evidence_type": "synthetic tiny untrained network; counters are fixture labels",
        "training_or_probe_updates": 0,
        "output": str(args.out),
        "inputs": record["inputs"],
        "metrics": {
            k: v
            for k, v in record["metrics"].items()
            if "/aggregate/" in k or "/analysis64/" in k
        },
    }
    (args.out / "example_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
