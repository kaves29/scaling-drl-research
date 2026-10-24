#!/usr/bin/env python3
"""CPU attribution controls against byte-verified pilot source; never submits jobs."""

import argparse
import hashlib
from importlib import util
import json
import os
from pathlib import Path
import subprocess
import sys

PILOT = "f8619dd5f494399d9febc79fc9092010234c714d"
FILES = (
    "experiments/exp12/probe.py",
    "experiments/exp12/trainer.py",
    "experiments/exp12/run_probes.py",
    "experiments/exp12/envs.py",
    "scale_rl/agents/sac/sac_agent.py",
    "scale_rl/agents/sac/sac_network.py",
    "scale_rl/agents/sac/sac_update.py",
    "scale_rl/networks/trainer.py",
    "scale_rl/networks/layers.py",
    "scale_rl/agents/wrappers/normalization.py",
    "configs/base_exp12.yaml",
    "configs/agent/sac_simba.yaml",
    "configs/env/myosuite_simba.yaml",
    "configs/env/dmc_hard.yaml",
    "configs/env/humanoid_bench.yaml",
    "configs/buffer/numpy_uniform.yaml",
    "scale_rl/networks/critics.py",
    "scale_rl/networks/policies.py",
    "scale_rl/networks/utils.py",
    "scale_rl/agents/__init__.py",
    "generate_manifest.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pilot-main", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    repository, source, out = map(Path, (args.repository, args.pilot_main, args.out))
    hashes = {}
    for name in FILES:
        expected = subprocess.check_output(
            ["git", "-C", str(repository), "show", PILOT + ":main/" + name]
        )
        actual = (source / name).read_bytes()
        if actual != expected:
            raise ValueError("pilot source mismatch: " + name)
        hashes[name] = hashlib.sha256(actual).hexdigest()
    os.environ.update(
        JAX_PLATFORMS="cpu",
        JAX_PLATFORM_NAME="cpu",
        JAX_ENABLE_X64="false",
        MUJOCO_GL="disable",
        WANDB_MODE="disabled",
        EXP12_JAX_CACHE_DIR="off",
    )
    sys.path.insert(0, str(repository / "main"))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import importlib.metadata
    from flax import serialization
    from experiments.exp12 import fixed_task as ft
    from experiments.exp12.precision import runtime_info
    from scale_rl.agents.sac.sac_network import SACCritic
    from experiments.exp1 import compose_config

    # Numerical network definitions must match the pilot too, not just the export.
    for name in (
        "scale_rl/agents/sac/sac_network.py",
        "scale_rl/networks/layers.py",
        "scale_rl/networks/critics.py",
        "scale_rl/networks/policies.py",
        "scale_rl/networks/utils.py",
        "scale_rl/agents/sac/sac_agent.py",
        "scale_rl/networks/trainer.py",
        "scale_rl/agents/__init__.py",
        "scale_rl/agents/wrappers/normalization.py",
    ):
        if (repository / "main" / name).read_bytes() != (source / name).read_bytes():
            raise ValueError("current network differs from pilot: " + name)
    for package, version in {
        "jax": "0.4.34",
        "jaxlib": "0.4.34",
        "flax": "0.8.4",
        "optax": "0.2.3",
        "numpy": "1.26.4",
    }.items():
        if importlib.metadata.version(package) != version:
            raise ValueError("wrong dependency: " + package)
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise ValueError("requires FP32 CPU")
    spec = util.spec_from_file_location(
        "verified_pilot_probe", source / "experiments/exp12/probe.py"
    )
    pilot = util.module_from_spec(spec)
    sys.modules[spec.name] = pilot
    spec.loader.exec_module(pilot)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "PARTIAL",
        "pilot_source": PILOT,
        "source_sha256": hashes,
        "diagnostic_source": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime": runtime_info(),
        "configurations": [],
        "cases": [],
    }

    def save():
        (out / "receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )

    save()
    # Resolve the seven later scheduler placements against the specified pilot
    # source. These are reconstructed plans, not seven unavailable runtime files.
    placements = [
        (4, 1536, "dog-run", "dmc_hard"),
        (4, 1536, "myo-key-turn", "myosuite_simba"),
        (2, 512, "humanoid-run", "dmc_hard"),
        (4, 1024, "h1-run-v0", "humanoid_bench"),
        (4, 1024, "myo-key-turn", "myosuite_simba"),
        (4, 1024, "dog-trot", "dmc_hard"),
        (4, 1024, "humanoid-walk", "dmc_hard"),
    ]
    for blocks, width, env, group in placements:
        cfg = compose_config(
            str(source.resolve() / "configs"),
            "base_exp12",
            [
                f"critic_num_blocks={blocks}",
                f"critic_hidden_dim={width}",
                f"env_name={env}",
                f"env={group}",
                "seed=1",
            ],
        )
        receipt["configurations"].append(
            __import__("omegaconf").OmegaConf.to_container(cfg, resolve=True)
        )
    from unittest import mock
    from omegaconf import OmegaConf
    from scale_rl.agents import create_agent
    import gymnasium as gym

    constructor_cfg = OmegaConf.create(receipt["configurations"][0]["agent"])
    constructor_cfg.critic_hidden_dim = 16
    constructor_cfg.actor_hidden_dim = 8
    with mock.patch.object(optax, "adamw", wraps=optax.adamw) as factory:
        agent = create_agent(
            observation_space=gym.spaces.Box(-np.inf, np.inf, (1, 6), np.float32),
            action_space=gym.spaces.Box(-1.0, 1.0, (1, 3), np.float32),
            cfg=constructor_cfg,
        )
    receipt["resized_training_constructor_calls"] = [
        call.kwargs for call in factory.call_args_list
    ]
    del agent
    save()
    # Scaled down residual fixtures retain normalization/residual/output paths.
    for blocks, width in ((2, 16), (4, 24), (4, 32)):
        for bias in (0.0, 10000.0):
            name = f"D{blocks}W{width}_bias{int(bias)}"
            directory = out / name
            directory.mkdir()
            net = SACCritic("residual", blocks, width, jnp.float32)
            obs = jax.random.normal(jax.random.PRNGKey(11), (128, 6))
            act = jax.random.uniform(
                jax.random.PRNGKey(12), (128, 3), minval=-1.0, maxval=1.0
            )
            params = net.init(
                jax.random.PRNGKey(13), observations=obs[:1], actions=act[:1]
            )["params"]
            # Deliberate offset stress condition, not a trained critic claim.
            params["predictor"]["Dense_0"]["bias"] = (
                params["predictor"]["Dense_0"]["bias"] + bias
            )
            target_key, index_key = jax.random.split(pilot.probe_key(1, 0, 0))
            base = pilot._base_targets(net, target_key, obs, act, 32, 100000.0)
            indices = ft.extended_indices(index_key, 2000, 16, 128)
            tx = optax.adamw(1e-4, weight_decay=0.01)
            initial_hash = ft.tree_sha256(params)
            losses, mse, offset = pilot._fit(
                net, tx, params, obs, act, base, indices, 32
            )
            targets = offset + base
            initial = ft.FitState(params, tx.init(params))
            whole, whole_curve = ft.fit_segment(
                net, tx, initial, obs, act, targets, indices
            )
            halfway, first = ft.fit_segment(
                net, tx, initial, obs, act, targets, indices[:1000]
            )
            raw = serialization.to_bytes(halfway)
            (directory / "halfway.msgpack").write_bytes(raw)
            restored = serialization.from_bytes(initial, raw)
            split, second = ft.fit_segment(
                net, tx, restored, obs, act, targets, indices[1000:]
            )
            whole_mse = ft.pool_metrics(net, whole.params, obs, act, targets, 32)["mse"]
            split_mse = ft.pool_metrics(net, split.params, obs, act, targets, 32)["mse"]
            split_curve = jnp.concatenate([first, second])
            q = ft.predictions(net, params, obs, act, 32)
            step = ft.first_update(net, tx, initial, obs, act, targets, indices[0], 32)
            row = {
                "name": name,
                "fixture_only": True,
                "optimizer": {"lr": 0.0001, "decay": 0.01},
                "first_update": {k: float(v) for k, v in step.items()},
                "initial_full_pool_mse": float(jnp.mean((q - targets) ** 2)),
                "frozen_final_pool_mse": float(mse),
                "materialized_final_pool_mse": float(whole_mse),
                "segmented_final_pool_mse": float(split_mse),
                "own_offset": float(offset),
                "initial_error_decomposition": {
                    "centered_q_variance": float(jnp.var(q)),
                    "base_variance": float(jnp.var(base)),
                    "q_base_covariance": float(
                        jnp.mean((q - q.mean()) * (base - base.mean()))
                    ),
                    "base_mean_squared": float(base.mean() ** 2),
                },
                "frozen_vs_materialized_whole": ft.equivalence_report(
                    losses, float(mse), whole_curve, float(whole_mse)
                ),
                "frozen_vs_segmented": ft.equivalence_report(
                    losses, float(mse), split_curve, float(split_mse)
                ),
                "materialized_whole_vs_segmented": ft.equivalence_report(
                    whole_curve, float(whole_mse), split_curve, float(split_mse)
                ),
                "restored_state_bitwise_equal": ft.tree_sha256(restored)
                == ft.tree_sha256(halfway),
                "materialized_final_state_bitwise_equal": ft.tree_sha256(whole)
                == ft.tree_sha256(split),
                "starting_parameters_unchanged": initial_hash == ft.tree_sha256(params),
                "optimizer_counts": [
                    int(s.optimizer[0].count)
                    for s in (initial, halfway, restored, split)
                ],
                "curve_summary": ft.curve_summary(losses),
            }
            np.savez(
                directory / "numeric_evidence.npz",
                observations=obs,
                actions=act,
                base=base,
                targets=targets,
                indices=indices,
                frozen_losses=losses,
                whole_losses=whole_curve,
                split_losses=split_curve,
                starting_predictions=q,
            )
            receipt["cases"].append(row)
            save()
            if (
                not row["restored_state_bitwise_equal"]
                or not row["starting_parameters_unchanged"]
                or row["optimizer_counts"] != [0, 1000, 1000, 2000]
            ):
                raise RuntimeError("CPU state contract failed")
    receipt["status"] = "COMPLETE"
    receipt["qualification"] = (
        "CPU attribution evidence only; failed equivalence guards remain failures; no GPU qualification"
    )
    save()
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "cases": len(receipt["cases"]),
                "out": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
