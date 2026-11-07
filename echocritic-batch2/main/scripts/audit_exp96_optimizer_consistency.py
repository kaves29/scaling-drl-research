#!/usr/bin/env python3
"""CPU-only fixed-task optimizer controls against verified frozen Exp96 source."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

FROZEN = "f597ee9898e4d401912092c336c3e5330cf2fda1"
SOURCE_FILES = (
    "experiments/exp12/probe.py",
    "experiments/exp12/run_probes.py",
    "scale_rl/agents/sac/sac_agent.py",
    "scale_rl/agents/sac/sac_network.py",
    "scale_rl/networks/trainer.py",
    "scale_rl/networks/layers.py",
    "configs/agent/sac.yaml",
    "configs/agent/sac_simba.yaml",
    "scripts/exp96_campaign.py",
    "generate_manifest.py",
    "experiments/exp1.py",
    "scale_rl/agents/__init__.py",
    "configs/base_exp12.yaml",
    "configs/base_exp12_ten.yaml",
    "configs/env/dmc_hard.yaml",
    "configs/env/myosuite_simba.yaml",
    "configs/buffer/numpy_uniform.yaml",
)


def source_check(repository, frozen_main):
    hashes = {}
    for name in SOURCE_FILES:
        expected = subprocess.check_output(
            ["git", "-C", str(repository), "show", FROZEN + ":main/" + name]
        )
        actual = (frozen_main / name).read_bytes()
        if expected != actual:
            raise ValueError("frozen source mismatch: " + name)
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return hashes


def write_curve_csv(out, results):
    """Export all observed minibatch losses without smoothing or interpolation."""
    import csv
    import numpy as np

    with (out / "loss_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "profile",
                "seed",
                "condition",
                "optimizer_updates_before_measurement",
                "pre_update_minibatch_mse",
            )
        )
        for row in results:
            case = out / f"cpu_{row['profile'].removeprefix('CPU ')}_seed{row['seed']}"
            with np.load(
                case / (row["condition"] + ".npz"), allow_pickle=False
            ) as values:
                losses = values["losses"]
                if losses.shape != (2000,) or not np.isfinite(losses).all():
                    raise ValueError("incomplete/nonfinite CPU control curve")
                writer.writerows(
                    (row["profile"], row["seed"], row["condition"], i, float(loss))
                    for i, loss in enumerate(losses)
                )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", required=True)
    p.add_argument("--frozen-main", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument(
        "--census-only",
        action="store_true",
        help="Compose all 96 frozen plans and trace a bounded real training constructor; no fits",
    )
    args = p.parse_args()
    source = Path(args.frozen_main).resolve()
    hashes = source_check(Path(args.repository).resolve(), source)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("unique CPU fixture seeds required")
    os.environ.update(
        JAX_PLATFORMS="cpu",
        JAX_PLATFORM_NAME="cpu",
        JAX_ENABLE_X64="false",
        MUJOCO_GL="disable",
        EXP12_JAX_CACHE_DIR="off",
        WANDB_MODE="disabled",
    )
    sys.path.insert(0, str(source))
    import csv
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.traverse_util import flatten_dict
    from omegaconf import OmegaConf
    from experiments.exp12 import probe
    from scale_rl.agents.sac import sac_network
    from scale_rl.networks.trainer import Trainer

    if jax.default_backend() != "cpu":
        raise RuntimeError("this audit never uses a GPU")
    for name, expected in {
        "jax": "0.4.34",
        "jaxlib": "0.4.34",
        "flax": "0.8.4",
        "optax": "0.2.3",
        "numpy": "1.26.4",
    }.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(
                "audit requires the frozen dependency pin: " + name + "==" + expected
            )
    if jax.config.jax_enable_x64:
        raise ValueError("audit requires float32 with JAX x64 disabled")
    for module in (probe, sac_network):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError("non-frozen numerical import")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.census_only:
        from scripts import exp96_campaign
        from experiments.exp1 import compose_config
        from scale_rl.agents import create_agent
        from unittest import mock
        import gymnasium as gym

        if not Path(exp96_campaign.__file__).resolve().is_relative_to(source):
            raise ValueError("non-frozen campaign source")
        entries = exp96_campaign.rows(
            Path(
                "/projects/biqc/skaveti1/exp96/campaign_owner_f597ee9_20261019T230748Z/campaign"
            )
        )
        records = []
        for row in entries:
            cfg = compose_config(
                str(source / "configs"), "base_exp12_ten", row["overrides"]
            )
            a = cfg.agent
            records.append(
                {
                    k: row[k]
                    for k in (
                        "run_key",
                        "environment",
                        "architecture",
                        "seed",
                        "config_hash",
                    )
                }
                | dict(
                    critic_lr=float(a.critic_learning_rate),
                    critic_decay=float(a.critic_weight_decay),
                    actor_lr=float(a.actor_learning_rate),
                    actor_decay=float(a.actor_weight_decay),
                    mixed_precision=bool(a.mixed_precision),
                    critic_sparsity=float(a.critic_sparsity),
                    optimizer_overrides=[
                        x
                        for x in row["overrides"]
                        if any(t in x for t in ("learning_rate", "weight_decay"))
                    ],
                )
            )
        assert len(records) == 96 and len({r["run_key"] for r in records}) == 96
        assert len({(r["critic_lr"], r["critic_decay"]) for r in records}) == 1
        # Bound memory while exercising the real frozen construction path; this
        # is explicitly a resized constructor fixture, not a historical run.
        cfg.agent.actor_hidden_dim = 16
        cfg.agent.critic_hidden_dim = 16
        cfg.agent.critic_num_blocks = 2
        with mock.patch("optax.adamw", wraps=optax.adamw) as factory:
            create_agent(
                gym.spaces.Box(-1, 1, (1, 6), dtype=np.float32),
                gym.spaces.Box(-1, 1, (1, 3), dtype=np.float32),
                cfg.agent,
            )
        calls = [dict(c.kwargs) for c in factory.call_args_list]
        assert [(c["learning_rate"], c["weight_decay"]) for c in calls] == [
            (1e-4, 0.01),
            (1e-4, 0.01),
            (1e-4, 0.0),
        ]
        with (out / "canonical_96.csv").open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        receipt = {
            "frozen_source": FROZEN,
            "source_sha256": hashes,
            "records": records,
            "resized_real_training_constructor_adamw_calls": calls,
            "scope": "Canonical configurations and executable CPU constructor only; not an independent read of all 96 archived runtime metadata files.",
        }
        (out / "census.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )
        print(
            "96 canonical configurations; optimizer tuples",
            sorted({(r["critic_lr"], r["critic_decay"]) for r in records}),
        )
        print("Frozen training constructor AdamW calls:", calls)
        return

    def array_hash(x):
        a = np.asarray(x)
        h = hashlib.sha256()
        h.update(a.dtype.str.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
        return h.hexdigest()

    def tree_hash(params):
        return hashlib.sha256(
            json.dumps(
                {str(k): array_hash(v) for k, v in flatten_dict(params).items()},
                sort_keys=True,
            ).encode()
        ).hexdigest()

    good = OmegaConf.load(source / "configs/agent/sac_simba.yaml")
    old = OmegaConf.load(source / "configs/agent/sac.yaml")
    lr, wd = float(good.critic_learning_rate), float(good.critic_weight_decay)
    old_lr, old_wd = float(old.critic_learning_rate), float(old.critic_weight_decay)
    settings = {
        "campaign": (lr, wd),
        "legacy_diagnostic": (old_lr, old_wd),
        "lr_only_change": (old_lr, wd),
        "decay_only_change": (lr, old_wd),
    }
    transforms = {
        name: probe.critic_optimizer(
            OmegaConf.create(dict(critic_learning_rate=a, critic_weight_decay=b))
        )
        for name, (a, b) in settings.items()
    }
    decay_checks = []
    for name, tx in transforms.items():
        params = {
            "kernel": jnp.array([2.0, -3.0], jnp.float32),
            "bias": jnp.array([1.0], jnp.float32),
        }
        state = Trainer(
            network_def=None, params=params, tx=tx, opt_state=tx.init(params)
        )
        zero = lambda p: (sum(jnp.sum(v * 0) for v in p.values()), {})
        following, _ = state.apply_gradient(zero)
        expected = jax.tree_util.tree_map(
            lambda x: x * (1 - settings[name][0] * settings[name][1]), params
        )
        for a, b in zip(
            jax.tree_util.tree_leaves(following.params),
            jax.tree_util.tree_leaves(expected),
        ):
            np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7)
        assert int(following.opt_state[0].count) == 1
        twice, _ = following.apply_gradient(zero)
        assert int(twice.opt_state[0].count) == 2
        decay_checks.append(
            {
                "condition": name,
                "adam_count_after_two_updates": 2,
                "kernel_and_bias_decay_reference_passed": True,
            }
        )
    results = []
    profiles = ((2, 16), (4, 24), (4, 32))
    for depth, width in profiles:
        net = sac_network.SACCritic("residual", depth, width, jnp.float32)
        teacher = sac_network.SACCritic("residual", 2, 16, jnp.float32)
        for seed in args.seeds:
            case = out / f"cpu_D{depth}W{width}_seed{seed}"
            case.mkdir()
            keys = jax.random.split(jax.random.PRNGKey(seed), 4)
            obs = jax.random.normal(keys[0], (128, 6))
            act = jax.random.normal(keys[1], (128, 3))
            params = net.init(keys[2], observations=obs[:1], actions=act[:1])["params"]
            parameter_hash = tree_hash(params)
            base = probe._base_targets(teacher, keys[3], obs, act, 32, 100000.0)
            q0 = probe._chunked(
                lambda o, a: net.apply({"params": params}, o, a).reshape(-1),
                obs,
                act,
                32,
            )
            target = jnp.asarray(jnp.mean(q0) + base, jnp.float32)
            key = jax.random.PRNGKey(3000 + seed)
            indices = jnp.concatenate(
                [
                    jax.random.randint(key, (1000, 16), 0, 128),
                    jax.random.randint(jax.random.fold_in(key, 1), (1000, 16), 0, 128),
                ]
            )
            arrays = dict(
                observations=np.asarray(obs),
                actions=np.asarray(act),
                targets=np.asarray(target),
                minibatch_indices=np.asarray(indices),
            )
            manifest = {name: array_hash(value) for name, value in arrays.items()}
            task_id = hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest()
            np.savez(case / "fixed_task.npz", **arrays)
            np.savez(
                case / "starting_params.npz",
                **{"/".join(k): np.asarray(v) for k, v in flatten_dict(params).items()},
            )
            (case / "task_manifest.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "arrays": manifest,
                        "starting_parameters_sha256": parameter_hash,
                    },
                    indent=2,
                )
                + "\n"
            )

            def first_loss(p):
                ix = indices[0]
                q = net.apply({"params": p}, obs[ix], act[ix]).reshape(-1)
                return jnp.mean((q - target[ix]) ** 2)

            initial_loss, gradient = jax.value_and_grad(first_loss)(params)
            gradient_norm = float(optax.global_norm(gradient))
            initial_pool = float(jnp.mean((q0 - target) ** 2))
            for name, tx in transforms.items():
                updates, _ = tx.update(gradient, tx.init(params), params)
                updated = optax.apply_updates(params, updates)
                wrapper = Trainer(
                    network_def=net, params=params, tx=tx, opt_state=tx.init(params)
                )
                wrapper_after, wrapper_info = wrapper.apply_gradient(
                    lambda p: (first_loss(p), {})
                )
                for a, b in zip(
                    jax.tree_util.tree_leaves(updated),
                    jax.tree_util.tree_leaves(wrapper_after.params),
                ):
                    np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(
                    wrapper_info["grad_norm"], gradient_norm, rtol=2e-5, atol=2e-6
                )
                assert int(wrapper_after.opt_state[0].count) == 1
                q1 = probe._chunked(
                    lambda o, a: net.apply({"params": updated}, o, a).reshape(-1),
                    obs,
                    act,
                    32,
                )
                change = float(jnp.sqrt(jnp.mean((q1 - q0) ** 2)))
                before = time.monotonic()
                # Materialized targets are shared exactly; no optimizer-dependent offset recomputation.
                losses, final, _ = probe._fit(
                    net,
                    tx,
                    params,
                    obs,
                    act,
                    target,
                    indices,
                    32,
                    offset=jnp.float32(0),
                )
                losses.block_until_ready()
                final.block_until_ready()
                curve = np.asarray(losses)
                assert (
                    curve.shape == (2000,)
                    and np.isfinite(curve).all()
                    and np.isfinite(float(final))
                )
                assert tree_hash(params) == parameter_hash
                np.testing.assert_allclose(curve[0], initial_loss, rtol=2e-5, atol=2e-6)
                np.savez(
                    case / (name + ".npz"),
                    losses=curve,
                    initial_predictions=np.asarray(q0),
                    first_update_predictions=np.asarray(q1),
                    final_mse=np.asarray(final),
                )
                results.append(
                    {
                        "profile": f"CPU D{depth}W{width}",
                        "parameters": sum(
                            int(x.size) for x in jax.tree_util.tree_leaves(params)
                        ),
                        "seed": seed,
                        "condition": name,
                        "critic_lr": settings[name][0],
                        "critic_decay": settings[name][1],
                        "task_id": task_id,
                        "starting_parameters_sha256": parameter_hash,
                        "initial_full_pool_mse": initial_pool,
                        "initial_minibatch_mse": float(initial_loss),
                        "first_update_gradient_l2": gradient_norm,
                        "first_update_prediction_change_rms": change,
                        "first_update_full_pool_mse": float(
                            jnp.mean((q1 - target) ** 2)
                        ),
                        "frozen_training_wrapper_first_update_reference_passed": True,
                        "first_100_mean_over_initial": float(
                            curve[:100].astype(np.float64).mean() / float(curve[0])
                        ),
                        "first_100_peak_over_initial": float(
                            curve[:100].max() / curve[0]
                        ),
                        "mean_updates_1900_1999": float(
                            curve[-100:].astype(np.float64).mean()
                        ),
                        "final_full_pool_mse_2000": float(final),
                        "all_losses_finite": True,
                        "starting_parameters_unchanged": True,
                        "fit_seconds_including_first_compile": time.monotonic()
                        - before,
                    }
                )
            print("completed", depth, width, seed, flush=True)
    with (out / "comparison.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    write_curve_csv(out, results)
    receipt = {
        "frozen_source": FROZEN,
        "source_sha256": hashes,
        "backend": jax.default_backend(),
        "python": sys.version,
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("jax", "jaxlib", "flax", "optax", "numpy")
        },
        "precision": {
            "dtype": "float32",
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "matmul_precision": jax.config.jax_default_matmul_precision,
        },
        "profiles": profiles,
        "seeds": args.seeds,
        "conditions": settings,
        "horizon": 2000,
        "rows": results,
        "frozen_training_wrapper_checks": decay_checks,
        "scope": "Small CPU numerical controls only; no production critic, original run, causal explanation or GPU qualification.",
    }
    (out / "comparison.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    (out / "SHA256SUMS.txt").write_text(
        "".join(
            hashlib.sha256(f.read_bytes()).hexdigest()
            + "  "
            + str(f.relative_to(out))
            + "\n"
            for f in sorted(out.rglob("*"))
            if f.is_file()
        )
    )


if __name__ == "__main__":
    main()
