"""Exploratory Method C controls on immutable Exp1 checkpoints.

This is deliberately separate from the production probe and never writes to a
source run. The official probe functions supply the five-round, 2,000-step
measurements. Extra optimization diagnostics and longer fits are labeled as
exploratory and cannot replace production scores.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import subprocess
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from omegaconf import OmegaConf
from scipy.stats import beta

from experiments.exp12.probe import (
    ORIGINAL_FIT_STEPS,
    _base_targets,
    _chunked,
    _mean_prediction,
    critic_optimizer,
    probe_config,
    probe_key,
    probe_rng,
    probe_round,
    summarize,
)
from experiments.exp12.run_probes import original_critic_def
from experiments.exp12.state import latest_state_dir, load_meta
from experiments.exp12.trigger import bootstrap_interval, triggered
from scale_rl.agents import create_agent
from utils.run_metadata import RUN_METADATA_FILENAME, load_run_metadata
from utils.atomic_io import atomic_write_text

PILOT_COMMIT = "f8619dd5f494399d9febc79fc9092010234c714d"
STREAM = 0x56414C49  # "VALI": separate from production probe keys.


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def source_metadata(run_dir: Path) -> dict:
    path = run_dir / RUN_METADATA_FILENAME
    metadata = load_run_metadata(path)
    if metadata is None or metadata.get("code", {}).get("commit") != PILOT_COMMIT:
        raise ValueError(f"{path}: expected source commit {PILOT_COMMIT}")
    if metadata["code"].get("dirty"):
        raise ValueError(f"{path}: source checkout was dirty")
    if metadata.get("identity", {}).get("run_role") != "confirmatory":
        raise ValueError(f"{path}: expected a confirmatory parent")
    identity = metadata["identity"]
    if identity.get("environment") != "myo-key-turn" or identity.get("seed") != 1:
        raise ValueError(f"{path}: expected the seed-1 Myo Key Turn pilot parent")
    if identity.get("architecture") not in ("D2W512", "D4W1024", "D4W1536"):
        raise ValueError(f"{path}: unexpected critic architecture")
    if run_dir.name != (f"run{ {'D2W512': 4, 'D4W1024': 5, 'D4W1536': 7}[identity['architecture']]}"
                        f"_{identity['architecture']}_myo-key-turn_seed1"):
        raise ValueError(f"{path}: pilot directory and run identity differ")
    return metadata


def resolve_state(run_dir: Path, point: str) -> Path:
    root = run_dir / ("fork/state" if point == "early" else "state")
    state = latest_state_dir(root)
    if state is None:
        raise FileNotFoundError(f"{root}/LATEST is missing")
    expected = 50_000 if point == "early" else 500_000
    if load_meta(state).get("interaction_step") != expected:
        raise ValueError(f"{state}: expected interaction step {expected}")
    for name in ("agent_ckpt", "obs_rms.pkl", "buffer.npz", "buffer_meta.pkl", "meta.pkl"):
        if not (state / name).exists():
            raise FileNotFoundError(f"{state / name} is missing")
    if point == "early" and not (run_dir / "fork/FORK_READY").exists():
        raise FileNotFoundError(f"{run_dir}/fork/FORK_READY is missing")
    return state


def artifact_inventory(run_dir: Path) -> dict:
    metadata = source_metadata(run_dir)
    states = {}
    for point in ("early", "late"):
        try:
            states[point] = str(resolve_state(run_dir, point))
        except (FileNotFoundError, ValueError) as exc:
            states[point] = {"unavailable": str(exc)}
    fresh = run_dir / "fresh_critic"
    return {
        "run_dir": str(run_dir.resolve()),
        "architecture": metadata["identity"]["architecture"],
        "environment": metadata["identity"]["environment"],
        "source_commit": metadata["code"]["commit"],
        "run_metadata_sha256": file_sha256(run_dir / RUN_METADATA_FILENAME),
        "states": states,
        "fresh_critic": str(fresh) if fresh.is_dir() else None,
    }


def exact_binomial_interval(fires: int, checks: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval; descriptive for independent pairs."""
    if not 0 <= fires <= checks or checks < 1:
        raise ValueError("invalid independent-pair count")
    tail = (1.0 - confidence) / 2.0
    low = 0.0 if fires == 0 else float(beta.ppf(tail, fires, checks - fires + 1))
    high = 1.0 if fires == checks else float(beta.ppf(1.0 - tail, fires + 1, checks - fires))
    return low, high


def followup_indices(key, batch_size: int, pool_size: int, steps: int):
    """Keep Method C's first 2,000 draws, then add disjoint follow-up streams."""
    if steps not in (2000, 4000, 8000):
        raise ValueError("exploratory extension must stop at 2000, 4000, or 8000")
    chunks = [
        jax.random.randint(key, (ORIGINAL_FIT_STEPS, batch_size), 0, pool_size),
        jax.random.randint(jax.random.fold_in(key, 1), (1000, batch_size), 0, pool_size),
    ]
    if steps >= 4000:
        chunks.append(jax.random.randint(jax.random.fold_in(key, 2), (2000, batch_size), 0, pool_size))
    if steps >= 8000:
        chunks.append(jax.random.randint(jax.random.fold_in(key, 3), (4000, batch_size), 0, pool_size))
    return chunks


def diagnostic_fit(network_def, tx, params, obs, act, base_targets, indices, chunk: int):
    """Probe-equivalent fit with ancillary gradient/parameter norms and state continuity.

    The official probe's result is calculated separately. These extra outputs
    can alter XLA lowering and must never determine an official trigger.
    """
    offset = _mean_prediction(network_def, params, obs, act, chunk)
    targets = offset + base_targets

    def loss_fn(p, batch_idx):
        q = network_def.apply({"params": p}, obs[batch_idx], act[batch_idx]).reshape(-1)
        return jnp.mean((q - targets[batch_idx]) ** 2)

    def step(carry, batch_idx):
        p, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(p, batch_idx)
        updates, new_opt_state = tx.update(grads, opt_state, p)
        return (optax.apply_updates(p, updates), new_opt_state), (loss, optax.global_norm(grads))

    carry = (params, tx.init(params))
    records = []
    for segment in indices:
        carry, (losses, grad_norms) = jax.lax.scan(step, carry, segment)
        fitted = carry[0]
        prediction = _chunked(lambda o, a: network_def.apply({"params": fitted}, o, a).reshape(-1),
                              obs, act, chunk)
        records.append({
            "losses": np.asarray(losses),
            "gradient_norms": np.asarray(grad_norms),
            "pool_mse": float(jnp.mean((prediction - targets) ** 2)),
            "parameter_norm": float(optax.global_norm(fitted)),
            "optimizer_step": int(jax.tree_util.tree_leaves(carry[1])[0]),
        })
    return records


def paired_decision(result: dict, current: str, fresh: str, seed: int, check_index: int, cfg) -> dict:
    summary = summarize(result, current=current, fresh=fresh)
    low, high = bootstrap_interval(summary["loss_rounds"], seed, check_index,
                                   int(cfg.trigger.resamples), float(cfg.trigger.confidence))
    return {
        "loss_rounds": np.asarray(summary["loss_rounds"]).tolist(),
        "loss_iqm": summary["loss_iqm"],
        "ci_low": low,
        "ci_high": high,
        "fires": triggered(low, float(cfg.trigger.null_threshold)) and summary["valid"],
    }


def init_fresh(network_def, key, obs, act):
    return network_def.init(key, observations=obs[:1], actions=act[:1])["params"]


def load_state(run_dir: Path, point: str, cfg):
    state = resolve_state(run_dir, point)
    with np.load(state / "buffer.npz", allow_pickle=False) as data:
        raw_obs, actions = data["observations"], data["actions"]
    with (state / "buffer_meta.pkl").open("rb") as stream:
        n = pickle.load(stream)["num_in_buffer"]
    if type(n) is not int or n < 1 or len(raw_obs) != n or len(actions) != n:
        raise ValueError(f"{state}: invalid replay size")
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=raw_obs.shape[1:], dtype=raw_obs.dtype)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=actions.shape[1:], dtype=actions.dtype)
    agent = create_agent(obs_space, act_space, cfg.agent)
    agent.load_checkpoint(str(state))
    critic = jax.device_get(agent.critic.params)
    rms = agent.obs_rms
    normalized = {
        "mean": np.asarray(rms.mean), "var": np.asarray(rms.var),
        "epsilon": float(agent.epsilon),
    }
    del agent
    gc.collect()
    return state, critic, raw_obs, actions, normalized


def pool(raw_obs, actions, normalizer: dict, seed: int, check_index: int, round_index: int, size: int):
    idx = probe_rng(seed, check_index, round_index).integers(0, len(raw_obs), size=size)
    obs = (raw_obs[idx] - normalizer["mean"]) / np.sqrt(normalizer["var"] + normalizer["epsilon"])
    return jnp.asarray(obs), jnp.asarray(actions[idx])


def publish(out: Path, name: str, summary: dict, arrays: dict):
    """Never overwrite a previous condition; completed evidence stays immutable."""
    target = out / name
    target.mkdir(parents=True, exist_ok=False)
    with (target / "curves.npz").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    summary["curves_sha256"] = file_sha256(target / "curves.npz")
    atomic_write_text(target / "summary.json", json.dumps(summary, indent=2, sort_keys=True, default=str))


def run_crossover(run_dir: Path, out: Path, cfg, states: dict, fresh):
    network_def, tx = original_critic_def(cfg), critic_optimizer(cfg.agent)
    for point in ("early", "late"):
        state, _, raw_obs, actions, normalizer = states[point]
        check_index = 2 if point == "early" else 20
        rounds, arrays = [], {}
        for r in range(int(cfg.probe.rounds)):
            obs, act = pool(raw_obs, actions, normalizer, int(cfg.seed), check_index, r,
                            int(cfg.probe.pool_size))
            result = probe_round(network_def, {
                "early": (network_def, states["early"][1]),
                "late": (network_def, states["late"][1]),
                "fresh": (network_def, fresh),
            }, tx, obs, act, probe_key(int(cfg.seed), check_index, r), probe_config(cfg))
            result = jax.device_get(result)
            rounds.append(result)
            for arm, fields in result.items():
                for field, value in fields.items():
                    arrays[f"round_{r}/{arm}_{field}"] = np.asarray(value)
        combined = {arm: {field: np.stack([one[arm][field] for one in rounds])
                          for field in rounds[0][arm]} for arm in rounds[0]}
        decisions = {arm: paired_decision(combined, arm, "fresh", int(cfg.seed), check_index, cfg)
                     for arm in ("early", "late")}
        original_path = run_dir / "probes" / f"check_{check_index:02d}.npz"
        if not original_path.exists():
            raise FileNotFoundError(f"original probe evidence missing: {original_path}")
        diagonal = "early" if point == "early" else "late"
        source_replay = {}
        with np.load(original_path, allow_pickle=False) as original:
            for arm, stored_arm in ((diagonal, "current"), ("fresh", "fresh")):
                for field in ("score", "final_loss", "losses", "b"):
                    observed, expected = combined[arm][field], original[f"{stored_arm}_{field}"]
                    source_replay[f"{arm}_{field}"] = {
                        "bitwise_equal": observed.dtype == expected.dtype and np.array_equal(
                            observed.view(np.uint8), expected.view(np.uint8)),
                        "max_abs": float(np.max(np.abs(observed - expected))),
                    }
        publish(out, f"crossover_{point}", {
            "pool_state": str(state), "check_index": check_index,
            "original_probe_file": str(original_path),
            "original_probe_sha256": file_sha256(original_path),
            "source_replay": source_replay,
            "source_replay_bitwise_verified": all(v["bitwise_equal"] for v in source_replay.values()),
            "teacher_base_is_shared_across_critics": True,
            "full_target_differs_by_each_critic_mean_offset": True,
            "decisions_are_exploratory": True, "paired": decisions,
        }, arrays)


def run_null(out: Path, cfg, states: dict, pairs: int, seed_base: int):
    network_def, tx = original_critic_def(cfg), critic_optimizer(cfg.agent)
    counts = {}
    for point in states:
        state, _, raw_obs, actions, normalizer = states[point]
        check_index = 2 if point == "early" else 20
        fires = {"a_as_current": 0, "b_as_current": 0}
        for pair_id in range(pairs):
            seed = seed_base + pair_id
            results, arrays = [], {}
            for r in range(int(cfg.probe.rounds)):
                obs, act = pool(raw_obs, actions, normalizer, seed, check_index, r,
                                int(cfg.probe.pool_size))
                key = jax.random.fold_in(jax.random.PRNGKey(seed), STREAM)
                a_key, b_key = jax.random.split(key)
                a = init_fresh(network_def, a_key, obs, act)
                b = init_fresh(network_def, b_key, obs, act)
                result = jax.device_get(probe_round(network_def, {
                    "fresh_a": (network_def, a), "fresh_b": (network_def, b),
                }, tx, obs, act, probe_key(seed, check_index, r), probe_config(cfg)))
                results.append(result)
                for arm, fields in result.items():
                    for field, value in fields.items():
                        arrays[f"round_{r}/{arm}_{field}"] = np.asarray(value)
            combined = {arm: {field: np.stack([one[arm][field] for one in results])
                              for field in results[0][arm]} for arm in results[0]}
            decisions = {
                "a_as_current": paired_decision(combined, "fresh_a", "fresh_b", seed, check_index, cfg),
                "b_as_current": paired_decision(combined, "fresh_b", "fresh_a", seed, check_index, cfg),
            }
            for direction in fires:
                fires[direction] += int(decisions[direction]["fires"])
            publish(out, f"null_{point}_{pair_id:03d}", {
                "pool_state": str(state), "pair_id": pair_id, "seed": seed,
                **decisions,
                "not_independent_directions": True,
            }, arrays)
        counts[point] = {direction: {
            "fires": count, "independent_pairs": pairs,
            "frequency": count / pairs,
            "exact_binomial_95_ci": exact_binomial_interval(count, pairs),
        } for direction, count in fires.items()}
    atomic_write_text(out / "null_summary.json", json.dumps(counts, indent=2, sort_keys=True))


def run_conditioning(out: Path, cfg, states: dict, fresh):
    """One round per state/pool; gradient traces are exploratory extra work."""
    network_def, tx = original_critic_def(cfg), critic_optimizer(cfg.agent)
    for point, (state, _, raw_obs, actions, normalizer) in states.items():
        check_index = 2 if point == "early" else 20
        obs, act = pool(raw_obs, actions, normalizer, int(cfg.seed), check_index, 0,
                        int(cfg.probe.pool_size))
        target_key, idx_key = jax.random.split(probe_key(int(cfg.seed), check_index, 0))
        base = _base_targets(network_def, target_key, obs, act, int(cfg.probe.eval_chunk),
                             float(cfg.probe.target_scale))
        perturbed = _base_targets(network_def, target_key, obs + 1e-5, act,
                                  int(cfg.probe.eval_chunk), float(cfg.probe.target_scale))
        indices = followup_indices(idx_key, int(cfg.probe.batch_size), int(cfg.probe.pool_size), 2000)
        arrays, summary = {}, {
            "pool_state": str(state), "round": 0, "exploratory_gradient_instrumentation": True,
            "teacher_base_variance": float(jnp.var(base)),
            "teacher_base_quantiles": np.quantile(np.asarray(base), [0, .01, .25, .5, .75, .99, 1]).tolist(),
            "teacher_base_finite": bool(np.isfinite(np.asarray(base)).all()),
            "teacher_perturbation_rms_at_1e_minus_5": float(jnp.sqrt(jnp.mean((base - perturbed) ** 2))),
            "teacher_perturbation_changed_fraction_at_1e_minus_5": float(np.mean(
                np.asarray(base) != np.asarray(perturbed))),
            "normalizer_mean_rms": float(np.sqrt(np.mean(np.square(normalizer["mean"])))),
            "normalizer_var_quantiles": np.quantile(normalizer["var"], [0, .5, 1]).tolist(),
            "conditions": {},
        }
        for arm, params in (("fresh", fresh), ("early", states["early"][1]),
                            ("late", states["late"][1])):
            prediction = _chunked(lambda o, a: network_def.apply({"params": params}, o, a).reshape(-1),
                                  obs, act, int(cfg.probe.eval_chunk))
            offset = jnp.mean(prediction)
            history = diagnostic_fit(network_def, tx, params, obs, act, base, indices,
                                     int(cfg.probe.eval_chunk))
            summary["conditions"][arm] = {
                "initial_prediction_mean": float(offset),
                "initial_prediction_variance": float(jnp.var(prediction)),
                "initial_full_pool_mse": float(jnp.mean((prediction - offset - base) ** 2)),
                "initial_parameter_norm": float(optax.global_norm(params)),
                "final_full_pool_mse": history[-1]["pool_mse"],
                "optimizer_count_at_2000": history[-1]["optimizer_step"],
            }
            arrays[f"{arm}_losses"] = np.concatenate([h["losses"] for h in history])
            arrays[f"{arm}_gradient_norms"] = np.concatenate([h["gradient_norms"] for h in history])
            losses = arrays[f"{arm}_losses"]
            gradients = arrays[f"{arm}_gradient_norms"]
            summary["conditions"][arm].update({
                "loss_auc_first_1000": float(np.mean(losses[:1000])),
                "loss_auc_last_1000": float(np.mean(losses[1000:])),
                "gradient_norm_quantiles": np.quantile(gradients, [0, .5, .95, 1]).tolist(),
                "gradients_finite": bool(np.isfinite(gradients).all()),
            })
        publish(out, f"conditioning_{point}", summary, arrays)


def run_scales(out: Path, cfg, states: dict, fresh, scales: list[float]):
    network_def, tx = original_critic_def(cfg), critic_optimizer(cfg.agent)
    point = "late"
    state, _, raw_obs, actions, normalizer = states[point]
    for scale in scales:
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("target scales must be finite and positive")
        rounds, arrays = [], {}
        probe_cfg = replace(probe_config(cfg), target_scale=float(scale))
        for r in range(probe_cfg.rounds):
            obs, act = pool(raw_obs, actions, normalizer, int(cfg.seed), 20, r, probe_cfg.pool_size)
            result = jax.device_get(probe_round(network_def, {
                "current": (network_def, states["late"][1]),
                "fresh": (network_def, fresh),
            }, tx, obs, act, probe_key(int(cfg.seed), 20, r), probe_cfg))
            rounds.append(result)
            for arm, fields in result.items():
                for field, value in fields.items():
                    arrays[f"round_{r}/{arm}_{field}"] = np.asarray(value)
        combined = {arm: {field: np.stack([one[arm][field] for one in rounds])
                          for field in rounds[0][arm]} for arm in rounds[0]}
        publish(out, f"scale_{scale:g}", {
            "pool_state": str(state), "target_scale": scale,
            "exploratory_only": True,
            "paired": paired_decision(combined, "current", "fresh", int(cfg.seed), 20, cfg),
        }, arrays)


def run_extension(out: Path, cfg, states: dict, fresh, steps: int):
    network_def, tx = original_critic_def(cfg), critic_optimizer(cfg.agent)
    state, _, raw_obs, actions, normalizer = states["late"]
    arrays, summaries = {}, []
    for r in range(int(cfg.probe.rounds)):
        obs, act = pool(raw_obs, actions, normalizer, int(cfg.seed), 20, r,
                        int(cfg.probe.pool_size))
        target_key, idx_key = jax.random.split(probe_key(int(cfg.seed), 20, r))
        base = _base_targets(network_def, target_key, obs, act, int(cfg.probe.eval_chunk),
                             float(cfg.probe.target_scale))
        indices = followup_indices(idx_key, int(cfg.probe.batch_size), int(cfg.probe.pool_size), steps)
        round_summary = {"round": r, "target_variance": float(jnp.var(base))}
        for arm, params in (("fresh", fresh), ("trained", states["late"][1])):
            history = diagnostic_fit(network_def, tx, params, obs, act, base, indices,
                                     int(cfg.probe.eval_chunk))
            arrays[f"round_{r}/{arm}_losses"] = np.concatenate([h["losses"] for h in history])
            arrays[f"round_{r}/{arm}_gradient_norms"] = np.concatenate(
                [h["gradient_norms"] for h in history])
            round_summary[arm] = [{k: v for k, v in h.items() if k not in ("losses", "gradient_norms")}
                                  for h in history]
        summaries.append(round_summary)
    publish(out, f"extension_{steps}", {
        "pool_state": str(state), "rounds": summaries,
        "exploratory_only": True,
        "first_2000_draws_match_method_c": True,
        "optimizer_parameters_and_state_continue_without_restart": True,
    }, arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("inventory", "crossover", "null", "conditioning",
                                           "scales", "extension"), required=True)
    parser.add_argument("--points", nargs="+", choices=("early", "late"), default=["early", "late"])
    parser.add_argument("--null-pairs", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=92000)
    parser.add_argument("--scales", nargs="+", type=float, default=[10000.0, 100000.0, 1000000.0])
    parser.add_argument("--extension-steps", type=int, choices=(4000, 8000), default=4000)
    parser.add_argument("--expected-harness-sha")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    inventory = artifact_inventory(run_dir)
    if args.mode == "inventory":
        print(json.dumps(inventory, indent=2))
        return
    if args.output is None or not args.output.is_absolute() or args.output.exists():
        raise ValueError("--output must be a new absolute directory")
    checkout = Path(__file__).resolve().parents[2]
    if run_dir == args.output or run_dir in args.output.parents or checkout in args.output.parents:
        raise ValueError("--output must be outside the source run and checkout")
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain",
                                     "--untracked-files=all"], text=True).strip()
    if args.expected_harness_sha != head or len(head) != 40 or dirty:
        raise ValueError("expected exact clean diagnostic-harness checkout")
    if args.null_pairs < 2 or args.null_pairs > 20:
        raise ValueError("--null-pairs must be in [2, 20]")
    if args.mode != "null" and not inventory["fresh_critic"]:
        raise FileNotFoundError("parent's fresh_critic checkpoint is missing")
    metadata = source_metadata(run_dir)
    cfg = OmegaConf.create(metadata["resolved_config"])
    probe_spec = (int(cfg.probe.rounds), int(cfg.probe.steps), int(cfg.probe.checks),
                  int(cfg.probe.pool_size), int(cfg.probe.batch_size),
                  float(cfg.probe.target_scale), int(cfg.probe.eval_chunk))
    trigger_spec = (int(cfg.trigger.resamples), float(cfg.trigger.confidence),
                    int(cfg.trigger.consecutive_checks), float(cfg.trigger.null_threshold),
                    float(cfg.trigger.eligible_fraction))
    if probe_spec != (5, 2000, 20, 25600, 256, 100000.0, 2560) or trigger_spec != (
            10000, 0.95, 2, 0.0, 0.95):
        raise ValueError("source configuration differs from approved Method C")
    points = (tuple(dict.fromkeys(args.points)) if args.mode == "null" else
              ("late",) if args.mode in ("scales", "extension") else ("early", "late"))
    states = {point: load_state(run_dir, point, cfg) for point in points}
    fresh = (orbax.checkpoint.PyTreeCheckpointer().restore(inventory["fresh_critic"])["params"]
             if args.mode != "null" else None)
    out = args.output
    out.mkdir(parents=True)
    evidence = {
        "mode": args.mode, "source": inventory,
        "analysis_commit": head,
        "state_inputs": {point: {
            **{name: file_sha256(path / name)
               for name in ("buffer.npz", "buffer_meta.pkl", "meta.pkl", "obs_rms.pkl")},
            "agent_ckpt_tree": tree_sha256(path / "agent_ckpt"),
        }
                         for point, (path, *_rest) in states.items()},
        "official_source_unchanged": True,
    }
    if fresh is not None:
        evidence["fresh_critic_tree_sha256"] = tree_sha256(Path(inventory["fresh_critic"]))
    atomic_write_text(out / "provenance.json", json.dumps(evidence, indent=2, sort_keys=True))
    if args.mode == "crossover":
        run_crossover(run_dir, out, cfg, states, fresh)
    elif args.mode == "null":
        run_null(out, cfg, states, args.null_pairs, args.seed_base)
    elif args.mode == "conditioning":
        run_conditioning(out, cfg, states, fresh)
    elif args.mode == "scales":
        run_scales(out, cfg, states, fresh, args.scales)
    else:
        run_extension(out, cfg, states, fresh, args.extension_steps)


if __name__ == "__main__":
    main()
