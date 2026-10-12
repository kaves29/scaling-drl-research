"""Exploratory paired 1000/2000-step fresh-probe diagnostic; no production writes.

The first 1000 minibatches use the original probe RNG call and the second 1000
continue the *same* parameters and AdamW state. Output never activates a rule.
"""

import argparse
import functools
import hashlib
import io
import json
import os
import resource
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.hardware import configure_hardware_env

configure_hardware_env()
from experiments.exp12.precision import configure_compilation_cache, runtime_info, set_matmul_precision

set_matmul_precision()
configure_compilation_cache()

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.exp12.probe import (
    _base_targets, _chunked, _mean_prediction, critic_optimizer, iqm, probe_config, probe_key, probe_rng, sample_pool,
)
from experiments.exp12.trigger import bootstrap_interval
from scripts.probe_fresh_checks import ARCHS, NULL_PAIR_KEY_OFFSET, _fresh_trainer, _stub_wandb
from scripts.sci_investigation.compare_probe_decisions import exact_binomial_interval


@functools.partial(jax.jit, static_argnames=("network_def", "tx", "chunk"))
def fit_stage(network_def, tx, carry, obs, act, base, idx, chunk, offset):
    """One uninterrupted optimizer stage, returning the live parameter/AdamW state."""
    def predict(p):
        return _chunked(lambda o, a: network_def.apply({"params": p}, o, a).reshape(-1),
                        obs, act, chunk)

    targets = offset + base

    def loss_fn(p, idx):
        q = network_def.apply({"params": p}, obs[idx], act[idx]).reshape(-1)
        return jnp.mean((q - targets[idx]) ** 2)

    def step(carry, idx):
        p, state = carry
        loss, grad = jax.value_and_grad(loss_fn)(p, idx)
        update, state = tx.update(grad, state, p)
        return (optax.apply_updates(p, update), state), loss

    carry, losses = jax.lax.scan(step, carry, idx)
    final = jnp.mean((predict(carry[0]) - targets) ** 2)
    return carry, losses, final


def staged_fit(network_def, tx, params, obs, act, base, first_idx, second_idx, chunk):
    """CPU reference-compatible first stage and optimizer-preserving continuation."""
    offset = _mean_prediction(network_def, params, obs, act, chunk)
    carry, first_losses, first_final = fit_stage(
        network_def, tx, (params, tx.init(params)), obs, act, base, first_idx, chunk, offset)
    carry, second_losses, second_final = fit_stage(
        network_def, tx, carry, obs, act, base, second_idx, chunk, offset)
    return first_losses, second_losses, first_final, second_final, offset


def _archive_arrays(archive, suffix):
    with tarfile.open(archive, "r:gz") as tar:
        names = [m for m in tar.getmembers() if m.name.endswith(suffix)]
        if len(names) != 1 or not names[0].isfile():
            raise ValueError(f"archive needs exactly one {suffix}")
        data = tar.extractfile(names[0]).read()
    return {k: v for k, v in np.load(io.BytesIO(data)).items()}


def _round_data(args, trainer, round_index, check_index, cfg):
    if args.mode == "hopper" and args.seed == 990:
        saved = _archive_arrays(args.capture_archive, f"round_{round_index}_teacher_pool.npz")
        return jnp.asarray(saved["observations"]), jnp.asarray(saved["actions"]), jnp.asarray(saved["base_targets"])
    obs, act = sample_pool(trainer.agent, trainer.buffer,
                           probe_rng(args.seed, check_index, round_index), cfg.pool_size)
    target_key, _ = jax.random.split(probe_key(args.seed, check_index, round_index))
    base = _base_targets(trainer._sac_agent.critic.network_def, target_key, obs, act,
                         cfg.eval_chunk, cfg.target_scale)
    return obs, act, base


def _save_state(path, carry):
    """Selected read-only diagnostic snapshot; leaf order follows the pinned source pytree."""
    leaves = jax.tree_util.tree_leaves(jax.device_get(carry))
    temp = path.with_suffix(".partial")
    with temp.open("xb") as stream:
        np.savez(stream, **{f"leaf_{i:04d}": np.asarray(leaf) for i, leaf in enumerate(leaves)})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "sha256": digest.hexdigest(),
            "leaf_count": len(leaves)}


def _one_check(args, trainer, tx, critic_params, check_index, cfg):
    network = trainer._sac_agent.critic.network_def
    fresh_scores = {1000: [], 2000: []}
    current_scores = {1000: [], 2000: []}
    baseline, curves = [], []
    snapshots = []
    target_keys = []
    params = {"fresh": critic_params}
    for rnd in range(args.round_start, args.round_start + args.rounds):
        round_wall = time.monotonic()
        round_cpu = time.process_time()
        obs, act, base = _round_data(args, trainer, rnd, check_index, cfg)
        key = probe_key(args.seed, check_index, rnd)
        target_key, idx_key = jax.random.split(key)
        target_keys.append(np.asarray(jax.device_get(target_key), dtype=np.uint32).tolist())
        first_idx = jax.random.randint(idx_key, (1000, cfg.batch_size), 0, cfg.pool_size)
        # Separate follow-up key preserves the exact original first-1000 stream.
        second_idx = jax.random.randint(jax.random.fold_in(idx_key, 1),
                                        (1000, cfg.batch_size), 0, cfg.pool_size)
        b = np.float32(jax.device_get(jnp.var(base)))
        baseline.append(float(b))
        if args.mode == "null" and rnd == args.round_start:
            obs0 = jnp.zeros((1, obs.shape[1]))
            act0 = jnp.zeros((1, act.shape[1]))
            keys = jax.random.split(jax.random.PRNGKey(NULL_PAIR_KEY_OFFSET + args.pair), 2)
            params = {"fresh": network.init(keys[0], obs0, act0)["params"],
                      "current": network.init(keys[1], obs0, act0)["params"]}
        round_curves = {}
        for name, p in params.items():
            offset = _mean_prediction(network, p, obs, act, cfg.eval_chunk)
            carry = (p, tx.init(p))
            fit_wall = time.monotonic()
            fit_cpu = time.process_time()
            carry, first, mse1 = fit_stage(network, tx, carry, obs, act, base,
                                           first_idx, cfg.eval_chunk, offset)
            jax.block_until_ready(mse1)
            first_wall = time.monotonic() - fit_wall
            second_wall_start = time.monotonic()
            carry, second, mse2 = fit_stage(network, tx, carry, obs, act, base,
                                            second_idx, cfg.eval_chunk, offset)
            first, second, mse1, mse2, offset = jax.device_get((first, second, mse1, mse2, offset))
            second_wall = time.monotonic() - second_wall_start
            scores = fresh_scores if name == "fresh" else current_scores
            # Production P=b-final is a float32 device subtraction, not host float64.
            scores[1000].append(float(np.float32(b - np.float32(mse1))))
            scores[2000].append(float(np.float32(b - np.float32(mse2))))
            fit_elapsed = time.monotonic() - fit_wall
            if args.mode == "hopper" and args.seed == 990 and rnd == 0 and name == "fresh":
                snapshots.append(_save_state(args.out_dir / "optimizer_round0_step2000.npz", carry))
            round_curves[name] = {"losses_1000": np.asarray(first), "losses_2000": np.asarray(second),
                                  "mse_1000": float(mse1), "mse_2000": float(mse2), "offset": float(offset),
                                  "first_stage_wall_s": first_wall, "second_stage_wall_s": second_wall,
                                  "fit_wall_s": fit_elapsed,
                                  "fit_updates_per_wall_s": 2000 / fit_elapsed,
                                  "fit_cpu_s": time.process_time() - fit_cpu}
        round_curves["performance"] = {"round_wall_s": time.monotonic() - round_wall,
                                        "round_cpu_s": time.process_time() - round_cpu,
                                        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        curves.append(round_curves)
        if args.mode == "hopper":
            _atomic_json(args.out_dir / f"round_{rnd}.json", {
                "seed": args.seed, "round": rnd, "baseline": float(b),
                "fresh": {k: v.tolist() if isinstance(v, np.ndarray) else v
                          for k, v in round_curves["fresh"].items()},
                "performance": round_curves["performance"],
            })
    result = {"pair": args.pair if args.mode == "null" else None, "check_index": check_index,
              "seed": args.seed, "round_indices": list(range(args.round_start, args.round_start + args.rounds)),
              "target_keys": target_keys, "b_rounds": baseline, "rounds": [],
              "optimizer_snapshots": snapshots}
    for rnd, rd in enumerate(curves):
        result["rounds"].append({n: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                     for k, v in fields.items()} if isinstance(fields, dict) else fields
                                 for n, fields in rd.items()})
    for steps in (1000, 2000):
        s = np.asarray(fresh_scores[steps], dtype=np.float32)
        if not np.isfinite(s).all() or not np.isfinite(baseline).all():
            raise ValueError(f"nonfinite fresh score/baseline at {steps} steps")
        result[f"fresh_{steps}"] = {"score_rounds": s.tolist(), "score_iqm": iqm(s),
                                     "range_ratio": iqm(s) / iqm(baseline)}
        if args.mode == "null":
            c = np.asarray(current_scores[steps], dtype=np.float32)
            loss = s - c
            if not np.isfinite(loss).all():
                raise ValueError(f"nonfinite paired null loss at {steps} steps")
            low, high = bootstrap_interval(loss, args.seed, check_index, 10000, .95)
            result[f"null_{steps}"] = {"loss_rounds": loss.tolist(), "loss_iqm": iqm(loss),
                                        "ci_low": low, "ci_high": high,
                                        "fired_zero": bool(np.isfinite(low) and low > 0)}
    if args.mode == "null":
        if args.pooled_threshold is not None:
            result["null_1000"]["fired_pooled_p95"] = bool(
                np.isfinite(result["null_1000"]["ci_low"])
                and result["null_1000"]["ci_low"] > args.pooled_threshold)
    return result


def _atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("x") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _summary(out_dir, mode, requested, threshold, started):
    files = sorted(out_dir.glob("pair_*.json")) if mode == "null" else list(out_dir.glob("hopper.json"))
    rows = [json.loads(path.read_text()) for path in files]
    summary = {"mode": mode, "completed": len(rows), "requested": requested,
               "complete": len(rows) == requested, "elapsed_wall_s": time.monotonic() - started}
    if mode == "null" and rows and threshold is not None:
        summary["pooled_threshold"] = threshold
        summary["methods"] = {}
        for label, field in (("A_zero_1000", "null_1000"), ("B_pooled_p95_1000", "null_1000"),
                             ("C_zero_2000", "null_2000")):
            key = "fired_pooled_p95" if label.startswith("B") else "fired_zero"
            n = sum(bool(row[field][key]) for row in rows)
            summary["methods"][label] = {"fires": n, "pairs": len(rows),
                                          "exact_binomial_95": exact_binomial_interval(n, len(rows))}
    if mode == "hopper" and rows:
        summary["range_ratio_1000"] = rows[0]["fresh_1000"]["range_ratio"]
        summary["range_ratio_2000"] = rows[0]["fresh_2000"]["range_ratio"]
    _atomic_json(out_dir / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hopper", "null"), required=True)
    parser.add_argument("--capture-archive", type=Path)
    parser.add_argument("--seed", type=int, default=990)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--round-start", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--pooled-threshold", type=float,
                        help="exploratory p95 IQM from two verified historical null datasets")
    parser.add_argument("--reference-null-jsonl", type=Path,
                        help="optional historical seed-991 rows for first-1000 comparison")
    parser.add_argument("--max-elapsed-seconds", type=int, default=14400)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    from utils.run_metadata import code_version

    code, runtime = code_version(), runtime_info()
    if code["commit"] != args.expected_commit or code["dirty"] is not False:
        parser.error("requires exact clean source revision")
    if runtime["platform"] != "gpu" or "A100" not in runtime["device_kind"]:
        parser.error("requires NVIDIA A100 GPU backend")
    if not args.out_dir.is_absolute() or args.out_dir.exists():
        parser.error("--out-dir must be a new absolute path")
    if args.mode == "hopper" and (args.capture_archive is None or not args.capture_archive.is_file()):
        if args.seed == 990:
            parser.error("historical hopper mode needs the original capture archive")
    if args.mode == "null" and args.round_start == 0:
        if args.pooled_threshold is None or not np.isfinite(args.pooled_threshold):
            parser.error("original null comparison needs a finite historical pooled threshold")
    if args.rounds != 5 or args.round_start not in (0, 5):
        parser.error("only the original rounds 0..4 or D-only rounds 5..9 are supported")
    if args.round_start == 5 and (args.mode != "null" or args.seed != 991 or args.pooled_threshold is not None):
        parser.error("extra target rounds are a separate seed-991 descriptive null stage")
    reference_null = {}
    if args.reference_null_jsonl:
        if args.mode != "null" or args.seed != 991:
            parser.error("historical null comparison requires seed-991 null mode")
        with args.reference_null_jsonl.open() as stream:
            reference_null = {row["pair"]: row for row in (json.loads(line) for line in stream)}
        if set(reference_null) != set(range(100)):
            parser.error("historical null input requires exactly 100 unique pairs")
    if args.max_elapsed_seconds < 1 or args.pairs < 1:
        parser.error("positive limits required")
    args.out_dir.mkdir(parents=True)
    started = time.monotonic()
    _stub_wandb()
    arch = "D4W1536" if args.mode == "hopper" else "D4W1024"
    env = "hopper-hop" if args.mode == "hopper" else "dog-run"
    blocks, width = ARCHS[arch]
    config_args = argparse.Namespace(env=env, env_group="dmc_hard", override=[])
    trainer = _fresh_trainer(config_args, blocks, width, args.seed)
    try:
        cfg = probe_config(trainer.cfg)
        if (cfg.rounds, cfg.steps, cfg.pool_size, cfg.batch_size, cfg.target_scale) != (5, 1000, 25600, 256, 1e5):
            raise ValueError("original probe settings differ")
        provenance = {"code": code, "runtime": runtime, "mode": args.mode,
                      "arch": arch, "env": env, "seed": args.seed,
                      "capture_sha256": hashlib.sha256(args.capture_archive.read_bytes()).hexdigest()
                      if args.capture_archive else None, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                      "reference_null_sha256": hashlib.sha256(args.reference_null_jsonl.read_bytes()).hexdigest()
                      if args.reference_null_jsonl else None,
                      "pooled_threshold": args.pooled_threshold,
                      "round_start": args.round_start, "rounds": args.rounds,
                      "note": "exploratory paired fitting; production settings unchanged"}
        _atomic_json(args.out_dir / "provenance.json", provenance)
        tx = critic_optimizer(trainer.cfg.agent)
        for pair in range(1 if args.mode == "hopper" else args.pairs):
            if time.monotonic() - started >= args.max_elapsed_seconds:
                _summary(args.out_dir, args.mode, 1 if args.mode == "hopper" else args.pairs,
                         args.pooled_threshold, started)
                raise TimeoutError("diagnostic reached its predeclared elapsed-time limit; partial results preserved")
            args.pair = pair
            result = _one_check(args, trainer, tx, trainer._sac_agent.critic.params,
                                0 if args.mode == "hopper" else pair + 1, cfg)
            if reference_null:
                old = reference_null[pair]
                old_loss = np.asarray(old["loss_rounds"], dtype=np.float32)
                new_loss = np.asarray(result["null_1000"]["loss_rounds"], dtype=np.float32)
                result["historical_1000_comparison"] = {
                    "loss_rounds_bitwise_equal": bool(np.array_equal(old_loss.view(np.uint32),
                                                                        new_loss.view(np.uint32))),
                    "ci_low_equal": old["ci_low"] == result["null_1000"]["ci_low"],
                    "max_abs_loss_difference": float(np.max(np.abs(old_loss - new_loss))),
                }
            if args.mode == "hopper" and args.seed == 990:
                old = _archive_arrays(args.capture_archive, "range_D4W1536_pool_25600_curves.npz")
                new_losses = np.asarray([rnd["fresh"]["losses_1000"] for rnd in result["rounds"]],
                                        dtype=np.float32)
                new_final = np.asarray([rnd["fresh"]["mse_1000"] for rnd in result["rounds"]],
                                       dtype=np.float32)
                result["historical_1000_comparison"] = {
                    "curves_bitwise_equal": bool(np.array_equal(new_losses.view(np.uint32),
                                                                   old["fresh_losses"].view(np.uint32))),
                    "final_mse_bitwise_equal": bool(np.array_equal(new_final.view(np.uint32),
                                                                      old["fresh_final_loss"].view(np.uint32))),
                    "max_abs_curve_difference": float(np.max(np.abs(new_losses - old["fresh_losses"]))),
                    "max_abs_final_mse_difference": float(np.max(np.abs(new_final - old["fresh_final_loss"]))),
                }
            path = args.out_dir / (f"pair_{pair:03d}.json" if args.mode == "null" else "hopper.json")
            _atomic_json(path, result)
            _summary(args.out_dir, args.mode, 1 if args.mode == "hopper" else args.pairs,
                     args.pooled_threshold, started)
            print(json.dumps({"file": str(path), "pair": pair,
                              "p1000": result["fresh_1000"]["range_ratio"],
                              "p2000": result["fresh_2000"]["range_ratio"]}), flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
