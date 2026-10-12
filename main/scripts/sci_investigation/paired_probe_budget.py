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
import sys
import tarfile
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
    _base_targets, _chunked, critic_optimizer, iqm, probe_config, probe_key, probe_rng, sample_pool,
)
from experiments.exp12.trigger import bootstrap_interval
from scripts.probe_fresh_checks import ARCHS, NULL_PAIR_KEY_OFFSET, _fresh_trainer, _stub_wandb


@functools.partial(jax.jit, static_argnames=("network_def", "tx", "chunk"))
def staged_fit(network_def, tx, params, obs, act, base, first_idx, second_idx, chunk):
    """Same _fit update order; return complete per-step curves and both pool MSEs."""
    def predict(p):
        return _chunked(lambda o, a: network_def.apply({"params": p}, o, a).reshape(-1),
                        obs, act, chunk)

    offset = jnp.mean(predict(params))
    targets = offset + base

    def loss_fn(p, idx):
        q = network_def.apply({"params": p}, obs[idx], act[idx]).reshape(-1)
        return jnp.mean((q - targets[idx]) ** 2)

    def step(carry, idx):
        p, state = carry
        loss, grad = jax.value_and_grad(loss_fn)(p, idx)
        update, state = tx.update(grad, state, p)
        return (optax.apply_updates(p, update), state), loss

    carry, first_losses = jax.lax.scan(step, (params, tx.init(params)), first_idx)
    first_final = jnp.mean((predict(carry[0]) - targets) ** 2)
    carry, second_losses = jax.lax.scan(step, carry, second_idx)
    second_final = jnp.mean((predict(carry[0]) - targets) ** 2)
    return first_losses, second_losses, first_final, second_final, offset


def _archive_arrays(archive, suffix):
    with tarfile.open(archive, "r:gz") as tar:
        names = [m for m in tar.getmembers() if m.name.endswith(suffix)]
        if len(names) != 1 or not names[0].isfile():
            raise ValueError(f"archive needs exactly one {suffix}")
        data = tar.extractfile(names[0]).read()
    return {k: v for k, v in np.load(io.BytesIO(data)).items()}


def _round_data(args, trainer, round_index, check_index, cfg):
    if args.mode == "hopper":
        saved = _archive_arrays(args.capture_archive, f"round_{round_index}_teacher_pool.npz")
        return jnp.asarray(saved["observations"]), jnp.asarray(saved["actions"]), jnp.asarray(saved["base_targets"])
    obs, act = sample_pool(trainer.agent, trainer.buffer,
                           probe_rng(args.seed, check_index, round_index), cfg.pool_size)
    target_key, _ = jax.random.split(probe_key(args.seed, check_index, round_index))
    base = _base_targets(trainer._sac_agent.critic.network_def, target_key, obs, act,
                         cfg.eval_chunk, cfg.target_scale)
    return obs, act, base


def _one_check(args, trainer, tx, critic_params, check_index, cfg):
    network = trainer._sac_agent.critic.network_def
    fresh_scores = {1000: [], 2000: []}
    current_scores = {1000: [], 2000: []}
    baseline, curves = [], []
    params = {"fresh": critic_params}
    for rnd in range(cfg.rounds):
        obs, act, base = _round_data(args, trainer, rnd, check_index, cfg)
        key = probe_key(args.seed, check_index, rnd)
        _, idx_key = jax.random.split(key)
        first_idx = jax.random.randint(idx_key, (1000, cfg.batch_size), 0, cfg.pool_size)
        # Separate follow-up key preserves the exact original first-1000 stream.
        second_idx = jax.random.randint(jax.random.fold_in(idx_key, 1),
                                        (1000, cfg.batch_size), 0, cfg.pool_size)
        b = float(jax.device_get(jnp.var(base)))
        baseline.append(b)
        if args.mode == "null" and rnd == 0:
            obs0 = jnp.zeros((1, obs.shape[1]))
            act0 = jnp.zeros((1, act.shape[1]))
            keys = jax.random.split(jax.random.PRNGKey(NULL_PAIR_KEY_OFFSET + args.pair), 2)
            params = {"fresh": network.init(keys[0], obs0, act0)["params"],
                      "current": network.init(keys[1], obs0, act0)["params"]}
        round_curves = {}
        for name, p in params.items():
            first, second, mse1, mse2, offset = staged_fit(network, tx, p, obs, act, base,
                                                            first_idx, second_idx, cfg.eval_chunk)
            first, second, mse1, mse2, offset = jax.device_get((first, second, mse1, mse2, offset))
            scores = fresh_scores if name == "fresh" else current_scores
            scores[1000].append(b - float(mse1))
            scores[2000].append(b - float(mse2))
            round_curves[name] = {"losses_1000": np.asarray(first), "losses_2000": np.asarray(second),
                                  "mse_1000": float(mse1), "mse_2000": float(mse2), "offset": float(offset)}
        curves.append(round_curves)
    result = {"pair": args.pair if args.mode == "null" else None, "check_index": check_index,
              "seed": args.seed, "b_rounds": baseline, "rounds": []}
    for rnd, rd in enumerate(curves):
        result["rounds"].append({n: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                     for k, v in fields.items()} for n, fields in rd.items()})
    for steps in (1000, 2000):
        s = np.asarray(fresh_scores[steps], dtype=np.float32)
        result[f"fresh_{steps}"] = {"score_rounds": s.tolist(), "score_iqm": iqm(s),
                                     "range_ratio": iqm(s) / iqm(baseline)}
        if args.mode == "null":
            c = np.asarray(current_scores[steps], dtype=np.float32)
            loss = s - c
            low, high = bootstrap_interval(loss, args.seed, check_index, 10000, .95)
            result[f"null_{steps}"] = {"loss_rounds": loss.tolist(), "loss_iqm": iqm(loss),
                                        "ci_low": low, "ci_high": high,
                                        "fired_zero": bool(np.isfinite(low) and low > 0)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hopper", "null"), required=True)
    parser.add_argument("--capture-archive", type=Path)
    parser.add_argument("--seed", type=int, default=990)
    parser.add_argument("--pairs", type=int, default=100)
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
        parser.error("hopper mode needs the original capture archive")
    args.out_dir.mkdir(parents=True)
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
                      "note": "exploratory paired fitting; production settings unchanged"}
        (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
        tx = critic_optimizer(trainer.cfg.agent)
        for pair in range(1 if args.mode == "hopper" else args.pairs):
            args.pair = pair
            result = _one_check(args, trainer, tx, trainer._sac_agent.critic.params,
                                0 if args.mode == "hopper" else pair + 1, cfg)
            if args.mode == "hopper":
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
            with path.open("x") as stream:
                json.dump(result, stream)
            print(json.dumps({"file": str(path), "pair": pair,
                              "p1000": result["fresh_1000"]["range_ratio"],
                              "p2000": result["fresh_2000"]["range_ratio"]}), flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
