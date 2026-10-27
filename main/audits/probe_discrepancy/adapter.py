"""Source-bound observational Exp1 pilot adapter.

This entrypoint replaces only the probe's data capture path. It retains the
original pool, teacher, own-offset targets, and one continuous AdamW scan.
Every output is written beneath the controller-assigned run directory.
"""

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import signal
import subprocess
import sys
import time
import traceback


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".pending")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def array_record(value):
    import numpy as np

    arr = np.asarray(value)
    return {
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "order": "C",
        "sha256": hashlib.sha256(arr.tobytes(order="C")).hexdigest(),
        "nonfinite": int(np.count_nonzero(~np.isfinite(arr)))
        if np.issubdtype(arr.dtype, np.number)
        else None,
    }


def tree_arrays(tree):
    import jax
    import numpy as np

    paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {f"leaf_{i:05d}": np.asarray(jax.device_get(value)) for i, (_, value) in enumerate(paths)}


def persist_tree(path, tree):
    import jax

    leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    record = persist_arrays(path, tree_arrays(tree))
    record["treedef"] = str(treedef)
    record["leaf_paths"] = {f"leaf_{i:05d}": str(key_path)
                            for i, (key_path, _) in enumerate(leaves)}
    return record


def persist_arrays(path, arrays):
    import numpy as np

    path = Path(path)
    temporary = path.with_name("." + path.name + ".pending")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": file_sha(path),
        "bytes": path.stat().st_size,
        "arrays": {name: array_record(value) for name, value in arrays.items()},
    }


def persist_state(path, state):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".pending")
    with temporary.open("xb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def expected_config(row, run_dir, project):
    """Compose the original campaign config and verify every scientific field."""
    from omegaconf import OmegaConf

    from experiments.exp1 import compose_config

    blocks = row["source_scientific_config"]["agent"]["critic_num_blocks"]
    width = row["source_scientific_config"]["agent"]["critic_hidden_dim"]
    group = "myosuite_simba" if row["environment"].startswith("myo-") else "dmc_hard"
    overrides = [
        f"critic_num_blocks={blocks}",
        f"critic_hidden_dim={width}",
        f"env_name={row['environment']}",
        f"env={group}",
        f"seed={row['seed']}",
        f"results_root={run_dir / 'results'}",
        "env.num_env_steps=1000000",
        f"project_name={project}",
    ]
    cfg = compose_config(str(Path(__file__).resolve().parents[2] / "configs"), "base_exp12_ten", overrides)
    actual = OmegaConf.to_container(cfg, resolve=True)
    scientific = {k: v for k, v in actual.items() if k not in ("results_root", "project_name", "server")}
    expected = row["source_scientific_config"]
    for key in ("num_params", "actor_num_params", "critic_num_params"):
        scientific[key] = expected[key]
    # OmegaConf resolves YAML zero sparsity as 0.0; the source snapshot used
    # JSON integer 0. Check numerical equality, then bind the pinned snapshot.
    if scientific != expected or digest(expected) != row["source_scientific_config_sha256"]:
        raise ValueError("composed scientific config differs from pinned source")
    return overrides, cfg


class Capture:
    def __init__(self, run_dir, row, expected_commit, plan_sha):
        self.root = Path(run_dir) / "diagnostic"
        self.root.mkdir(exist_ok=False)
        self.row = row
        self.commit = expected_commit
        self.plan_sha = plan_sha
        self.checks = []
        self.started = time.time()

    def event(self, event, **fields):
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": time.time(), "event": event, **fields}, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def trainer_state(self, trainer, check_index):
        import numpy as np

        from experiments.exp12.envs import env_restore_state

        if check_index not in self.row["retained_checks"]:
            return
        check_dir = self.root / f"check_{check_index:02d}"
        check_dir.mkdir(exist_ok=True)
        state = {
            "interaction_step": trainer.interaction_step,
            "update_step": trainer.update_step,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "train_env": env_restore_state(trainer.train_env),
            "eval_env": env_restore_state(trainer.eval_env),
            "agent_rng": tree_arrays(trainer._sac_agent._rng),
            "replay_n_step_transitions": list(trainer.buffer._n_step_transitions),
            "observations": trainer.observations,
            "timestep": trainer.timestep,
        }
        record = persist_state(check_dir / "trainer_before_probe.pkl", state)
        self.event("trainer_state", check=check_index, record=record)

    def checkpoint_state(self, trainer, path, seconds):
        path = Path(path)
        files = {str(item.relative_to(path)): {"sha256": file_sha(item), "bytes": item.stat().st_size}
                 for item in sorted(path.rglob("*")) if item.is_file()}
        record = {"interaction_step": trainer.interaction_step, "path": str(path),
                  "save_seconds": seconds, "files": files,
                  "total_bytes": sum(item["bytes"] for item in files.values())}
        with (self.root / "checkpoints.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.event("checkpoint_saved", step=trainer.interaction_step,
                   seconds=seconds, bytes=record["total_bytes"])

    def run_probe(self, agent, buffer, target_def, critics, tx, seed, check_index, cfg, shared_offset=None):
        """One source-equivalent probe pass with retained values, never a refit."""
        import jax
        import jax.numpy as jnp
        import numpy as np

        from experiments.exp12 import probe

        if shared_offset is not None:
            raise ValueError("pilot does not authorize shared-offset sensitivity")
        if cfg.steps != 2000 or cfg.rounds != 10:
            raise ValueError("pilot probe schedule changed")
        selected = check_index in self.row["retained_checks"]
        check_dir = self.root / f"check_{check_index:02d}"
        check_dir.mkdir(exist_ok=True)
        if (check_dir / "partial.json").exists() or (check_dir / "complete.json").exists():
            raise ValueError(f"probe check {check_index} already captured; refusing overwrite")
        check_start = time.perf_counter()
        checkpoint = {"check_index": check_index, "selected": selected, "rounds": []}
        if selected:
            checkpoint["replay"] = {
                "size": int(buffer._num_in_buffer),
                "capacity": int(buffer._max_length),
                "current_idx": int(buffer._current_idx),
            }
            if hasattr(agent, "obs_rms"):
                checkpoint["normalizer"] = persist_arrays(
                    check_dir / "normalizer.npz",
                    {
                        "mean": np.asarray(agent.obs_rms.mean),
                        "var": np.asarray(agent.obs_rms.var),
                        "count": np.asarray(agent.obs_rms.count),
                        "epsilon": np.asarray(agent.epsilon),
                    },
                )
            checkpoint["rng"] = {"probe_seed": int(seed),
                                  "trainer_state": str(check_dir / "trainer_before_probe.pkl")}
            checkpoint["initial_trees"] = {}
            for name, critic in critics.items():
                checkpoint["initial_trees"][name] = persist_tree(
                    check_dir / f"initial_{name}.npz", critic[1]
                )
        rounds = []
        for round_index in range(cfg.rounds):
            round_start = time.perf_counter()
            rng = probe.probe_rng(seed, check_index, round_index)
            sample_index = rng.integers(0, buffer._num_in_buffer, size=cfg.pool_size)
            raw_obs = buffer._observations[sample_index]
            raw_act = buffer._actions[sample_index]
            normalized = agent._normalize(raw_obs) if hasattr(agent, "_normalize") else raw_obs
            obs, act = jnp.asarray(normalized), jnp.asarray(raw_act)
            key = probe.probe_key(seed, check_index, round_index)
            target_key, idx_key = jax.random.split(key)
            if selected:
                base, prescale, teacher = teacher_values(target_def, target_key, obs, act, cfg.eval_chunk, cfg.target_scale)
                teacher_record = persist_tree(
                    check_dir / f"round_{round_index:02d}_teacher.npz", teacher
                )
                del teacher
            else:
                base = probe._base_targets(target_def, target_key, obs, act, cfg.eval_chunk, cfg.target_scale)
            b = jnp.var(base)
            first = jax.random.randint(idx_key, (probe.ORIGINAL_FIT_STEPS, cfg.batch_size), 0, cfg.pool_size)
            second = jax.random.randint(jax.random.fold_in(idx_key, 1),
                                        (cfg.steps - probe.ORIGINAL_FIT_STEPS, cfg.batch_size), 0, cfg.pool_size)
            fit_index = jnp.concatenate((first, second), axis=0)
            details = {}
            result = {}
            for name, critic in critics.items():
                network, params, own_tx = probe._unpack(critic, tx)
                fit_start = time.perf_counter()
                if selected:
                    losses, final, offset, q_trace, targets = fit_values(
                        network, own_tx, params, obs, act, base, fit_index, cfg.eval_chunk
                    )
                    score = b - final
                    losses, final, offset, q_trace, targets, score = jax.device_get(
                        (losses, final, offset, q_trace, targets, score)
                    )
                else:
                    losses, final, offset = probe._fit(
                        network, own_tx, params, obs, act, base, fit_index, cfg.eval_chunk
                    )
                    score = b - final
                    losses, final, offset, score = jax.device_get((losses, final, offset, score))
                fit_seconds = time.perf_counter() - fit_start
                result[name] = {"losses": losses, "final_loss": final, "offset": offset,
                                "b": jax.device_get(b), "score": score}
                details[name] = {"fit_seconds": fit_seconds, "losses": losses, "offset": offset}
                if selected:
                    details[name].update(full_pool_q=q_trace, literal_targets=targets)
            rounds.append(result)
            retained = {"round": round_index, "duration_seconds": time.perf_counter() - round_start,
                        "score": {name: float(fields["score"]) for name, fields in result.items()},
                        "final_mse": {name: float(fields["final_loss"]) for name, fields in result.items()},
                        "b": float(jax.device_get(b)),
                        "fit_seconds": {name: details[name]["fit_seconds"] for name in details}}
            arrays = {}
            for name, fields in details.items():
                arrays[f"{name}_losses_preupdate"] = fields["losses"]
                arrays[f"{name}_offset"] = np.asarray(fields["offset"])
                if selected:
                    arrays[f"{name}_literal_targets"] = fields["literal_targets"]
                    arrays[f"{name}_full_pool_q_0_100_500_1000_2000"] = fields["full_pool_q"]
                    residual = fields["full_pool_q"] - fields["literal_targets"]
                    arrays[f"{name}_full_pool_mse_0_100_500_1000_2000"] = np.mean(
                        residual * residual, axis=1
                    )
                    arrays[f"{name}_centered_residual_0"] = residual[0] - np.mean(residual[0])
                    arrays[f"{name}_q_variance_0"] = np.asarray(np.var(fields["full_pool_q"][0]))
                    arrays[f"{name}_q_target_covariance_0"] = np.asarray(np.mean(
                        (fields["full_pool_q"][0] - np.mean(fields["full_pool_q"][0])) *
                        (fields["literal_targets"] - np.mean(fields["literal_targets"]))
                    ))
            if selected:
                arrays.update(
                    replay_index=sample_index,
                    observation_raw=raw_obs,
                    observation_normalized=jax.device_get(obs),
                    action_raw=raw_act,
                    probe_key=jax.device_get(key),
                    target_key=jax.device_get(target_key),
                    index_key=jax.device_get(idx_key),
                    minibatch_index=jax.device_get(fit_index),
                    teacher_prescale=jax.device_get(prescale),
                    sine_base=jax.device_get(base),
                )
                retained["teacher_tree"] = teacher_record
            retained["arrays"] = persist_arrays(check_dir / f"round_{round_index:02d}.npz", arrays)
            checkpoint["rounds"].append(retained)
            atomic_json(check_dir / "partial.json", checkpoint)
            self.event("round_complete", check=check_index, round=round_index)
        checkpoint["duration_seconds"] = time.perf_counter() - check_start
        checkpoint["complete"] = True
        atomic_json(check_dir / "complete.json", checkpoint)
        self.checks.append(check_index)
        return {
            name: {key: np.stack([rd[name][key] for rd in rounds]) for key in rounds[0][name]}
            for name in critics
        }


def teacher_values(network, key, obs, act, chunk, scale):
    return _teacher_compiled(network, chunk, scale)(key, obs, act)


@functools.lru_cache(maxsize=None)
def _teacher_compiled(network, chunk, scale):
    import jax

    return jax.jit(lambda key, obs, act: _teacher_values(network, key, obs, act, chunk, scale))


def _teacher_values(network, key, obs, act, chunk, scale):
    import jax.numpy as jnp

    from experiments.exp12.probe import _chunked

    omega = network.init(key, observations=obs[:1], actions=act[:1])["params"]
    f = _chunked(lambda o, a: network.apply({"params": omega}, o, a).reshape(-1), obs, act, chunk)
    return jnp.sin(scale * f), f, omega


def fit_values(network, tx, params, obs, act, base, idx, chunk):
    """Original one-scan fit with observational full-pool Q at fixed updates."""
    return _fit_compiled(network, tx, chunk)(params, obs, act, base, idx)


@functools.lru_cache(maxsize=None)
def _fit_compiled(network, tx, chunk):
    import jax
    import jax.numpy as jnp
    import optax

    from experiments.exp12.probe import _chunked

    def compute(params, obs, act, base, idx):
        def predict(p):
            return _chunked(lambda o, a: network.apply({"params": p}, o, a).reshape(-1), obs, act, chunk)

        initial_q = predict(params)
        offset = jnp.mean(initial_q)
        targets = offset + base

        def loss_fn(p, batch_idx):
            q = network.apply({"params": p}, obs[batch_idx], act[batch_idx]).reshape(-1)
            return jnp.mean((q - targets[batch_idx]) ** 2)

        def step(carry, item):
            p, opt_state, traces = carry
            batch_idx, step_index = item
            loss, grads = jax.value_and_grad(loss_fn)(p, batch_idx)
            updates, opt_state = tx.update(grads, opt_state, p)
            p = optax.apply_updates(p, updates)

            def capture(traces):
                slot = jnp.where(step_index == 99, 1,
                                 jnp.where(step_index == 499, 2, 3))
                return traces.at[slot].set(predict(p))

            traces = jax.lax.cond(
                (step_index == 99) | (step_index == 499) | (step_index == 999),
                capture, lambda traces: traces, traces,
            )
            return (p, opt_state, traces), loss

        traces = jnp.zeros((5, obs.shape[0]), dtype=initial_q.dtype).at[0].set(initial_q)
        (fitted, _, traces), losses = jax.lax.scan(
            step, (params, tx.init(params), traces), (idx, jnp.arange(idx.shape[0]))
        )
        final_q = predict(fitted)
        traces = traces.at[4].set(final_q)
        final = jnp.mean((final_q - targets) ** 2)
        return losses, final, offset, traces, targets

    return jax.jit(compute)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--pilot-manifest", required=True)
    parser.add_argument("--pilot-run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wandb-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    from audits.probe_discrepancy.plan import plan_hash, validate_design

    manifest = json.loads(Path(args.pilot_manifest).read_text())
    validate_design(manifest)
    rows = [r for r in manifest["runs"] if r["pilot_id"] == args.pilot_run_id]
    if len(rows) != 1:
        raise ValueError("pilot run identity missing or duplicated")
    row = rows[0]
    run_dir = Path(args.run_dir).resolve()
    if run_dir != Path(manifest["new_campaign_root"]) / row["output_relative"]:
        raise ValueError("run directory differs from manifest")
    if os.environ.get("WANDB_RUN_ID") != args.wandb_id:
        raise ValueError("W&B identity differs from controller")
    checkout = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if source != args.expected_commit or manifest["diagnostic_implementation_source"] != source:
        raise ValueError("source SHA differs from approved executable")
    capture = Capture(run_dir, row, source, plan_hash(manifest))
    capture.event("adapter_start", source=source, config=row["source_scientific_config_sha256"])

    def interrupted(signum, _frame):
        raise RuntimeError(f"adapter interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        # JAX and training modules are imported only after the controller bound
        # this process to one allocation token.
        import wandb
        from experiments import exp1
        from experiments.exp12 import run_probes
        from experiments.exp12.trainer import Exp12Trainer
        from omegaconf import OmegaConf

        project = os.environ["WANDB_PROJECT"]
        overrides, _ = expected_config(row, run_dir, project)
        original_init = wandb.init
        starts = 0

        def pilot_wandb_init(*a, **kw):
            nonlocal starts
            starts += 1
            if starts == 1 and kw.get("id") is not None:
                raise ValueError("first W&B session tried to resume")
            if starts > 1 and kw.get("id") != args.wandb_id:
                raise ValueError("fork W&B session changed run identity")
            kw.update(id=args.wandb_id, project=project,
                      group=row["wandb_group"],
                      name=f"{args.pilot_run_id}-{row['environment']}-{row['architecture']}-seed{row['seed']}",
                      resume="never" if starts == 1 else "allow")
            return original_init(*a, **kw)

        wandb.init = pilot_wandb_init
        run_probes.run_probe = capture.run_probe
        original_trainer_init = Exp12Trainer.__init__
        original_probe = run_probes.RunProbes._probe
        original_save = Exp12Trainer.save

        def checked_trainer_init(trainer, cfg, path):
            original_trainer_init(trainer, cfg, path)
            actual = OmegaConf.to_container(cfg, resolve=True)
            scientific = {k: v for k, v in actual.items()
                          if k not in ("results_root", "project_name", "server")}
            if scientific != row["source_scientific_config"]:
                raise ValueError("instantiated scientific config differs from pinned source")
            capture.event("trainer_created", resolved_config_sha256=digest(actual),
                          scientific_config_sha256=digest(row["source_scientific_config"]),
                          runtime_path=str(path))

        def observed_probe(probes, check_index, critics):
            capture.trainer_state(probes.trainer, check_index)
            return original_probe(probes, check_index, critics)

        def observed_save(trainer, root, keep_previous=False):
            started = time.perf_counter()
            path = original_save(trainer, root, keep_previous=keep_previous)
            capture.checkpoint_state(trainer, path, time.perf_counter() - started)
            return path

        Exp12Trainer.__init__ = checked_trainer_init
        RunProbes = run_probes.RunProbes
        RunProbes._probe = observed_probe
        Exp12Trainer.save = observed_save
        exp1.run({
            "experiment": "exp1",
            "config_path": str(checkout / "main/configs"),
            "config_name": "base_exp12_ten",
            "overrides": overrides,
            "checkpoint_dir": str(run_dir),
            "checkpoint_interval": 10000,
            "checkpoint_start_frac": 0.0,
        })
        done = json.loads((run_dir / "DONE").read_text())
        expected_endpoint = row["nominal_endpoint_interactions"]
        fork_file = run_dir / "fork" / "fork.json"
        if fork_file.is_file():
            expected_endpoint = json.loads(fork_file.read_text())["control_end_step"]
        if done["interaction_step"] != expected_endpoint:
            raise ValueError("scientific endpoint incomplete")
        check_interval = row["nominal_endpoint_interactions"] // 50
        expected_checks = set(range(expected_endpoint // check_interval + 1))
        if set(capture.checks) != expected_checks or len(capture.checks) != len(expected_checks):
            raise ValueError("required original probe checks missing or duplicated")
        for check_index in capture.checks:
            receipt = json.loads((capture.root / f"check_{check_index:02d}" / "complete.json").read_text())
            if not receipt.get("complete") or len(receipt["rounds"]) != 10:
                raise ValueError(f"incomplete check {check_index}")
            for round_record in receipt["rounds"]:
                for record in [round_record["arrays"], *(
                    [round_record["teacher_tree"]] if "teacher_tree" in round_record else [])]:
                    if file_sha(record["path"]) != record["sha256"]:
                        raise ValueError(f"changed retained arrays at check {check_index}")
        if not (capture.root / "checkpoints.jsonl").is_file():
            raise ValueError("training checkpoints missing")
        completion = {
            "protocol": "echocritic-observation-only-v1",
            "expected_commit": source,
            "plan_sha256": capture.plan_sha,
            "pilot_id": row["pilot_id"],
            "config_sha256": row["source_scientific_config_sha256"],
            "diagnostics_complete": True,
            "scientific_endpoint_complete": True,
            "interaction_step": done["interaction_step"],
            "recorded_checks": capture.checks,
            "wandb_id": args.wandb_id,
            "duration_seconds": time.time() - capture.started,
            "done_sha256": file_sha(run_dir / "DONE"),
        }
        atomic_json(run_dir / "diagnostic_completion.json", completion)
        capture.event("adapter_complete", completion=completion)
    except BaseException as exc:
        atomic_json(run_dir / "diagnostic_partial.json", {
            "pilot_id": row["pilot_id"], "expected_commit": source,
            "error": repr(exc), "traceback": traceback.format_exc(),
            "recorded_checks": capture.checks,
        })
        capture.event("adapter_failure", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
