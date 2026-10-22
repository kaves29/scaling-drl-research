"""Budgeted single-A100 diagnostics on immutable saved Exp96 critics; no SAC training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from experiments.exp12 import fixed_task as ft
from experiments.exp12.diagnostic_artifacts import (
    atomic_json,
    initial_normalizer,
    load_critic,
    normalizer,
    replay_array,
    state_dir,
)
from experiments.exp12.precision import runtime_info, set_matmul_precision
from experiments.exp12.probe import (
    _base_targets,
    _fit,
    critic_optimizer,
    probe_key,
    probe_rng,
)
from scale_rl.agents.sac.sac_network import SACCritic


FROZEN = "f597ee9898e4d401912092c336c3e5330cf2fda1"
ARCHS = {"D2W512": (2, 512), "D4W1024": (4, 1024), "D4W1536": (4, 1536)}


def sync(value):
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), value)
    return value


def array_hash(value):
    a = np.asarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()


class Investigation:
    def __init__(self, args):
        self.args, self.root = args, Path(args.root)
        self.start = time.monotonic()
        self.deadline = self.start + args.cap_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows, self.timing, self.inputs = [], [], []
        self.seconds_per_1000 = 120.0
        self.queue = json.loads((Path(args.campaign) / "queue/queue.json").read_text())
        if self.queue["commit"] != FROZEN:
            raise ValueError("wrong frozen campaign source")
        self.parents = {
            (
                p["row"]["environment"],
                p["row"]["architecture"],
                int(p["row"]["seed"]),
            ): p["row"]
            for p in self.queue["parents"]
        }
        if len(self.parents) != 96:
            raise ValueError("not the canonical96-parent queue")
        self.host_models = {}
        self.segment_executables = {}
        optimizer_config = (
            Path(__file__).resolve().parents[1] / "configs/agent/sac.yaml"
        )
        self.tx = critic_optimizer(OmegaConf.load(optimizer_config))
        self.inputs.append(
            {
                "optimizer_config": str(optimizer_config),
                "sha256": hashlib.sha256(optimizer_config.read_bytes()).hexdigest(),
            }
        )

    def time_call(self, label, call):
        before = time.monotonic()
        result = sync(call())
        self.timing.append({"label": label, "seconds": time.monotonic() - before})
        return result

    def allowed(self, reserve=180):
        return time.monotonic() + reserve < self.deadline

    def model(self, env, arch, seed, kind):
        key = env, arch, seed, kind
        if key not in self.host_models:
            row = self.parents[env, arch, seed]
            params, provenance = load_critic(row["checkpoint_dir"], kind)
            self.host_models[key] = params
            self.inputs.append({"parent": row["run_key"], "kind": kind, **provenance})
        params = self.time_call(
            "host_to_device/" + "/".join(map(str, key)),
            lambda: jax.device_put(self.host_models[key]),
        )
        return SACCritic("residual", *ARCHS[arch], jnp.float32), params

    def pool(self, env, seed, check=50, period="late", norm="latest", support=None):
        # Large-parent replay is shared for controlled architecture comparisons.
        row = self.parents[env, "D4W1536", seed]
        cp = Path(row["checkpoint_dir"])
        st = state_dir(cp, "latest")
        obs, act = replay_array(st, "observations"), replay_array(st, "actions")
        n = 5000 if period == "early" else 500000
        if len(obs) < n:
            raise ValueError("requested replay prefix is unavailable")
        rms = (
            initial_normalizer(obs)
            if norm == "initial_reconstructed"
            else normalizer(state_dir(cp, norm))
        )
        rng = np.random.default_rng([20261022, seed, 0x504F4F4C])
        if support is None:
            indices = rng.integers(0, n, 25600)
        else:
            members = rng.choice(n, support, replace=False)
            indices = members[
                np.random.default_rng([20261022, seed, 0x53555050]).integers(
                    0, support, 25600
                )
            ]
        raw, actions = np.asarray(obs[indices]), np.asarray(act[indices])
        normalized = (raw - rms["mean"]) / np.sqrt(rms["var"] + 1e-8)
        tag = f"{env}_{seed}_{period}_{norm}_{support}"
        path = self.root / ("pool_" + tag + ".npz")
        if not path.exists():
            np.savez(
                path,
                raw_obs=raw,
                obs=normalized,
                act=actions,
                replay_indices=indices,
                normalizer_mean=rms["mean"],
                normalizer_var=rms["var"],
            )
        self.inputs.append(
            {
                "pool": tag,
                "checkpoint": str(cp),
                "n": n,
                "unique_indices": int(len(np.unique(indices))),
                "obs_sha256": array_hash(normalized),
                "act_sha256": array_hash(actions),
                "normalizer": norm,
                "reconstructed": norm == "initial_reconstructed",
            }
        )
        return tag, jnp.asarray(normalized), jnp.asarray(actions)

    def save(self):
        atomic_json(
            self.root / "results.json",
            {
                "source": self.args.commit,
                "frozen_source": FROZEN,
                "rows": self.rows,
                "timings": self.timing,
                "inputs": self.inputs,
                "elapsed_seconds": time.monotonic() - self.start,
                "jax_memory_stats": jax.devices()[0].memory_stats(),
            },
        )

    def fit(self, name, env, arch, seed, kind, obs, act, z, idx, offset_mode, steps):
        estimated = 180 + steps / 1000 * self.seconds_per_1000
        if not self.allowed(estimated):
            return False
        net, params = self.model(env, arch, seed, kind)
        q = sync(ft.predictions(net, params, obs, act, 2560))
        offset = jnp.mean(q) if offset_mode == "own" else jnp.array(0, jnp.float32)
        targets = q + z if offset_mode == "incremental" else offset + z
        # Numeric targets are durable; incremental tasks explicitly differ by starting function.
        out = self.root / name
        out.mkdir(exist_ok=False)
        np.savez(
            out / "task.npz",
            targets=np.asarray(targets),
            base=np.asarray(z),
            indices=np.asarray(idx[:steps]),
            offset=np.asarray(offset),
        )
        metrics = self.time_call(
            "initial_pool_metrics/" + name,
            lambda: ft.pool_metrics(net, params, obs, act, targets, 2560),
        )
        loss, grad_norm = self.time_call(
            "initial_gradient/" + name,
            lambda: ft.initial_gradient(
                net, params, obs[idx[0]], act[idx[0]], targets[idx[0]]
            ),
        )
        rows = [{"step": 0, **{k: float(v) for k, v in metrics.items()}}]
        state, curves = ft.FitState(params, self.tx.init(params)), []
        compile_key = (env, arch)
        if compile_key not in self.segment_executables:
            before = time.monotonic()
            self.segment_executables[compile_key] = ft.fit_segment.lower(
                net, self.tx, state, obs, act, targets, idx[:1000]
            ).compile()
            self.timing.append(
                {
                    "label": "segment_lower_compile/" + env + "/" + arch,
                    "seconds": time.monotonic() - before,
                }
            )
        executable = self.segment_executables[compile_key]
        for lo in range(0, steps, 1000):
            if not self.allowed(90):
                break
            before = time.monotonic()
            state, curve = self.time_call(
                "fit_segment/" + name + "/" + str(lo),
                lambda: executable(state, obs, act, targets, idx[lo : lo + 1000]),
            )
            curves.append(np.asarray(curve))
            metrics = self.time_call(
                "pool_metrics/" + name + "/" + str(lo + 1000),
                lambda: ft.pool_metrics(net, state.params, obs, act, targets, 2560),
            )
            rows.append(
                {
                    "step": lo + 1000,
                    "segment_seconds_including_dispatch": time.monotonic() - before,
                    **{k: float(v) for k, v in metrics.items()},
                }
            )
            np.savez(
                out / f"curve_through_{lo+1000:04d}.npz", losses=np.concatenate(curves)
            )
            atomic_json(
                out / "progress.json",
                {
                    "measurements": rows,
                    "offset_mode": offset_mode,
                    "initial_minibatch_loss": float(loss),
                    "initial_gradient_norm": float(grad_norm),
                    "target_sha256": array_hash(targets),
                    "base_sha256": array_hash(z),
                    "optimizer_count": int(state.optimizer[0].count),
                },
            )
            self.save()
        self.rows.append(
            {
                "name": name,
                "environment": env,
                "architecture": arch,
                "seed": seed,
                "kind": kind,
                "offset_mode": offset_mode,
                "steps": rows[-1]["step"],
                "measurements": rows,
                "initial_gradient_norm": float(grad_norm),
            }
        )
        self.save()
        print("FIT", name, rows[-1], flush=True)
        return rows[-1]["step"] == steps

    def original_reproduction(self, env, arch, seed, round_index=0):
        row = self.parents[env, arch, seed]
        cp = Path(row["checkpoint_dir"])
        st = state_dir(cp, "latest")
        obs, act = replay_array(st, "observations"), replay_array(st, "actions")
        rms = normalizer(st)
        indices = probe_rng(seed, 50, round_index).integers(0, 500000, 25600)
        o = jnp.asarray((obs[indices] - rms["mean"]) / np.sqrt(rms["var"] + 1e-8))
        a = jnp.asarray(act[indices])
        net, fresh = self.model(env, arch, seed, "fresh")
        target_key, index_key = jax.random.split(probe_key(seed, 50, round_index))
        base = self.time_call(
            "original_target/" + row["run_key"],
            lambda: _base_targets(net, target_key, o, a, 2560, 100000.0),
        )
        idx = ft.extended_indices(index_key, 8000, 256, 25600)
        out = self.root / f"original_{row['run_key']}_r{round_index}"
        out.mkdir(exist_ok=False)
        np.savez(
            out / "task.npz",
            obs=np.asarray(o),
            act=np.asarray(a),
            base=np.asarray(base),
            indices=np.asarray(idx),
            replay_indices=indices,
        )
        self.inputs.append(
            {
                "original_probe_path": str(cp / "probes/check_50.npz"),
                "original_probe_sha256": hashlib.sha256(
                    (cp / "probes/check_50.npz").read_bytes()
                ).hexdigest(),
                "reconstructed_task": str(out / "task.npz"),
                "base_sha256": array_hash(base),
            }
        )
        with np.load(cp / "probes/check_50.npz") as original:
            for kind in ["fresh", "latest"]:
                params = (
                    fresh if kind == "fresh" else self.model(env, arch, seed, kind)[1]
                )
                label = "fresh" if kind == "fresh" else "current"
                before = time.monotonic()
                executable = _fit.lower(
                    net, self.tx, params, o, a, base, idx[:2000], 2560
                ).compile()
                self.timing.append(
                    {
                        "label": "original_lower_compile/"
                        + row["run_key"]
                        + "/"
                        + kind,
                        "seconds": time.monotonic() - before,
                    }
                )
                loss, mse, offset = self.time_call(
                    "original_fit/" + row["run_key"] + "/" + kind,
                    lambda: executable(params, o, a, base, idx[:2000]),
                )
                err = float(
                    np.max(
                        np.abs(
                            np.asarray(loss) - original[label + "_losses"][round_index]
                        )
                    )
                )
                reference = float(original[label + "_final_loss"][round_index])
                record = dict(
                    environment=env,
                    architecture=arch,
                    seed=seed,
                    kind=kind,
                    round=round_index,
                    final_mse=float(mse),
                    reference_mse=reference,
                    curve_max_abs_error=err,
                    offset_error=float(
                        abs(offset - original[label + "_offset"][round_index])
                    ),
                    variance_error=float(
                        abs(jnp.var(base) - original[label + "_b"][round_index])
                    ),
                )
                np.savez(
                    out / (kind + "_original_fit.npz"),
                    losses=np.asarray(loss),
                    final_mse=np.asarray(mse),
                    offset=np.asarray(offset),
                )
                self.rows.append({"original_reproduction": record})
                self.save()
                if arch == "D4W1536":
                    self.seconds_per_1000 = max(10.0, self.timing[-1]["seconds"])
                print("REPRODUCTION", record, flush=True)
                # Do not assume a small mismatch is a bug. Stop if the historical task cannot be reconstructed.
                if (
                    record["variance_error"] > 0.002
                    or record["offset_error"] > 0.05
                    or abs(float(mse) - reference) > max(0.02, 0.1 * abs(reference))
                ):
                    atomic_json(self.root / "reproduction_review_required.json", record)
                    raise RuntimeError(
                        "historical reproduction mismatch requires analysis before further fits"
                    )
                if kind == "fresh":
                    again = self.time_call(
                        "identity_repeat/" + row["run_key"],
                        lambda: executable(params, o, a, base, idx[:2000]),
                    )
                    if not np.array_equal(np.asarray(again[0]), np.asarray(loss)):
                        raise RuntimeError(
                            "identical frozen-fit repetition changed losses"
                        )
                    state = ft.FitState(params, self.tx.init(params))
                    segments = []
                    for lo in [0, 1000]:
                        state, curve = self.time_call(
                            "segmented_equivalence/" + row["run_key"] + "/" + str(lo),
                            lambda: ft.fit_segment(
                                net,
                                self.tx,
                                state,
                                o,
                                a,
                                offset + base,
                                idx[lo : lo + 1000],
                            ),
                        )
                        segments.append(np.asarray(curve))
                    final = sync(
                        ft.pool_metrics(net, state.params, o, a, offset + base, 2560)
                    )["mse"]
                    delta = float(
                        np.max(np.abs(np.concatenate(segments) - np.asarray(loss)))
                    )
                    final_delta = float(abs(final - mse))
                    atomic_json(
                        out / "segmented_equivalence.json",
                        {
                            "curve_max_abs_error": delta,
                            "final_abs_error": final_delta,
                            "bitwise_curve_equal": bool(
                                np.array_equal(
                                    np.concatenate(segments), np.asarray(loss)
                                )
                            ),
                        },
                    )
                    if final_delta > max(1e-4, 0.005 * abs(float(mse))):
                        raise RuntimeError(
                            "segmented fit differs materially from frozen implementation"
                        )

    def run(self):
        self.original_reproduction("myo-pen-twirl", "D2W512", 5)
        if self.allowed(600):
            self.original_reproduction("humanoid-run", "D4W1536", 1)
        # Common numeric tasks and minibatches; teacher is fixed across learners and normalizers.
        for r in range(3):
            for env, seed in [
                (e, s) for e in ["myo-key-turn", "myo-pen-twirl"] for s in [1, 2]
            ]:
                teacher = SACCritic("residual", *ARCHS["D4W1536"], jnp.float32)
                if self.allowed(300):
                    key = jax.random.PRNGKey(20261022 + seed * 100 + r)
                    target_key, index_key = jax.random.split(key)
                    idx = ft.extended_indices(index_key, 8000, 256, 25600)
                    # The same keyed teacher weights are generated for every crossed condition.
                    prior_base = None
                    for period, norm, support in [
                        ("late", "latest", None),
                        ("late", "latest", 5000),
                        ("early", "latest", 5000),
                        ("early", "early", 5000),
                    ]:
                        if not self.allowed(300):
                            return
                        tag, o, a = self.pool(
                            env, seed, period=period, norm=norm, support=support
                        )
                        z = self.time_call(
                            "target/" + tag,
                            lambda: _base_targets(
                                teacher, target_key, o, a, 2560, 100000.0
                            ),
                        )
                        name = f"{tag}_r{r}_fresh_zero"
                        horizon = (
                            8000
                            if period == "late"
                            and norm == "latest"
                            and support is None
                            and seed == 1
                            else 2000
                        )
                        self.fit(
                            name,
                            env,
                            "D4W1536",
                            seed,
                            "fresh",
                            o,
                            a,
                            z,
                            idx,
                            "zero",
                            horizon,
                        )
                        if period == "early" and norm == "latest":
                            prior_base = z
                        if period == "early" and norm == "early":
                            self.fit(
                                f"{tag}_r{r}_fresh_input_only",
                                env,
                                "D4W1536",
                                seed,
                                "fresh",
                                o,
                                a,
                                prior_base,
                                idx,
                                "zero",
                                2000,
                            )
                        if period == "late" and norm == "latest" and support is None:
                            for kind in ["early", "latest"]:
                                self.fit(
                                    f"{tag}_r{r}_{kind}_shared_zero",
                                    env,
                                    "D4W1536",
                                    seed,
                                    kind,
                                    o,
                                    a,
                                    z,
                                    idx,
                                    "zero",
                                    horizon,
                                )
                            self.fit(
                                f"{tag}_r{r}_latest_own",
                                env,
                                "D4W1536",
                                seed,
                                "latest",
                                o,
                                a,
                                z,
                                idx,
                                "own",
                                2000,
                            )
                            for arch in ["D2W512", "D4W1024"]:
                                self.fit(
                                    f"{tag}_r{r}_{arch}_fresh_zero",
                                    env,
                                    arch,
                                    seed,
                                    "fresh",
                                    o,
                                    a,
                                    z,
                                    idx,
                                    "zero",
                                    2000,
                                )
        self.save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--cap-seconds", type=int, default=6600)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU diagnostics require the authorized Slurm allocation")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if actual != args.commit or dirty:
        raise RuntimeError("diagnostic checkout is not clean and source matched")
    if (
        jax.default_backend() != "gpu"
        or len(jax.devices()) != 1
        or "A100" not in jax.devices()[0].device_kind
    ):
        raise RuntimeError("requires exactly one visible A100")
    if args.cap_seconds > 13800:
        raise ValueError("work cap exceeds the4physicalGPUh authorization margin")
    set_matmul_precision()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        root / "runtime.json",
        {
            "runtime": runtime_info(),
            "job_id": os.environ["SLURM_JOB_ID"],
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "expected_commit": args.commit,
        },
    )
    runner = Investigation(args)
    stop = threading.Event()

    def telemetry():
        with (root / "telemetry.jsonl").open("x") as stream:
            while not stop.is_set():
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "-q", "-x"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode:
                        raise RuntimeError(result.stderr[:500])
                    for gpu in ET.fromstring(result.stdout).findall("gpu"):
                        processes = [
                            x
                            for x in gpu.findall("processes/process_info")
                            if x.findtext("pid") == str(os.getpid())
                        ]
                        if processes:
                            row = {
                                "elapsed_seconds": time.monotonic() - runner.start,
                                "uuid": gpu.findtext("uuid"),
                                "name": gpu.findtext("product_name"),
                                "total_memory": gpu.findtext("fb_memory_usage/total"),
                                "own_process_memory": processes[0].findtext(
                                    "used_memory"
                                ),
                                "shared_gpu_utilization": gpu.findtext(
                                    "utilization/gpu_util"
                                ),
                            }
                            stream.write(json.dumps(row) + "\n")
                            stream.flush()
                except Exception as exc:
                    stream.write(
                        json.dumps({"telemetry_unavailable": repr(exc)}) + "\n"
                    )
                    stream.flush()
                    return
                stop.wait(5)

    monitor = threading.Thread(target=telemetry, daemon=True)
    monitor.start()
    try:
        runner.run()
        atomic_json(
            root / "status.json",
            {
                "status": "COMPLETED_BUDGETED_STAGE",
                "elapsed_seconds": time.monotonic() - runner.start,
            },
        )
    except Exception as exc:
        runner.save()
        atomic_json(root / "status.json", {"status": "FAILED", "error": repr(exc)})
        raise
    finally:
        stop.set()
        monitor.join(timeout=12)


if __name__ == "__main__":
    main()
