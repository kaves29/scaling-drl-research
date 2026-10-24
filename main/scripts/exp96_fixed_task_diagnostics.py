"""Budgeted single-A100 diagnostics on immutable saved Exp96 critics; no SAC training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from experiments.exp12.instrumentation import (
    entrypoint, diagnostic_state, diagnostic_output,
)
from experiments.exp12 import fixed_task as ft
from experiments.exp12 import probe as probe_module
from experiments.exp12 import probe_tasks as task_module
from experiments.exp12.diagnostic_plan import load_plan
from experiments.exp12.diagnostic_artifacts import (
    atomic_json,
    atomic_npz,
    initial_normalizer,
    load_critic,
    normalizer,
    optimizer_binding,
    replay_array,
    state_dir,
)
from experiments.exp12.precision import runtime_info, set_matmul_precision
from experiments.exp12.probe_tasks import FrozenTask
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
        self.plan, self.plan_sha256 = (
            load_plan(args.plan) if getattr(args, "plan", None) else (None, None)
        )
        self.seconds_per_1000 = 120.0
        self.queue = json.loads((Path(args.campaign) / "queue/queue.json").read_text())
        if self.queue["commit"] != FROZEN:
            raise ValueError("wrong frozen campaign source")
        parent_rows = [p["row"] for p in self.queue["parents"]]
        self.parents = {
            (
                p["row"]["environment"],
                p["row"]["architecture"],
                int(p["row"]["seed"]),
            ): p["row"]
            for p in self.queue["parents"]
        }
        expected = {
            (e, a, s)
            for e in ("dog-run", "humanoid-run", "myo-key-turn", "myo-pen-twirl")
            for a in ARCHS
            for s in range(1, 9)
        }
        if len(parent_rows) != 96 or set(self.parents) != expected:
            raise ValueError("not the canonical96-parent queue")
        self.host_models = {}
        self.segment_executables = {}
        self.optimizer_provenance = []
        self.optimizer_config = None
        for row in parent_rows:
            settings, binding = optimizer_binding(row, FROZEN)
            if self.optimizer_config is not None and settings != self.optimizer_config:
                raise ValueError("selected parent optimizer settings disagree")
            self.optimizer_config = settings
            self.optimizer_provenance.append(binding)
        self.tx = critic_optimizer(OmegaConf.create(self.optimizer_config))
        self.optimizer_sha256 = hashlib.sha256(
            json.dumps(
                self.optimizer_provenance[0]["optimizer"], sort_keys=True
            ).encode()
        ).hexdigest()
        self.inputs.extend(self.optimizer_provenance)

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
            atomic_npz(
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
                "optimizer": self.optimizer_provenance[0]["optimizer"],
                "optimizer_sha256": self.optimizer_sha256,
                "plan": self.plan,
                "plan_sha256": self.plan_sha256,
                "conditions": getattr(self, "condition_status", []),
                "elapsed_seconds": time.monotonic() - self.start,
                "jax_memory_stats": jax.devices()[0].memory_stats(),
            },
        )

        diagnostic_state(self)

    def bank_task(self, directory, obs, act, targets, indices, base, offset_mode):
        """Persist exact numeric inputs; learner identity cannot change task identity."""
        task = FrozenTask(
            np.asarray(obs),
            np.asarray(act),
            np.asarray(targets),
            np.asarray(indices),
            {
                "diagnostic_source": self.args.commit,
                "frozen_source": FROZEN,
                "optimizer_config_sha256": self.optimizer_sha256,
                "base_sha256": array_hash(base),
                "offset_mode": offset_mode,
                "precision": "float32",
            },
        )
        task.save(directory)
        restored = FrozenTask.load(directory)
        self.inputs.append(
            {"numeric_task": str(directory), "task_id": restored.task_id}
        )
        return restored.task_id

    def fit(
        self,
        name,
        env,
        arch,
        seed,
        kind,
        obs,
        act,
        z,
        idx,
        offset_mode,
        steps,
        shared_offset=None,
    ):
        if offset_mode not in ("own", "zero", "shared", "incremental"):
            raise ValueError("unknown target construction")
        if offset_mode == "shared" and shared_offset is None:
            raise ValueError("shared targets require an explicit reference offset")
        if steps not in (2000, 4000, 8000) or len(idx) < steps:
            raise ValueError("unsupported or incomplete fitting horizon")
        estimated = 180 + steps / 1000 * self.seconds_per_1000
        if not self.allowed(estimated):
            self.rows.append(
                {"name": name, "status": "SKIPPED_BUDGET", "requested_steps": steps}
            )
            self.save()
            return False
        net, params = self.model(env, arch, seed, kind)
        chunk = getattr(self, "eval_chunk", 2560)
        initial_hash = ft.tree_sha256(params)
        q = sync(ft.predictions(net, params, obs, act, chunk))
        offset = (
            jnp.mean(q)
            if offset_mode == "own"
            else jnp.asarray(
                shared_offset if offset_mode == "shared" else 0, jnp.float32
            )
        )
        targets = q + z if offset_mode == "incremental" else offset + z
        # Numeric targets are durable; incremental tasks explicitly differ by starting function.
        out = self.root / name
        out.mkdir(exist_ok=False)
        atomic_npz(
            out / "task.npz",
            targets=np.asarray(targets),
            base=np.asarray(z),
            indices=np.asarray(idx[:steps]),
            offset=np.asarray(offset),
            starting_predictions=np.asarray(q),
        )
        task_id = self.bank_task(
            out / "numeric_task", obs, act, targets, idx[:steps], z, offset_mode
        )
        metrics = self.time_call(
            "initial_pool_metrics/" + name,
            lambda: ft.pool_metrics(net, params, obs, act, targets, chunk),
        )
        rows = [{"step": 0, **{k: float(v) for k, v in metrics.items()}}]
        state, curves = ft.FitState(params, self.tx.init(params)), []
        first_update = self.time_call(
            "first_update/" + name,
            lambda: ft.first_update(
                net, self.tx, state, obs, act, targets, idx[0], chunk
            ),
        )
        record = {
            "name": name,
            "environment": env,
            "architecture": arch,
            "seed": seed,
            "kind": kind,
            "offset_mode": offset_mode,
            "requested_steps": steps,
            "steps": 0,
            "status": "PARTIAL",
            "measurements": rows,
            "initial_gradient_norm": float(first_update["gradient_l2"]),
            "first_update": {k: float(v) for k, v in first_update.items()},
            "starting_parameters_sha256": initial_hash,
            "optimizer_sha256": self.optimizer_sha256,
            "task_id": task_id,
            "base_sha256": array_hash(z),
            "target_mean": float(jnp.mean(targets)),
            "target_variance": float(jnp.var(targets)),
            "initial_function_mismatch_rms": float(jnp.sqrt(metrics["mse"])),
            "pairing": (
                "identical_numeric_task_required"
                if offset_mode in ("zero", "shared")
                else "learner_specific_targets; not a common fixed-task comparison"
            ),
        }
        self.rows.append(record)
        atomic_json(out / "progress.json", record)
        self.save()
        compile_key = (
            env,
            arch,
            obs.shape,
            act.shape,
            obs.dtype,
            act.dtype,
            self.optimizer_sha256,
        )
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
                lambda: ft.pool_metrics(net, state.params, obs, act, targets, chunk),
            )
            rows.append(
                {
                    "step": lo + 1000,
                    "segment_seconds_including_dispatch": time.monotonic() - before,
                    "mse_reduction_per_update": (
                        rows[-1]["mse"] - float(metrics["mse"])
                    )
                    / 1000,
                    **{k: float(v) for k, v in metrics.items()},
                }
            )
            atomic_npz(
                out / f"curve_through_{lo+1000:04d}.npz", losses=np.concatenate(curves)
            )
            record.update(
                steps=lo + 1000,
                optimizer_count=int(state.optimizer[0].count),
                curve_summary=ft.curve_summary(np.concatenate(curves)),
            )
            atomic_json(out / "progress.json", record)
            self.save()
        if ft.tree_sha256(params) != initial_hash:
            raise RuntimeError("diagnostic mutated original checkpoint parameters")
        record.update(
            status="COMPLETE" if rows[-1]["step"] == steps else "PARTIAL",
            final_state_sha256=ft.tree_sha256(state),
            starting_parameters_unchanged=True,
        )
        atomic_json(out / "progress.json", record)
        self.save()
        print("FIT", name, rows[-1], flush=True)
        return rows[-1]["step"] == steps

    def original_reproduction(self, env, arch, seed, round_index=0):
        row = self.parents[env, arch, seed]
        cp = Path(row["checkpoint_dir"])
        st = state_dir(cp, "latest")
        obs, act = replay_array(st, "observations"), replay_array(st, "actions")
        if len(obs) < 500000 or len(act) < 500000:
            raise ValueError("historical check 50 replay prefix is unavailable")
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
        atomic_npz(
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
                if not self.allowed(180):
                    self.rows.append(
                        {
                            "original_reproduction": {
                                "environment": env,
                                "architecture": arch,
                                "seed": seed,
                                "kind": kind,
                                "round": round_index,
                                "status": "SKIPPED_BUDGET",
                            }
                        }
                    )
                    self.save()
                    return False
                params = (
                    fresh if kind == "fresh" else self.model(env, arch, seed, kind)[1]
                )
                label = "fresh" if kind == "fresh" else "current"
                for field, shape in (
                    ("losses", (10, 2000)),
                    ("final_loss", (10,)),
                    ("offset", (10,)),
                    ("b", (10,)),
                ):
                    values = original[label + "_" + field]
                    if values.shape != shape or not np.isfinite(values).all():
                        raise ValueError("incomplete/nonfinite archived original probe")
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
                    optimizer_sha256=self.optimizer_sha256,
                )
                if not all(
                    np.isfinite(record[key])
                    for key in (
                        "final_mse",
                        "reference_mse",
                        "curve_max_abs_error",
                        "offset_error",
                        "variance_error",
                    )
                ):
                    raise ValueError("nonfinite historical reconstruction")
                atomic_npz(
                    out / (kind + "_original_fit.npz"),
                    losses=np.asarray(loss),
                    final_mse=np.asarray(mse),
                    offset=np.asarray(offset),
                )
                record["task_id"] = self.bank_task(
                    out / (kind + "_numeric_task"),
                    o,
                    a,
                    offset + base,
                    idx[:2000],
                    base,
                    "own",
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
                    report = ft.equivalence_report(
                        loss, float(mse), np.concatenate(segments), float(final)
                    )
                    atomic_npz(
                        out / "segmented_fit.npz",
                        losses=np.concatenate(segments),
                        final_mse=np.asarray(final),
                    )
                    atomic_json(
                        out / "segmented_equivalence.json",
                        {
                            **report,
                            "optimizer_sha256": self.optimizer_sha256,
                            "starting_parameters_sha256": ft.tree_sha256(params),
                            "segmented_state_sha256": ft.tree_sha256(state),
                        },
                    )
                    if not report["passed_existing_final_guard"]:
                        raise RuntimeError(
                            "segmented fit differs materially from frozen implementation"
                        )
        return True

    def run(self):
        """Execute only a reviewed explicit plan, retaining budget omissions."""
        if self.plan is None:
            raise ValueError("an explicit diagnostic plan is required")
        self.condition_status = []
        chunk = getattr(self, "eval_chunk", 2560)
        for condition in self.plan["conditions"]:
            status = {"id": condition["id"], "status": "PARTIAL"}
            self.condition_status.append(status)
            if not self.allowed(300):
                status["status"] = "SKIPPED_BUDGET"
                self.save()
                continue
            env, seed, r = (
                condition["environment"],
                condition["seed"],
                condition["round"],
            )
            if condition["type"] == "original":
                done = self.original_reproduction(
                    env, condition["architecture"], seed, r
                )
                status["status"] = "COMPLETE" if done else "PARTIAL_BUDGET"
                self.save()
                continue
            pool = condition["pool"]
            _, obs, act = self.pool(
                env,
                seed,
                period=pool["period"],
                norm=pool["normalizer"],
                support=pool["support"],
            )
            target_obs = obs
            if pool["target_normalizer"] != "same":
                _, target_obs, target_act = self.pool(
                    env,
                    seed,
                    period=pool["period"],
                    norm=pool["target_normalizer"],
                    support=pool["support"],
                )
                if not np.array_equal(np.asarray(act), np.asarray(target_act)):
                    raise ValueError("input-only crossover changed sampled actions")
            target_key, index_key = jax.random.split(
                jax.random.PRNGKey(20261022 + seed * 100 + r)
            )
            teacher = SACCritic("residual", *ARCHS["D4W1536"], jnp.float32)
            base = self.time_call(
                "target/" + condition["id"],
                lambda: _base_targets(
                    teacher, target_key, target_obs, act, chunk, 100000.0
                ),
            )
            horizon = max(condition["horizons"])
            idx = ft.extended_indices(index_key, horizon, 256, len(obs))
            shared = None
            if "shared" in condition["offset_modes"]:
                ref = condition["shared_reference"]
                net, params = self.model(env, ref["architecture"], seed, ref["kind"])
                shared = sync(jnp.mean(ft.predictions(net, params, obs, act, chunk)))
            done = True
            for arch in condition["architectures"]:
                for kind in condition["learners"]:
                    for mode in condition["offset_modes"]:
                        name = f"{condition['id']}_{arch}_{kind}_{mode}"
                        complete = self.fit(
                            name,
                            env,
                            arch,
                            seed,
                            kind,
                            obs,
                            act,
                            base,
                            idx,
                            mode,
                            horizon,
                            shared_offset=shared,
                        )
                        done = complete and done
            status["status"] = "COMPLETE" if done else "PARTIAL_BUDGET"
            self.save()
        return all(c["status"] == "COMPLETE" for c in self.condition_status)


@entrypoint("diagnostic")
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--cap-seconds", type=int, default=6600)
    p.add_argument(
        "--plan", required=True, help="Explicit reviewed diagnostic plan JSON"
    )
    args = p.parse_args()
    diagnostic_output(args)
    load_plan(args.plan)
    if not 0 < args.cap_seconds <= 6600:
        raise ValueError("work cap must be positive and at most 6600 seconds")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU diagnostics require the authorized Slurm allocation")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if actual != args.commit or dirty:
        raise RuntimeError("diagnostic checkout is not clean and source matched")
    from scripts.exp12_launch_plan import require_pinned_stack

    require_pinned_stack()
    source_root = Path(__file__).resolve().parents[1]
    if not all(
        Path(path).resolve().is_relative_to(source_root)
        for path in [ft.__file__, probe_module.__file__, task_module.__file__]
    ):
        raise RuntimeError("diagnostic imported a module from a different checkout")
    root = Path(args.root).resolve()
    if root.is_relative_to(source_root.parent) or root.is_relative_to(
        Path(args.campaign).resolve()
    ):
        raise ValueError("diagnostic output must be outside source and frozen campaign")
    if (
        jax.default_backend() != "gpu"
        or len(jax.devices()) != 1
        or "A100" not in jax.devices()[0].device_kind
    ):
        raise RuntimeError("requires exactly one visible A100")
    set_matmul_precision()
    runtime = runtime_info()
    if jax.config.jax_enable_x64 or any(
        runtime.get(name) != "0.4.34"
        for name in ("jax_cuda12_plugin", "jax_cuda12_pjrt")
    ):
        raise RuntimeError("requires the pinned CUDA plugins and FP32 policy")
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        root / "runtime.json",
        {
            "runtime": runtime,
            "job_id": os.environ["SLURM_JOB_ID"],
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "expected_commit": args.commit,
            "plan_sha256": load_plan(args.plan)[1],
            "module_paths": {"fixed_task": ft.__file__, "probe": probe_module.__file__},
        },
    )
    try:
        runner = Investigation(args)
        diagnostic_state(runner)
    except Exception as exc:
        atomic_json(
            root / "status.json", {"status": "FAILED_PREFLIGHT", "error": repr(exc)}
        )
        raise
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
        complete = runner.run()
        atomic_json(
            root / "status.json",
            {
                "status": "COMPLETED_BUDGETED_STAGE" if complete else "PARTIAL_BUDGET",
                "elapsed_seconds": time.monotonic() - runner.start,
            },
        )
        if not complete:
            return 3
    except Exception as exc:
        status = {"status": "FAILED", "error": repr(exc)}
        try:
            runner.save()
        except Exception as save_error:
            status["result_save_error"] = repr(save_error)
        atomic_json(root / "status.json", status)
        raise
    finally:
        stop.set()
        monitor.join(timeout=12)


if __name__ == "__main__":
    sys.exit(main())
