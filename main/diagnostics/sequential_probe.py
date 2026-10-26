"""Opt-in eight-task extension; distinct from the approved independent probe.

A bank is shared by a matched environment/seed block (including learner sizes).
A reference is per learner architecture/run; checkpoint outputs are separate.
No live trainer, replay writer, optimizer or training RNG is accepted by fit().
"""

from dataclasses import asdict, dataclass
import functools
from types import SimpleNamespace
import importlib.metadata
import hashlib
import json
import os
import platform
from pathlib import Path
import uuid

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np

from experiments.exp12.probe import _mean_prediction, critic_optimizer

STREAM = 0x53514551


def precision_contract(network=None, params=None, observations=None, actions=None):
    """Fail closed before JAX can silently narrow unsupported inputs/weights."""
    backend = jax.default_backend()
    if backend not in ("cpu", "gpu") or jax.config.jax_enable_x64:
        raise ValueError(
            "unsupported backend/precision for creation or resume: requires CPU or GPU, x64 off"
        )
    if jax.config.jax_default_matmul_precision not in (
        None,
        "default",
        "high",
        "highest",
        "float32",
        "tensorfloat32",
    ):
        raise ValueError("unsupported matmul precision")
    if network is not None and np.dtype(
        getattr(network, "dtype", np.float32)
    ) != np.dtype("float32"):
        raise ValueError("network computation dtype must be float32")
    arrays = {}
    if observations is not None:
        arrays["observations"] = observations
    if actions is not None:
        arrays["actions"] = actions
    if params is not None:
        leaves = jax.tree_util.tree_leaves(params)
        if not leaves:
            raise ValueError("empty parameter tree")
        arrays.update({f"parameter_{i}": x for i, x in enumerate(leaves)})
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"{name} requires finite float32 values")
    if observations is not None and actions is not None:
        if (
            observations.ndim != 2
            or actions.ndim != 2
            or len(observations) != len(actions)
            or not len(observations)
        ):
            raise ValueError("matching nonempty rank-two inputs required")
        if params is not None:
            shape = jax.eval_shape(
                lambda p, o, a: network.apply({"params": p}, o, a),
                params,
                jax.ShapeDtypeStruct(observations.shape, np.float32),
                jax.ShapeDtypeStruct(actions.shape, np.float32),
            )
            if shape.dtype != np.float32 or shape.shape not in (
                (len(observations),),
                (len(observations), 1),
            ):
                raise ValueError("network must return one float32 scalar per input")
    return {
        "backend": backend,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "prng_impl": str(jax.config.jax_default_prng_impl),
        "dtype_promotion": str(jax.config.jax_numpy_dtype_promotion),
        "devices": [
            list(x)
            for x in sorted({(d.platform, d.device_kind) for d in jax.devices()})
        ],
        "input_parameter_compute_dtype": "float32",
        "python": platform.python_version(),
        "backend_platform_version": jax.extend.backend.get_backend().platform_version,
        "runtime_flags": {
            k: os.environ.get(k) for k in ("XLA_FLAGS", "NVIDIA_TF32_OVERRIDE")
        },
        "runtime_versions": {
            k: importlib.metadata.version(k)
            for k in ("jax", "jaxlib", "flax", "optax", "numpy", "orbax-checkpoint")
        },
        "GPU_qualified": False,
    }


def digest(data):
    return hashlib.sha256(data).hexdigest()


def array_hash(x):
    x = np.asarray(x)
    return digest(json.dumps([x.dtype.str, list(x.shape)]).encode() + x.tobytes())


def tree_hash(tree):
    # Dictionary insertion order may change in Orbax restore without changing
    # any parameter. Hash keyed numerical leaves, not msgpack dictionary order.
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    records = [(str(path), array_hash(value)) for path, value in leaves]
    return digest(json.dumps(sorted(records), separators=(",", ":")).encode())


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def load_json(path):
    return json.loads(Path(path).read_text())


def commit_directory(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.parent / ("." + path.name + "-" + uuid.uuid4().hex)
    pending.mkdir()
    writer(pending)
    for artifact in pending.iterdir():
        if artifact.is_file():
            with artifact.open("rb") as stream:
                os.fsync(stream.fileno())
    descriptor = os.open(pending, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    # Refuse to replace any previously committed evidence.
    if path.exists():
        raise FileExistsError(path)
    os.rename(pending, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify(path):
    path = Path(path)
    receipt = load_json(path / "receipt.json")
    for name, expected in receipt["files"].items():
        if Path(name).name != name or digest((path / name).read_bytes()) != expected:
            raise ValueError("artifact hash mismatch")
    return receipt


def seal(path, metadata):
    metadata = dict(metadata)
    metadata["files"] = {
        p.name: digest(p.read_bytes()) for p in path.iterdir() if p.is_file()
    }
    write_json(path / "receipt.json", metadata)


@dataclass(frozen=True)
class Settings:
    steps: int = 2000
    batch_size: int = 256
    chunk: int = 2560
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    def validate(self, n):
        if (
            self.steps != 2000
            or self.batch_size <= 0
            or self.chunk <= 0
            or n % self.chunk
        ):
            raise ValueError(
                "requires 2000 steps and positive batch/chunk dividing pool"
            )
        if (
            not np.isfinite([self.learning_rate, self.weight_decay]).all()
            or self.learning_rate <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("invalid optimizer configuration")


def create_bank(
    path,
    *,
    observations,
    actions,
    num_filled,
    warmup,
    rms,
    environment,
    seed,
    teacher,
    teacher_architecture,
    pool_size=25600,
    scale=1e5,
    teacher_chunk=None,
    source,
):
    """Call once at initial warmup; later users load this exact shared bank.

    rms is None for unnormalized inputs, otherwise mean/var/count/epsilon.
    Raw replay arrays are read/copied, never normalized in place.
    """
    if Path(path).exists():
        raise FileExistsError("load existing bank; never regenerate it")
    if (
        num_filled < warmup
        or num_filled > len(observations)
        or num_filled > len(actions)
        or pool_size <= 0
    ):
        raise ValueError("invalid warmup/replay size")
    execution = precision_contract(
        teacher,
        observations=np.asarray(observations)[:num_filled],
        actions=np.asarray(actions)[:num_filled],
    )
    if (
        not environment
        or not source
        or not teacher_architecture
        or not np.isfinite(scale)
    ):
        raise ValueError("explicit source/environment/fixed teacher identity required")
    env_stream = int.from_bytes(
        hashlib.sha256(environment.encode()).digest()[:4], "little"
    )
    rng = np.random.default_rng([seed, STREAM, env_stream])
    indices = rng.integers(0, num_filled, pool_size)
    raw = np.asarray(observations)[indices].copy()
    act = np.asarray(actions)[indices].astype(np.float32, copy=True)
    normalized = raw.copy()
    rms_copy = None
    if rms is not None:
        rms_copy = {k: np.asarray(v).tolist() for k, v in rms.items()}
        if (
            set(rms) != {"mean", "var", "count", "epsilon"}
            or np.any(np.asarray(rms["var"]) < 0)
            or float(rms["epsilon"]) <= 0
        ):
            raise ValueError("invalid observation normalizer")
        normalized = (raw - rms["mean"]) / np.sqrt(
            np.asarray(rms["var"]) + rms["epsilon"]
        )
    obs = np.asarray(normalized, np.float32)
    if (
        obs.ndim != 2
        or act.ndim != 2
        or not np.isfinite(obs).all()
        or not np.isfinite(act).all()
    ):
        raise ValueError("finite rank-two observation/action arrays required")
    if teacher_chunk is None:
        teacher_chunk = 2560 if pool_size % 2560 == 0 else pool_size
    if teacher_chunk <= 0 or pool_size % teacher_chunk:
        raise ValueError("teacher chunk must divide fixed pool")
    # Teacher stream is independent of learner architecture and checkpoint.
    key = jax.random.fold_in(jax.random.PRNGKey(seed), STREAM)
    key = jax.random.fold_in(key, env_stream)
    keys = [jax.random.fold_in(key, k) for k in range(8)]
    evaluated = [
        _teacher_values(
            teacher, k, jnp.asarray(obs), jnp.asarray(act), teacher_chunk, scale
        )
        for k in keys
    ]
    base = np.stack([np.asarray(item[0]) for item in evaluated])
    teachers = [item[1] for item in evaluated]
    for parameters in teachers:
        precision_contract(teacher, parameters, obs, act)
    if base.shape != (8, pool_size) or not np.isfinite(base).all():
        raise ValueError("invalid teacher bank")

    def writer(tmp):
        np.savez_compressed(
            tmp / "bank.npz",
            raw_observations=raw,
            observations=obs,
            actions=act,
            replay_indices=indices,
            base=base,
            keys=np.asarray(keys),
            **(
                {"rms_" + k: np.asarray(v) for k, v in rms.items()}
                if rms is not None
                else {}
            ),
        )
        (tmp / "teachers.msgpack").write_bytes(serialization.to_bytes(teachers))
        seal(
            tmp,
            {
                "environment": environment,
                "seed": seed,
                "source": source,
                "execution": execution,
                "teacher_architecture": teacher_architecture,
                "teacher_eval_chunk": teacher_chunk,
                "teacher_definition": repr(teacher),
                "scale": scale,
                "rms": rms_copy,
                "num_filled": num_filled,
                "warmup": warmup,
                "array_hashes": {
                    "raw_observations": array_hash(raw),
                    "replay_indices": array_hash(indices),
                    "observations": array_hash(obs),
                    "actions": array_hash(act),
                    "base": array_hash(base),
                },
                "base_variances": np.var(base, axis=1).tolist(),
                "distinct_base_arrays": len({array_hash(x) for x in base}),
            },
        )

    commit_directory(path, writer)
    return verify(path)


def create_reference(path, bank, network, fresh, *, architecture, sac_updates):
    """Call before SAC updates using the exact saved fresh parameter tree."""
    if sac_updates != 0 or not architecture:
        raise ValueError("reference must be captured before SAC updates")
    verify(bank)
    with np.load(Path(bank) / "bank.npz", allow_pickle=False) as data:
        obs, act, base = data["observations"], data["actions"], data["base"]
    execution = precision_contract(network, fresh, obs, act)
    offset = _mean_prediction(
        network, fresh, jnp.asarray(obs), jnp.asarray(act), len(obs)
    )
    targets = np.asarray(jnp.asarray(base) + offset)
    if not np.isfinite(targets).all():
        raise ValueError("nonfinite literal targets")
    fresh_bytes = serialization.to_bytes(fresh)

    def writer(tmp):
        (tmp / "fresh.msgpack").write_bytes(fresh_bytes)
        np.savez_compressed(
            tmp / "targets.npz",
            targets=targets,
            offset=np.asarray(offset),
            variances=np.asarray([jnp.var(jnp.asarray(t)) for t in base]),
            literal_variances=np.var(targets, axis=1),
        )
        seal(
            tmp,
            {
                "architecture": architecture,
                "network": repr(network),
                "fresh_hash": tree_hash(fresh),
                "bank_hash": digest((Path(bank) / "receipt.json").read_bytes()),
                "literal_target_hashes": [array_hash(t) for t in targets],
                "sac_updates": 0,
                "execution": execution,
            },
        )

    commit_directory(path, writer)
    return verify(path)


def scores(variance, errors):
    values = {
        arm: float(np.float32(variance) - np.float32(err))
        for arm, err in errors.items()
    }
    return {
        "scores": values,
        "C_minus_A": float(np.float32(values["C"]) - np.float32(values["A"])),
        "B_minus_A": float(np.float32(values["B"]) - np.float32(values["A"])),
    }


def evaluate(
    *,
    enabled=False,
    bank=None,
    reference=None,
    output=None,
    network=None,
    current=None,
    fresh=None,
    checkpoint=None,
    settings=None,
):
    """Explicit checkpoint API. Disabled means no reads, RNG draws or model work."""
    if not enabled:
        return None
    if checkpoint is None or settings is None:
        raise ValueError("explicit checkpoint and configured settings required")
    verify(bank)
    ref = verify(reference)
    if tree_hash(fresh) != ref["fresh_hash"] or repr(network) != ref["network"]:
        raise ValueError("fresh initialization/architecture mismatch")
    if digest((Path(bank) / "receipt.json").read_bytes()) != ref["bank_hash"]:
        raise ValueError("fixed bank changed")
    with np.load(Path(bank) / "bank.npz", allow_pickle=False) as d:
        raw_obs, raw_act = d["observations"], d["actions"]
    execution = precision_contract(network, current, raw_obs, raw_act)
    precision_contract(network, fresh, raw_obs, raw_act)
    if execution != ref["execution"] or execution != verify(bank)["execution"]:
        raise ValueError("resume bank/reference backend or precision mismatch")
    obs, act = jnp.asarray(raw_obs), jnp.asarray(raw_act)
    with np.load(Path(reference) / "targets.npz", allow_pickle=False) as d:
        targets, variances = d["targets"], d["variances"]
        offset = float(d["offset"])
    if targets.dtype != np.float32 or variances.dtype != np.float32:
        raise ValueError("stored targets/variances require float32")
    if [array_hash(t) for t in targets] != ref["literal_target_hashes"]:
        raise ValueError("literal targets changed")
    settings.validate(len(obs))
    signature = {
        "checkpoint": checkpoint,
        "implementation_sha256": digest(Path(__file__).read_bytes()),
        "runtime_versions": {
            k: importlib.metadata.version(k)
            for k in ("jax", "jaxlib", "flax", "optax", "numpy")
        },
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "execution": execution,
        "settings": asdict(settings),
        "current_hash": tree_hash(current),
        "reference_hash": digest((Path(reference) / "receipt.json").read_bytes()),
    }
    output = Path(output)
    if not output.exists():
        commit_directory(
            output, lambda tmp: write_json(tmp / "identity.json", signature)
        )
    if load_json(output / "identity.json") != signature:
        raise ValueError("resume source/config/checkpoint mismatch")
    # Model arrays are functional; private copies make ownership explicit.
    state = {
        "A": jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), current),
        "B": jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), fresh),
    }
    tx = _optimizer(settings.learning_rate, settings.weight_decay)
    bank_seed = verify(bank)["seed"]
    summaries = []
    for k in range(8):
        phase = output / f"phase_{k+1:02d}"
        idx = jnp.asarray(
            np.random.default_rng([bank_seed, STREAM, k]).integers(
                0, len(obs), (settings.steps, settings.batch_size)
            ),
            jnp.int32,
        )
        if phase.exists():
            rec = verify(phase)
            if rec["summary"]["K"] != k + 1 or rec["updates_per_arm"] != settings.steps:
                raise ValueError("phase sequence/update metadata mismatch")
            if (
                rec["target_sha256"] != array_hash(targets[k])
                or rec["indices_sha256"] != array_hash(idx)
                or rec["optimizer_initializations"] != {a: 1 for a in "ABC"}
                or rec["milestones"] != [0, 100, 500, 1000, 2000]
            ):
                raise ValueError("phase target/index/optimizer contract mismatch")
            for arm in "ABC":
                initial = fresh if arm == "C" else state[arm]
                if rec["initial_parameter_hashes"][arm] != tree_hash(initial):
                    raise ValueError("phase continuation parameter mismatch")
                with np.load(phase / f"{arm}.npz", allow_pickle=False) as arrays:
                    if (
                        array_hash(arrays["targets"]) != array_hash(targets[k])
                        or array_hash(arrays["indices"]) != array_hash(idx)
                        or arrays["losses"].shape != (settings.steps,)
                    ):
                        raise ValueError("phase arm arrays mismatch")
            for arm in ("A", "B"):
                state[arm] = serialization.from_bytes(
                    state[arm], (phase / f"{arm}.msgpack").read_bytes()
                )
                if tree_hash(state[arm]) != rec["final_parameter_hashes"][arm]:
                    raise ValueError("restored phase parameter mismatch")
            summaries.append(rec["summary"])
            continue
        pending_record = {"K": k + 1}
        try:
            # Same phase-specific stream in A/B/C and at all checkpoints.
            target = jnp.asarray(targets[k])
            results, fitted = {}, {}
            for arm in ("A", "B", "C"):
                start = fresh if arm == "C" else state[arm]
                # Reuse the verified captured fit primitive with an explicit
                # literal target and zero offset: no learner-dependent recentering.
                parameters, losses, final, trace, literal = _fit_literal(
                    network, tx, start, obs, act, target, idx, settings.chunk
                )
                if array_hash(literal) != ref["literal_target_hashes"][k]:
                    raise ValueError("arm target is not bitwise identical")
                arrays = {
                    "losses": np.asarray(losses),
                    "final_mse": np.asarray(final),
                    "pool_predictions": np.asarray(trace),
                    "pool_losses": np.mean(
                        (np.asarray(trace) - targets[k]) ** 2, axis=1
                    ),
                    "targets": np.asarray(literal),
                    "indices": np.asarray(idx),
                }
                if not all(np.isfinite(v).all() for v in arrays.values()):
                    raise ValueError("nonfinite fitting result")
                fitted[arm], results[arm] = parameters, arrays
            summary = {
                "K": k + 1,
                "frozen_fresh_offset": offset,
                "variance": float(variances[k]),
                "final_mse": {a: float(v["final_mse"]) for a, v in results.items()},
            }
            summary.update(scores(variances[k], summary["final_mse"]))

            def writer(tmp):
                for arm in results:
                    np.savez_compressed(tmp / f"{arm}.npz", **results[arm])
                for arm in ("A", "B"):
                    (tmp / f"{arm}.msgpack").write_bytes(
                        serialization.to_bytes(fitted[arm])
                    )
                seal(
                    tmp,
                    {
                        "summary": summary,
                        "optimizer_initializations": {a: 1 for a in results},
                        "updates_per_arm": settings.steps,
                        "milestones": [0, 100, 500, 1000, 2000],
                        "target_sha256": array_hash(target),
                        "indices_sha256": array_hash(idx),
                        "initial_parameter_hashes": {
                            a: tree_hash(fresh if a == "C" else state[a])
                            for a in results
                        },
                        "final_parameter_hashes": {
                            a: tree_hash(fitted[a]) for a in results
                        },
                    },
                )

            commit_directory(phase, writer)
            state = {a: fitted[a] for a in ("A", "B")}
            summaries.append(summary)
        except BaseException as exc:
            write_json(
                output / f"failure_{k+1:02d}_{uuid.uuid4().hex}.json",
                {
                    **pending_record,
                    "error": repr(exc),
                    "committed_phases": len(summaries),
                },
            )
            raise
    result = {
        "checkpoint": checkpoint,
        "environment": verify(bank)["environment"],
        "seed": bank_seed,
        "architecture": ref["architecture"],
        "unit": "independent training seed; K/tasks are repeated measurements",
        "sign": "C-A or B-A > 0 means A has larger final fitting error",
        "phases": summaries,
    }
    summary_path = output / "trajectory.json"
    if summary_path.exists():
        if load_json(summary_path) != result:
            raise ValueError("trajectory summary mismatch")
    else:
        write_json(summary_path, result)
    return summaries


@functools.partial(jax.jit, static_argnames=("network", "tx", "chunk"))
def _fit_literal(network, tx, params, obs, act, target, idx, chunk):
    from experiments.exp12.probe import _chunked

    def predict(p):
        return _chunked(
            lambda o, a: network.apply({"params": p}, o, a).reshape(-1), obs, act, chunk
        )

    def loss(p, indices):
        q = network.apply({"params": p}, obs[indices], act[indices]).reshape(-1)
        return jnp.mean((q - target[indices]) ** 2)

    def step(carry, item):
        p, opt, trace = carry
        indices, i = item
        value, grads = jax.value_and_grad(loss)(p, indices)
        updates, opt = tx.update(grads, opt, p)
        p = jax.tree_util.tree_map(lambda a, b: a + b, p, updates)
        slot = jnp.where(i == 99, 1, jnp.where(i == 499, 2, 3))
        trace = jax.lax.cond(
            (i == 99) | (i == 499) | (i == 999),
            lambda t: t.at[slot].set(predict(p)),
            lambda t: t,
            trace,
        )
        return (p, opt, trace), value

    trace = jnp.zeros((5, len(obs)), dtype=target.dtype).at[0].set(predict(params))
    (fitted, _, trace), losses = jax.lax.scan(
        step, (params, tx.init(params), trace), (idx, jnp.arange(len(idx)))
    )
    trace = trace.at[4].set(predict(fitted))
    return fitted, losses, jnp.mean((trace[4] - target) ** 2), trace, target


@functools.partial(jax.jit, static_argnames=("network", "chunk", "scale"))
def _teacher_values(network, key, obs, act, chunk, scale):
    from experiments.exp12.probe import _chunked

    parameters = network.init(key, observations=obs[:1], actions=act[:1])["params"]
    outputs = _chunked(
        lambda o, a: network.apply({"params": parameters}, o, a).reshape(-1),
        obs,
        act,
        chunk,
    )
    return jnp.sin(scale * outputs), parameters


@functools.lru_cache(maxsize=None)
def _optimizer(learning_rate, weight_decay):
    # Cache only the stateless transformation, never any optimizer state.
    return critic_optimizer(
        SimpleNamespace(
            critic_learning_rate=learning_rate, critic_weight_decay=weight_decay
        )
    )
