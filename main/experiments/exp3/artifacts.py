"""Strict, trusted-local Exp1/Exp2 checkpoint adapters. Pickles are not sandboxed."""

import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import gymnasium as gym
import jax
import numpy as np
from omegaconf import OmegaConf

from experiments.exp12.injection import inject, inject_twin, injection_key
from experiments.exp12.state import load_buffer, load_meta
from experiments.exp12.twin import is_twin
from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from utils.run_metadata import _hash, _strip_locations


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_fingerprint(state):
    state = Path(state).resolve()
    return _hash(
        {
            str(p.relative_to(state)): digest(p)
            for p in sorted(state.rglob("*"))
            if p.is_file()
        }
    )


def tree_equal(a, b):
    from flax.serialization import to_state_dict
    from experiments.exp12.state import _deep_equal

    # Flax static fields include newly constructed Python optimizer closures.
    # Check their protocol separately; compare serialized dynamic state exactly.
    a, b = to_state_dict(a), to_state_dict(b)
    return jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b) and all(
        _deep_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def core(agent):
    return getattr(agent, "agent", agent)


def inspect_artifact(state, metadata):
    """Fail before construction for missing provenance or checkpoint components."""
    import orbax.checkpoint

    from experiments.exp3.retention import read_snapshot

    state, metadata = Path(state).resolve(), Path(metadata).resolve()
    snapshot = read_snapshot(state)  # verified checksums; None for ordinary states
    lean = snapshot is not None and snapshot["replay"] == "omitted"
    required = ("agent_ckpt", "meta.pkl", "obs_rms.pkl") + (
        () if lean else ("buffer.npz", "buffer_meta.pkl")
    )
    for name in required:
        if not (state / name).exists():
            raise ValueError(f"missing checkpoint component: {name}")
    info = json.loads(metadata.read_text())
    for name in ("resolved_config", "config_hash", "protocol_hash", "identity", "code"):
        if name not in info:
            raise ValueError(f"missing provenance: {name}")
    cfg = info["resolved_config"]
    if (
        cfg["agent"].get("mixed_precision") is not False
        or cfg["agent"].get("normalize_observation") is not True
    ):
        raise ValueError(
            "Exp3 adapter requires the existing normalized FP32 SAC artifact contract"
        )
    if info["config_hash"] != _hash(_strip_locations(cfg)):
        raise ValueError("resolved configuration fingerprint mismatch")
    if "protocol" not in info or info["protocol_hash"] != _hash(info["protocol"]):
        raise ValueError("protocol fingerprint mismatch")
    sha = info["code"].get("commit")
    if (
        not isinstance(sha, str)
        or len(sha) != 40
        or any(c not in "0123456789abcdef" for c in sha)
    ):
        raise ValueError("missing exact source revision")
    identity = info["identity"]
    if (
        identity.get("environment") != cfg["env_name"]
        or identity.get("seed") != cfg["seed"]
    ):
        raise ValueError("environment/seed provenance differs from config")
    meta = load_meta(state)
    for key in (
        "interaction_step",
        "update_step",
        "update_counter",
        "numpy_rng_state",
        "python_rng_state",
        "train_env",
        "eval_env",
        "extra_state",
        "observations",
        "timestep",
        "actor_diagnostics",
        "agent_window_buffers",
    ):
        if key not in meta:
            raise ValueError(f"missing complete-state field: {key}")
    for name in ("interaction_step", "update_step"):
        if type(meta[name]) is not int or meta[name] < 0:
            raise ValueError(f"invalid counter: {name}")
    structure = orbax.checkpoint.PyTreeCheckpointer().metadata(
        str(state / "agent_ckpt")
    )
    for name in (
        "actor",
        "critic",
        "target_critic",
        "temperature",
        "rng",
        "churn_ref_batch",
    ):
        if name not in structure:
            raise ValueError(f"missing agent field: {name}")
    for name in ("actor", "critic", "temperature"):
        if not all(
            k in structure[name] for k in ("params", "opt_state", "update_step")
        ):
            raise ValueError(f"missing parameters/optimizer/counter: {name}")
        if structure[name]["opt_state"] is None:
            raise ValueError(f"missing optimizer state: {name}")
    with open(state / "obs_rms.pkl", "rb") as f:
        rms = pickle.load(f)
    if set(rms) != {"mean", "var", "count"} or not all(
        np.isfinite(x).all() for x in rms.values()
    ):
        raise ValueError("invalid observation normalization")
    if np.asarray(rms["mean"]).shape != np.asarray(rms["var"]).shape or np.any(
        rms["var"] < 0
    ):
        raise ValueError("invalid normalization shape/variance")
    if lean:
        # Replay deliberately omitted at retention; dimensions were read from the
        # source state's replay headers when the snapshot was published.
        obs_dim, act_dim = snapshot["observation_dim"], snapshot["action_dim"]
        if np.asarray(rms["mean"]).shape not in ((obs_dim,), (1, obs_dim)):
            raise ValueError("normalization/snapshot observation dimensions differ")
        return info, meta, obs_dim, act_dim
    with zipfile.ZipFile(state / "buffer.npz") as archive:
        headers = {}
        for name in (
            "observations",
            "actions",
            "rewards",
            "terminateds",
            "truncateds",
            "next_observations",
        ):
            with archive.open(name + ".npy") as member:
                version = np.lib.format.read_magic(member)
                shape, _, dtype = np.lib.format._read_array_header(member, version)
            if dtype.hasobject or not shape:
                raise ValueError("invalid replay array header")
            headers[name] = (shape, dtype)
        if len(archive.namelist()) != len(headers):
            raise ValueError("unexpected/duplicate replay arrays")
    with open(state / "buffer_meta.pkl", "rb") as stream:
        buffer_meta = pickle.load(stream)
    n, idx = buffer_meta["num_in_buffer"], buffer_meta["current_idx"]
    capacity = cfg["buffer"]["max_length"]
    if (
        type(n) is not int
        or type(idx) is not int
        or not 0 <= n <= capacity
        or not 0 <= idx < capacity
        or "n_step_transitions" not in buffer_meta
    ):
        raise ValueError("invalid complete replay metadata")
    obs_dim, act_dim = headers["observations"][0][-1], headers["actions"][0][-1]
    for name, (shape, dtype) in headers.items():
        expected = (
            (n, obs_dim)
            if name in ("observations", "next_observations")
            else (n, act_dim) if name == "actions" else (n,)
        )
        if shape != expected or dtype != np.dtype("float32"):
            raise ValueError(f"invalid replay shape/dtype: {name}")
    if np.asarray(rms["mean"]).shape not in ((obs_dim,), (1, obs_dim)):
        raise ValueError("normalization/replay observation dimensions differ")
    return info, meta, obs_dim, act_dim


class Artifact:
    """Reuse the exact agent/buffer restoration path without allocating an environment."""

    def __init__(self, state, metadata, load_replay=True):
        self.state = Path(state).resolve()
        self.metadata = Path(metadata).resolve()
        self.info, self.meta, obs_dim, act_dim = inspect_artifact(
            self.state, self.metadata
        )
        if load_replay and not (self.state / "buffer.npz").exists():
            raise ValueError("replay was omitted from this retained snapshot")
        self.cfg = OmegaConf.create(self.info["resolved_config"])
        obs = gym.spaces.Box(-np.inf, np.inf, (1, obs_dim), dtype=np.float32)
        act = gym.spaces.Box(-1, 1, (1, act_dim), dtype=np.float32)
        self.agent = create_agent(obs, act, self.cfg.agent)
        self.buffer = None
        if load_replay:
            self.buffer = create_buffer(
                observation_space=obs, action_space=act, **self.cfg.buffer
            )
            self.buffer.reset()
        intervention = self.meta["extra_state"].get("injection")
        self.intervention = None
        if intervention:
            self.inject(intervention["m"], intervention["seed"])
        a = core(self.agent)
        templates = {
            name: jax.tree_util.tree_map(
                lambda x: (x.shape, np.dtype(x.dtype).str), getattr(a, name).params
            )
            for name in ("_actor", "_critic", "_target_critic", "_temperature")
        }
        self.agent.load_checkpoint(str(self.state))
        if (
            int(core(self.agent)._actor.update_step) != self.meta["update_step"]
            or int(core(self.agent)._temperature.update_step)
            != self.meta["update_step"]
        ):
            raise ValueError(
                "agent actor/temperature counters differ from complete-state metadata"
            )
        for name, expected in templates.items():
            actual = jax.tree_util.tree_map(
                lambda x: (x.shape, np.dtype(x.dtype).str),
                getattr(core(self.agent), name).params,
            )
            if expected != actual:
                raise ValueError(
                    f"checkpoint parameters incompatible with resolved config: {name}"
                )
        if load_replay:
            load_buffer(self.buffer, self.state)

    def inject(self, m, seed):
        a = core(self.agent)
        fn = inject_twin if is_twin(a._critic.params) else inject
        a._critic, a._target_critic = fn(
            a._critic,
            a._target_critic,
            m,
            injection_key(seed),
            float(self.cfg.agent.critic_learning_rate),
            float(self.cfg.agent.critic_weight_decay),
        )
        self.intervention = {"m": m, "seed": int(seed)}

    def provenance(self):
        return {
            "state": str(self.state),
            "metadata": str(self.metadata),
            "metadata_sha256": digest(self.metadata),
            "source": self.info["code"],
            "identity": self.info["identity"],
            "config_hash": self.info["config_hash"],
            "interaction_step": self.meta["interaction_step"],
            "update_step": self.meta["update_step"],
            "intervention": self.intervention,
        }


def require_matched(u, i, fork=None):
    """Post-fork checkpoints must refer to the same parent and absolute time."""
    for key in ("environment", "seed", "architecture"):
        if u.info["identity"].get(key) != i.info["identity"].get(key):
            raise ValueError(f"unmatched artifacts: {key}")
    # Agent structures may differ due to injection; ordinary training settings may not.
    if u.info["protocol_hash"] != i.info["protocol_hash"]:
        raise ValueError("unmatched training protocol")
    if u.meta["interaction_step"] != i.meta["interaction_step"]:
        raise ValueError("unmatched post-fork checkpoint steps")
    if (u.meta["update_step"], u.meta["update_counter"]) != (
        i.meta["update_step"],
        i.meta["update_counter"],
    ):
        raise ValueError("unmatched post-fork update counters")
    if fork is not None:
        for key in ("environment", "seed", "architecture"):
            if u.info["identity"].get(key) != fork.info["identity"].get(key):
                raise ValueError(f"fork provenance differs: {key}")
        if fork.meta["interaction_step"] > u.meta["interaction_step"]:
            raise ValueError("fork occurs after panel checkpoint")
        for name, a in (("untreated", u), ("injected", i)):
            if a.meta["interaction_step"] > fork.meta["interaction_step"]:
                plan = a.meta["extra_state"].get("fork")
                if (
                    not plan
                    or plan.get("run_key") != fork.info["identity"].get("run_key")
                    or plan.get("fork_step") != fork.meta["interaction_step"]
                ):
                    raise ValueError(f"missing/mismatched actual fork lineage: {name}")
        if not i.meta["extra_state"].get("injection"):
            raise ValueError("injected checkpoint lacks actual injection provenance")
        if u.meta["extra_state"].get("injection") or fork.meta["extra_state"].get(
            "injection"
        ):
            raise ValueError("untreated/fork checkpoint is already injected")
