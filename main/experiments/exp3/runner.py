"""Isolated, resumable Exp3 execution; no scheduler, implicit budget or source run."""

import copy
import json
import pickle
import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.exp12.envs import create_eval_env
from experiments.exp12.fork import check1, panel_q_and_grad
from experiments.exp12.trainer import evaluate_episodes
from experiments.exp3.artifacts import (
    Artifact,
    core,
    digest,
    require_matched,
    state_fingerprint,
    tree_equal,
)
from experiments.exp3.guidance import (
    assert_fork_equivalence,
    intervention_gradient,
    measure_chunk,
    optimizer_step,
    updated_policy_arrays,
)
from experiments.exp3.passive import PassivePair, replay_batch
from experiments.exp3.protocol import validate_protocol
from experiments.exp3.streams import Benchmark, StreamReader, atomic_pickle, write_json
from experiments.exp3.target_policy import TargetPair
from utils.run_metadata import code_version


def artifact(spec, name, load_replay=True):
    data = spec["artifacts"][name]
    return Artifact(data["state"], data["metadata"], load_replay=load_replay)


def evaluate_agent(agent, cfg, episodes, seed):
    """Evaluation-only environments and common RNGs, with all training state preserved."""
    import random

    np_state, py_state, key = np.random.get_state(), random.getstate(), core(agent)._rng
    env = None
    try:
        np.random.seed(seed)
        random.seed(seed)
        core(agent)._rng = jax.random.PRNGKey(seed)
        env_cfg = dict(cfg.env)
        env_cfg["seed"] = seed
        env = create_eval_env(**env_cfg)
        _, returns, lengths, successes = evaluate_episodes(agent, env, episodes)
        return {
            "returns": list(map(float, returns)),
            "lengths": list(map(int, lengths)),
            "successes": list(map(float, successes)),
        }
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        core(agent)._rng = key
        if env is not None:
            env.close()


def panel(spec, f):
    """One explicit common raw-state panel, saved once and fingerprinted."""
    if spec.get("panel"):
        with np.load(spec["panel"], allow_pickle=False) as data:
            raw = np.asarray(data["observation_raw"])
        if len(raw) != spec["panel_size"]:
            raise ValueError("fixed panel size differs from protocol")
        return raw, {
            "path": str(Path(spec["panel"]).resolve()),
            "sha256": digest(spec["panel"]),
        }
    rng = np.random.RandomState(spec["rng_seed"])
    if not len(f.buffer):
        raise ValueError("fork replay has no panel states")
    idx = rng.randint(0, len(f.buffer), size=spec["panel_size"])
    return f.buffer._observations[idx].copy(), {
        "source": "fork_replay",
        "indices": idx.tolist(),
    }


def array_file(path, values):
    path = Path(path)
    values = {k: np.asarray(v) for k, v in values.items()}
    if path.exists():
        # A crash after output publication but before checkpoint publication can
        # replay this step. Verify identical evidence, never silently overwrite.
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(values) or not all(
                (
                    existing[k].shape == v.shape
                    and existing[k].dtype == v.dtype
                    and not v.dtype.hasobject
                    and existing[k].tobytes() == v.tobytes()
                )
                for k, v in values.items()
            ):
                raise ValueError(f"replayed diagnostic output differs: {path}")
        return {"name": path.name, "sha256": digest(path)}
    # Never expose a partial NPZ after a crash; only complete output is compared
    # on deterministic replay from the last committed diagnostic checkpoint.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".pending", delete=False
        ) as f:
            temporary = Path(f.name)
            np.savez(f, **values)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return {"name": path.name, "sha256": digest(path)}


def pilot1(spec, out, bench):
    f, u, i = (
        artifact(spec, k, load_replay=(k == "fork"))
        for k in ("fork", "untreated", "injected")
    )
    require_matched(u, i, f)
    raw, origin = panel(spec, f)
    array_file(out / "common_panel.npz", {"observation_raw": raw})
    report = {
        "panel": origin,
        "artifacts": {
            k: a.provenance()
            for k, a in (("fork", f), ("untreated", u), ("injected", i))
        },
        "actors": {},
    }
    if u.meta["interaction_step"] == f.meta["interaction_step"]:
        assert_fork_equivalence(core(u.agent), core(i.agent))
        obs = f.agent._normalize(raw)
        action = (
            core(f.agent)
            .actor(observations=obs)
            .sample(seed=jax.random.PRNGKey(spec["rng_seed"]))
        )
        check_panel = {
            "observation": np.asarray(obs, np.float32),
            "action": np.asarray(action),
        }
        pre = panel_q_and_grad(core(f.agent).critic, check_panel)
        result = check1(
            pre,
            panel_q_and_grad(core(i.agent).critic, check_panel),
            panel_q_and_grad(core(u.agent).critic, check_panel),
            float(f.cfg.checks.check1_tolerance_eps),
            injected=True,
        )
        if not result["pass"]:
            raise ValueError("existing approved fork-time Check1 failed")
        report["fork_equivalence"] = result
    else:
        report["fork_equivalence"] = (
            "post-fork snapshots: separate actual fork-time proof required"
        )
    for name, owner in (("u", u), ("i", i), ("f", f)):
        actor = core(owner.agent).actor
        alpha = (
            core(f.agent).temperature()
            if spec["alpha_source"] == "fork"
            else core(owner.agent).temperature()
        )
        norm = f.agent if spec["normalization"] == "common_fork" else owner.agent
        # Match raw states/actions. Either use a declared common normalization
        # intervention or preserve each artifact's original preprocessing.
        obs = jnp.asarray(norm._normalize(raw))
        critic_inputs = (
            None
            if spec["normalization"] == "common_fork"
            else {
                "u": jnp.asarray(u.agent._normalize(raw)),
                "i": jnp.asarray(i.agent._normalize(raw)),
            }
        )
        totals = {mode: None for mode in spec["interventions"]}
        ordinary = {"u": None, "i": None}
        chunks = []
        for start in range(0, len(obs), spec["chunk_size"]):
            part = obs[start : start + spec["chunk_size"]]
            cpart = (
                None
                if critic_inputs is None
                else {
                    k: v[start : start + spec["chunk_size"]]
                    for k, v in critic_inputs.items()
                }
            )
            key = jax.random.fold_in(jax.random.PRNGKey(spec["rng_seed"]), start)
            values, grads = bench.measure(
                "guidance",
                measure_chunk,
                actor,
                core(u.agent).critic,
                core(i.agent).critic,
                part,
                key,
                alpha,
                bool(f.cfg.agent.critic_use_cdq),
                cpart,
            )
            chunks.append(array_file(out / f"actor_{name}_{start:09d}.npz", values))
            weight = len(part) / len(obs)
            for label, g in grads.items():
                ordinary[label] = (
                    jax.tree_util.tree_map(lambda x: x * weight, g)
                    if ordinary[label] is None
                    else jax.tree_util.tree_map(
                        lambda a, b: a + b * weight, ordinary[label], g
                    )
                )
            for mode in totals:
                g = intervention_gradient(
                    actor,
                    core(u.agent).critic,
                    core(i.agent).critic,
                    part,
                    key,
                    alpha,
                    bool(f.cfg.agent.critic_use_cdq),
                    mode,
                    spec["zero_signal_policy"],
                    cpart,
                    spec["intervention_space"],
                )
                totals[mode] = (
                    jax.tree_util.tree_map(lambda x: x * weight, g)
                    if totals[mode] is None
                    else jax.tree_util.tree_map(
                        lambda a, b: a + b * weight, totals[mode], g
                    )
                )
        outcomes = {}
        for mode, gradient in {
            **{f"ordinary_{k}": v for k, v in ordinary.items()},
            **totals,
        }.items():
            updated = optimizer_step(
                actor, gradient
            )  # starts from SAME params/Adam state every condition
            array_file(
                out / f"actor_{name}_{mode}_update.npz",
                updated_policy_arrays(actor, updated, obs),
            )
            saved = core(owner.agent)._actor
            try:
                core(owner.agent)._actor = updated
                outcomes[mode] = bench.measure(
                    "evaluation",
                    evaluate_agent,
                    owner.agent,
                    owner.cfg,
                    spec["outcome_eval_episodes"],
                    spec["rng_seed"],
                )
            finally:
                core(owner.agent)._actor = saved
        report["actors"][name] = {
            "chunks": chunks,
            "outcomes": outcomes,
            "alpha": float(alpha),
            "optimizer_matching": "same immutable actor Trainer for all conditions",
        }
    if spec.get("source_panels"):
        report["source_panels"] = {}
        for label, value in spec["source_panels"].items():
            if not label or Path(label).name != label or label in (".", ".."):
                raise ValueError(
                    "source panel label must be a single safe path component"
                )
            nested = copy.deepcopy(spec)
            nested.pop("source_panels")
            nested["panel"], nested["panel_size"] = value["path"], value["size"]
            validate_protocol(nested)
            directory = out / f"source_panel_{label}"
            directory.mkdir()
            report["source_panels"][label] = pilot1(nested, directory, bench)
    return report


def make_passive(spec, reader, source="u", benchmark=None):
    u, i = artifact(spec, "fork"), artifact(spec, "fork")
    if u.meta["extra_state"].get("injection"):
        raise ValueError("passive common fork must be pre-injection")
    provenance = reader.manifest["provenance"]
    if provenance.get("arm") not in (
        {"control", "untreated"} if source == "u" else {"injected"}
    ):
        raise ValueError("source stream arm assignment differs")
    for key, value in (
        ("start_step", u.meta["interaction_step"]),
        ("fork_metadata_sha256", digest(u.metadata)),
        ("config_hash", u.info["config_hash"]),
        ("fork_state_sha256", state_fingerprint(u.state)),
    ):
        if provenance.get(key) != value:
            raise ValueError(f"source stream/fork provenance mismatch: {key}")
    i.inject(spec["injection_m"], spec["injection_seed"])
    pair = PassivePair(
        u,
        i,
        u.meta["numpy_rng_state"],
        spec["normalization"],
        reader.fingerprint,
        benchmark,
    )
    # Reuse the approved Exp2 fixed panel; do not choose a new Check1 panel size.
    with np.load(spec["fork_panel"], allow_pickle=False) as saved:
        fixed = {k: saved[k] for k in saved.files}
    from experiments.exp12.fork import PANEL_SIZE

    if len(fixed["action"]) != PANEL_SIZE:
        raise ValueError("passive fork panel must be the existing 256-pair Exp2 panel")
    expected_obs = np.asarray(u.agent._normalize(fixed["observation_raw"]), np.float32)
    if not tree_equal(expected_obs, fixed["observation"]):
        raise ValueError("fixed fork panel normalization differs from restored fork")
    pre = panel_q_and_grad(core(u.agent).critic, fixed)
    pair.initial_check = check1(
        pre,
        panel_q_and_grad(core(i.agent).critic, fixed),
        pre,
        float(u.cfg.checks.check1_tolerance_eps),
        injected=True,
    )
    if not pair.initial_check["pass"]:
        raise ValueError("passive initial injection fails the existing Check1 rule")
    return pair


def pilot2(spec, out, bench, resume=False):
    labels = ("u", "i") if spec["stage_b"] else ("u",)
    report = {
        "sources": {},
        "active_references": spec.get("active_references", {}),
        "note": "active trajectories are references, never passive substitutes",
    }
    for source in labels:
        reader = StreamReader(spec["streams"][source])
        if reader.manifest["count"] < spec["arrivals"]:
            raise ValueError("ordered source stream too short")
        pair = make_passive(spec, reader, source, bench)
        root = out / f"source_{source}"
        root.mkdir(exist_ok=True)
        checkpoint = root / "checkpoints"
        if resume and (checkpoint / "LATEST.json").exists():
            pair.restore(checkpoint)
        for arrival, record in enumerate(reader, 1):
            if arrival <= pair.cursor:
                continue
            if arrival > spec["arrivals"]:
                break
            bench.measure(
                "transition_delivery_and_sac",
                pair.deliver,
                record,
                spec["updates_per_arrival"],
            )
            if pair.last_metrics:
                write_json(
                    root / f"metrics_{arrival:09d}.json",
                    {
                        k: {field: float(value) for field, value in v.items()}
                        for k, v in pair.last_metrics.items()
                    },
                )
            if arrival % spec["evaluation_every"] == 0:
                value = bench.measure(
                    "evaluation",
                    pair.evaluate,
                    lambda agent: evaluate_agent(
                        agent, pair.cfg, spec["evaluation_episodes"], spec["rng_seed"]
                    ),
                )
                write_json(root / f"evaluation_{arrival:09d}.json", value)
            if arrival % spec["checkpoint_every"] == 0:
                bench.measure("checkpoint", pair.save, checkpoint)
        bench.measure("checkpoint", pair.save, checkpoint)
        report["sources"][source] = {
            "stream_sha256": reader.fingerprint,
            "arrivals": pair.cursor,
            "update_step": pair.update_step,
            "checkpoint": str(checkpoint),
            "passive": True,
            "fork": pair.learners["u"].provenance(),
            "initial_check": pair.initial_check,
        }
    return report


def target_checkpoint(root, pairs, sampler):
    """Private diagnostic checkpoint stores immutable Trainer state and both RNGs."""
    state = {
        k: {
            "critics": {
                label: {
                    "params": c.params,
                    "opt_state": c.opt_state,
                    "update_step": c.update_step,
                }
                for label, c in p.critics.items()
            },
            "key": np.asarray(p.key),
            "update_step": p.update_step,
        }
        for k, p in pairs.items()
    }
    atomic_pickle(
        root / "target_state.pkl", {"pairs": state, "sampler": sampler.get_state()}
    )


def pilot3(spec, out, bench, resume=False):
    owners = {
        k: artifact(spec, k, load_replay=(k == "fork"))
        for k in ("fork", "untreated", "injected")
    }
    require_matched(owners["untreated"], owners["injected"], owners["fork"])
    f = owners["fork"]
    raw, origin = panel(spec, f)
    critic_owner = owners[spec["critic_source"]]
    common = spec["normalization"] == "common_fork"
    obs = jnp.asarray((f if common else critic_owner).agent._normalize(raw))
    alpha = core(owners[spec["alpha_source"]].agent).temperature()
    start = core(owners[spec["critic_source"]].agent).critic
    choices = [spec["evaluator_source"]]
    if (
        spec["alternative_evaluator"] is not None
        and spec["alternative_evaluator"] not in choices
    ):
        choices.append(spec["alternative_evaluator"])
    pairs = {
        label: TargetPair(
            start,
            core(owners["untreated"].agent).actor,
            core(owners["injected"].agent).actor,
            core(owners[label].agent)._target_critic,
            alpha,
            float(f.cfg.gamma),
            int(f.cfg.n_step),
            bool(f.cfg.agent.critic_use_cdq),
            jax.random.PRNGKey(spec["rng_seed"]),
        )
        for label in choices
    }
    sampler = np.random.RandomState(spec["rng_seed"])
    if resume and (out / "target_state.pkl").exists():
        with open(out / "target_state.pkl", "rb") as stream:
            saved = pickle.load(stream)
        if set(saved["pairs"]) != set(pairs):
            raise ValueError("target sensitivity resume mismatch")
        for label, p in pairs.items():
            row = saved["pairs"][label]
            p.critics = {
                k: p.critics[k].replace(**jax.tree_util.tree_map(jnp.asarray, v))
                for k, v in row["critics"].items()
            }
            p.key, p.update_step = jnp.asarray(row["key"]), row["update_step"]
        sampler.set_state(saved["sampler"])
    begin = next(iter(pairs.values())).update_step
    if any(p.update_step != begin for p in pairs.values()):
        raise ValueError("target pair resume counters differ")
    for step in range(begin, spec["updates"]):
        idx = sampler.randint(0, len(f.buffer), size=spec["batch_size"])
        batch = replay_batch(f.buffer, idx)
        raw_next = batch["next_observation"].copy()
        for name in ("observation", "next_observation"):
            batch[name] = (f if common else critic_owner).agent._normalize(batch[name])
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        for label, p in pairs.items():
            actor_obs = (
                None
                if common
                else {
                    k: jnp.asarray(owners[name].agent._normalize(raw_next))
                    for k, name in (("u", "untreated"), ("i", "injected"))
                }
            )
            eval_obs = (
                None
                if common
                else jnp.asarray(owners[label].agent._normalize(raw_next))
            )
            values = bench.measure(
                "target_diagnostic", p.update, batch, actor_obs, eval_obs
            )
            array_file(
                out / f"targets_{label}_{step:09d}.npz",
                {
                    "noise": values["noise"],
                    "target_u": values["u"]["target"],
                    "target_i": values["i"]["target"],
                    "target_difference": values["target_difference"],
                    "indices": idx,
                    "loss_u": values["u"]["loss"],
                    "loss_i": values["i"]["loss"],
                },
            )
        if (step + 1) % spec["checkpoint_every"] == 0:
            target_checkpoint(out, pairs, sampler)
    target_checkpoint(out, pairs, sampler)
    for label, p in pairs.items():
        actor_obs = jnp.asarray(f.agent._normalize(raw))
        for start in range(0, len(obs), spec["chunk_size"]):
            values = measure_chunk(
                core(f.agent).actor,
                p.critics["u"],
                p.critics["i"],
                actor_obs[start : start + spec["chunk_size"]],
                jax.random.fold_in(jax.random.PRNGKey(spec["rng_seed"]), start),
                alpha,
                bool(f.cfg.agent.critic_use_cdq),
                {
                    "u": obs[start : start + spec["chunk_size"]],
                    "i": obs[start : start + spec["chunk_size"]],
                },
            )[0]
            array_file(out / f"final_panel_{label}_{start:09d}.npz", values)
    return {
        "updates": spec["updates"],
        "alpha": float(alpha),
        "alpha_source": spec["alpha_source"],
        "panel": origin,
        "evaluators": choices,
        "actors_frozen": True,
        "artifacts": {k: a.provenance() for k, a in owners.items()},
    }


def run(spec, out, resume=False):
    from experiments.exp12.precision import (
        configure_compilation_cache,
        runtime_info,
        set_matmul_precision,
    )

    validate_protocol(spec)
    if not Path(out).is_absolute():
        raise ValueError("absolute output directory required")
    out = Path(out).resolve()
    checkout = Path(__file__).resolve().parents[3]
    if out == checkout or checkout in out.parents:
        raise ValueError("pilot output must be outside the source checkout")
    source = code_version()
    # This interface intentionally has no submission, source interaction or budget defaults.
    inputs = {}
    for name, data in spec["artifacts"].items():
        state = Path(data["state"]).resolve()
        inputs[name] = {
            "metadata": digest(data["metadata"]),
            "state": {
                str(p.relative_to(state)): digest(p)
                for p in state.rglob("*")
                if p.is_file()
            },
        }
    receipt = {"spec": spec, "code": source, "inputs": inputs}
    receipt["auxiliary_inputs"] = {
        name: digest(spec[name]) for name in ("panel", "fork_panel") if spec.get(name)
    }
    for label, value in spec.get("source_panels", {}).items():
        receipt["auxiliary_inputs"]["source_panel_" + label] = digest(value["path"])
    for label, value in spec.get("streams", {}).items():
        if value:
            receipt["auxiliary_inputs"]["stream_" + label] = digest(
                Path(value) / "manifest.json"
            )
    if resume:
        previous = json.loads((out / "run.json").read_text())
        if previous != receipt:
            raise ValueError("protocol/source revision differs on resume")
        if (out / "DONE").exists():
            raise ValueError("pilot already complete")
        if spec["pilot"] == 1:
            raise ValueError(
                "Pilot1 is immutable measurement; restart in a new output directory"
            )
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "run.json", receipt)
    set_matmul_precision()
    configure_compilation_cache()
    write_json(out / "runtime.json", runtime_info())
    bench = Benchmark()
    with bench.compilation():
        report = (
            pilot1(spec, out, bench)
            if spec["pilot"] == 1
            else (
                pilot2(spec, out, bench, resume)
                if spec["pilot"] == 2
                else pilot3(spec, out, bench, resume)
            )
        )
    write_json(out / "report.json", report)
    write_json(out / "benchmark.json", bench.report(out))
    files = {
        str(p.relative_to(out)): digest(p)
        for p in out.rglob("*")
        if p.is_file() and p.name not in ("SHA256SUMS.json", "DONE")
    }
    write_json(out / "SHA256SUMS.json", files)
    write_json(
        out / "DONE",
        {
            "status": "development pilot complete; not scientific qualification",
            "code": source,
        },
    )
    return report
