"""Synthetic, complete study evidence; no CUDA execution or scientific results."""

import copy
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import exp12_validation as v
from experiments.exp12 import exp2_ledger, fork, ledger
from experiments.exp12.probe import iqm
from scripts.freeze_exp12_analysis_manifest import build
from utils.run_metadata import _hash, _strip_locations, training_protocol

COMMIT = "1" * 40
DIAGNOSTICS = (
    "train/churn",
    "train/policy_kl",
    "train/actor_gnorm",
    "train/actor_gnorm_std",
    "train/actor_action",
    "train/actor_saturation",
)


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def metadata(source, cfg, study, a, e, s, experiment="exp1"):
    protocol = training_protocol(cfg)
    runtime = study["runtime_by_suite"][v.suite(e)]
    protocol["simulator_versions"] = runtime["simulator_versions"]
    launch = {
        "commit": COMMIT,
        "dirty": False,
        "device": {"platform": "gpu", "device_kind": "SYNTHETIC"},
        "runtime": {**runtime, "platform": "gpu", "matmul_precision": "tensorfloat32"},
        "critic_count": 2 if v.suite(e) == "humanoid_bench" else 1,
    }
    identity = dict(
        experiment=experiment,
        architecture=a,
        environment=e,
        seed=s,
        run_role="confirmatory",
        run_key=v.run_key(a, e, s).replace("exp1_", experiment + "_", 1),
    )
    json_write(
        source / "run_metadata.json",
        dict(
            schema_version=1,
            identity=identity,
            code={"commit": COMMIT, "dirty": False},
            resolved_config=cfg,
            config_hash=_hash(_strip_locations(cfg)),
            protocol=protocol,
            protocol_hash=_hash(protocol),
            launches=[launch],
        ),
    )


def state(root, meta):
    path = root / f"step_{meta['interaction_step']}"
    path.mkdir(parents=True, exist_ok=True)
    (root / "LATEST").write_text(path.name)
    (path / "meta.pkl").write_bytes(pickle.dumps(meta))
    for name in ("buffer.npz", "buffer_meta.pkl", "obs_rms.pkl", "agent_ckpt/dummy"):
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"synthetic persistence evidence")


def records(n, end, twin=False, eligible=False, invalid=False, fork_check=2):
    result, curves = [], {}
    for k in range(end // (n // 20) + 1):
        fresh = np.ones(5, np.float32)
        current = (
            np.zeros(5, np.float32)
            if eligible and k in (fork_check - 1, fork_check)
            else np.full(5, 2.0, np.float32)
        )
        if invalid and k == 3:
            current[0] = np.nan
        loss = fresh - current
        valid = bool(np.isfinite(loss).all())
        row = dict(
            check_index=k,
            interaction_step=5000 if k == 0 else k * n // 20,
            **{f"score_fresh_r{r}": float(x) for r, x in enumerate(fresh)},
            score_fresh_iqm=iqm(fresh),
        )
        if k:
            low = float(loss[0]) if valid else float("nan")
            row.update(
                **{f"score_current_r{r}": float(x) for r, x in enumerate(current)},
                **{f"loss_r{r}": float(x) for r, x in enumerate(loss)},
                score_current_iqm=iqm(current) if valid else float("nan"),
                score_fresh_iqm=iqm(fresh) if valid else float("nan"),
                loss_iqm=iqm(loss) if valid else float("nan"),
                ci_low=low,
                ci_high=low,
                valid=valid,
                triggered=valid and low > 0,
            )
        for name, values in [("fresh", fresh)] + ([("current", current)] if k else []):
            for part in [f"{name}_q{q}" for q in (1, 2)] if twin else [name]:
                curves[f"check_{k:02d}/{part}_losses"] = np.zeros((5, 1000), np.float32)
                curves[f"check_{k:02d}/{part}_score"] = values
                if twin:
                    row.update(
                        {f"score_{part}_r{r}": float(x) for r, x in enumerate(values)}
                    )
        if twin and k:
            for q in (1, 2):
                row.update({f"loss_q{q}_r{r}": float(x) for r, x in enumerate(loss)})
                row[f"loss_q{q}_iqm"] = iqm(loss) if valid else float("nan")
        result.append(row)
    return result, curves


def metrics(end):
    return [
        {"env_step": k * 2, **({x: 0.1 for x in DIAGNOSTICS} if k else {})}
        for k in range(0, end + 1, 2000)
    ]


def evals(n, end):
    return [
        dict(interaction_step=k, episode=e, **{"return": 1.0}, length=1)
        for k in range(0, end + 1, 50_000)
        for e in range(5 if k == 0 or k + 50_000 <= n else 50)
    ]


def parent(base, study, a, e, s, eligible=False, invalid=False, fork_check=2):
    results, checkpoints = base / "results", base / "checkpoints"
    key, n = v.run_key(a, e, s), v.budget(e) // 2
    source = checkpoints / "exp1" / a / e / f"seed_{s}"
    source.mkdir(parents=True, exist_ok=True)
    cfg = v.seeded(study["configs"][f"{a}/{e}"], s)
    metadata(source, cfg, study, a, e, s)
    fs = (
        {"check_index": fork_check, "interaction_step": fork_check * n // 20}
        if eligible
        else None
    )
    end = max(n, fs["interaction_step"] + n // 4) if fs else n
    recs, curves = records(
        n, end, v.suite(e) == "humanoid_bench", eligible, invalid, fork_check
    )
    m = dict(
        interaction_step=end,
        metrics_rows=metrics(end),
        eval_rows=evals(n, end),
        extra_state={"probe_records": recs, "f_star": fs},
    )
    state(source / "state", m)
    json_write(source / "DONE", {"interaction_step": end})
    (source / "fresh_critic").mkdir()
    (source / "fresh_critic/dummy").write_bytes(b"synthetic reference")
    identity = dict(
        run_key=key,
        run_role="confirmatory",
        architecture=a,
        environment=e,
        seed=s,
        budget_env_steps=2 * n,
        num_interaction_steps=n,
        num_checks=20,
        code_commit=COMMIT,
    )
    directory = ledger.write_run(
        identity,
        recs,
        fs,
        "complete",
        source / "probes",
        str(results),
        None if fs is None else fs["interaction_step"],
        m["metrics_rows"],
        2,
    )
    np.savez_compressed(directory / "probe_curves.npz", **curves)
    if eligible:
        make_fork(base, study, a, e, s, cfg, source, m, fs)
    return directory, source


def make_fork(base, study, a, e, s, cfg, source, meta, fs):
    results = str(base / "results")
    key, n, f = v.run_key(a, e, s), v.budget(e) // 2, fs["interaction_step"]
    plan = dict(
        fork_step=f,
        fork_check_index=fs["check_index"],
        num_interaction_steps=n,
        horizon_steps=n // 4,
        arm_end_step=f + n // 4,
        control_end_step=max(n, f + n // 4),
        eval_every_steps=n // 100,
        eval_episodes=10,
        run_key=key,
        device={"platform": "gpu", "device_kind": "SYNTHETIC"},
    )
    json_write(source / "fork/fork.json", plan)
    (source / "fork/FORK_READY").write_text(key)
    fork_meta = copy.deepcopy(meta)
    fork_meta["interaction_step"] = f
    state(source / "fork/state", fork_meta)
    exp2_ledger.write_json(key, "fork.json", plan, results)
    panel = dict(
        observation=np.ones((256, 6), np.float32), action=np.ones((256, 3), np.float32)
    )
    np.savez(source / "fork/panel.npz", **panel)
    pre = dict(
        q=np.ones(512 if v.suite(e) == "humanoid_bench" else 256, np.float32),
        dq_da=np.ones((256, 3), np.float32),
    )
    for name in ("pre", "control"):
        np.savez(source / f"fork/check1_{name}.npz", **pre)
    for arm in ("control", "injected"):
        arm_source = (
            source
            if arm == "control"
            else base / "checkpoints/exp2_arm" / a / e / f"seed_{s}/injected"
        )
        arm_source.mkdir(parents=True, exist_ok=True)
        end = plan["control_end_step" if arm == "control" else "arm_end_step"]
        m = copy.deepcopy(meta)
        m["interaction_step"] = end
        m["metrics_rows"] = metrics(end)
        m["extra_state"]["probe_records"] = [
            r
            for r in meta["extra_state"]["probe_records"]
            if r["interaction_step"] <= end
        ]
        m["extra_state"]["post_fork_evals"] = [
            dict(
                arm=arm,
                eval_index=j,
                episode=ep,
                steps_since_fork=j * n // 100,
                interaction_step=f + j * n // 100,
                **{"return": 1.0 + (arm == "injected")},
                length=1,
                success=float("nan"),
            )
            for j in range(26)
            for ep in range(10)
        ]
        if arm == "injected":
            arm_cfg = copy.deepcopy(cfg)
            arm_cfg["fork"].update(source=str(source), arm=arm)
            arm_cfg["injection"]["m"] = study["injection_m"]
            metadata(arm_source, arm_cfg, study, a, e, s, "exp2_arm")
            np.savez(arm_source / "check1_after.npz", **pre)
        state(arm_source / "state", m)
        json_write(arm_source / "DONE", {"interaction_step": end})
        exp2_ledger.write_arm(
            key,
            arm,
            plan,
            m["extra_state"]["probe_records"],
            m["extra_state"]["post_fork_evals"],
            m["metrics_rows"],
            2,
            results,
            run_dir=arm_source,
        )
        exp2_ledger.write_json(
            key,
            "check1_" + arm + ".json",
            fork.check1(
                pre, pre, pre, 0.0 if arm == "control" else 64.0, arm == "injected"
            ),
            results,
        )
    scores = {
        name: np.full(5, val, np.float32)
        for name, val in [("fresh", 1), ("control", 2), ("injected", 0)]
    }
    check2 = dict(
        check_index=fs["check_index"],
        **{f"score_{k}_rounds": x.tolist() for k, x in scores.items()},
        **{f"score_{k}_iqm": iqm(x) for k, x in scores.items()},
        loss_injected_rounds=[1.0] * 5,
        loss_control_rounds=[-1.0] * 5,
        paired_difference_rounds=[-2.0] * 5,
        paired_difference_iqm=-2.0,
        paired_difference_ci_low=-2.0,
        paired_difference_ci_high=-2.0,
        confidence=0.95,
        bootstrap_resamples=10000,
        **{"pass": False},
    )
    curves = {}
    for name, score in scores.items():
        for part in (
            [f"{name}_q{q}" for q in (1, 2)]
            if v.suite(e) == "humanoid_bench"
            else [name]
        ):
            curves[f"{part}_losses"] = np.zeros((5, 1000), np.float32)
            curves[f"{part}_score"] = score
            check2[f"score_{part}_rounds"] = score.tolist()
    (arm_source / "probes").mkdir()
    np.savez_compressed(arm_source / "probes/check2_curves.npz", **curves)
    exp2_ledger.write_json(key, "check2.json", check2, results)


def study(base):
    runtime = {
        s: {
            **{k: "SYNTHETIC" for k in v.RUNTIME_KEYS},
            "simulator_versions": {"synthetic": "1"},
        }
        for s in ("dmc", "myosuite", "humanoid_bench")
    }
    contract = build(str(base / "checkpoints"), COMMIT, "SYNTHETIC", "last", runtime)
    json_write(base / "study.json", contract)
    return contract


def populate(base, contract):
    for a in v.ARCHITECTURES:
        for e in v.ENVIRONMENTS:
            for s in v.SEEDS:
                parent(
                    base,
                    contract,
                    a,
                    e,
                    s,
                    eligible=a == "D4W1024" and e == "dog-run" and s in (1, 2),
                    invalid=e == "dog-trot",
                )
