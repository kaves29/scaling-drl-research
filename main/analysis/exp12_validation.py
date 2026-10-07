"""Fail-closed certification of the approved Exp1/2 population and saved evidence."""

import copy
import hashlib
import io
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from experiments.exp12 import exp2_ledger, ledger, trigger
from utils.run_metadata import _hash, _strip_locations, training_protocol

ARCHITECTURES = ("D2W512", "D4W1024", "D4W1536")
SCALED = ARCHITECTURES[1:]
ENVIRONMENTS = (
    "dog-run",
    "dog-trot",
    "humanoid-run",
    "humanoid-walk",
    "humanoid-stand",
    "swimmer-swimmer15",
    "hopper-hop",
    "myo-key-turn",
    "myo-pen-twirl",
    "myo-pose-hard",
    "myo-reach",
    "h1-run-v0",
    "h1-reach-v0",
)
SEEDS = (1, 2, 3, 4, 5)
RUNTIME_KEYS = (
    "jax",
    "jaxlib",
    "backend_platform_version",
    "jax_cuda12_plugin",
    "jax_cuda12_pjrt",
)


class ValidationError(ValueError):
    def __init__(self, errors, census=()):
        self.errors, self.census = list(errors), list(census)
        super().__init__("Confirmatory analysis refused: " + "; ".join(self.errors))


class ReportingDecisionRequired(ValidationError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run_key(architecture, environment, seed):
    return f"exp1_{architecture}_{environment}_seed{seed}"


def suite(environment):
    return (
        "humanoid_bench"
        if environment.startswith("h1-")
        else "myosuite" if environment.startswith("myo-") else "dmc"
    )


def budget(environment):
    return (
        2_000_000
        if suite(environment) == "humanoid_bench"
        else (
            500_000 if environment in ("hopper-hop", "swimmer-swimmer15") else 1_000_000
        )
    )


def seeded(config, seed):
    if isinstance(config, dict):
        return {k: seed if k == "seed" else seeded(v, seed) for k, v in config.items()}
    if isinstance(config, list):
        return [seeded(v, seed) for v in config]
    return copy.deepcopy(config)


class Snapshot:
    def __init__(self):
        self.files = {}
        self.directories = {}

    def read(self, path):
        path = Path(path).resolve()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(
            str(path) not in self.files or self.files[str(path)] == digest,
            f"input changed: {path}",
        )
        self.files[str(path)] = digest
        return content

    def json(self, path):
        return json.loads(self.read(path))

    def csv(self, path):
        return pd.read_csv(io.BytesIO(self.read(path)))

    def directory(self, path):
        path = Path(path).resolve()
        names = sorted(p.name for p in path.iterdir()) if path.exists() else []
        self.directories[str(path)] = names
        return names

    def unchanged(self):
        for name, expected in self.directories.items():
            path = Path(name)
            actual = sorted(p.name for p in path.iterdir()) if path.exists() else []
            require(actual == expected, f"input directory changed: {name}")
        for name, digest in self.files.items():
            require(
                hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest,
                f"input changed: {name}",
            )


def _source_state(snapshot, source, expected_end):
    done = snapshot.json(source / "DONE")
    require(
        done["interaction_step"] == expected_end, f"{source}: incorrect terminal step"
    )
    state = _state(snapshot, source / "state")
    meta = pickle.loads(snapshot.read(state / "meta.pkl"))
    require(meta["interaction_step"] == expected_end, f"{source}: stale terminal state")
    return meta


def _state(snapshot, root):
    name = snapshot.read(root / "LATEST").decode().strip()
    require(name and Path(name).name == name, f"{root}: invalid LATEST")
    state = root / name
    for name in ("meta.pkl", "buffer.npz", "buffer_meta.pkl", "obs_rms.pkl"):
        require(
            (state / name).is_file() and (state / name).stat().st_size > 0,
            f"{state}: missing {name}",
        )
    require(
        any(p.is_file() for p in (state / "agent_ckpt").rglob("*")),
        f"{state}: missing agent checkpoint",
    )
    return state


def _metadata(snapshot, source, expected, study, env, key, experiment="exp1"):
    expected = dict(expected)
    architecture = expected.pop("_architecture")
    meta = snapshot.json(source / "run_metadata.json")
    require(meta["schema_version"] == 1, f"{key}: unknown metadata schema")
    require(
        _strip_locations(meta["resolved_config"]) == _strip_locations(expected),
        f"{key}: incompatible config",
    )
    require(
        meta["config_hash"] == _hash(_strip_locations(meta["resolved_config"])),
        f"{key}: stale config hash",
    )
    require(
        meta["protocol_hash"] == _hash(meta["protocol"]), f"{key}: stale protocol hash"
    )
    protocol = training_protocol(expected)
    protocol["simulator_versions"] = study["runtime_by_suite"][suite(env)][
        "simulator_versions"
    ]
    require(meta["protocol"] == protocol, f"{key}: incompatible protocol")
    ident = meta["identity"]
    require(
        ident["experiment"] == experiment
        and ident["architecture"] == architecture
        and ident["environment"] == env
        and ident["seed"] == expected["seed"]
        and ident["run_role"] == "confirmatory",
        f"{key}: incompatible metadata identity",
    )
    expected_key = key if experiment == "exp1" else key.replace("exp1_", "exp2_arm_", 1)
    require(ident["run_key"] == expected_key, f"{key}: metadata run-key mismatch")
    require(
        meta["code"]["commit"] == study["code_commit"]
        and meta["code"]["dirty"] is False,
        f"{key}: incompatible source revision",
    )
    runtime = study["runtime_by_suite"][suite(env)]
    require(
        meta["protocol"]["simulator_versions"] == runtime["simulator_versions"],
        f"{key}: simulator mismatch",
    )
    require(meta["launches"], f"{key}: missing launch provenance")
    for launch in meta["launches"]:
        require(
            launch["commit"] == study["code_commit"] and launch["dirty"] is False,
            f"{key}: incompatible resume revision",
        )
        require(
            launch["device"]["platform"] == "gpu"
            and launch["device"]["device_kind"] == study["gpu_model"],
            f"{key}: incompatible GPU model",
        )
        require(
            launch["runtime"]["platform"] == "gpu"
            and launch["runtime"]["matmul_precision"] == "tensorfloat32",
            f"{key}: incompatible precision/backend",
        )
        require(
            all(launch["runtime"][k] == runtime[k] for k in RUNTIME_KEYS),
            f"{key}: runtime mismatch",
        )
        require(
            launch.get("critic_count", 1)
            == (2 if suite(env) == "humanoid_bench" else 1),
            f"{key}: critic count mismatch",
        )
    return meta


def _csv_records(records, columns):
    return pd.read_csv(
        io.StringIO(pd.DataFrame(records).reindex(columns=columns).to_csv(index=False))
    )


def _checks(snapshot, path, meta, n, end, seed, config, architecture, environment):
    frame = snapshot.csv(path)
    expected_indices = list(range(end // (n // 20) + 1))
    require(
        list(frame.check_index) == expected_indices,
        f"{path}: missing/duplicate/wrong check indices",
    )
    steps = [5000] + [k * n // 20 for k in expected_indices[1:]]
    require(list(frame.interaction_step) == steps, f"{path}: incorrect check times")
    for column, value in (
        ("run_key", run_key(architecture, environment, seed)),
        ("experiment", "exp1"),
        ("run_role", "confirmatory"),
        ("architecture", architecture),
        ("environment", environment),
        ("seed", seed),
    ):
        require((frame[column] == value).all(), f"{path}: incorrect check identity")
    require(
        np.array_equal(frame.budget_fraction, frame.interaction_step / n),
        f"{path}: incorrect check fractions",
    )
    original = meta["extra_state"]["probe_records"]
    fields = [
        k
        for k in ledger.CHECK_COLUMNS
        if k
        not in (
            "run_key",
            "experiment",
            "run_role",
            "architecture",
            "environment",
            "seed",
            "budget_fraction",
        )
    ]
    saved = _csv_records(original, fields)
    pd.testing.assert_frame_equal(
        frame[fields].reset_index(drop=True), saved, check_dtype=False, check_exact=True
    )
    invalid = []
    for row in original:
        k = row["check_index"]
        fresh = np.array([row[f"score_fresh_r{r}"] for r in range(5)])
        if suite(environment) == "humanoid_bench":
            for name in ("fresh", "current") if k else ("fresh",):
                parts = np.array(
                    [
                        [row[f"score_{name}_q{q}_r{r}"] for r in range(5)]
                        for q in (1, 2)
                    ],
                    np.float32,
                )
                aggregate = np.array(
                    [row[f"score_{name}_r{r}"] for r in range(5)], np.float32
                )
                require(
                    np.array_equal(aggregate, parts.mean(0), equal_nan=True),
                    f"{path}: inconsistent per-Q mean at {k}",
                )
            if k:
                for q in (1, 2):
                    f = np.array(
                        [row[f"score_fresh_q{q}_r{r}"] for r in range(5)], np.float32
                    )
                    c = np.array(
                        [row[f"score_current_q{q}_r{r}"] for r in range(5)], np.float32
                    )
                    values = np.array([row[f"loss_q{q}_r{r}"] for r in range(5)])
                    require(
                        np.array_equal(values, f - c, equal_nan=True),
                        f"{path}: inconsistent per-Q loss",
                    )
                    expected_iqm = (
                        float(trim_mean(values, 0.25))
                        if np.isfinite(values).all()
                        else np.nan
                    )
                    require(
                        np.array_equal(
                            [row[f"loss_q{q}_iqm"]], [expected_iqm], equal_nan=True
                        ),
                        f"{path}: inconsistent per-Q IQM",
                    )
        if k == 0:
            require(np.isfinite(fresh).all(), f"{path}: invalid initial fresh probe")
            require(
                row["score_fresh_iqm"] == float(trim_mean(fresh, 0.25)),
                f"{path}: stale initial score",
            )
            continue
        loss = np.array([row[f"loss_r{r}"] for r in range(5)])
        current = np.array([row[f"score_current_r{r}"] for r in range(5)])
        valid = bool(
            np.isfinite(loss).all()
            and np.isfinite(fresh).all()
            and np.isfinite(current).all()
        )
        require(row["valid"] == valid, f"{path}: invalid validity marker at {k}")
        if not valid:
            invalid.append(k)
            require(not row["triggered"], f"{path}: invalid check fires at {k}")
        else:
            require(
                np.array_equal(
                    loss, fresh.astype(np.float32) - current.astype(np.float32)
                ),
                f"{path}: inconsistent round losses at {k}",
            )
            lo, hi = trigger.bootstrap_interval(
                loss, seed, k, config["trigger"]["resamples"], 0.95
            )
            require(
                row["ci_low"] == lo and row["ci_high"] == hi,
                f"{path}: inconsistent interval at {k}",
            )
            require(
                row["triggered"] == (lo > 0), f"{path}: inconsistent trigger at {k}"
            )
            for name, values in (
                ("score_current_iqm", current),
                ("score_fresh_iqm", fresh),
                ("loss_iqm", loss),
            ):
                require(
                    row[name] == float(trim_mean(values, 0.25)),
                    f"{path}: inconsistent {name} at {k}",
                )
        require(not row.get("forced", False), f"{path}: forced confirmatory trigger")
        if suite(environment) == "humanoid_bench":
            for q in (1, 2):
                require(
                    all(f"loss_q{q}_r{r}" in row for r in range(5)),
                    f"{path}: missing per-Q measurements",
                )
    final = next(r for r in original if r["check_index"] == 20)
    require(
        final["valid"] and np.isfinite(final["loss_iqm"]),
        f"{path}: invalid final endpoint",
    )
    return frame, original, invalid


def _curves(snapshot, path, checks, twin, records):
    with np.load(io.BytesIO(snapshot.read(path))) as arrays:
        for k in checks:
            names = [f"fresh_q{q}" for q in (1, 2)] if twin else ["fresh"]
            if k:
                names += [f"current_q{q}" for q in (1, 2)] if twin else ["current"]
            for name in names:
                require(
                    arrays[f"check_{k:02d}/{name}_losses"].shape == (5, 1000),
                    f"{path}: wrong curve shape",
                )
                record = next(row for row in records if row["check_index"] == k)
                score_name = "score_" + name
                values = np.array([record[f"{score_name}_r{r}"] for r in range(5)])
                require(
                    np.array_equal(
                        arrays[f"check_{k:02d}/{name}_score"], values, equal_nan=True
                    ),
                    f"{path}: score evidence mismatch",
                )


def _parent(snapshot, directory, architecture, environment, seed, study):
    key = run_key(architecture, environment, seed)
    runs = snapshot.csv(directory / "run.csv")
    require(
        len(runs) == 1 and list(runs.columns) == ledger.RUN_COLUMNS,
        f"{key}: duplicate/invalid run record",
    )
    row = runs.iloc[0]
    require(
        (
            row.run_key,
            row.experiment,
            row.run_role,
            row.architecture,
            row.environment,
            row.seed,
        )
        == (key, "exp1", "confirmatory", architecture, environment, seed),
        f"{key}: conflicting identity",
    )
    n = budget(environment) // 2
    require(
        (row.budget_env_steps, row.num_interaction_steps, row.num_checks, row.status)
        == (2 * n, n, 20, "complete"),
        f"{key}: incomplete/wrong budget",
    )
    require(row.code_commit == study["code_commit"], f"{key}: ledger revision mismatch")
    source = (
        Path(study["checkpoint_root"])
        / "exp1"
        / architecture
        / environment
        / f"seed_{seed}"
    )
    require(
        Path(snapshot.json(directory / "source.json")["run_dir"]).resolve()
        == source.resolve(),
        f"{key}: source path mismatch",
    )
    config = seeded(study["configs"][f"{architecture}/{environment}"], seed)
    expected = {**config, "_architecture": architecture}
    _metadata(snapshot, source, expected, study, environment, key)
    end = (
        n
        if pd.isna(row.fork_interaction_step)
        else max(n, int(row.fork_interaction_step) + n // 4)
    )
    meta = _source_state(snapshot, source, end)
    require(
        any(p.is_file() for p in (source / "fresh_critic").rglob("*")),
        f"{key}: missing fresh reference",
    )
    frame, records, invalid = _checks(
        snapshot,
        directory / "checks.csv",
        meta,
        n,
        end,
        seed,
        config,
        architecture,
        environment,
    )
    require(
        row.initial_fresh_score_iqm == frame.iloc[0].score_fresh_iqm,
        f"{key}: stale initial score",
    )
    final = frame[frame.check_index == 20].iloc[0]
    require(
        np.isfinite(row.final_loss_iqm) and row.final_loss_iqm == final.loss_iqm,
        f"{key}: stale endpoint",
    )
    fstar = trigger.f_star(records, 20, 2, 0.95)
    require(
        (
            fstar is None
            and pd.isna(row.f_star_check)
            and pd.isna(row.f_star_interaction_step)
            and pd.isna(row.f_star_fraction)
        )
        or (
            fstar is not None
            and row.f_star_check == fstar["check_index"]
            and row.f_star_interaction_step == fstar["interaction_step"]
            and row.f_star_fraction == fstar["check_index"] / 20
        ),
        f"{key}: inconsistent eligibility",
    )
    eligible = architecture in SCALED and fstar is not None
    require(
        (eligible and row.fork_interaction_step == fstar["interaction_step"])
        or (not eligible and pd.isna(row.fork_interaction_step)),
        f"{key}: missing/unexpected fork",
    )
    if eligible:
        _control_fork(snapshot, source, directory, key, fstar, study, environment, n)
    _curves(
        snapshot,
        directory / "probe_curves.npz",
        frame.check_index,
        suite(environment) == "humanoid_bench",
        records,
    )
    _metrics(snapshot, directory / "metrics.csv", meta, n, end, key)
    evaluations = pd.DataFrame(meta["eval_rows"])
    expected_evaluations = {
        (step, episode)
        for step in range(0, end + 1, 50_000)
        for episode in range(5 if step == 0 or step + 50_000 <= n else 50)
    }
    require(
        not evaluations.duplicated(["interaction_step", "episode"]).any()
        and set(map(tuple, evaluations[["interaction_step", "episode"]].to_numpy()))
        == expected_evaluations,
        f"{key}: incomplete nominal evaluation evidence",
    )
    require(
        np.isfinite(evaluations["return"]).all(), f"{key}: nonfinite nominal return"
    )
    return (
        row,
        frame,
        {
            "run_key": key,
            "architecture": architecture,
            "environment": environment,
            "seed": seed,
            "status": (
                "eligible_trigger"
                if eligible
                else "valid_no_trigger" if fstar is None else "valid_default_trigger"
            ),
            "invalid_checks": invalid,
            "fork_step": None if not eligible else fstar["interaction_step"],
            "source": source,
            "config": config,
            "nominal_steps": n,
        },
    )


def _control_fork(snapshot, source, directory, key, fstar, study, environment, n):
    from experiments.exp12.fork import check1

    root = source / "fork"
    require(
        snapshot.read(root / "FORK_READY").decode().strip() == key,
        f"{key}: incomplete control fork",
    )
    plan = snapshot.json(root / "fork.json")
    expected = {
        "run_key": key,
        "fork_check_index": fstar["check_index"],
        "fork_step": fstar["interaction_step"],
        "num_interaction_steps": n,
        "horizon_steps": n // 4,
        "eval_every_steps": n // 100,
        "eval_episodes": 10,
        "arm_end_step": fstar["interaction_step"] + n // 4,
        "control_end_step": max(n, fstar["interaction_step"] + n // 4),
    }
    require(
        all(plan.get(k) == value for k, value in expected.items()),
        f"{key}: incompatible control plan",
    )
    require(
        plan["device"]["platform"] == "gpu"
        and plan["device"]["device_kind"] == study["gpu_model"],
        f"{key}: incompatible fork device",
    )
    fork_state = _state(snapshot, root / "state")
    require(
        pickle.loads(snapshot.read(fork_state / "meta.pkl"))["interaction_step"]
        == fstar["interaction_step"],
        f"{key}: stale control fork state",
    )
    with np.load(io.BytesIO(snapshot.read(root / "panel.npz"))) as panel:
        action_shape = panel["action"].shape
        require(
            len(action_shape) == 2
            and action_shape[0] == 256
            and panel["observation"].shape[0] == 256,
            f"{key}: invalid control panel",
        )
    values = []
    for name in ("pre", "control"):
        with np.load(io.BytesIO(snapshot.read(root / f"check1_{name}.npz"))) as data:
            parts = dict(data)
        require(
            set(parts) == {"q", "dq_da"}
            and all(
                v.dtype == np.float32 and np.isfinite(v).all() for v in parts.values()
            )
            and parts["q"].shape
            == (256 * (2 if suite(environment) == "humanoid_bench" else 1),)
            and parts["dq_da"].shape == action_shape,
            f"{key}: invalid control Check 1 values",
        )
        values.append(parts)
    computed = check1(values[0], values[1], values[1], 0.0, False)
    ledger_root = Path(directory).parents[3]
    stored = snapshot.json(
        Path(exp2_ledger.run_root(key, ledger_root)) / "check1_control.json"
    )
    require(
        stored == computed and computed["pass"],
        f"{key}: unresolved original/control Check 1",
    )


def _metrics(snapshot, path, meta, n, end, key, arm=None, fork_step=None):
    frame = snapshot.csv(path)
    rows = pd.DataFrame(meta["metrics_rows"])
    if arm is None:
        rows.insert(0, "run_key", key)
        rows["interaction_step"] = rows.env_step / 2
        rows["budget_fraction"] = rows.interaction_step / n
        expected_steps = np.arange(0, end + 1, 2000) * 2
    else:
        rows = rows[rows.env_step >= fork_step * 2].copy()
        rows.insert(0, "arm", arm)
        rows["steps_since_fork"] = rows.env_step / 2 - fork_step
        expected_steps = np.arange(2000, end + 1, 2000) * 2
        expected_steps = expected_steps[expected_steps >= fork_step * 2]
    require(
        np.array_equal(frame.env_step, expected_steps),
        f"{path}: missing/duplicate logging windows",
    )
    pd.testing.assert_frame_equal(
        frame,
        _csv_records(rows.to_dict("records"), list(frame.columns)),
        check_dtype=False,
        check_exact=True,
    )
    for column in (
        "train/churn",
        "train/policy_kl",
        "train/actor_gnorm",
        "train/actor_gnorm_std",
        "train/actor_action",
        "train/actor_saturation",
    ):
        require(column in frame, f"{path}: missing required diagnostic {column}")


def _evaluations(snapshot, path, arm, fork_step, n, environment):
    frame = snapshot.csv(path)
    keys = ["eval_index", "episode"]
    require(
        len(frame) == 260 and not frame.duplicated(keys).any(),
        f"{path}: missing/extra/duplicate episodes",
    )
    require(
        set(map(tuple, frame[keys].to_numpy()))
        == {(j, e) for j in range(26) for e in range(10)},
        f"{path}: incorrect episode grid",
    )
    require(
        (frame.arm == arm).all()
        and (frame.steps_since_fork == frame.eval_index * (n // 100)).all()
        and (frame.interaction_step == fork_step + frame.steps_since_fork).all(),
        f"{path}: wrong arm/times",
    )
    require(np.isfinite(frame["return"]).all(), f"{path}: nonfinite return")
    limit = (
        25
        if environment == "myo-pen-twirl"
        else 50 if suite(environment) == "myosuite" else 500
    )
    require(
        (frame.length >= 1).all()
        and (frame.length <= limit).all()
        and (frame.length == frame.length.astype(int)).all(),
        f"{path}: impossible episode length",
    )
    return frame


def _fork(snapshot, root, item, study):
    from experiments.exp12.fork import check1

    key, source, n, f = (
        item["run_key"],
        item["source"],
        item["nominal_steps"],
        item["fork_step"],
    )
    directory = Path(exp2_ledger.run_root(key, root))
    expected = {
        "fork_step": f,
        "fork_check_index": f // (n // 20),
        "num_interaction_steps": n,
        "horizon_steps": n // 4,
        "arm_end_step": f + n // 4,
        "control_end_step": max(n, f + n // 4),
        "eval_every_steps": n // 100,
        "eval_episodes": 10,
        "run_key": key,
    }
    plan = snapshot.json(directory / "fork.json")
    require(
        all(plan.get(k) == v for k, v in expected.items()),
        f"{key}: incompatible fork plan",
    )
    stored = snapshot.json(source / "fork/fork.json")
    require(
        all(stored.get(k) == v for k, v in expected.items()),
        f"{key}: source plan mismatch",
    )
    require(
        snapshot.read(source / "fork/FORK_READY").decode().strip() == key,
        f"{key}: incomplete fork",
    )
    fork_state = _state(snapshot, source / "fork/state")
    require(
        pickle.loads(snapshot.read(fork_state / "meta.pkl"))["interaction_step"] == f,
        f"{key}: incorrect fork state",
    )

    def arrays(path):
        with np.load(io.BytesIO(snapshot.read(path))) as data:
            return dict(data)

    panel = arrays(source / "fork/panel.npz")
    pre = arrays(source / "fork/check1_pre.npz")
    control = arrays(source / "fork/check1_control.npz")
    q_count = 2 if suite(item["environment"]) == "humanoid_bench" else 1
    require(
        panel["action"].shape[0] == 256
        and pre["q"].shape == (256 * q_count,)
        and pre["dq_da"].shape == panel["action"].shape,
        f"{key}: invalid panel shape",
    )
    after_path = (
        Path(study["checkpoint_root"])
        / "exp2_arm"
        / item["architecture"]
        / item["environment"]
        / f"seed_{item['seed']}"
        / "injected/check1_after.npz"
    )
    after = arrays(after_path)
    for values in (pre, control, after):
        require(
            set(values) == {"q", "dq_da"}
            and all(
                value.dtype == np.float32
                and np.isfinite(value).all()
                and value.shape == pre[field].shape
                for field, value in values.items()
            ),
            f"{key}: invalid panel values",
        )
    for name in ("check1_control.json", "check1_injected.json"):
        check = snapshot.json(directory / name)
        require(
            check["pass"] is True and check["matmul_precision"] == "highest",
            f"{key}: unresolved {name}",
        )
        require(
            all(pair["pass"] is True for pair in check["pairs"].values()),
            f"{key}: failed Check 1 pair",
        )
        computed = check1(
            pre,
            control if name == "check1_control.json" else after,
            control,
            0.0 if name == "check1_control.json" else 64.0,
            name == "check1_injected.json",
        )
        require(check == computed and computed["pass"], f"{key}: stale/failed {name}")
    check2 = snapshot.json(directory / "check2.json")
    _check2(
        snapshot,
        check2,
        after_path.parent / "probes/check2_curves.npz",
        expected["fork_check_index"],
        item["seed"],
        item["config"],
        q_count == 2,
        key,
    )
    for arm in ("control", "injected"):
        data = directory / f"arm_{arm}"
        arm_source = (
            source
            if arm == "control"
            else Path(study["checkpoint_root"])
            / "exp2_arm"
            / item["architecture"]
            / item["environment"]
            / f"seed_{item['seed']}"
            / "injected"
        )
        require(
            Path(snapshot.json(data / "source.json")["run_dir"]).resolve()
            == arm_source.resolve(),
            f"{key}: arm source mismatch",
        )
        end = expected["control_end_step" if arm == "control" else "arm_end_step"]
        state = _source_state(snapshot, arm_source, end)
        if arm == "injected":
            cfg = copy.deepcopy(item["config"])
            cfg["fork"].update(source=str(source), arm="injected")
            cfg["injection"]["m"] = study["injection_m"]
            _metadata(
                snapshot,
                arm_source,
                {**cfg, "_architecture": item["architecture"]},
                study,
                item["environment"],
                key,
                "exp2_arm",
            )
            snapshot.read(arm_source / "check1_after.npz")
        evals = _evaluations(
            snapshot, data / "eval_episodes.csv", arm, f, n, item["environment"]
        )
        pd.testing.assert_frame_equal(
            evals,
            _csv_records(state["extra_state"]["post_fork_evals"], list(evals.columns)),
            check_dtype=False,
            check_exact=True,
        )
        checks = snapshot.csv(data / "checks.csv")
        require(
            list(checks.check_index)
            == list(range(expected["fork_check_index"], end // (n // 20) + 1)),
            f"{key}: incomplete arm probes",
        )
        require(
            (checks.interaction_step == checks.check_index * (n // 20)).all()
            and (checks.steps_since_fork == checks.interaction_step - f).all(),
            f"{key}: wrong arm probe times",
        )
        records = [
            r
            for r in state["extra_state"]["probe_records"]
            if r["check_index"] >= expected["fork_check_index"]
        ]
        canonical = [
            {**r, "arm": arm, "steps_since_fork": r["interaction_step"] - f}
            for r in records
        ]
        pd.testing.assert_frame_equal(
            checks,
            _csv_records(canonical, list(checks.columns)),
            check_dtype=False,
            check_exact=True,
        )
        _metrics(snapshot, data / "metrics.csv", state, n, end, key, arm, f)
        require(
            len(state["extra_state"]["post_fork_evals"]) == 260,
            f"{key}: incomplete terminal evaluation evidence",
        )

    return check2


def _check2(snapshot, result, path, k, seed, config, twin, key):
    require(
        result["check_index"] == k and isinstance(result["pass"], bool),
        f"{key}: incomplete Check 2",
    )
    with np.load(io.BytesIO(snapshot.read(path))) as data:
        scores = {}
        for name in ("control", "injected", "fresh"):
            parts = [f"{name}_q{q}" for q in (1, 2)] if twin else [name]
            for part in parts:
                require(
                    data[f"{part}_losses"].shape == (5, 1000),
                    f"{key}: incomplete Check 2 curve",
                )
                require(
                    data[f"{part}_score"].shape == (5,),
                    f"{key}: incomplete Check 2 rounds",
                )
                require(
                    np.array_equal(
                        data[f"{part}_score"],
                        result[f"score_{part}_rounds"],
                        equal_nan=True,
                    ),
                    f"{key}: inconsistent Check 2 scores",
                )
            scores[name] = (
                np.mean([data[f"{part}_score"] for part in parts], axis=0)
                if twin
                else data[f"{name}_score"]
            )
            require(
                np.array_equal(
                    scores[name], result[f"score_{name}_rounds"], equal_nan=True
                ),
                f"{key}: Check 2 mean mismatch",
            )
            require(
                np.isclose(
                    result[f"score_{name}_iqm"],
                    trim_mean(scores[name].astype(np.float64), 0.25),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                ),
                f"{key}: stale Check 2 IQM",
            )
        diff = scores["injected"] - scores["control"]
        for field, values in (
            ("paired_difference_rounds", diff),
            ("loss_injected_rounds", scores["fresh"] - scores["injected"]),
            ("loss_control_rounds", scores["fresh"] - scores["control"]),
        ):
            require(
                np.array_equal(result[field], values, equal_nan=True),
                f"{key}: inconsistent Check 2 {field}",
            )
        require(
            np.isclose(
                result["paired_difference_iqm"],
                trim_mean(diff.astype(np.float64), 0.25),
                rtol=0,
                atol=0,
                equal_nan=True,
            ),
            f"{key}: stale Check 2 difference",
        )
        require(
            result["confidence"] == 0.95
            and result["bootstrap_resamples"] == config["trigger"]["resamples"],
            f"{key}: incompatible Check 2 protocol",
        )
        low, high = trigger.bootstrap_interval(
            diff, seed, k, result["bootstrap_resamples"], 0.95
        )
        require(
            np.array_equal(
                [low, high],
                [
                    result["paired_difference_ci_low"],
                    result["paired_difference_ci_high"],
                ],
                equal_nan=True,
            ),
            f"{key}: stale Check 2 interval",
        )
        require(
            result["pass"] == bool(np.isfinite(low) and low > 0),
            f"{key}: stale Check 2 marker",
        )


def validate(results_root, study_manifest, experiment):
    snapshot, errors, census = Snapshot(), [], []
    try:
        require(
            study_manifest is not None,
            "--study-manifest is required; use --exploratory for progress data",
        )
        study = snapshot.json(study_manifest)
        require(
            study["schema_version"] == 1
            and study["injection_m"] in ("last", "half", "all"),
            "invalid frozen study manifest",
        )
        require(
            Path(study["checkpoint_root"]).is_absolute()
            and len(study["code_commit"]) == 40
            and study["gpu_model"],
            "invalid study provenance",
        )
        require(
            set(study["configs"])
            == {f"{a}/{e}" for a in ARCHITECTURES for e in ENVIRONMENTS},
            "incorrect study config population",
        )
        from scripts.freeze_exp12_analysis_manifest import build

        canonical = build(
            study["checkpoint_root"],
            study["code_commit"],
            study["gpu_model"],
            study["injection_m"],
            study["runtime_by_suite"],
        )
        require(
            _strip_locations(study["configs"])
            == _strip_locations(canonical["configs"]),
            "frozen settings differ from the approved study configurations",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValidationError([str(exc)]) from exc
    architectures = ARCHITECTURES if experiment == "exp1" else SCALED
    expected = {
        run_key(a, e, s) for a in architectures for e in ENVIRONMENTS for s in SEEDS
    }
    observed = set()
    root = ledger.ledger_root(results_root)
    snapshot.directory(root)
    for directory in sorted(root.glob("*")):
        if not directory.is_dir():
            continue
        try:
            rows = snapshot.csv(directory / "run.csv")
            if (rows.run_role == "dev").all():
                continue
            if experiment == "exp2" and (rows.architecture == ARCHITECTURES[0]).all():
                continue
            require(
                directory.name in expected,
                f"unexpected confirmatory directory {directory.name}",
            )
            observed.add(directory.name)
        except (OSError, ValueError, KeyError, AttributeError) as exc:
            errors.append(str(exc))
    if expected != observed:
        errors.append(
            f"population mismatch: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    for a in architectures:
        for e in ENVIRONMENTS:
            for s in SEEDS:
                key = run_key(a, e, s)
                try:
                    _, _, item = _parent(snapshot, root / key, a, e, s, study)
                    if experiment == "exp2" and item["status"] == "eligible_trigger":
                        item["check2_pass"] = _fork(
                            snapshot, results_root, item, study
                        )["pass"]
                    census.append(
                        {k: v for k, v in item.items() if k not in ("source", "config")}
                    )
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    AttributeError,
                    AssertionError,
                    pickle.UnpicklingError,
                ) as exc:
                    errors.append(f"{key}: {exc}")
                    census.append(
                        {
                            "run_key": key,
                            "architecture": a,
                            "environment": e,
                            "seed": s,
                            "status": "invalid_incomplete",
                            "error": str(exc),
                        }
                    )
    if experiment == "exp2":
        eligible = {r["run_key"] for r in census if r["status"] == "eligible_trigger"}
        exp2_root = Path(exp2_ledger.run_root("x", results_root)).parent
        snapshot.directory(exp2_root)
        for directory in exp2_root.glob("*"):
            if directory.is_dir() and directory.name not in eligible:
                try:
                    parent = snapshot.csv(root / directory.name / "run.csv")
                    if (parent.run_role == "dev").all():
                        continue
                except (OSError, ValueError, AttributeError):
                    pass
                errors.append(f"unexpected/unresolved fork directory {directory.name}")
    if errors:
        raise ValidationError(errors, census)
    snapshot.unchanged()
    return snapshot, {
        "pass": True,
        "validated_confirmatory": True,
        "mode": "confirmatory",
        "experiment": experiment,
        "study_manifest_sha256": _hash(study),
        "census": census,
        "input_files": snapshot.files,
        "input_directories": snapshot.directories,
    }


def publish(
    out_dir, writer, results_root, experiment, study_manifest=None, exploratory=False
):
    out = Path(out_dir).resolve()
    require(
        not out.exists() or not any(out.iterdir()),
        f"refusing to overwrite existing analysis artifacts: {out}",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        if exploratory:
            snapshot, certification = None, {
                "validated_confirmatory": False,
                "mode": "exploratory",
            }
        else:
            snapshot, certification = validate(results_root, study_manifest, experiment)
        outputs = writer(stage, certification)
        if snapshot is not None:
            snapshot.unchanged()
        certification["output_files"] = {
            str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in stage.rglob("*")
            if p.is_file()
        }
        (stage / ("EXPLORATORY.json" if exploratory else "validation.json")).write_text(
            json.dumps(certification, indent=2)
        )
        if out.exists():
            out.rmdir()
        os.replace(stage, out)
        return outputs
    except ValidationError as exc:
        out.with_name(out.name + ".validation_failed.json").write_text(
            json.dumps(
                {"pass": False, "errors": exc.errors, "census": exc.census}, indent=2
            )
        )
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
