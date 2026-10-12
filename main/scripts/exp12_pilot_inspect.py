#!/usr/bin/env python3
"""READ-ONLY inspection of a completed Exp1 development (pilot) run. Lightweight CPU only.

Dependencies: Python >= 3.10, numpy, scipy, omegaconf, hydra-core, and git (for the
compatibility certificate). It never imports JAX, Flax, Optax, Orbax, MuJoCo,
simulators, W&B, pandas or the `experiments` package (whose __init__ registers every
training entry point).

Production code reused byte-for-byte, loaded by file path under private names:
  - experiments/exp12/trigger.py   (bootstrap interval, triggered, f*_run, eligibility)
  - scripts/exp12_source_compatibility.py
  - utils/run_metadata.py          (protocol projection used by the positive control)
Small production helpers re-stated here (equivalence-tested against production in
tests/test_exp12_pilot_inspect.py): iqm, check_steps, fork_plan, latest_state_dir,
the architecture id and the base_exp12 Hydra composition. meta.pkl is read with a
restricted unpickler that rebuilds pickled JAX arrays as the identical NumPy values.

Answers, with evidence paths and SHA-256 digests:
  1. did the run complete its training horizon (DONE, LATEST state, all checks)?
  2. did a natural plasticity-loss trigger occur (recomputed from the saved rounds)?
  3. if so: trigger check, statistics, eligibility;
  4. does a complete, natural Exp2 fork exist (the positive control's own validation);
  5. optional: science-source compatibility with --final-sha (git certificate);
  6. whether the run meets the positive control's input requirements.
It never writes inside --root, never runs training or probes, never forces or
manufactures a trigger, and never declares the positive control passed.
"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import trim_mean

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))  # only for the dependency-free `utils` package

SMALL = 64 * 1024 * 1024  # hash evidence files up to 64 MiB; larger ones are sized only
STATE_NAME = re.compile(r"step_[0-9]{9}_[0-9a-f]{8}")
FORK_LOG_PATTERNS = {
    "fork_state_saved": re.compile(r"\[exp1\] fork state saved at interaction_step (\d+)"),
    "restored_after_fork": re.compile(r"\[exp1\] restored interaction_step (\d+)"),
    "post_fork_evaluation": re.compile(r"post-fork evaluation \d+"),
}


def _load_file(name, relative):
    """Load one production module by path, without importing its package."""
    spec = importlib.util.spec_from_file_location(f"_exp12_inspect_{name}", MAIN / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trigger = _load_file("trigger", "experiments/exp12/trigger.py")
compatibility = _load_file("compat", "scripts/exp12_source_compatibility.py")


# --- Re-stated production helpers (equivalence-tested) ----------------------------
def iqm(values):
    """experiments/exp12/probe.py: iqm."""
    return float(trim_mean(np.asarray(values, dtype=np.float64), proportiontocut=0.25, axis=None))


def check_steps(num_interaction_steps, checks):
    """experiments/exp12/probe.py: check_steps."""
    if num_interaction_steps % checks:
        raise ValueError(f"{num_interaction_steps} interaction steps are not divisible into {checks} checks.")
    return [k * num_interaction_steps // checks for k in range(1, checks + 1)]


def fork_plan(cfg, fork_step, check_index):
    """experiments/exp12/fork.py: fork_plan."""
    n = int(cfg.num_interaction_steps)
    horizon = int(round(float(cfg.fork.horizon_fraction) * n))
    every = int(round(float(cfg.fork.eval_every_fraction) * n))
    if horizon % every:
        raise ValueError("fork.horizon_fraction must be a multiple of fork.eval_every_fraction")
    arm_end = fork_step + horizon
    if arm_end > float(cfg.fork.max_total_fraction) * n + 1e-9:
        raise ValueError(f"fork at {fork_step} leaves no full {horizon}-step horizon within the maximum budget")
    return {"fork_step": int(fork_step), "fork_check_index": int(check_index), "num_interaction_steps": n,
            "horizon_steps": horizon, "arm_end_step": arm_end, "control_end_step": max(n, arm_end),
            "eval_every_steps": every, "eval_episodes": int(cfg.fork.eval_episodes)}


def latest_state_dir(root):
    """experiments/exp12/state.py: latest_state_dir."""
    pointer = Path(root) / "LATEST"
    if not pointer.exists():
        return None
    name = pointer.read_text().strip()
    path = Path(root) / name
    if not name or Path(name).name != name or name in (".", "..") or not path.is_dir():
        raise ValueError(f"{pointer}: invalid or missing state directory")
    if path.resolve().parent != Path(root).resolve():
        raise ValueError(f"{pointer}: state directory escapes root")
    return path


_CRITIC_SIZE_MAP = {(128, 1): "Small", (1024, 3): "XXL"}


def architecture_id(cfg):
    """scale_rl/common/logger.py: get_architecture_id."""
    key = (cfg.agent.critic_hidden_dim, cfg.agent.critic_num_blocks)
    return _CRITIC_SIZE_MAP.get(key, f"D{cfg.agent.critic_num_blocks}W{cfg.agent.critic_hidden_dim}")


def compose_config(overrides):
    """experiments/exp1.py: compose_config (base_exp12)."""
    import hydra
    import omegaconf
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(version_base=None, config_dir=str(MAIN / "configs"))
    cfg = hydra.compose(config_name="base_exp12", overrides=list(overrides))
    omegaconf.OmegaConf.register_new_resolver("eval", eval, replace=True)
    omegaconf.OmegaConf.resolve(cfg)
    return cfg


# --- Restricted, read-only meta.pkl reader ----------------------------------------
def _numpy_from_jax(fun, args, arr_state, aval_state):
    """jax._src.array._reconstruct_array without device_put: the identical NumPy value."""
    value = fun(*args)
    value.__setstate__(arr_state)
    return value


class _Opaque:
    """Placeholder for a class the inspector does not need (never used in a verdict)."""

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class _MetaUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("numpy", "dtype"), ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"), ("numpy._core.numeric", "_frombuffer"),
        ("collections", "OrderedDict"), ("builtins", "set"), ("builtins", "frozenset"),
        ("builtins", "complex"), ("builtins", "bytearray"), ("builtins", "slice"),
    }

    def __init__(self, stream):
        super().__init__(stream)
        self.opaque = set()

    def find_class(self, module, name):
        if (module, name) == ("jax._src.array", "_reconstruct_array"):
            return _numpy_from_jax
        if (module, name) in self.ALLOWED:
            if module.startswith("numpy.core.") and int(np.__version__.split(".")[0]) >= 2:
                module = "numpy._core." + module[len("numpy.core."):]  # same objects, no warning
            return super().find_class(module, name)
        self.opaque.add(f"{module}.{name}")
        return _Opaque


def read_meta(state_dir):
    with open(Path(state_dir) / "meta.pkl", "rb") as stream:
        reader = _MetaUnpickler(stream)
        meta = reader.load()
    return meta, sorted(reader.opaque)


def _plain(value):
    if isinstance(value, _Opaque):
        raise ValueError("required checkpoint field uses an unsupported pickled class")
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# --- Natural-fork validation (scripts/positive_control.py: _validate_natural_fork) --
def validate_natural_fork(cfg, plan, meta):
    records = meta["extra_state"]["probe_records"]
    k = plan["fork_check_index"]
    if type(k) is not int or not 1 <= k <= int(cfg.probe.checks):
        raise ValueError("positive control: invalid fork check index")
    if sorted(r["check_index"] for r in records) != list(range(k + 1)):
        raise ValueError("positive control: missing or duplicate fork probe records")
    tc = trigger.trigger_config(cfg)
    schedule = check_steps(int(cfg.num_interaction_steps), int(cfg.probe.checks))
    for row in records:
        index = row["check_index"]
        if index == 0:
            continue
        if row.get("forced") or row["interaction_step"] != schedule[index - 1]:
            raise ValueError("positive control: forced or off-schedule trigger evidence")
        n = int(cfg.probe.rounds)
        current = np.asarray([row[f"score_current_r{r}"] for r in range(n)], np.float32)
        fresh = np.asarray([row[f"score_fresh_r{r}"] for r in range(n)], np.float32)
        loss = fresh - current
        recorded = np.asarray([row[f"loss_r{r}"] for r in range(n)], np.float32)
        if not np.array_equal(loss, recorded, equal_nan=True):
            raise ValueError("positive control: saved paired losses differ")
        valid = bool(np.isfinite(loss).all())
        low, high = trigger.bootstrap_interval(loss, int(cfg.seed), index, tc.resamples, tc.confidence)
        expected = {"valid": valid, "loss_iqm": iqm(loss) if valid else float("nan"),
                    "ci_low": low, "ci_high": high,
                    "triggered": trigger.triggered(low, tc.null_threshold) and valid}
        for field, value in expected.items():
            if field not in row or not np.array_equal(row[field], value, equal_nan=True):
                raise ValueError(f"positive control: inconsistent saved {field}")
    natural = trigger.f_star(records, int(cfg.probe.checks), tc.consecutive_checks, tc.eligible_fraction)
    if natural != {"check_index": k, "interaction_step": plan["fork_step"]}:
        raise ValueError("positive control: fork is not the first eligible natural trigger")
    if meta["interaction_step"] != plan["fork_step"] or meta["extra_state"].get("f_star") != natural:
        raise ValueError("positive control: state and natural fork disagree")
    expected_plan = fork_plan(cfg, natural["interaction_step"], k)
    if any(plan.get(field) != value for field, value in expected_plan.items()):
        raise ValueError("positive control: fork horizons disagree with configuration")


# --- Positive-control input settings (scripts/positive_control.py: _setting_errors) --
def setting_errors(cfg):
    import omegaconf
    from utils.run_metadata import _strip_locations, differing_keys, training_protocol

    errors = []
    if cfg.run_role != "dev":
        errors.append(f"run_role is {cfg.run_role!r}, not 'dev'")
    if 1 <= int(cfg.seed) <= 5:
        errors.append(f"seed {cfg.seed} is a main-run seed (1-5)")
    pc = cfg.positive_control
    if architecture_id(cfg) != pc.architecture:
        errors.append(f"architecture {architecture_id(cfg)} is not {pc.architecture}")
    if cfg.env_name != pc.env_name:
        errors.append(f"environment {cfg.env_name} is not {pc.env_name}")
    if cfg.testing.force_trigger_check is not None:
        errors.append("forced triggers cannot qualify a positive control")
    canonical = compose_config(["env_name=dog-run", "env=dmc_hard", "critic_num_blocks=4",
                                "critic_hidden_dim=1536", "run_role=dev", f"seed={cfg.seed}"])
    actual = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    approved = omegaconf.OmegaConf.to_container(canonical, resolve=True)

    def protocol(config):
        return {**training_protocol(_strip_locations(config)),
                **{k: config[k] for k in ("probe", "trigger", "positive_control", "testing", "diagnostics")}}

    differences = differing_keys(protocol(approved), protocol(actual))
    if differences:
        errors.append(f"noncanonical positive-control settings: {sorted(differences)}")
    return errors


# --- Evidence ----------------------------------------------------------------------
def _evidence(path, ledger):
    path = Path(path)
    if not path.exists():
        return None
    entry = {"path": str(path), "bytes": path.stat().st_size}
    if path.is_file() and path.stat().st_size <= SMALL:
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    ledger.append(entry)
    return entry


def _run_dirs(root):
    found = []
    for meta in sorted(Path(root).rglob("run_metadata.json")):
        try:
            data = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        if data.get("identity", {}).get("experiment") == "exp1":
            found.append(meta.parent)
    return found


def _float(text):
    return float("nan") if text in ("", None) else float(text)


def _bool(text):
    return str(text).strip().lower() in ("true", "1", "1.0")


def _check_records(run, cfg):
    """probe_checks.csv rows, each re-derived from its saved rounds with production trigger code."""
    tc = trigger.trigger_config(cfg)
    rounds = int(cfg.probe.rounds)
    with open(run / "probes" / "probe_checks.csv", newline="") as stream:
        table = list(csv.DictReader(stream))
    rows, problems = [], []
    for record in table:
        k = int(float(record["check_index"]))
        step = int(float(record["interaction_step"]))
        if k == 0:
            rows.append({"check_index": 0, "interaction_step": step,
                         "score_fresh_iqm": _float(record.get("score_fresh_iqm"))})
            continue
        single = [record.get(f"loss_r{r}") for r in range(rounds)]
        if all(v not in ("", None) for v in single):
            loss = np.array([float(v) for v in single], dtype=np.float32)
        else:  # twin critics: paired loss is the mean of the two networks (amendment (z))
            loss = np.mean([[float(record[f"loss_q{q}_r{r}"]) for r in range(rounds)] for q in (1, 2)],
                           axis=0).astype(np.float32)
        low, high = trigger.bootstrap_interval(loss, int(cfg.seed), k, tc.resamples, tc.confidence)
        valid = bool(np.isfinite(loss).all())
        fired = bool(trigger.triggered(low, tc.null_threshold) and valid)
        forced = _bool(record.get("forced", ""))
        recorded_low = _float(record.get("ci_low"))
        rows.append({
            "check_index": k,
            "interaction_step": step,
            "loss_rounds": loss.tolist(),
            "loss_iqm": iqm(loss) if valid else None,
            "ci_low": low,
            "ci_high": high,
            "fired_recomputed": fired,
            "triggered_recorded": _bool(record["triggered"]),
            "forced": forced,
            "score_current_iqm": _float(record.get("score_current_iqm")),
            "score_fresh_iqm": _float(record.get("score_fresh_iqm")),
        })
        if not forced and fired != _bool(record["triggered"]):
            problems.append(f"check {k}: recorded trigger differs from recomputation")
        if not forced and not (math.isclose(low, recorded_low, rel_tol=1e-12, abs_tol=1e-15)
                               or (math.isnan(low) and math.isnan(recorded_low))):
            problems.append(f"check {k}: recorded ci_low differs from recomputation")
    return rows, problems, tc


def inspect(root, final_sha=None, extra_logs=(), compat_out=None):
    from omegaconf import OmegaConf

    root = Path(root).resolve()
    evidence = []
    report = {"root": str(root), "evidence": evidence, "problems": [], "unsupported_pickled_classes": []}
    runs = _run_dirs(root)
    report["exp1_run_dirs"] = [str(r) for r in runs]
    if len(runs) != 1:
        report["verdict"] = "INCONCLUSIVE"
        report["problems"].append(f"expected exactly one exp1 run directory, found {len(runs)}")
        report["next_step"] = "Point --root at exactly one exp1 run."
        return report
    run = runs[0]
    meta_file = _evidence(run / "run_metadata.json", evidence)
    metadata = json.loads(Path(meta_file["path"]).read_text())
    cfg = OmegaConf.create(metadata["resolved_config"])
    identity = metadata["identity"]
    report.update(identity=identity, source_code=metadata.get("code"),
                  launches=metadata.get("launches"), config_hash=metadata.get("config_hash"))

    # 1. Horizon.
    n = int(cfg.num_interaction_steps)
    done = _evidence(run / "DONE", evidence)
    done_body = json.loads((run / "DONE").read_text()) if done else None
    latest = latest_state_dir(run / "state")
    latest_meta = None
    if latest is not None:
        _evidence(run / "state" / "LATEST", evidence)
        _evidence(latest / "meta.pkl", evidence)
        raw, opaque = read_meta(latest)
        report["unsupported_pickled_classes"] += opaque
        latest_meta = {"interaction_step": _plain(raw["interaction_step"]),
                       "extra_state": _plain(raw["extra_state"])}
    records, problems, tc = _check_records(run, cfg)
    _evidence(run / "probes" / "probe_checks.csv", evidence)
    for k in range(int(cfg.probe.checks) + 1):
        _evidence(run / "probes" / f"check_{k:02d}.npz", evidence)
    report["problems"] += problems
    checks_seen = sorted(r["check_index"] for r in records)
    report["horizon"] = {
        "num_interaction_steps": n,
        "done_marker": done_body,
        "latest_state": str(latest) if latest else None,
        "latest_interaction_step": latest_meta.get("interaction_step") if latest_meta else None,
        "checks_present": checks_seen,
        "complete": bool(
            done_body
            and latest_meta
            and checks_seen == list(range(int(cfg.probe.checks) + 1))
            and latest_meta.get("interaction_step") == done_body.get("interaction_step")
            and done_body.get("interaction_step", -1) >= n
        ),
    }

    # 2-3. Trigger: production f*_run over the recomputed natural records.
    natural = [{"check_index": r["check_index"], "interaction_step": r["interaction_step"],
                "triggered": r.get("fired_recomputed", False)} for r in records]
    star = trigger.f_star(natural, int(cfg.probe.checks), tc.consecutive_checks, tc.eligible_fraction)
    report["trigger"] = {
        "rule": {"null_threshold": tc.null_threshold, "consecutive_checks": tc.consecutive_checks,
                 "eligible_fraction": tc.eligible_fraction,
                 "eligible_checks": f"1..{trigger.last_eligible_check(int(cfg.probe.checks), tc.eligible_fraction)}",
                 "resamples": tc.resamples, "confidence": tc.confidence},
        "fired_checks": [r["check_index"] for r in records if r.get("fired_recomputed")],
        "forced_checks": [r["check_index"] for r in records if r.get("forced")],
        "recorded_f_star": (latest_meta or {}).get("extra_state", {}).get("f_star"),
        "recomputed_f_star": star,
        "checks": records,
    }
    results_root = cfg.get("results_root")
    if results_root:
        for name in ("run.csv", "checks.csv"):
            for path in Path(results_root).glob(f"exp12/exp1/runs/*/{name}"):
                _evidence(path, evidence)

    # 4. Fork artifacts and natural-fork validation.
    fork = run / "fork"
    fork_state = latest_state_dir(fork / "state") if (fork / "state").is_dir() else None
    report["fork"] = {
        "fork_dir_exists": fork.is_dir(),
        "fork_json": _evidence(fork / "fork.json", evidence),
        "fork_ready": _evidence(fork / "FORK_READY", evidence),
        "fork_state": str(fork_state) if fork_state else None,
    }
    plan = None
    if report["fork"]["fork_json"]:
        plan = json.loads((fork / "fork.json").read_text())
        report["fork"]["plan"] = plan
    natural_fork_valid = None
    if report["fork"]["fork_ready"] and fork_state is not None and plan is not None:
        _evidence(fork_state / "meta.pkl", evidence)
        try:
            raw, opaque = read_meta(fork_state)
            report["unsupported_pickled_classes"] += opaque
            meta = {"interaction_step": _plain(raw["interaction_step"]),
                    "extra_state": _plain(raw["extra_state"])}
            # experiments/exp12/fork.py: read_fork drops FORK_JSON_INFO.
            fork_fields = {k: v for k, v in plan.items() if k not in ("run_dir", "fresh_critic_dir", "device")}
            validate_natural_fork(cfg, fork_fields, meta)
            natural_fork_valid = True
        except Exception as exc:  # report, never repair
            natural_fork_valid = False
            report["problems"].append(f"natural-fork validation: {type(exc).__name__}: {exc}")
    report["fork"]["natural_fork_valid"] = natural_fork_valid

    # Logs: corroboration only (the recomputed ledger is the evidence).
    logs = {k: [] for k in FORK_LOG_PATTERNS}
    log_files = [p for p in root.rglob("*") if p.suffix in (".log", ".out", ".err") and p.is_file()]
    log_files += [Path(p) for p in extra_logs]
    for path in log_files:
        text = path.read_text(errors="replace")
        for name, pattern in FORK_LOG_PATTERNS.items():
            for match in pattern.finditer(text):
                logs[name].append({"file": str(path), "match": match.group(0)})
    report["log_corroboration"] = {"files_scanned": len(log_files), "matches": logs}

    # 6. Positive-control input requirements.
    report["positive_control_inputs"] = {
        "setting_errors": setting_errors(cfg),
        "run_role": identity.get("run_role"),
        "seed": identity.get("seed"),
        "architecture": identity.get("architecture"),
        "environment": identity.get("environment"),
    }

    # 5. Source compatibility (git only).
    if final_sha:
        cert = compatibility.certify(metadata["code"]["commit"], final_sha, ["main/experiments/exp2_arm.py"])
        report["source_compatibility"] = {k: cert[k] for k in ("verdict", "source_commit", "target_commit",
                                                               "differing", "unexplained_differences")}
        if compat_out:
            with Path(compat_out).open("x") as stream:
                json.dump(cert, stream, indent=2, sort_keys=True)
                stream.write("\n")
            report["source_compatibility"]["certificate"] = str(compat_out)
            report["source_compatibility"]["certificate_sha256"] = hashlib.sha256(
                Path(compat_out).read_bytes()).hexdigest()

    # Verdict.
    if not report["horizon"]["complete"]:
        verdict = "INCOMPLETE_RUN"
    elif report["trigger"]["forced_checks"] or cfg.testing.force_trigger_check is not None:
        verdict = "FORCED_FORK_NOT_ELIGIBLE"
    elif report["problems"]:
        verdict = "INCONSISTENT_EVIDENCE"
    elif star is None and not report["fork"]["fork_ready"]:
        verdict = "NO_NATURAL_TRIGGER"
    elif star is not None and natural_fork_valid:
        verdict = "NATURAL_FORK_AVAILABLE"
    else:
        verdict = "INCONSISTENT_EVIDENCE"
    report["verdict"] = verdict
    report["next_step"] = {
        "NO_NATURAL_TRIGGER": "Methodology (e): 'If it never triggers, stop and consult. The trigger is never "
        "loosened.' No positive control can run on this pilot; do not force a fork.",
        "NATURAL_FORK_AVAILABLE": "Eligible input for scripts/positive_control.py (run it; its exit status "
        "decides). Owner decision D4 governs reuse across sources.",
        "INCOMPLETE_RUN": "The pilot did not reach its horizon; inspect before any reuse.",
        "FORCED_FORK_NOT_ELIGIBLE": "Forced trigger evidence can never qualify a positive control.",
        "INCONSISTENT_EVIDENCE": "Recorded and recomputed evidence disagree; preserve and review manually.",
    }[verdict]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="pilot root (read only)")
    parser.add_argument("--out", required=True, help="new report path OUTSIDE --root")
    parser.add_argument("--final-sha", help="campaign commit for the compatibility certificate")
    parser.add_argument("--compat-out", help="new path for the full certificate (outside --root)")
    parser.add_argument("--log", action="append", default=[], help="extra Slurm .out/.err files to scan")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    for target in (args.out, args.compat_out):
        if target and Path(target).resolve().is_relative_to(root):
            raise SystemExit("outputs must be outside the pilot root (read-only inspection)")
    report = inspect(root, args.final_sha, args.log, args.compat_out)
    with Path(args.out).open("x") as stream:
        json.dump(report, stream, indent=2, default=str)
        stream.write("\n")
    print(json.dumps({k: report.get(k) for k in ("verdict", "next_step")}, indent=2))
    summary = report.get("trigger", {})
    print(json.dumps({"fired_checks": summary.get("fired_checks"),
                      "recomputed_f_star": summary.get("recomputed_f_star"),
                      "horizon_complete": report.get("horizon", {}).get("complete"),
                      "source_compatibility": report.get("source_compatibility", {}).get("verdict")},
                     default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
