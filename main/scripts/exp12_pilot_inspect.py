#!/usr/bin/env python3
"""READ-ONLY inspection of a completed Exp1 development (pilot) run. CPU only.

Answers, with evidence paths and SHA-256 digests:
  1. did the run complete its training horizon (DONE, LATEST state, all checks)?
  2. did a natural plasticity-loss trigger occur (recomputed from the saved rounds
     with the production bootstrap and f*_run rule, not read from a log)?
  3. if so: trigger check, statistics, eligibility;
  4. does a complete Exp2 fork checkpoint exist (fork.json, FORK_READY, state);
  5. optional: science-source compatibility with --final-sha (git certificate);
  6. whether the run meets the positive control's input requirements.
It never writes inside --root, never runs training or probes, never forces or
manufactures a trigger, and never declares the positive control passed.
"""

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))

SMALL = 64 * 1024 * 1024  # hash evidence files up to 64 MiB; larger ones are sized only
FORK_LOG_PATTERNS = {
    "fork_state_saved": re.compile(r"\[exp1\] fork state saved at interaction_step (\d+)"),
    "restored_after_fork": re.compile(r"\[exp1\] restored interaction_step (\d+)"),
    "post_fork_evaluation": re.compile(r"post-fork evaluation \d+"),
}


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


def _check_records(run, cfg):
    """probe_checks.csv rows, each re-derived from its saved rounds."""
    import numpy as np
    import pandas as pd
    from experiments.exp12.probe import iqm
    from experiments.exp12.trigger import bootstrap_interval, trigger_config, triggered

    tc = trigger_config(cfg)
    table = pd.read_csv(run / "probes" / "probe_checks.csv")
    rows, problems = [], []
    for record in table.to_dict("records"):
        k = int(record["check_index"])
        if k == 0:
            rows.append({"check_index": 0, "interaction_step": int(record["interaction_step"]),
                         "score_fresh_iqm": record.get("score_fresh_iqm")})
            continue
        rounds = int(cfg.probe.rounds)
        loss_cols = [f"loss_r{r}" for r in range(rounds)]
        if all(c in record and np.isfinite(record[c]) for c in loss_cols):
            loss = np.array([record[c] for c in loss_cols], dtype=np.float32)
        else:  # twin critics: the paired loss is the mean of the two networks (amendment (z))
            loss = np.mean([[record[f"loss_q{q}_r{r}"] for r in range(rounds)] for q in (1, 2)],
                           axis=0).astype(np.float32)
        low, high = bootstrap_interval(loss, int(cfg.seed), k, tc.resamples, tc.confidence)
        valid = bool(np.isfinite(loss).all())
        fired = bool(triggered(low, tc.null_threshold) and valid)
        forced = bool(record.get("forced")) if record.get("forced") == record.get("forced") else False
        row = {
            "check_index": k,
            "interaction_step": int(record["interaction_step"]),
            "loss_rounds": loss.tolist(),
            "loss_iqm": iqm(loss) if valid else None,
            "ci_low": low,
            "ci_high": high,
            "fired_recomputed": fired,
            "triggered_recorded": bool(record["triggered"]),
            "forced": forced,
            "score_current_iqm": record.get("score_current_iqm"),
            "score_fresh_iqm": record.get("score_fresh_iqm"),
        }
        if not forced and fired != bool(record["triggered"]):
            problems.append(f"check {k}: recorded trigger differs from recomputation")
        if not np.isclose(low, record["ci_low"], rtol=1e-12, atol=1e-15) and not forced:
            problems.append(f"check {k}: recorded ci_low differs from recomputation")
        rows.append(row)
    return rows, problems, tc


def inspect(root, final_sha=None, extra_logs=(), compat_out=None):
    from omegaconf import OmegaConf

    root = Path(root).resolve()
    evidence = []
    report = {"root": str(root), "evidence": evidence, "problems": []}
    runs = _run_dirs(root)
    report["exp1_run_dirs"] = [str(r) for r in runs]
    if len(runs) != 1:
        report["verdict"] = "INCONCLUSIVE"
        report["problems"].append(f"expected exactly one exp1 run directory, found {len(runs)}")
        return report
    run = runs[0]
    meta_file = _evidence(run / "run_metadata.json", evidence)
    metadata = json.loads(Path(meta_file["path"]).read_text())
    cfg = OmegaConf.create(metadata["resolved_config"])
    identity = metadata["identity"]
    report["identity"] = identity
    report["source_code"] = metadata.get("code")
    report["launches"] = metadata.get("launches")
    report["config_hash"] = metadata.get("config_hash")

    # 1. Horizon.
    from experiments.exp12.state import latest_state_dir

    n = int(cfg.num_interaction_steps)
    done = _evidence(run / "DONE", evidence)
    done_body = json.loads((run / "DONE").read_text()) if done else None
    latest = latest_state_dir(run / "state")
    latest_meta = None
    if latest is not None:
        _evidence(run / "state" / "LATEST", evidence)
        _evidence(Path(latest) / "meta.pkl", evidence)
        with open(Path(latest) / "meta.pkl", "rb") as stream:
            latest_meta = pickle.load(stream)
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

    # 2-3. Trigger: production f*_run rule over the recomputed natural records.
    from experiments.exp12.trigger import f_star, last_eligible_check

    natural = [
        {"check_index": r["check_index"], "interaction_step": r["interaction_step"],
         "triggered": r.get("fired_recomputed", False)}
        for r in records
    ]
    star = f_star(natural, int(cfg.probe.checks), tc.consecutive_checks, tc.eligible_fraction)
    eligible_last = last_eligible_check(int(cfg.probe.checks), tc.eligible_fraction)
    fired = [r["check_index"] for r in records if r.get("fired_recomputed")]
    report["trigger"] = {
        "rule": {"null_threshold": tc.null_threshold, "consecutive_checks": tc.consecutive_checks,
                 "eligible_fraction": tc.eligible_fraction, "eligible_checks": f"1..{eligible_last}",
                 "resamples": tc.resamples, "confidence": tc.confidence},
        "fired_checks": fired,
        "forced_checks": [r["check_index"] for r in records if r.get("forced")],
        "recorded_f_star": (latest_meta or {}).get("extra_state", {}).get("f_star"),
        "recomputed_f_star": star,
        "checks": records,
    }
    for name in ("run.csv", "checks.csv"):
        for path in Path(cfg.results_root).glob(f"exp12/exp1/runs/*/{name}") if cfg.get("results_root") else ():
            _evidence(path, evidence)

    # 4. Fork artifacts.
    fork = run / "fork"
    report["fork"] = {
        "fork_dir_exists": fork.is_dir(),
        "fork_json": _evidence(fork / "fork.json", evidence),
        "fork_ready": _evidence(fork / "FORK_READY", evidence),
        "fork_state": str(latest_state_dir(fork / "state")) if (fork / "state").is_dir() else None,
    }
    if report["fork"]["fork_json"]:
        report["fork"]["plan"] = json.loads((fork / "fork.json").read_text())

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

    # 6. Positive-control input requirements (methodology (e), (q); scripts/positive_control.py).
    from scripts.positive_control import _setting_errors

    report["positive_control_inputs"] = {
        "setting_errors": _setting_errors(cfg, False),
        "run_role": identity.get("run_role"),
        "seed": identity.get("seed"),
        "architecture": identity.get("architecture"),
        "environment": identity.get("environment"),
    }
    natural_fork_valid = None
    if report["fork"]["fork_ready"] and report["fork"]["fork_state"]:
        from experiments.exp12 import fork as fork_mod
        from experiments.exp12.state import load_meta
        from scripts.positive_control import _validate_natural_fork

        try:
            _validate_natural_fork(cfg, fork_mod.read_fork(run), load_meta(Path(report["fork"]["fork_state"])))
            natural_fork_valid = True
        except Exception as exc:  # report, never repair
            natural_fork_valid = False
            report["problems"].append(f"natural-fork validation: {type(exc).__name__}: {exc}")
    report["fork"]["natural_fork_valid"] = natural_fork_valid

    # 5. Source compatibility (git only).
    if final_sha:
        from scripts.exp12_source_compatibility import certify

        cert = certify(metadata["code"]["commit"], final_sha, ["main/experiments/exp2_arm.py"])
        report["source_compatibility"] = {k: cert[k] for k in ("verdict", "source_commit", "target_commit",
                                                               "differing", "unexplained_differences")}
        if compat_out:
            with Path(compat_out).open("x") as stream:
                json.dump(cert, stream, indent=2, sort_keys=True)
                stream.write("\n")

    # Verdict.
    if not report["horizon"]["complete"]:
        verdict = "INCOMPLETE_RUN"
    elif report["trigger"]["forced_checks"] or cfg.testing.force_trigger_check is not None:
        verdict = "FORCED_FORK_NOT_ELIGIBLE"
    elif report["problems"]:
        verdict = "INCONSISTENT_EVIDENCE"
    elif star is None and not report["fork"]["fork_ready"]:
        verdict = "NO_NATURAL_TRIGGER"
    elif star is not None and natural_fork_valid and not report["trigger"]["forced_checks"]:
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
        "INCONCLUSIVE": "Point --root at exactly one exp1 run.",
    }[verdict]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
                      "horizon_complete": report.get("horizon", {}).get("complete")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
