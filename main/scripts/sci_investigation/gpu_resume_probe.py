#!/usr/bin/env python3
"""Cross-process kill-and-resume bit-exactness on the CURRENT backend (the A100 when run on one).

tests.test_exp12_foundations.KillAndResumeEntryPointTest forces JAX_PLATFORMS=cpu in its training subprocesses, so
it never exercises GPU save/restore. This driver runs the same scenario (tiny hopper-hop exp1, crash at interaction
steps 5, 60 and 95, relaunch) through tests/exp12_subprocess_runner.py WITHOUT overriding the backend. Each child
records backend startup and observes actual trained/saved/restored parameter placement; no arrays are moved.
Each resumed final state is compared with two uninterrupted references using experiments.exp12.state.state_differences.
No production code is modified; the crash hook is the existing test runner's.

    python scripts/sci_investigation/gpu_resume_probe.py --out /abs/new_dir --require-backend gpu   # from main/

Verdict and exit status (gpu_resume_probe.json, "verdict"):
  0 PASS                      both references identical; every crash exits 3 without DONE; every relaunch exits 0
                              with DONE and a final state identical to the reference.
  1 RESUME_DEFECT             two references identical, but a relaunch failed or its final state differs:
                              suspected resume-path defect; cause is not established.
  2 NONDETERMINISTIC_BACKEND  the two uninterrupted references differ, so bit-exact resume cannot be judged; the
                              resume cases are still run and reported against both references for diagnosis.
  3 INCOMPLETE                harness/environment failure: a reference run failed, a crash run did not die as
                              scripted, a state is missing, or a child ran on a backend other than --require-backend.
Every child's command, environment summary, backend, exit status, wall time, stdout/stderr log and run directory are
kept under --out.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

MAIN = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(MAIN), str(MAIN / "tests")]

CRASH_STEPS = (5, 60, 95)
CRASH_EXIT = 3  # os._exit code of the runner's crash hook
ENV_KEYS = (
    "JAX_PLATFORMS",
    "JAX_PLATFORM_NAME",
    "CUDA_VISIBLE_DEVICES",
    "XLA_FLAGS",
    "NVIDIA_TF32_OVERRIDE",
    "JAX_DEFAULT_MATMUL_PRECISION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "EXP12_JAX_CACHE_DIR",
    "MUJOCO_GL",
    "OMP_NUM_THREADS",
)
PASS, RESUME_DEFECT, NONDETERMINISTIC, INCOMPLETE = (
    "PASS",
    "RESUME_DEFECT",
    "NONDETERMINISTIC_BACKEND",
    "INCOMPLETE",
)
EXIT = {PASS: 0, RESUME_DEFECT: 1, NONDETERMINISTIC: 2, INCOMPLETE: 3}


def child(spec_path, backend_path):
    """Observe real training/restore without moving arrays or changing initialization."""
    import jax
    from experiments.exp12.trainer import Exp12Trainer
    from experiments.exp12.state import load_meta
    from utils.atomic_io import atomic_write_text

    record = {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "device_kinds": [d.device_kind for d in jax.devices()],
        "training": None,
        "restores": [],
        "saves": [],
    }

    def publish():
        atomic_write_text(Path(backend_path), json.dumps(record, indent=1))

    def snapshot(trainer):
        agent = trainer._sac_agent
        params = {
            name: getattr(agent, name).params
            for name in ("_actor", "_critic", "_target_critic", "_temperature")
        }
        return {
            "interaction_step": trainer.interaction_step,
            "update_step": trainer.update_step,
            "parameter_platforms": {
                name: sorted(
                    {
                        d.platform
                        for x in jax.tree_util.tree_leaves(tree)
                        for d in x.devices()
                    }
                )
                for name, tree in params.items()
            },
        }

    original_restore, original_save, original_train = (
        Exp12Trainer.restore,
        Exp12Trainer.save,
        Exp12Trainer.train,
    )

    def restore(trainer, state_dir, **kwargs):
        meta = load_meta(Path(state_dir))
        attempt = {
            "source": str(Path(state_dir).resolve()),
            "saved_interaction_step": meta["interaction_step"],
            "saved_update_step": meta["update_step"],
            "completed": False,
        }
        record["restores"].append(attempt)
        publish()
        try:
            original_restore(trainer, state_dir, **kwargs)
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            publish()
            raise
        attempt.update(completed=True, **snapshot(trainer))
        publish()

    def save(trainer, *args, **kwargs):
        observation = snapshot(trainer)
        path = original_save(trainer, *args, **kwargs)
        record["saves"].append({"path": str(Path(path).resolve()), **observation})
        publish()
        return path

    def train(trainer, last_step, after_step=None, **kwargs):
        def hook(t):
            if record["training"] is None and t.update_step > 0:
                record["training"] = snapshot(t)
                publish()
            if after_step is not None:
                after_step(t)

        return original_train(trainer, last_step, after_step=hook, **kwargs)

    Exp12Trainer.restore, Exp12Trainer.save, Exp12Trainer.train = restore, save, train
    publish()
    import exp12_subprocess_runner

    exp12_subprocess_runner.exp1_run(spec_path)


def run(out, name, crash_step=None):
    from exp12_helpers import tiny_overrides

    run_dir = out / name
    tag = f"{name}_{'crash' if crash_step else 'run'}"
    spec = {
        "overrides": tiny_overrides(
            "hopper-hop",
            "dmc_medium",
            steps=300,
            extra=[f"results_root={run_dir}_results"],
        ),
        "checkpoint_dir": str(run_dir),
        "checkpoint_interval": 60,
        "crash_step": crash_step,
    }
    spec_path = out / f"{tag}.json"
    spec_path.write_text(json.dumps(spec))
    backend_path = out / f"{tag}_backend.json"
    expected = final_state(run_dir)
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        str(spec_path),
        str(backend_path),
    ]
    env = dict(
        os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "disable")
    )  # backend NOT overridden
    start = time.time()
    log = out / f"{tag}.log"
    receipt = {
        "command": cmd,
        "run_directory": str(run_dir),
        "log": log.name,
        "environment": {k: env.get(k) for k in ENV_KEYS},
        "expected_restore": str(expected.resolve()) if expected else None,
    }
    receipt_path = out / f"{tag}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=1))
    # Stream to disk: an outer timeout must not discard a running child's output.
    with log.open("w") as stream:
        r = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    try:
        backend = (
            json.loads(backend_path.read_text()) if backend_path.exists() else None
        )
    except (OSError, ValueError):
        backend = None
    receipt.update(
        exit=r.returncode,
        seconds=round(time.time() - start, 1),
        backend=backend and backend.get("backend"),
        device_kinds=backend and backend.get("device_kinds"),
        observations=backend,
        done=(run_dir / "DONE").exists(),
        state_present=final_state(run_dir) is not None,
    )
    try:
        from experiments.exp12.state import load_meta

        state = final_state(run_dir)
        meta = load_meta(state) if state is not None else {}
        receipt.update(
            state_interaction_step=meta.get("interaction_step"),
            state_update_step=meta.get("update_step"),
        )
    except Exception as exc:
        receipt["state_error"] = f"{type(exc).__name__}: {exc}"
    receipt_path.write_text(json.dumps(receipt, indent=1))
    return receipt, run_dir


def final_state(run_dir):
    from experiments.exp12.state import latest_state_dir

    try:
        return latest_state_dir(run_dir / "state")
    except ValueError:
        return None


def diff(a, b):
    from experiments.exp12.state import state_differences

    if a is None or b is None:
        return ["missing state"]
    try:
        return state_differences(a, b)
    except Exception as exc:
        return [f"unreadable state: {type(exc).__name__}: {exc}"]


def classify(report, require_backend=None):
    """Verdict and reasons from a filled report; pure, so it is unit-tested without running training."""
    reasons = []
    selected = report.get("crash_steps", CRASH_STEPS)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected).issubset(CRASH_STEPS)
        or set(report["cases"]) != {str(s) for s in selected}
    ):
        reasons.append("missing or unexpected crash scenarios")
    children = [report["reference"], report["reference_repeat"]]
    for case in report["cases"].values():
        children += [case["crash"], case["resume"]]
    for c in children:
        if require_backend and c["backend"] != require_backend:
            reasons.append(
                f"child {c['log']} ran on backend {c['backend']!r}, required {require_backend!r}"
            )
        obs = c.get("observations") or {}
        if require_backend:
            snapshots = [obs.get("training")] if c["exit"] == 0 else []
            snapshots += [
                r for r in obs.get("restores", []) if r.get("completed", False)
            ]
            if c.get("expected_restore") and len(obs.get("restores", [])) != 1:
                reasons.append(f"{c['log']}: existing checkpoint was not restored")
            for snapshot in snapshots:
                if snapshot is None or any(
                    snapshot.get("parameter_platforms", {}).get(name)
                    != [require_backend]
                    for name in ("_actor", "_critic", "_target_critic", "_temperature")
                ):
                    reasons.append(
                        f"{c['log']}: actual training/restore parameter backend not verified"
                    )
            for restore in obs.get("restores", []):
                complete = restore.get("completed", False)
                if (
                    restore["source"] != c.get("expected_restore")
                    or (not complete and c["exit"] == 0)
                    or (
                        complete
                        and (
                            restore["interaction_step"]
                            != restore["saved_interaction_step"]
                            or restore["update_step"] != restore["saved_update_step"]
                        )
                    )
                ):
                    reasons.append(f"{c['log']}: wrong restore source or counters")
    for key in ("reference", "reference_repeat"):
        if report[key]["exit"] != 0 or not report[key]["done"]:
            reasons.append(
                f"{key} exit {report[key]['exit']}, DONE={report[key]['done']}"
            )
        if not report[key].get("state_present"):
            reasons.append(f"{key}: missing final state")
        if (
            report[key].get("state_interaction_step"),
            report[key].get("state_update_step"),
        ) != (300, 582):
            reasons.append(
                f"{key}: wrong endpoint (expected300 interactions/582 updates)"
            )
    for step, case in report["cases"].items():
        if case["crash"]["exit"] != CRASH_EXIT or case["crash"]["done"]:
            reasons.append(
                f"crash{step}: crash run exit {case['crash']['exit']} (expected {CRASH_EXIT}), "
                f"DONE={case['crash']['done']}"
            )
        if step in ("60", "95") and not case["resume"].get("expected_restore"):
            reasons.append(f"crash{step}: no checkpoint selected for resume")
        if step in ("60", "95"):
            saved = (case["crash"].get("observations") or {}).get("saves", [])
            source = case["resume"].get("expected_restore")
            matched = [
                s
                for s in saved
                if s["path"] == source
                and s["interaction_step"] == 60
                and s["update_step"] > 0
            ]
            if len(matched) != 1 or (
                require_backend
                and any(
                    matched[0]["parameter_platforms"].get(name) != [require_backend]
                    for name in ("_actor", "_critic", "_target_critic", "_temperature")
                )
            ):
                reasons.append(
                    f"crash{step}: trained checkpoint backend/source not verified"
                )
        if case["resume"]["exit"] == 0 and not case["resume"].get("state_present"):
            reasons.append(f"crash{step}: missing final state")
        if case["resume"]["exit"] == 0 and (
            case["resume"].get("state_interaction_step"),
            case["resume"].get("state_update_step"),
        ) != (300, 582):
            reasons.append(
                f"crash{step}: wrong endpoint (expected300 interactions/582 updates)"
            )
    for differences in [report["reference_repeat_differences"]] + [
        c[k]
        for c in report["cases"].values()
        if c["resume"]["exit"] == 0 and c["resume"]["done"]
        for k in ("differences", "differences_vs_repeat")
    ]:
        if any(
            d == "missing state" or d.startswith("unreadable state:")
            for d in differences
        ):
            reasons.append("missing/unreadable comparison state")
    if reasons:
        return INCOMPLETE, reasons
    if report["reference_repeat_differences"]:
        return NONDETERMINISTIC, [
            f"uninterrupted references differ (cause unproven): {report['reference_repeat_differences']}",
            *[
                f"crash{s}: relaunch also failed"
                for s, c in report["cases"].items()
                if c["resume"]["exit"] != 0 or not c["resume"]["done"]
            ],
        ]
    for step, case in report["cases"].items():
        if case["resume"]["exit"] != 0 or not case["resume"]["done"]:
            reasons.append(
                f"crash{step}: relaunch exit {case['resume']['exit']}, DONE={case['resume']['done']}"
            )
        elif case["differences"] or case["differences_vs_repeat"]:
            reasons.append(
                f"crash{step}: resumed state differs from reference: {case['differences']}"
            )
    return (RESUME_DEFECT, reasons) if reasons else (PASS, [])


def provenance():
    import jax
    import jaxlib

    def git(*args):
        r = subprocess.run(
            ["git", "-C", str(MAIN), *args], capture_output=True, text=True
        )
        return r.stdout.strip() if r.returncode == 0 else None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty_paths": git("status", "--porcelain", "--untracked-files=all"),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "env": {k: os.environ.get(k) for k in ENV_KEYS},
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out")
    p.add_argument(
        "--require-backend",
        choices=("gpu", "cpu"),
        help="classify as INCOMPLETE if any child initialised a different JAX backend",
    )
    p.add_argument(
        "--crash-step",
        type=int,
        action="append",
        choices=CRASH_STEPS,
        help="explicit test selection; default runs all three scenarios",
    )
    p.add_argument(
        "--child", nargs=2, metavar=("SPEC", "BACKEND_JSON"), help=argparse.SUPPRESS
    )
    args = p.parse_args(argv)
    if args.child:
        child(*args.child)
        return 0
    if not args.out:
        p.error("--out is required")
    out = Path(args.out)
    if not out.is_absolute() or out.exists():
        raise SystemExit("--out must be a new absolute directory")
    out.mkdir(parents=True)
    selected = args.crash_step or list(CRASH_STEPS)
    if len(set(selected)) != len(selected):
        p.error("--crash-step selections must be unique")
    report = {
        "provenance": provenance(),
        "require_backend": args.require_backend,
        "crash_steps": selected,
        "cases": {},
    }

    def publish():
        from utils.atomic_io import atomic_write_text

        atomic_write_text(out / "gpu_resume_probe.json", json.dumps(report, indent=1))

    publish()
    report["reference"], ref_dir = run(out, "ref")
    publish()
    # A second reference detects observed repetition differences; it does not establish their cause.
    report["reference_repeat"], ref2_dir = run(out, "ref2")
    publish()
    ref, ref2 = final_state(ref_dir), final_state(ref2_dir)
    report["reference_repeat_differences"] = diff(ref, ref2)
    for step in selected:
        case = {}
        case["crash"], run_dir = run(out, f"crash{step}", crash_step=step)
        state = final_state(run_dir)
        if state is not None:
            shutil.copytree(state, out / f"crash{step}_checkpoint")
        report["cases"][str(step)] = case
        publish()
        case["resume"], _ = run(out, f"crash{step}")
        res = final_state(run_dir)
        case["differences"] = diff(ref, res)
        case["differences_vs_repeat"] = diff(ref2, res)
        report["cases"][str(step)] = case
        publish()
    report["verdict"], report["reasons"] = classify(report, args.require_backend)
    report["pass"] = report["verdict"] == PASS
    publish()
    print(json.dumps(report, indent=1))
    return EXIT[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
