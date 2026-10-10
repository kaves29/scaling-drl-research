#!/usr/bin/env python3
"""Cross-process kill-and-resume bit-exactness on the CURRENT backend (the A100 when run on one).

tests.test_exp12_foundations.KillAndResumeEntryPointTest forces JAX_PLATFORMS=cpu in its training subprocesses, so
it never exercises GPU save/restore. This driver runs the same scenario (tiny hopper-hop exp1, crash at interaction
steps 5, 60 and 95, relaunch) through tests/exp12_subprocess_runner.py WITHOUT overriding the backend. Each child
first records the backend and devices it actually initialised, so a silent CPU fallback cannot pass as GPU evidence.
Each resumed final state is compared with two uninterrupted references using experiments.exp12.state.state_differences.
No production code is modified; the crash hook is the existing test runner's.

    python scripts/sci_investigation/gpu_resume_probe.py --out /abs/new_dir --require-backend gpu   # from main/

Verdict and exit status (gpu_resume_probe.json, "verdict"):
  0 PASS                      both references identical; every crash exits 3 without DONE; every relaunch exits 0
                              with DONE and a final state identical to the reference.
  1 RESUME_DEFECT             references identical (backend run-to-run deterministic), but a relaunch failed or its
                              final state differs: a save/restore defect.
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
import subprocess
import sys
import time
from pathlib import Path

MAIN = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(MAIN), str(MAIN / "tests")]

RUNNER = MAIN / "tests" / "exp12_subprocess_runner.py"
CRASH_STEPS = (5, 60, 95)
CRASH_EXIT = 3  # os._exit code of the runner's crash hook
ENV_KEYS = ("JAX_PLATFORMS", "JAX_PLATFORM_NAME", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "NVIDIA_TF32_OVERRIDE",
            "JAX_DEFAULT_MATMUL_PRECISION", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "EXP12_JAX_CACHE_DIR", "MUJOCO_GL", "OMP_NUM_THREADS")
PASS, RESUME_DEFECT, NONDETERMINISTIC, INCOMPLETE = "PASS", "RESUME_DEFECT", "NONDETERMINISTIC_BACKEND", "INCOMPLETE"
EXIT = {PASS: 0, RESUME_DEFECT: 1, NONDETERMINISTIC: 2, INCOMPLETE: 3}


def child(spec_path, backend_path):
    """Records the backend this process actually uses, then runs the unmodified test-runner entry."""
    import jax

    Path(backend_path).write_text(json.dumps(
        {"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
         "device_kinds": [d.device_kind for d in jax.devices()]}))
    import exp12_subprocess_runner

    exp12_subprocess_runner.exp1_run(spec_path)


def run(out, name, crash_step=None):
    from exp12_helpers import tiny_overrides

    run_dir = out / name
    tag = f"{name}_{'crash' if crash_step else 'run'}"
    spec = {"overrides": tiny_overrides("hopper-hop", "dmc_medium", steps=300,
                                        extra=[f"results_root={run_dir}_results"]),
            "checkpoint_dir": str(run_dir), "checkpoint_interval": 60, "crash_step": crash_step}
    spec_path = out / f"{tag}.json"
    spec_path.write_text(json.dumps(spec))
    backend_path = out / f"{tag}_backend.json"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(spec_path), str(backend_path)]
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "disable"))  # backend NOT overridden
    start = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log = out / f"{tag}.log"
    log.write_text(r.stdout + r.stderr)
    backend = json.loads(backend_path.read_text()) if backend_path.exists() else None
    return {"exit": r.returncode, "seconds": round(time.time() - start, 1), "log": log.name,
            "backend": backend and backend["backend"], "device_kinds": backend and backend["device_kinds"],
            "done": (run_dir / "DONE").exists()}, run_dir


def final_state(run_dir):
    from experiments.exp12.state import latest_state_dir

    try:
        return latest_state_dir(run_dir / "state")
    except ValueError:
        return None


def diff(a, b):
    from experiments.exp12.state import state_differences

    return state_differences(a, b) if a is not None and b is not None else ["missing state"]


def classify(report, require_backend=None):
    """Verdict and reasons from a filled report; pure, so it is unit-tested without running training."""
    reasons = []
    children = [report["reference"], report["reference_repeat"]]
    for case in report["cases"].values():
        children += [case["crash"], case["resume"]]
    for c in children:
        if require_backend and c["backend"] != require_backend:
            reasons.append(f"child {c['log']} ran on backend {c['backend']!r}, required {require_backend!r}")
    for key in ("reference", "reference_repeat"):
        if report[key]["exit"] != 0 or not report[key]["done"]:
            reasons.append(f"{key} exit {report[key]['exit']}, DONE={report[key]['done']}")
    for step, case in report["cases"].items():
        if case["crash"]["exit"] != CRASH_EXIT or case["crash"]["done"]:
            reasons.append(f"crash{step}: crash run exit {case['crash']['exit']} (expected {CRASH_EXIT}), "
                           f"DONE={case['crash']['done']}")
    if reasons:
        return INCOMPLETE, reasons
    if report["reference_repeat_differences"]:
        return NONDETERMINISTIC, [f"uninterrupted references differ: {report['reference_repeat_differences']}"]
    for step, case in report["cases"].items():
        if case["resume"]["exit"] != 0 or not case["resume"]["done"]:
            reasons.append(f"crash{step}: relaunch exit {case['resume']['exit']}, DONE={case['resume']['done']}")
        elif case["differences"]:
            reasons.append(f"crash{step}: resumed state differs from reference: {case['differences']}")
    return (RESUME_DEFECT, reasons) if reasons else (PASS, [])


def provenance():
    import jax
    import jaxlib

    def git(*args):
        r = subprocess.run(["git", "-C", str(MAIN), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {"commit": git("rev-parse", "HEAD"), "dirty_paths": git("status", "--porcelain", "--untracked-files=no"),
            "python": platform.python_version(), "jax": jax.__version__, "jaxlib": jaxlib.__version__,
            "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "env": {k: os.environ.get(k) for k in ENV_KEYS}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out")
    p.add_argument("--require-backend", choices=("gpu", "cpu"),
                   help="classify as INCOMPLETE if any child initialised a different JAX backend")
    p.add_argument("--child", nargs=2, metavar=("SPEC", "BACKEND_JSON"), help=argparse.SUPPRESS)
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
    report = {"provenance": provenance(), "require_backend": args.require_backend, "cases": {}}
    report["reference"], ref_dir = run(out, "ref")
    # Same command again, uninterrupted: separates run-to-run backend nondeterminism from a resume defect.
    report["reference_repeat"], ref2_dir = run(out, "ref2")
    ref, ref2 = final_state(ref_dir), final_state(ref2_dir)
    report["reference_repeat_differences"] = diff(ref, ref2)
    for step in CRASH_STEPS:
        case = {}
        case["crash"], run_dir = run(out, f"crash{step}", crash_step=step)
        case["resume"], _ = run(out, f"crash{step}")
        res = final_state(run_dir)
        case["differences"] = diff(ref, res)
        case["differences_vs_repeat"] = diff(ref2, res)
        report["cases"][str(step)] = case
    report["verdict"], report["reasons"] = classify(report, args.require_backend)
    report["pass"] = report["verdict"] == PASS
    (out / "gpu_resume_probe.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return EXIT[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
