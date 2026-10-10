#!/usr/bin/env python3
"""Cross-process kill-and-resume bit-exactness on the CURRENT backend (the A100 when run on one).

tests.test_exp12_foundations.KillAndResumeEntryPointTest forces JAX_PLATFORMS=cpu in its training subprocesses, so
it never exercises GPU save/restore. This driver runs the same scenario (tiny hopper-hop exp1, crash at interaction
steps 5, 60 and 95, relaunch) through tests/exp12_subprocess_runner.py WITHOUT overriding the backend, then compares
each resumed final state with an uninterrupted reference using experiments.exp12.state.state_differences.

    python scripts/sci_investigation/gpu_resume_probe.py --out /abs/new_dir     # from main/

Pass: both uninterrupted references exit 0 and are identical (otherwise the backend is not run-to-run
deterministic and resume cannot be judged bit-exact); every crash run exits 3 without DONE; every relaunch exits 0
with DONE; every state_differences(reference, resumed) == []. Exit status 0 only if all pass.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(MAIN), str(MAIN / "tests")]

RUNNER = MAIN / "tests" / "exp12_subprocess_runner.py"


def run(out, name, crash_step=None):
    from exp12_helpers import tiny_overrides

    run_dir = out / name
    spec = {"overrides": tiny_overrides("hopper-hop", "dmc_medium", steps=300,
                                        extra=[f"results_root={run_dir}_results"]),
            "checkpoint_dir": str(run_dir), "checkpoint_interval": 60, "crash_step": crash_step}
    spec_path = out / f"{name}_{'crash' if crash_step else 'run'}.json"
    spec_path.write_text(json.dumps(spec))
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "disable"))  # backend NOT overridden
    r = subprocess.run([sys.executable, str(RUNNER), "exp1", str(spec_path)], env=env, capture_output=True, text=True)
    (out / f"{spec_path.stem}.log").write_text(r.stdout + r.stderr)
    return r.returncode, run_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    out = Path(p.parse_args(argv).out)
    if not out.is_absolute() or out.exists():
        raise SystemExit("--out must be a new absolute directory")
    out.mkdir(parents=True)
    import jax

    from experiments.exp12.state import latest_state_dir, state_differences

    report = {"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()], "cases": {}}
    code, ref_dir = run(out, "ref")
    report["reference_exit"] = code
    ok = code == 0
    ref = latest_state_dir(ref_dir / "state") if ok else None
    # Same command again, uninterrupted: separates run-to-run GPU nondeterminism from a resume defect.
    code2, ref2_dir = run(out, "ref2")
    ref2 = latest_state_dir(ref2_dir / "state") if code2 == 0 else None
    report["reference_repeat_exit"] = code2
    report["reference_repeat_differences"] = (state_differences(ref, ref2) if ref is not None and ref2 is not None
                                              else ["missing state"])
    ok &= code2 == 0 and report["reference_repeat_differences"] == []
    for step in (5, 60, 95):
        case = {}
        case["crash_exit"], run_dir = run(out, f"crash{step}", crash_step=step)
        case["done_after_crash"] = (run_dir / "DONE").exists()
        case["resume_exit"], _ = run(out, f"crash{step}")
        case["done_after_resume"] = (run_dir / "DONE").exists()
        res = latest_state_dir(run_dir / "state")
        case["differences"] = state_differences(ref, res) if ref is not None and res is not None else ["missing state"]
        case["pass"] = (case["crash_exit"] == 3 and not case["done_after_crash"] and case["resume_exit"] == 0
                        and case["done_after_resume"] and case["differences"] == [])
        ok &= case["pass"]
        report["cases"][str(step)] = case
    report["pass"] = bool(ok)
    (out / "gpu_resume_probe.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
