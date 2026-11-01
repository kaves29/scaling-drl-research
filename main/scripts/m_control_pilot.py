"""Serial, bounded development m-control stages. Delta supplies allocation and binding.

Run each stage once using the same --root; failed/interrupted reservations are
never refunded or retried. One allocated GPU only. No Slurm submission occurs.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MAIN = Path(__file__).resolve().parents[1]
LIMITS = {"startup": 900, "generate": 21600, "calibrate": 13500}
BASE = "38e5abb7325a949e23440d6197b5d7d9ecd57c43"


def reserve(ledger, stage, source):
    """Conservative physical GPU seconds: retain full reservation on every outcome."""
    if ledger.get("source", source) != source:
        raise ValueError("source changed; do not reuse pilot receipts")
    stages = ledger.get("stages", {})
    if stage in stages:
        raise ValueError("stage already reserved; no automatic retry")
    if stage != "startup" and stages.get("startup", {}).get("status") != "complete":
        raise ValueError("exact-source startup must complete first")
    if (
        stage == "calibrate"
        and "generate" in stages
        and stages["generate"]["status"] != "complete"
    ):
        raise ValueError("generation failed/incomplete; stop for owner review")
    if stage == "generate" and "calibrate" in stages:
        raise ValueError("cannot generate after calibration")
    if sum(v["reserved_seconds"] for v in stages.values()) + LIMITS[stage] > 36000:
        raise ValueError("10 total physical GPU-hour ceiling exceeded")
    stages[stage] = {"reserved_seconds": LIMITS[stage], "status": "reserved"}
    return {"source": source, "base": BASE, "allocated_gpus": 1, "stages": stages}


def worker(stage, root, run_dir):
    # Explicit CUDA rejects CPU fallback and avoids JAX 0.4.34's gpu->ROCm alias.
    os.environ["JAX_PLATFORMS"] = "cuda"
    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("exactly one visible CUDA GPU required")
    if "A100" not in devices[0].device_kind:
        raise ValueError("reviewed pilot requires an A100; do not substitute hardware")
    value = jnp.sum(
        jnp.ones((16, 16), dtype=jnp.float32) @ jnp.ones((16, 16), dtype=jnp.float32)
    )
    if float(value.block_until_ready()) != 4096:
        raise ValueError("CUDA startup kernel failed")
    if stage == "startup":
        from experiments.exp12.precision import runtime_info

        print(json.dumps(runtime_info(), default=str), flush=True)
        return 0
    if stage == "generate":
        from experiments import exp1
        from experiments.exp12 import fork

        if fork.is_ready(run_dir):
            raise ValueError("checkpoint already exists; omit generation stage")
        exp1.run(
            {
                "config_path": str(MAIN / "configs"),
                "config_name": "base_exp12_m_control_ten",
                "overrides": [f"results_root={root / 'results'}"],
                "checkpoint_dir": str(run_dir),
                "checkpoint_interval": 50000,
                "checkpoint_start_frac": 0.0,
                "stop_at_natural_fork": True,
            }
        )
        return 0 if fork.is_ready(run_dir) else 3
    from scripts import positive_control

    return positive_control.main(
        [
            "--protocol",
            "m-control-ten",
            "--run_dir",
            str(run_dir),
            "--out_dir",
            str(root / "calibration"),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="existing dev checkpoint; omit to generate under root",
    )
    parser.add_argument("--stage", choices=LIMITS, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root
    run_dir = args.run_dir or root / "development"
    if not root.is_absolute() or not run_dir.is_absolute():
        parser.error("root and run-dir must be absolute")
    if args.worker:
        return worker(args.stage, root, run_dir)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=MAIN, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=MAIN, text=True
    ).strip()
    if source != args.source_sha or dirty:
        raise ValueError("clean exact reviewed source required")
    binding = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not binding or len(binding.split(",")) != 1 or binding == "-1":
        raise ValueError("Delta must bind one allocated GPU before starting")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "budget.json"
        ledger = json.loads(path.read_text()) if path.exists() else {}
        if ledger.get("run_dir", str(run_dir)) != str(run_dir):
            raise ValueError("development input changed")
        ledger = reserve(ledger, args.stage, source)
        ledger["run_dir"] = str(run_dir)

        def save():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(ledger, indent=2) + "\n")
            tmp.replace(path)

        save()
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.m_control_pilot",
            *sys.argv[1:],
            "--worker",
        ]
        row = ledger["stages"][args.stage]
        row.update(command=command, cuda_visible_devices=binding)

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"pilot received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        start = time.monotonic()
        with (root / f"{args.stage}.log").open("xb") as log:
            child = subprocess.Popen(
                command,
                cwd=MAIN,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = child.wait(timeout=max(1, LIMITS[args.stage] - 10))
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                row["status"] = "interrupted_or_timeout"
                raise
            finally:
                row["elapsed_seconds"] = time.monotonic() - start
                row["log_sha256"] = hashlib.sha256(
                    (root / f"{args.stage}.log").read_bytes()
                ).hexdigest()
                save()
        row.update(exit_code=code, status="complete" if code == 0 else "failed")
        save()
        return code


if __name__ == "__main__":
    sys.exit(main())
