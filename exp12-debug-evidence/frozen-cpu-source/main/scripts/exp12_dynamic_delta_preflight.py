#!/usr/bin/env python3
"""Two-node Delta filesystem and device-binding check; never runs an experiment."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp12_dynamic_scheduler import GPU_KIND, Ledger, _write_new


def _item(root):
    return {
        "id": "delta_preflight_claim_only",
        "kind": "exp1",
        "row": {
            "config_hash": "diagnostic_only",
            "checkpoint_dir": str(root / "no_run"),
        },
    }


def worker(root):
    import jax

    rank = int(os.environ["SLURM_PROCID"])
    job = os.environ["SLURM_JOB_ID"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    devices = jax.devices()
    if (
        len(visible) != 4
        or len(devices) != 4
        or any(d.platform != "gpu" or d.device_kind != GPU_KIND for d in devices)
        or not jax.local_devices(backend="cpu")
        or len(os.sched_getaffinity(0)) < 16
    ):
        raise ValueError("four A100s, CPU backend or 16 bound CPUs unavailable")
    host = socket.gethostname()
    owner = {"job_id": job, "token": uuid.uuid4().hex, "host": host, "pid": os.getpid()}
    ready = root / f"ready_{rank}.json"
    _write_new(ready, {"host": host, "visible": visible})
    until = time.monotonic() + 60
    while len(list(root.glob("ready_*.json"))) != 2:
        if time.monotonic() >= until:
            raise TimeoutError("second node did not reach the shared filesystem")
        time.sleep(0.2)
    token = root / f"link_source_{rank}"
    token.write_text(owner["token"])
    try:
        os.link(token, root / "atomic_link_winner")
        linked = True
    except FileExistsError:
        linked = False
    claimed = Ledger(root).claim(_item(root), owner, lambda _: None)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.total", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    _write_new(
        root / f"result_{rank}.json",
        {
            "rank": rank,
            "host": host,
            "job_id": job,
            "visible": visible,
            "device_count": len(devices),
            "cpu_affinity": len(os.sched_getaffinity(0)),
            "gpu_inventory": gpu,
            "linked": linked,
            "claimed": claimed,
        },
    )


def verify(root):
    rows = [json.loads(path.read_text()) for path in sorted(root.glob("result_*.json"))]
    if (
        len(rows) != 2
        or {r["rank"] for r in rows} != {0, 1}
        or len({r["host"] for r in rows}) != 2
        or len({r["job_id"] for r in rows}) != 1
        or sum(r["linked"] for r in rows) != 1
        or sum(r["claimed"] for r in rows) != 1
        or any(r["device_count"] != 4 or r["cpu_affinity"] < 16 for r in rows)
    ):
        raise ValueError("two-node binding or shared atomic publication failed")
    link_owner = next(row for row in rows if row["linked"])
    link = root / "atomic_link_winner"
    source = root / f"link_source_{link_owner['rank']}"
    if link.read_text() != source.read_text() or (
        link.stat().st_dev,
        link.stat().st_ino,
    ) != (source.stat().st_dev, source.stat().st_ino):
        raise ValueError("atomic hard link content differs from its winning writer")
    # Simulate an ended allocation for this disposable claim only. The normal
    # scheduler still requires a real squeue check before reclaiming a run.
    diagnostic = Ledger(root, query=lambda _: False)
    recovery_owner = {
        "job_id": rows[0]["job_id"],
        "token": uuid.uuid4().hex,
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    if not diagnostic.claim(_item(root), recovery_owner, lambda _: None):
        raise ValueError("simulated stale-claim recovery failed on shared mount")
    retired = list((diagnostic._row_dir(_item(root)) / "retired_claims").iterdir())
    if len(retired) != 1 or not (retired[0] / "owner.json").is_file():
        raise ValueError("old claim was not preserved during recovery")
    _write_new(
        root / "verdict.json",
        {"verdict": "PASS", "job_id": rows[0]["job_id"], "nodes": rows},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("worker", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if not args.root.is_absolute() or not args.root.is_dir():
        parser.error("root must be a new external absolute directory")
    (worker if args.role == "worker" else verify)(args.root)


if __name__ == "__main__":
    main()
