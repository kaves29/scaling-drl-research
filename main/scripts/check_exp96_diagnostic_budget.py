#!/usr/bin/env python3
"""Prepare or recheck a source/plan-bound budget receipt. Never submits jobs."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.exp12.diagnostic_budget import admission, reconcile
from experiments.exp12.diagnostic_plan import load_plan


def walltime_seconds(value):
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    fields = clock.split(":")
    if len(fields) != 3 or any(not x.isdigit() for x in [days, *fields]):
        raise ValueError("Slurm walltime must be finite [days-]HH:MM:SS")
    h, m, s = map(int, fields)
    if m >= 60 or s >= 60:
        raise ValueError("invalid Slurm walltime")
    return int(days) * 86400 + h * 3600 + m * 60 + s


def prepare(raw, jobs, plan, commit, allocation_seconds, reserved, start):
    if len(commit) != 40 or any(x not in "0123456789abcdef" for x in commit):
        raise ValueError("full lowercase source SHA required")
    datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
    result = reconcile(raw, jobs)
    return {
        "schema": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": commit,
        "plan_sha256": load_plan(plan)[1],
        "investigation_start_utc": start,
        "job_inventory": list(map(str, jobs)),
        "accounting_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "accounting": result,
        "admission": admission(result, allocation_seconds, reserved),
    }


def check_live(receipt, plan, commit, job_id, cap_seconds):
    """Reconcile account-wide records again inside the allocation before fitting."""
    if (
        receipt["source"] != commit
        or receipt["plan_sha256"] != load_plan(plan)[1]
        or receipt["schema"] != 1
    ):
        raise ValueError("budget receipt source/plan mismatch")
    fields = dict(
        x.split("=", 1)
        for x in subprocess.check_output(
            ["scontrol", "show", "job", str(job_id), "--oneliner"], text=True
        ).split()
        if "=" in x
    )
    seconds = walltime_seconds(fields["TimeLimit"])
    resources = dict(
        x.split("=", 1) for x in fields["AllocTRES"].split(",") if "=" in x
    )
    if (
        fields.get("Requeue") != "0"
        or fields.get("Restarts") != "0"
        or resources.get("gres/gpu") != "1"
        or resources.get("node") != "1"
        or not fields.get("Partition", "").startswith("gpuA100")
        or not 0 < cap_seconds
        or cap_seconds + 360 > seconds
        or seconds > receipt["admission"]["requested_gpu_seconds"]
    ):
        raise ValueError("allocation resources/requeue/walltime violate budget receipt")
    user = os.environ["USER"]
    queue = subprocess.check_output(
        ["squeue", "-h", "-u", user, "-o", "%A"], text=True
    ).split()
    if any(x != str(job_id) for x in queue):
        raise ValueError("other queued/running allocations need reconciliation")
    raw = subprocess.check_output(
        [
            "sacct",
            "-D",
            "-nP",
            "-u",
            user,
            "--starttime",
            receipt["investigation_start_utc"],
            "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,Restarts,Partition",
        ],
        text=True,
        env={**os.environ, "TZ": "UTC"},
    )
    prior = "\n".join(
        line
        for line in raw.splitlines()
        if line.split("|", 1)[0].split(".", 1)[0] != str(job_id)
    )
    accounting = reconcile(prior, receipt["job_inventory"])
    return {
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_id": str(job_id),
        "source": commit,
        "plan_sha256": receipt["plan_sha256"],
        "accounting": accounting,
        "admission": admission(
            accounting, seconds, receipt["admission"]["reserved_gpu_seconds"]
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--accounting")
    p.add_argument("--jobs", nargs="+")
    p.add_argument("--allocation-seconds", type=int, default=3600)
    p.add_argument("--reserved-gpu-seconds", type=int, default=0)
    p.add_argument("--investigation-start-utc", default="2026-10-22T22:00:00")
    p.add_argument("--receipt")
    p.add_argument("--live-job")
    p.add_argument("--cap-seconds", type=int)
    a = p.parse_args()
    if a.live_job:
        if not a.receipt or a.cap_seconds is None:
            p.error("live check needs --receipt and --cap-seconds")
        result = check_live(
            json.loads(Path(a.receipt).read_text()),
            a.plan,
            a.commit,
            a.live_job,
            a.cap_seconds,
        )
    else:
        if not a.accounting or not a.jobs:
            p.error("preparation needs --accounting and --jobs")
        result = prepare(
            Path(a.accounting).read_text(),
            a.jobs,
            a.plan,
            a.commit,
            a.allocation_seconds,
            a.reserved_gpu_seconds,
            a.investigation_start_utc,
        )
    with Path(a.out).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
