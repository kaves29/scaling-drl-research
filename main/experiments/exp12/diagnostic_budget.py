"""Conservative physical-GPU accounting for diagnostic execution preparation."""

import re

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def reconcile(raw, expected_jobs):
    """Parse sacct -D -nP allocation rows; refuse duplicates and unsettled jobs.

    Required fields: JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,Restarts,Partition.
    Step rows are evidence about their parent, not additional GPU allocations.
    expected_jobs must include every investigation attempt, not a convenient subset.
    """
    expected = set(map(str, expected_jobs))
    jobs = {}
    if not expected:
        raise ValueError("explicit investigation job inventory is required")
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 7:
            raise ValueError("incomplete accounting row")
        job, state, exit_code, seconds, tres, restarts, partition = parts[:7]
        if "." in job:
            continue
        if not re.fullmatch(r"\d+", job):
            raise ValueError("unknown allocation identity")
        if job not in expected:
            raise ValueError("unreconciled additional investigation allocation")
        if job in jobs:
            raise ValueError("duplicate/requeued accounting requires review")
        if state.split()[0] not in TERMINAL:
            raise ValueError("allocation is not terminal")
        if restarts != "0":
            raise ValueError("restarts require interval-level accounting")
        elapsed = int(seconds)
        if elapsed < 0:
            raise ValueError("negative allocation duration")
        resources = dict(x.split("=", 1) for x in tres.split(",") if "=" in x)
        gpu = int(resources.get("gres/gpu", "0"))
        if elapsed and (gpu != 1 or not partition.startswith("gpuA100")):
            raise ValueError("allocated hardware must be one physical A100")
        if gpu not in (0, 1):
            raise ValueError("unexpected GPU allocation")
        jobs[job] = {
            "state": state,
            "exit_code": exit_code,
            "elapsed_seconds": elapsed,
            "gpu_count": gpu,
            "physical_gpu_seconds": elapsed * gpu,
            "partition": partition,
        }
    if set(jobs) != expected:
        raise ValueError("missing investigation accounting")
    used = sum(j["physical_gpu_seconds"] for j in jobs.values())
    return {
        "jobs": jobs,
        "physical_gpu_seconds": used,
        "physical_A100_hours": used / 3600,
    }


def admission(accounting, allocation_seconds, reserved_gpu_seconds=0):
    """A planning calculation, not scientific approval or an atomic reservation."""
    if (
        type(allocation_seconds) is not int
        or not 0 < allocation_seconds <= 7200
        or type(reserved_gpu_seconds) is not int
        or reserved_gpu_seconds < 0
    ):
        raise ValueError("invalid requested or reserved allocation time")
    total = (
        accounting["physical_gpu_seconds"] + reserved_gpu_seconds + allocation_seconds
    )
    if total > 14400:
        raise ValueError("four physical A100-hour ceiling exceeded")
    return {
        "used_gpu_seconds": accounting["physical_gpu_seconds"],
        "reserved_gpu_seconds": reserved_gpu_seconds,
        "requested_gpu_seconds": allocation_seconds,
        "ceiling_gpu_seconds": 14400,
        "remaining_after_request_seconds": 14400 - total,
        "authorization": "NOT_GRANTED; owner approval and fresh complete accounting required",
    }
