#!/usr/bin/env python3
"""Source-bound, single-node multi-GPU Exp1 orchestration and SU reservations.

Scientific rows and result validation remain in exp96_campaign. Historical
launchers retain their original policies. No parent may start without the new
qualification, owner study decision, allocation approval and spending reservation.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))
SCHEMA = "exp96-multigpu-754-v1"
ARCHS = ("D2W512", "D4W1024", "D4W1536")
LIMITS = dict(zip(ARCHS, (7, 5, 4)))
CPUS = dict(zip(ARCHS, (2, 3, 4)))
# Only these measured compositions and their subsets may be used. This does not
# infer interference behavior for all nominally feasible mixed configurations.
COMPOSITIONS = (
    (7, 0, 0),
    (0, 5, 0),
    (0, 0, 4),
    (1, 0, 3),
    (0, 1, 3),
    (0, 2, 2),
    (2, 1, 2),
    (3, 1, 1),
)
BASE_SMOKE_SHA = "be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574"
BASE_SMOKE_DIGEST = "effc8c6366b5de04c8bfff682b1886cbbd7c7c0c19a063a765013ac135a59c4d"
TRAINING_PATHS = (
    "main/run.py",
    "main/experiments",
    "main/scale_rl",
    "main/utils",
    "main/configs",
    "main/generate_manifest.py",
    "main/requirements.txt",
    "main/pyproject.toml",
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj, *, replace=False):
    """Fsync an external evidence file; immutable unless explicitly budget state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with path.open("x") as stream:
            json.dump(obj, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temp = path.with_name(path.name + "." + uuid.uuid4().hex)
        write(temp, obj)
        os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checked(args, **kwargs):
    return subprocess.check_output(args, text=True, timeout=30, **kwargs).strip()


def core():
    from scripts import exp96_campaign

    return exp96_campaign


def source(commit):
    core()._source(commit)


def training_fingerprint(commit):
    raw = checked(
        ["git", "ls-tree", "-r", commit, "--", *TRAINING_PATHS], cwd=MAIN.parent
    )
    # This added seven-parent deployment draft is not composed by any Exp96 row.
    excluded = (
        "main/configs/launch_readiness/exp1_seven_parent_resource_estimates.draft.json"
    )
    raw = "\n".join(
        line for line in raw.splitlines() if line.split("\t")[-1] != excluded
    )
    if not raw or "main/run.py" not in raw or len(raw.splitlines()) < 30:
        raise ValueError("scientific source fingerprint is incomplete")
    return hashlib.sha256(raw.encode()).hexdigest()


def composition_allowed(counts):
    return all(type(n) is int and n >= 0 for n in counts) and any(
        all(n <= ceiling for n, ceiling in zip(counts, tested))
        for tested in COMPOSITIONS
    )


def cpu_partition(affinity, count):
    affinity = sorted(affinity)
    if count <= 0 or len(affinity) < 16 * count:
        raise ValueError("at least sixteen allocated CPUs per GPU required")
    return [affinity[i * 16 : (i + 1) * 16] for i in range(count)]


def gpu_tokens(value):
    result = []
    for token in value.split(","):
        match = re.fullmatch(r"(\d+)-(\d+)", token)
        if match:
            first, last = map(int, match.groups())
            if not 0 <= first <= last < 32:
                raise ValueError("invalid Slurm GPU range")
            result.extend(map(str, range(first, last + 1)))
        else:
            result.append(token)
    if len(set(result)) != len(result):
        raise ValueError("duplicate GPU tokens")
    return result


def gpu_inventory(count):
    lines = checked(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    devices = []
    for line in lines:
        index, token, name, memory = [x.strip() for x in line.split(",")]
        if name != "NVIDIA A100-SXM4-40GB" or not 39000 <= int(memory) <= 42000:
            raise ValueError("unqualified GPU model/capacity: " + line)
        devices.append({"index": index, "uuid": token, "memory_mb": int(memory)})
    visible = gpu_tokens(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    allocated = gpu_tokens(
        os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS", "")
    )
    # The reviewed shape owns every physical GPU on this one node. Never map a
    # partial-node local CUDA ordinal to an unallocated physical GPU.
    if (
        len(devices) != count
        or len(set(visible)) != count
        or len(set(allocated)) != count
    ):
        raise ValueError("allocation, visibility and physical GPU counts disagree")
    indices = {x["index"] for x in devices}
    tokens = {x["uuid"] for x in devices}
    if not (set(visible) == indices or set(visible) == tokens):
        raise ValueError("CUDA visibility does not cover the allocated node")
    if not (set(allocated) == indices or set(allocated) == tokens):
        raise ValueError("Slurm GPU ownership does not match the physical devices")
    return devices


def rate(resources):
    """Delta gpuA100x8 live MAX_TRES weights, normalized by accounts' factor1000."""
    if resources["partition"] != "gpuA100x8":
        raise ValueError("unreviewed partition billing")
    return (
        math.ceil(
            max(
                resources["cpus"] * 93.75,
                resources["memory_gb"] * 6,
                resources["gpus"] * 1500,
            )
        )
        / 1000
    )


def resources_valid(resources, *, qualification=False):
    if resources != {
        "partition": "gpuA100x8",
        "nodes": 1,
        "gpus": 8,
        "cpus": 128,
        "memory_gb": 2007,
        "wall_minutes": 15 if qualification else resources.get("wall_minutes"),
    }:
        raise ValueError("reviewed full-node GPU/CPU/RAM shape required")
    wall = resources["wall_minutes"]
    if type(wall) is not int or not 1 <= wall <= 2880:
        raise ValueError("invalid walltime")


def live_request(resources):
    resources_valid(resources, qualification=resources["wall_minutes"] == 15)
    partition = checked(["scontrol", "show", "partition", resources["partition"]])
    config = checked(["scontrol", "show", "config"])
    if "CPU=93.75,Mem=6G,GRES/gpu=1500" not in partition or "MAX_TRES" not in config:
        raise ValueError("billing configuration changed; re-review required")
    nodes = checked(
        ["sinfo", "-p", resources["partition"], "-N", "-h", "-o", "%c %m %G"]
    )
    if not nodes or any(
        int(line.split()[0]) != 128
        or int(line.split()[1]) < 2063600
        or ":8" not in line.split()[2]
        for line in nodes.splitlines()
    ):
        raise ValueError("partition topology changed")
    text = checked(
        ["scontrol", "show", "assoc", "qos=biqc-delta-gpu", "flags=qos", "-o"]
    )
    match = re.search(r"GrpTRESMins=.*?\bbilling=(\d+)\((\d+)\)", text)
    if (
        not match
        or (int(match[1]) - int(match[2])) / 60000
        < rate(resources) * resources["wall_minutes"] / 60
    ):
        raise ValueError(
            "current account balance does not cover this worst-case allocation"
        )
    return rate(resources)


def reserved_job(budget_root, key, resources):
    for _ in range(30):
        entry = read(Path(budget_root) / "budget.json")["entries"][key]
        if entry.get("job_id"):
            break
        time.sleep(1)
    if (
        entry.get("job_id") != os.environ["SLURM_JOB_ID"]
        or entry["resources"] != resources
    ):
        raise ValueError("allocation is not bound to its spending reservation")


class Budget:
    """Reserve worst-case charges BEFORE sbatch. Ambiguous submissions stay reserved."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "budget.json"

    def initialize(self, reference, prior_spent_su=0):
        if (
            not reference
            or not math.isfinite(prior_spent_su)
            or not 0 <= prior_spent_su <= 600
        ):
            raise ValueError(
                "valid authorization and nonnegative prior charges required"
            )
        self.root.mkdir(parents=True, exist_ok=False)
        write(
            self.path,
            {
                "schema": SCHEMA,
                "limit_su": 600,
                "qualification_gpu_hour_limit": 2,
                "authorization_reference": reference,
                "prior_spent_su": prior_spent_su,
                "entries": {},
            },
        )

    @contextlib.contextmanager
    def lock(self):
        guard = self.root / "transition_guard"
        for _ in range(50):
            try:
                guard.mkdir()
                break
            except FileExistsError:
                time.sleep(0.1)
        else:
            raise ValueError("budget transition busy/orphaned; do not submit")
        try:
            yield
        finally:
            guard.rmdir()

    def reserve(self, key, resources, category):
        resources_valid(resources, qualification=category == "qualification")
        cost = rate(resources) * resources["wall_minutes"] / 60
        physical = resources["gpus"] * resources["wall_minutes"] / 60
        with self.lock():
            data = read(self.path)
            if data["limit_su"] != 600 or not data["authorization_reference"]:
                raise ValueError("invalid owner spending authorization")
            if key in data["entries"]:
                raise ValueError(
                    "submission key already reserved; reconcile, never retry blindly"
                )
            used = data.get("prior_spent_su", 0) + sum(
                x.get("actual_su", x["reserved_su"]) for x in data["entries"].values()
            )
            q = sum(
                x.get("actual_gpu_hours", x["physical_gpu_hours"])
                for x in data["entries"].values()
                if x["category"] == "qualification"
            )
            if (
                used + cost > 600 + 1e-9
                or category == "qualification"
                and q + physical > 2 + 1e-9
            ):
                raise ValueError("cumulative authorized budget would be exceeded")
            entry = {
                "category": category,
                "resources": resources,
                "reserved_su": cost,
                "physical_gpu_hours": physical,
                "status": "RESERVED",
                "created": time.time(),
            }
            data["entries"][key] = entry
            write(self.root / "events" / (key + ".reserve.json"), entry)
            write(self.path, data, replace=True)
        return key

    def submitted(self, key, job_id):
        if not str(job_id).isdigit():
            raise ValueError("invalid Slurm ID; reservation retained")
        with self.lock():
            data = read(self.path)
            item = data["entries"][key]
            if item["status"] != "RESERVED":
                raise ValueError("already submitted")
            item.update(status="SUBMITTED", job_id=str(job_id))
            write(self.root / "events" / (key + ".submitted.json"), item)
            write(self.path, data, replace=True)

    def settle(self, key):
        with self.lock():
            data = read(self.path)
            item = data["entries"][key]
            if "actual_su" in item:
                return item
            if not item.get("job_id"):
                raise ValueError(
                    "ambiguous submission: reconcile scheduler before settling"
                )
            text = checked(
                [
                    "sacct",
                    "-X",
                    "-nP",
                    "-j",
                    item["job_id"],
                    "--format=JobIDRaw,State,ElapsedRaw,AllocTRES",
                ]
            )
            rows = [
                line.split("|")
                for line in text.splitlines()
                if line.split("|")[0] == item["job_id"]
            ]
            if len(rows) != 1 or rows[0][1].split()[0] not in {
                "COMPLETED",
                "FAILED",
                "TIMEOUT",
                "CANCELLED",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
            }:
                raise ValueError("terminal unambiguous Slurm accounting not available")
            row = rows[0]
            billing = re.search(r"(?:^|,)billing=(\d+)(?:,|$)", row[3])
            if not billing:
                raise ValueError("missing actual billed TRES")
            seconds = int(row[2])
            actual = int(billing[1]) * seconds / 3_600_000
            if actual > item["reserved_su"] + 0.01:
                raise ValueError(
                    "actual charge exceeds reservation; freeze further spending"
                )
            item.update(
                status=row[1],
                actual_su=actual,
                actual_gpu_hours=seconds * item["resources"]["gpus"] / 3600,
                accounting_row=row,
            )
            write(self.root / "events" / (key + ".settled.json"), item)
            write(self.path, data, replace=True)
            return item


def qualification(path, commit):
    q = read(path)
    if (
        q.get("schema") != SCHEMA
        or q.get("verdict") != "PASS"
        or q.get("expected_commit") != commit
        or q.get("limits") != LIMITS
        or q.get("compositions") != [list(c) for c in COMPOSITIONS]
        or q.get("training_fingerprint") != training_fingerprint(commit)
        or training_fingerprint(commit) != training_fingerprint(BASE_SMOKE_SHA)
    ):
        raise ValueError("source-bound multi-GPU qualification required")
    smoke = Path(q["original_smoke"])
    if (
        digest(smoke) != BASE_SMOKE_DIGEST
        or read(smoke)["expected_commit"] != BASE_SMOKE_SHA
    ):
        raise ValueError("original smoke provenance changed")
    for file, expected in q["evidence_sha256"].items():
        if digest(file) != expected:
            raise ValueError("qualification artifact changed: " + file)
    if (
        not q.get("claims_race_passed")
        or not q.get("all_devices_isolated")
        or not q.get("production_orchestration_passed")
    ):
        raise ValueError("orchestration qualification incomplete")
    resources_valid(q["resources"], qualification=True)
    accounting = checked(
        [
            "sacct",
            "-X",
            "-nP",
            "-j",
            str(q["job_id"]),
            "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES",
        ]
    )
    rows = [
        line.split("|")
        for line in accounting.splitlines()
        if line.split("|")[0] == str(q["job_id"])
    ]
    if (
        len(rows) != 1
        or rows[0][1:3] != ["COMPLETED", "0:0"]
        or not 0 < int(rows[0][3]) <= 900
        or "gres/gpu=8" not in rows[0][4]
        or q.get("resources", {}).get("wall_minutes") != 15
    ):
        raise ValueError("successful bounded Slurm qualification accounting required")
    return q


def study(plan):
    a = read(plan["approval"])
    if (
        a.get("approved") is not True
        or a.get("expected_commit") != plan["commit"]
        or a.get("study_version") != core().VERSION
        or a.get("parent_count") != 96
        or not a.get("decision_reference")
    ):
        raise ValueError("explicit source-bound study decision required")
    if (
        plan["approval_sha256"] is not None
        and digest(plan["approval"]) != plan["approval_sha256"]
    ):
        raise ValueError("study approval changed")
    for gate in (
        "ten_function_method",
        "scientific_calibration",
        "gpu_smoke",
        "allocation_and_storage",
        "cross_seed_analysis",
    ):
        decision = a.get("gates", {}).get(gate, {})
        if decision.get("status") != "APPROVED" or not decision.get(
            "decision_reference"
        ):
            raise ValueError("unapproved study gate: " + gate)
    evidence = a["gates"]["gpu_smoke"]["evidence"]
    if digest(evidence["path"]) != evidence["sha256"]:
        raise ValueError("qualification receipt hash changed")
    return qualification(evidence["path"], plan["commit"])


def allocation(queue_root, approval_path):
    c = core()
    queue, plan = c.load(queue_root, require_approval=False)
    source(queue["commit"])
    q = study(plan)
    a = read(approval_path)
    if (
        a.get("approved") is not True
        or a.get("schema") != SCHEMA
        or a.get("expected_commit") != queue["commit"]
        or a.get("queue_sha256") != digest(Path(queue_root) / "queue.json")
        or a.get("study_approval_sha256") != digest(plan["approval"])
        or not a.get("decision_reference")
    ):
        raise ValueError("source/queue/study-bound allocation approval required")
    r = a["resources"]
    resources_valid(r)
    adm = a["admission"]
    if (
        adm["limits"] != LIMITS
        or adm["packing"] != SCHEMA
        or adm["profiles"] != q["profiles"]
        or adm["compositions"] != q["compositions"]
        or adm["gpu_headroom_mb"] != q["gpu_headroom_mb"]
        or adm["host_headroom_gb"] < 16
        or not 1 <= adm["stop_margin_minutes"] <= 60
    ):
        raise ValueError("admission differs from measured qualification")
    for arch, p in adm["profiles"].items():
        if (
            arch not in ARCHS
            or p["cpus"] != CPUS[arch]
            or p["gpu_memory_mb"] <= 0
            or p["host_ram_gb"] <= 0
            or p["checkpoint_minutes"] + adm["stop_margin_minutes"] >= r["wall_minutes"]
        ):
            raise ValueError("incomplete or unfit qualified profile")
    if set(adm["profiles"]) != set(ARCHS):
        raise ValueError("all architectures required")
    return queue, plan, a


def verify_actual(r):
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise ValueError("real Slurm allocation required")
    description = checked(["scontrol", "show", "job", "-o", job])
    fields = dict(re.findall(r"(?:^|\s)(\w+)=(\S+)", description))
    if (
        fields.get("Partition") != r["partition"]
        or fields.get("Account") != "biqc-delta-gpu"
        or int(fields["NumNodes"]) != 1
        or int(fields["NumCPUs"]) != r["cpus"]
        or int(os.environ.get("SLURM_GPUS_ON_NODE", 0)) != r["gpus"]
        or f"gres/gpu={r['gpus']}" not in fields.get("AllocTRES", "")
        or f"mem={r['memory_gb']}G" not in fields.get("AllocTRES", "")
    ):
        raise ValueError("Slurm allocation differs from approved resources")
    # Avoid relying on optional site-specific formatted time helpers.
    value = fields["TimeLimit"]
    days, _, tail = value.rpartition("-")
    hh, mm, ss = map(int, (tail if days else value).split(":"))
    minutes = (int(days) * 1440 if days else 0) + hh * 60 + mm + ss / 60
    if minutes != r["wall_minutes"]:
        raise ValueError("Slurm walltime differs")
    billing = re.search(r"(?:^|,)billing=(\d+)(?:,|$)", fields["AllocTRES"])
    if not billing or int(billing[1]) > rate(r) * 1000:
        raise ValueError("actual Slurm billing exceeds the reserved SU rate")
    return description


def start_child(item, gpu, approval, queue_root, ledger, job_id, cpu_ids):
    from scripts import exp12_dynamic_scheduler as d

    return d._start(
        item,
        gpu,
        approval,
        queue_root,
        ledger,
        job_id,
        cpu_ids=cpu_ids,
        child_env={
            "JAX_PLATFORMS": "cuda,cpu",
            "PYTHONPATH": str(MAIN),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


def pause(job, ledger, commit):
    """Planned allocation boundary: retire only a stopped, resumable owner.

    This is not a successful completion and never creates a completion receipt.
    Unexpected process errors continue through the existing blocked-result path.
    """
    from scripts import exp12_dynamic_scheduler as d

    p = job["process"]
    if p.poll() is not None:
        return d._finish(job, ledger, commit, ledger.root.parent)
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait(timeout=10)
    try:
        resumed = d.validate_resume(job["item"], commit)
        write(
            job["attempt"] / "planned_pause.json",
            {
                "source": commit,
                "id": job["item"]["id"],
                "returncode": p.returncode,
                "checkpoint_available": resumed,
                "reason": "approved allocation boundary",
                "ended_at": time.time(),
            },
        )
        ledger.finish(job["item"], job["owner"])
    except Exception as exc:
        ledger.finish(
            job["item"], job["owner"], error=f"planned pause validation: {exc}"
        )


def can_admit(
    item, token, active, validating, admission, resources, deadline, snapshot, free_host
):
    if any(j["item"]["id"] == item["id"] for j in list(active) + list(validating)):
        return False
    arch = item["row"]["architecture"]
    profile = admission["profiles"][arch]
    if (
        time.time()
        + 60 * (profile["checkpoint_minutes"] + admission["stop_margin_minutes"])
        >= deadline
    ):
        return False
    occupied = list(active) + list(validating)
    lane = [j for j in occupied if j["gpu"] == token]
    counts = [
        sum(j["item"]["row"]["architecture"] == a for j in lane) + int(a == arch)
        for a in ARCHS
    ]
    if not composition_allowed(counts):
        return False
    if sum(j["profile"]["cpus"] for j in lane) + profile["cpus"] > 16:
        return False
    host = sum(j["profile"]["host_ram_gb"] for j in occupied)
    if (
        host + profile["host_ram_gb"] + admission["host_headroom_gb"]
        > resources["memory_gb"]
    ):
        return False
    if free_host < profile["host_ram_gb"] + admission["host_headroom_gb"]:
        return False
    total, used = snapshot
    reserved = sum(j["profile"]["gpu_memory_mb"] for j in lane)
    return (
        max(used, reserved) + profile["gpu_memory_mb"] + admission["gpu_headroom_mb"]
        <= total
    )


def worker(queue_root, approval_path, budget_root, reservation):
    from scripts import exp12_dynamic_scheduler as d
    from scripts.exp12_launch_plan import require_pinned_stack

    queue, plan, approval = allocation(queue_root, approval_path)
    require_pinned_stack()
    r = approval["resources"]
    verify_actual(r)
    reserved_job(budget_root, reservation, r)
    expected_hash = os.environ["EXP96_APPROVAL_SHA256"]
    if digest(approval_path) != expected_hash:
        raise ValueError("approval changed after submission")
    gpus = gpu_inventory(r["gpus"])
    lanes = {
        g["uuid"]: ids
        for g, ids in zip(gpus, cpu_partition(os.sched_getaffinity(0), len(gpus)))
    }
    ledger = d.Ledger(queue_root)
    admission = approval["admission"]
    # _start/_finish use this historical field name only for process metadata.
    approval["admission"]["architectures"] = admission["profiles"]
    from datetime import datetime

    end = re.search(
        r"\bEndTime=(\S+)",
        checked(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]]),
    )
    if not end:
        raise ValueError("Slurm allocation endpoint unavailable")
    deadline = datetime.strptime(end[1], "%Y-%m-%dT%H:%M:%S").timestamp()
    active, validating, unrecorded, seen, unhealthy = [], {}, [], set(), set()
    receipt = (
        Path(queue_root) / "slurm" / f"startup_gate_{os.environ['SLURM_JOB_ID']}.json"
    )
    write(
        receipt,
        {
            "verdict": "PASS",
            "source": queue["commit"],
            "approval_sha256": expected_hash,
            "queue_sha256": digest(Path(queue_root) / "queue.json"),
            "gpus": gpus,
            "cpu_lanes": lanes,
            "budget_reservation": reservation,
        },
    )
    # Interleave sizes and environments without changing immutable queue identities.
    ordered = sorted(
        queue["parents"],
        key=lambda x: (
            x["row"]["seed"],
            [e[0] for e in core().ENVS].index(x["row"]["environment"]),
            ARCHS.index(x["row"]["architecture"]),
        ),
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            while True:
                if (
                    digest(approval_path) != expected_hash
                    or digest(plan["approval"]) != approval["study_approval_sha256"]
                    or digest(Path(queue_root) / "queue.json")
                    != approval["queue_sha256"]
                ):
                    raise ValueError("immutable launch inputs changed")
                for job in list(active):
                    if job["process"].poll() is not None:
                        active.remove(job)
                        validating[
                            pool.submit(
                                d._finish, job, ledger, queue["commit"], queue_root
                            )
                        ] = job
                d._collect_finished(validating, unrecorded, time.time())
                for job in list(unrecorded):
                    if job["retry_finish_at"] <= time.time():
                        unrecorded.remove(job)
                        validating[
                            pool.submit(
                                d._finish, job, ledger, queue["commit"], queue_root
                            )
                        ] = job
                pending = [
                    x
                    for x in ordered
                    if x["id"] not in seen and ledger.state(x) == "pending"
                ]
                if not active and not validating and not unrecorded and not pending:
                    return 0
                if time.time() >= deadline - 90:
                    for job in list(active):
                        pause(job, ledger, queue["commit"])
                        active.remove(job)
                    if not validating and not unrecorded:
                        return 0
                    if time.time() >= deadline - 30:
                        return 4
                started = False
                if time.time() < deadline - admission["stop_margin_minutes"] * 60:
                    for gpu in sorted(
                        lanes, key=lambda t: sum(j["gpu"] == t for j in active)
                    ):
                        if gpu in unhealthy:
                            continue
                        try:
                            snapshot = d.gpu_snapshot(gpu)
                        except Exception as exc:
                            unhealthy.add(gpu)
                            print(f"GPU {gpu} admission disabled: {exc}", flush=True)
                            continue
                        occupied = active + list(validating.values()) + unrecorded
                        used = set(
                            c for j in occupied if j["gpu"] == gpu for c in j["cpu_ids"]
                        )
                        available = [c for c in lanes[gpu] if c not in used]
                        for item in pending:
                            p = admission["profiles"][item["row"]["architecture"]]
                            if not can_admit(
                                item,
                                gpu,
                                active,
                                list(validating.values()) + unrecorded,
                                admission,
                                r,
                                deadline,
                                snapshot,
                                d.available_host_gb(),
                            ):
                                continue
                            if len(available) < p["cpus"]:
                                continue
                            ids = available[: p["cpus"]]
                            job = start_child(
                                item,
                                gpu,
                                approval,
                                queue_root,
                                ledger,
                                os.environ["SLURM_JOB_ID"],
                                ids,
                            )
                            if job:
                                job["cpu_ids"] = ids
                                active.append(job)
                                seen.add(item["id"])
                                started = True
                                break
                        if started:
                            break
                if not started and not active and not validating and not unrecorded:
                    # No fitting work: release GPUs instead of idling until walltime.
                    return 0
                time.sleep(1 if started else 3)
    finally:
        for job in active:
            if job["process"].poll() is None:
                try:
                    os.killpg(job["process"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        # Slurm step cleanup is a final fence; claims/checkpoints are never erased.


def submit(
    commit, root, resources, budget_root, category, *, queue=None, approval=None
):
    source(commit)
    live_request(resources)
    root = Path(root).resolve()
    if category == "campaign":
        _, _, a = allocation(queue, approval)
        if a["resources"] != resources:
            raise ValueError("submission differs from approved request")
    key = category + "_" + uuid.uuid4().hex
    budget = Budget(budget_root)
    budget.reserve(key, resources, category)
    root.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        EXPECTED_COMMIT=commit,
        EXP96_ACTION="qualify" if category == "qualification" else "worker",
        EXP96_ROOT=str(root),
        EXP96_BUDGET=str(Path(budget_root).resolve()),
        EXP96_RESERVATION=key,
        EXP96_QUEUE=str(queue or ""),
        EXP96_ALLOCATION_APPROVAL=str(approval or ""),
        EXP96_APPROVAL_SHA256=digest(approval) if approval else "",
    )
    for keyname in ("SBATCH_HOLD", "SBATCH_BEGIN", "SBATCH_DEPENDENCY"):
        env.pop(keyname, None)
    cmd = [
        "sbatch",
        "--parsable",
        "--export=ALL",
        "--account=biqc-delta-gpu",
        "--partition=gpuA100x8",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:8",
        "--cpus-per-task=128",
        "--mem=2007G",
        f"--time={resources['wall_minutes']}",
        f"--job-name=exp96-{category}-{key[-8:]}",
        f"--output={root}/slurm_%j.out",
        f"--error={root}/slurm_%j.err",
        "scripts/sbatch_exp96_multigpu.sh",
    ]
    write(
        root / (key + ".submission_intent.json"),
        {
            "command": cmd,
            "source": commit,
            "reservation": key,
            "resources": resources,
            "category": category,
            "approval_sha256": env["EXP96_APPROVAL_SHA256"],
            "queue": str(queue or ""),
        },
    )
    # A timeout, unparsable reply or exception leaves the complete reservation in
    # place and raises. Never automatically retry sbatch.
    output = checked(cmd, cwd=MAIN, env=env)
    job_id = output.split(";")[0]
    budget.submitted(key, job_id)
    write(
        root / (key + ".submission_receipt.json"),
        {
            "job_id": job_id,
            "source": commit,
            "reservation": key,
            "command": cmd,
            "approval_sha256": env["EXP96_APPROVAL_SHA256"],
        },
    )
    print(
        json.dumps({"job_id": job_id, "reservation": key, "root": str(root)}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "worker",
            "qualify",
            "submit-qualification",
            "submit-campaign",
            "init-budget",
            "settle",
        ),
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--commit", default=os.getenv("EXPECTED_COMMIT"))
    parser.add_argument("--queue", default=os.getenv("EXP96_QUEUE"))
    parser.add_argument("--approval", default=os.getenv("EXP96_ALLOCATION_APPROVAL"))
    parser.add_argument("--budget", default=os.getenv("EXP96_BUDGET"))
    parser.add_argument("--reservation", default=os.getenv("EXP96_RESERVATION"))
    parser.add_argument("--reference")
    parser.add_argument("--prior-spent-su", type=float, default=0)
    args = parser.parse_args()
    if args.action == "init-budget":
        Budget(args.root).initialize(args.reference, args.prior_spent_su)
    elif args.action == "settle":
        print(json.dumps(Budget(args.root).settle(args.reservation)))
    elif args.action == "qualify":
        from scripts.exp96_multigpu_qualification import run

        return run(args.root, args.commit)
    elif args.action == "worker":
        return worker(args.queue, args.approval, args.budget, args.reservation)
    else:
        cat = "qualification" if args.action == "submit-qualification" else "campaign"
        r = {
            "partition": "gpuA100x8",
            "nodes": 1,
            "gpus": 8,
            "cpus": 128,
            "memory_gb": 2007,
            "wall_minutes": 15 if cat == "qualification" else 2880,
        }
        submit(
            args.commit,
            args.root,
            r,
            args.budget,
            cat,
            queue=args.queue,
            approval=args.approval,
        )


if __name__ == "__main__":
    # Controllers and completion validators use CPU; only isolated training/
    # qualification children initialize CUDA. No parent reserves all GPU memory.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.pop("JAX_PLATFORM_NAME", None)
    raise SystemExit(main())
