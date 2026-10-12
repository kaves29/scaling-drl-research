#!/usr/bin/env python3
"""Persistent, source-bound Exp1/Exp2 queue for independently queued allocations.

The filesystem containing the queue MUST support atomic directory creation and
rename across its clients. Claims are never timed out by wall clock: a claim is
recoverable only after Slurm confirms that its owning allocation has ended.
An orphan transition guard blocks that row for manual review.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))

from scripts.exp12_launch_plan import (  # noqa: E402
    DEFERRED,
    GATES,
    IDENTITY_REQUIRED,
    cells,
    config_hash,
    evidence_coverage,
    postvalidate,
    require_pinned_stack,
    sha,
    submission_command,
)

WEIGHTS = {"D2W512": 1 / 7, "D4W1024": 1 / 4, "D4W1536": 1 / 2}
CEILINGS = {"D2W512": 7, "D4W1024": 4, "D4W1536": 2}
# Legacy named shapes stay exact; any other approved name is a generic allocation of
# 1..MAX_NODES four-A100 gpuA100x4 nodes for at most the partition's 48 hours.
ALLOCATIONS = {"A": (1, 30 * 60), "B": (2, 48 * 60)}
ALLOCATION_NAME = re.compile(r"[A-Za-z0-9_-]{1,32}")
MAX_NODES = 4
MAX_WALL_MINUTES = 48 * 60
GROUPS = ("dmc_hard", "dmc_medium", "myosuite_simba", "humanoid_bench")
FORKING = ("D4W1024", "D4W1536")
GLOBAL_GATES = (
    "positive_control",
    "suite_runtime_storage",
    "grid_allocation",
    "cuda_numeric_bounds",
    "study_methodology_freeze",
)
NULL_ARCH = "D4W1024"
HOPPER_CELL = ("D4W1536", "hopper-hop")
GPU_KIND = "NVIDIA A100-SXM4-40GB"
POLL_SECONDS = 15
_VERIFIED_ARM_FILES = {}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_new(path, value):
    """Publish one immutable JSON file without exposing partial bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    return path


def _read(path):
    return json.loads(Path(path).read_text())


def _evidence_digests(item):
    run = Path(item["row"]["checkpoint_dir"])
    paths = [run / "DONE", run / "run_metadata.json", run / "state" / "LATEST"]
    if item["kind"] == "exp2_injected":
        row = item["row"]
        results = (
            Path(_read(run / "run_metadata.json")["resolved_config"]["results_root"])
            / "exp12"
            / "exp2"
            / row["run_key"]
        )
        paths.extend(
            [
                Path(row["source_dir"]) / "fork" / "fork.json",
                results / "check1_injected.json",
                results / "check2.json",
                *(
                    results / "arm_injected" / name
                    for name in ("checks.csv", "eval_episodes.csv", "metrics.csv")
                ),
            ]
        )
    return {str(path): sha(path) for path in paths}


def identity(kind, row, commit, *, source_id=None, injection_m=None):
    fields = {
        "kind": kind,
        "environment": row["environment"],
        "architecture": row["architecture"],
        "seed": row["seed"],
        "config_hash": row["config_hash"],
        "commit": commit,
    }
    if kind == "exp2_injected":
        fields.update(
            source_id=source_id,
            fork_id=row["fork_id"],
            injection_m=injection_m,
        )
    return _digest(fields)


def _check_plan(plan_path):
    # Reuse the existing approval, source, full-population and config checks.
    submission_command(plan_path, per_job_policy=True)
    plan = _read(plan_path)
    if plan["stage"] != "parents" or len(plan["entries"]) != 195:
        raise ValueError("dynamic queue requires the complete approved parent plan")
    return plan


def prepare(plan_path, queue_root):
    plan_path, queue_root = Path(plan_path).resolve(), Path(queue_root).resolve()
    plan = _check_plan(plan_path)
    if queue_root.exists() or queue_root.is_relative_to(MAIN.parent):
        raise ValueError("queue must be a new absolute directory outside the checkout")
    rows = []
    for row in plan["entries"]:
        rows.append(
            {
                "id": identity("exp1", row, plan["commit"]),
                "kind": "exp1",
                "row": row,
            }
        )
    if len({r["id"] for r in rows}) != 195:
        raise ValueError("parent logical identities are not unique")
    campaign_root = Path(plan["root"])
    campaign_root.mkdir(parents=True, exist_ok=True)
    _write_new(
        campaign_root / "EXP12_DYNAMIC_QUEUE.json",
        {
            "queue": str(queue_root),
            "plan_sha256": sha(plan_path),
            "commit": plan["commit"],
        },
    )
    queue_root.parent.mkdir(parents=True, exist_ok=True)
    queue_root.mkdir()
    queue = {
        "schema_version": 1,
        "plan": str(plan_path),
        "plan_sha256": sha(plan_path),
        "commit": plan["commit"],
        "parents": rows,
    }
    _write_new(queue_root / "queue.json", queue)
    return queue


def load_queue(queue_root):
    queue_root = Path(queue_root).resolve()
    queue = _read(queue_root / "queue.json")
    plan_path = Path(queue["plan"])
    plan = _check_plan(plan_path)
    if sha(plan_path) != queue["plan_sha256"] or plan["commit"] != queue["commit"]:
        raise ValueError("queue plan/source differs from approved immutable plan")
    if _read(Path(plan["root"]) / "EXP12_DYNAMIC_QUEUE.json") != {
        "queue": str(queue_root),
        "plan_sha256": queue["plan_sha256"],
        "commit": queue["commit"],
    }:
        raise ValueError("campaign root is bound to a different scheduler queue")
    expected = [
        {"id": identity("exp1", row, plan["commit"]), "kind": "exp1", "row": row}
        for row in plan["entries"]
    ]
    if queue["parents"] != expected or len(expected) != 195:
        raise ValueError("queue population differs from all 195 approved parents")
    return queue


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} requires a positive integer")
    return value


def allocation_shape(allocation, resources):
    """Legacy A/B keep their exact shapes; other names are bounded generic allocations."""
    if not isinstance(allocation, str) or not ALLOCATION_NAME.fullmatch(allocation):
        raise ValueError("invalid allocation name")
    nodes, minutes = resources.get("nodes"), resources.get("wall_minutes")
    if resources.get("gpus_per_node") != 4:
        raise ValueError("allocations use whole four-A100 nodes")
    if allocation in ALLOCATIONS:
        if (nodes, minutes) != ALLOCATIONS[allocation]:
            raise ValueError("allocation must be exactly A=4 A100/30h or B=8 A100/48h")
    elif (
        type(nodes) is not int
        or not 1 <= nodes <= MAX_NODES
        or type(minutes) is not int
        or not 1 <= minutes <= MAX_WALL_MINUTES
    ):
        raise ValueError(
            f"generic allocation needs 1..{MAX_NODES} nodes and at most 48 hours"
        )
    return nodes, minutes


def validate_positive_control(evidence, commit, require_m=None):
    """Owner-referenced positive-control evidence for this campaign source.

    The fork may come from an earlier commit only with a hash-pinned, re-verified
    science-source compatibility certificate (scripts/exp12_source_compatibility.py)."""
    path, digest = evidence.get("path"), evidence.get("sha256")
    if not path or not digest or sha(path) != digest:
        raise ValueError("positive-control evidence missing or changed")
    control = _read(path)
    if (
        control.get("status") != "m_chosen"
        or control.get("chosen_m") not in ("last", "half", "all")
        or (require_m is not None and control.get("chosen_m") != require_m)
        or control.get("allow_any_setting") is not False
        or control.get("trigger_check", {}).get("forced") is not False
        or control.get("analysis_code", {}).get("dirty") is not False
    ):
        raise ValueError("positive control does not qualify this final source/m")
    source = control.get("source_code") or {}
    if source.get("dirty") is not False:
        raise ValueError("positive control does not qualify this final source/m")
    if source.get("commit") != commit:
        certificate = evidence.get("source_compatibility")
        if not certificate:
            raise ValueError("positive control does not qualify this final source/m")
        from scripts.exp12_source_compatibility import verify

        verify(certificate["path"], certificate["sha256"], source.get("commit"), commit)
    return control


def validate_approval(approval, queue_root, allocation):
    queue = load_queue(queue_root)
    if (
        approval.get("approved") is not True
        or approval.get("allocation") != allocation
        or approval.get("expected_commit") != queue["commit"]
        or approval.get("queue_sha256") != sha(Path(queue_root) / "queue.json")
        or not approval.get("decision_reference")
    ):
        raise ValueError("explicit allocation/source/queue approval is missing")
    resources = approval.get("resources", {})
    _, expected_minutes = allocation_shape(allocation, resources)
    cpus = _positive_int(resources.get("cpus_per_node"), "allocated CPUs per node")
    ram = _positive_int(
        resources.get("memory_gb_per_node"), "allocated host RAM per node"
    )
    admission = approval.get("admission", {})
    pack_limit = admission.get("max_processes_per_gpu", 1)
    if type(pack_limit) is not int or not 1 <= pack_limit <= max(CEILINGS.values()):
        raise ValueError("approved GPU process cap must be an integer from 1 to 7")
    margin_gpu = _positive_int(admission.get("gpu_headroom_gb"), "GPU headroom")
    margin_host = _positive_int(admission.get("host_headroom_gb"), "host headroom")
    stop = _positive_int(admission.get("stop_margin_minutes"), "stop margin")
    if stop >= expected_minutes:
        raise ValueError("stop margin consumes the allocation")
    profiles = admission.get("architectures", {})
    if set(profiles) != set(WEIGHTS):
        raise ValueError("all three architecture resource profiles are required")
    for arch, profile in profiles.items():
        _positive_int(profile.get("cpus"), f"{arch} CPUs")
        _positive_int(profile.get("host_ram_gb"), f"{arch} host RAM")
        _positive_int(profile.get("gpu_memory_gb"), f"{arch} GPU memory")
        estimates = profile.get("estimated_minutes_by_group", {})
        if set(estimates) != set(GROUPS):
            raise ValueError(
                f"{arch} needs runtime estimates for every environment group"
            )
        for group, estimate in estimates.items():
            _positive_int(estimate, f"{arch}/{group} runtime")
        # Exp2 may be released while this allocation runs, so forking sizes always
        # carry arm estimates.
        if arch in FORKING:
            arms = profile.get("estimated_arm_minutes_by_group", {})
            if set(arms) != set(GROUPS):
                raise ValueError(f"{arch} needs injected-arm runtime estimates")
            for group, estimate in arms.items():
                _positive_int(estimate, f"{arch}/{group} injected-arm runtime")
        floor = profile.get("min_interactions_per_second")
        if type(floor) not in (float, int) or floor <= 0:
            raise ValueError(f"{arch} measured-throughput floor is required")
        if profile["cpus"] > cpus or profile["host_ram_gb"] + margin_host > ram:
            raise ValueError(f"{arch} profile cannot fit the host allocation")
        if profile["gpu_memory_gb"] + margin_gpu > 40:
            raise ValueError(f"{arch} profile cannot fit a 40 GB A100")
    if approval.get("exp2", {}).get("approved") is True:
        exp2 = approval["exp2"]
        if (
            exp2.get("injection_m") not in ("last", "half", "all")
            or not exp2.get("decision_reference")
            or not exp2.get("positive_control_evidence")
            or not exp2.get("positive_control_sha256")
            or sha(exp2["positive_control_evidence"]) != exp2["positive_control_sha256"]
        ):
            raise ValueError(
                "Exp2 requires owner-frozen m and intact positive-control evidence"
            )
        validate_positive_control(
            {
                "path": exp2["positive_control_evidence"],
                "sha256": exp2["positive_control_sha256"],
                "source_compatibility": exp2.get("source_compatibility"),
            },
            queue["commit"],
            require_m=exp2["injection_m"],
        )
    hb_python = resources.get("humanoid_bench_python")
    if not isinstance(hb_python, str) or not Path(hb_python).is_absolute():
        raise ValueError("approved absolute HumanoidBench Python path required")
    return resources, admission


# ---------------------------------------------------------------------------
# Per-run releases (plan approval release_mode = "per_run").
#
# The approved plan may mark gates DEFERRED_TO_RELEASE. A deferred gate is satisfied
# only by an immutable, owner-signed release published under <queue>/releases, which
# is re-validated against the queue source and hash-pinned evidence on every read.
# Before each start the scheduler evaluates, per run, every prerequisite applicable to
# that run; an unsatisfied prerequisite keeps the run pending, never failed.
# ---------------------------------------------------------------------------
RELEASE_TYPES = ("gate", "identity", "a2_staged_identity", "fresh_null", "hopper_range", "exp2")
_RELEASE_CACHE = {}
_EFFECTIVE_CACHE = {}


def _plan_approval(queue):
    plan = _read(queue["plan"])
    return _read(plan["approval"])


def _deferred(approval, gate):
    from scripts.exp12_launch_plan import per_run_release

    return (
        per_run_release(approval, "parents")
        and approval.get("gates", {}).get(gate, {}).get("status") == DEFERRED
    )


def release_name(release):
    kind = release.get("type")
    if kind == "gate":
        return f"gate_{release.get('gate')}.json"
    if kind == "identity":
        return f"identity_{_digest(release)[:16]}.json"
    return f"{kind}.json"


def validate_release(decision, queue, queue_root):
    """One owner decision -> normalized release; raises on anything not provable."""
    approval = _plan_approval(queue)
    commit = queue["commit"]
    if (
        decision.get("approved") is not True
        or decision.get("expected_commit") != commit
        or decision.get("queue_sha256") != sha(Path(queue_root) / "queue.json")
        or not decision.get("decision_reference")
    ):
        raise ValueError("release needs an approved, source- and queue-bound owner decision")
    release = decision.get("release") or {}
    kind = release.get("type")
    if kind not in RELEASE_TYPES:
        raise ValueError("unknown release type")
    if kind == "gate":
        gate = release.get("gate")
        if gate not in GLOBAL_GATES or not _deferred(approval, gate):
            raise ValueError("gate release requires a gate deferred by the approved plan")
        if gate == "positive_control":
            validate_positive_control(release.get("positive_control") or {}, commit)
    elif kind in ("identity", "a2_staged_identity"):
        if not _deferred(approval, "cuda_identity_scaled_suites"):
            raise ValueError("identity is not deferred by the approved plan")
        if kind == "identity":
            coverage = evidence_coverage(release.get("evidence", []), commit, "cuda_identity_fork")
            if not {(a, s) for a, s, _ in coverage} <= IDENTITY_REQUIRED:
                raise ValueError("identity evidence covers a case outside amendment (p)")
        elif not release.get("amendment_reference"):
            raise ValueError("staged identity requires the approved amendment A2 reference")
    elif kind == "fresh_null":
        if not _deferred(approval, "fresh_null_disposition"):
            raise ValueError("fresh-null disposition is not deferred by the approved plan")
        choice = release.get("decision")
        if choice == "adopt_threshold":
            computation = release.get("computation") or {}
            if not computation.get("path") or sha(computation["path"]) != computation.get("sha256"):
                raise ValueError("threshold computation report missing or changed")
            report = _read(computation["path"])
            if (
                report.get("status") != "COMPUTED_NOT_ACTIVATED"
                or report.get("architecture") != NULL_ARCH
                or report.get("selection") != release.get("selection")
                or report.get("override") != release.get("override")
                or report.get("override")
                != f"trigger.null_threshold={report.get('selected_threshold')!r}"
            ):
                raise ValueError("threshold release differs from the computed report")
        elif choice != "keep_zero" or "override" in release:
            raise ValueError("fresh-null decision must be keep_zero or adopt_threshold")
    elif kind == "hopper_range":
        if not _deferred(approval, "hopper_range_disposition"):
            raise ValueError("hopper disposition is not deferred by the approved plan")
        if release.get("decision") not in ("release", "exclude"):
            raise ValueError("hopper decision must be release or exclude")
    else:  # exp2
        if release.get("injection_m") not in ("last", "half", "all"):
            raise ValueError("Exp2 release requires the owner-frozen m")
        validate_positive_control(
            release.get("positive_control") or {}, commit, require_m=release["injection_m"]
        )
    return release


def publish_release(queue_root, decision_path):
    """Validate one owner decision and publish it write-once. Never invents a decision."""
    from scripts.exp12_launch_plan import per_run_release

    queue_root = Path(queue_root).resolve()
    queue = load_queue(queue_root)
    if not per_run_release(_plan_approval(queue), "parents"):
        raise ValueError("the approved plan does not use per-run releases")
    decision = _read(decision_path)
    release = validate_release(decision, queue, queue_root)
    target = queue_root / "releases" / release_name(release)
    record = {
        "decision": decision,
        "decision_path": str(Path(decision_path).resolve()),
        "decision_sha256": sha(decision_path),
        "published_at": time.time(),
    }
    try:
        _write_new(target, record)
    except FileExistsError:
        raise ValueError(f"release {target.name} already published; releases are immutable")
    return target


def release_state(queue_root, queue):
    """Satisfied prerequisites: plan-approved gates plus valid published releases."""
    approval = _plan_approval(queue)
    gates = approval.get("gates", {})
    state = {
        "gates": {g for g in GATES["parents"] if gates.get(g, {}).get("status") == "APPROVED"},
        "identity": set(),
        "a2": False,
        "fresh_null": None,
        "hopper": None,
        "exp2": None,
        "releases": [],
    }
    if "cuda_identity_scaled_suites" in state["gates"]:
        state["identity"] |= IDENTITY_REQUIRED
    if "fresh_null_disposition" in state["gates"]:
        state["fresh_null"] = {"decision": "keep_zero"}
    if "hopper_range_disposition" in state["gates"]:
        state["hopper"] = "release"
    directory = Path(queue_root) / "releases"
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
        if path.is_symlink():
            raise ValueError("release record cannot be a symlink")
        key = (str(path), sha(path))
        if key not in _RELEASE_CACHE:
            # The record embeds the owner decision; its evidence is re-verified here.
            release = validate_release(_read(path)["decision"], queue, queue_root)
            if path.name != release_name(release):
                raise ValueError(f"release file name differs from its content: {path.name}")
            _RELEASE_CACHE[key] = release
        release = _RELEASE_CACHE[key]
        state["releases"].append(path.name)
        kind = release["type"]
        if kind == "gate":
            state["gates"].add(release["gate"])
        elif kind == "identity":
            state["identity"] |= {
                (a, s)
                for a, s, _ in evidence_coverage(release["evidence"], queue["commit"], "cuda_identity_fork")
            }
        elif kind == "a2_staged_identity":
            state["a2"] = True
        elif kind == "fresh_null":
            state["fresh_null"] = release
        elif kind == "hopper_range":
            state["hopper"] = release["decision"]
        else:
            state["exp2"] = release
    return state


def run_blockers(item, state, exp2_active=False):
    """Every unsatisfied prerequisite applicable to this run (empty = may start)."""
    from analysis.exp12_validation import suite

    row = item["row"]
    arch, environment = row["architecture"], row["environment"]
    blockers = [f"gate:{g}" for g in GLOBAL_GATES if g not in state["gates"]]
    if state["a2"]:
        # Proposed amendment A2: D2W512 never forks, so identity cannot affect it;
        # a forking run needs both scaled identity cases of its own suite.
        needed = {(a, suite(environment)) for a in FORKING} if arch in FORKING else set()
    else:
        needed = set(IDENTITY_REQUIRED)  # (b): all six before any grid run
    if not needed <= state["identity"]:
        missing = sorted(f"{a}/{s}" for a, s in needed - state["identity"])
        blockers.append("identity:" + ",".join(missing))
    if arch == NULL_ARCH and state["fresh_null"] is None:
        blockers.append("fresh_null_disposition")
    if (arch, environment) == HOPPER_CELL and state["hopper"] != "release":
        blockers.append(
            "hopper_range_excluded" if state["hopper"] == "exclude" else "hopper_range_disposition"
        )
    if item["kind"] == "exp2_injected" and not exp2_active:
        blockers.append("exp2_release")
    return blockers


def effective_parents(queue, state):
    """Queue rows with only release-approved overrides applied (D4W1024 threshold)."""
    release = state["fresh_null"]
    override = release.get("override") if release and release.get("decision") == "adopt_threshold" else None
    key = (queue["commit"], queue["plan_sha256"], override)
    if key in _EFFECTIVE_CACHE:
        return _EFFECTIVE_CACHE[key]
    parents = []
    for parent in queue["parents"]:
        row = parent["row"]
        if override is not None and row["architecture"] == NULL_ARCH:
            row = dict(row)
            row["overrides"] = list(row["overrides"]) + [override]
            row["release_overrides"] = [override]
            row["config_hash"] = config_hash(row["overrides"])
            row["command"] = list(row["command"]) + ["--overrides", override]
            parent = {"id": identity("exp1", row, queue["commit"]), "kind": "exp1", "row": row}
        parents.append(parent)
    if len({p["id"] for p in parents}) != len(parents):
        raise ValueError("effective parent identities are not unique")
    _EFFECTIVE_CACHE[key] = parents
    return parents


def estimated_minutes(item, profile):
    """Approved estimate scaled by the unfinished fraction of a resumable parent."""
    row = item["row"]
    if item["kind"] == "exp2_injected":
        return profile["estimated_arm_minutes_by_group"][row["group"]]
    full = profile["estimated_minutes_by_group"][row["group"]]
    try:
        point = checkpoint_progress(item)
    except (OSError, ValueError):
        point = None
    if point is None or not row.get("budget"):
        return full
    remaining = 1 - point[0] / (row["budget"] // 2)
    return full * min(1.0, max(0.05, remaining))


def priority(item, admission):
    """Arms first (they gate Exp2 completion), then longest remaining work first."""
    profile = admission["architectures"][item["row"]["architecture"]]
    return (0 if item["kind"] == "exp2_injected" else 1, -estimated_minutes(item, profile))


def allocation_command(queue_root, approval_path, allocation):
    queue_root = Path(queue_root).resolve()
    approval_path = Path(approval_path).resolve()
    approval = _read(approval_path)
    resources, _ = validate_approval(approval, queue_root, allocation)
    logs = queue_root / "slurm"
    reservation = queue_root / "submissions" / allocation
    command = [
        "env",
        f"EXPECTED_COMMIT={approval['expected_commit']}",
        f"EXP12_QUEUE={queue_root}",
        f"EXP12_ALLOCATION_APPROVAL={approval_path}",
        f"EXP12_APPROVAL_SHA256={sha(approval_path)}",
        f"EXP12_ALLOCATION={allocation}",
        "sbatch",
        "--parsable",
        "--export=ALL",
        f"--nodes={resources['nodes']}",
        "--ntasks-per-node=1",
        "--gpus-per-node=4",
        f"--cpus-per-task={resources['cpus_per_node']}",
        f"--mem={resources['memory_gb_per_node']}G",
        f"--time={resources['wall_minutes']}",
        f"--output={logs}/%x_%j.out",
        f"--error={logs}/%x_%j.err",
        "scripts/sbatch_exp12_dynamic_allocation.sh",
    ]
    return (
        shlex.join(command)
        + " > "
        + shlex.quote(str(reservation / "job-id.txt"))
        + " 2> "
        + shlex.quote(str(reservation / "submission.stderr"))
    )


def _owner_alive(owner, query):
    job = owner.get("job_id")
    if not job or not re.fullmatch(r"[0-9]+", str(job)):
        raise ValueError("claim lacks a valid Slurm owner; manual recovery required")
    return query(str(job))


def slurm_alive(job):
    result = subprocess.run(
        ["squeue", "-h", "-j", job, "-o", "%T"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("squeue failed; claim recovery refused")
    return bool(result.stdout.strip())


class Ledger:
    def __init__(self, root, query=slurm_alive):
        self.root = Path(root).resolve() / "ledger"
        self.query = query

    def _row_dir(self, item):
        return self.root / item["kind"] / item["id"]

    def _guard(self, directory):
        guard = directory / "transition_guard"
        try:
            guard.mkdir()
        except FileExistsError:
            return None
        return guard

    def _unguard(self, guard):
        guard.rmdir()

    def claim(self, item, owner, reconcile):
        """Atomic cross-allocation claim. Never infer completion from DONE alone."""
        directory = self._row_dir(item)
        directory.mkdir(parents=True, exist_ok=True)
        guard = self._guard(directory)
        if guard is None:
            return False
        try:
            completed = directory / "completed.json"
            if completed.exists():
                if _read(completed)["id"] != item["id"]:
                    raise ValueError("completed ledger identity mismatch")
                return False
            if (directory / "blocked.json").exists():
                return False
            claim = directory / "claim"
            if claim.exists():
                if not (claim / "ready").exists():
                    raise ValueError("partial claim requires manual recovery")
                prior = _read(claim / "owner.json")
                if _owner_alive(prior, self.query):
                    return False
                retired = directory / "retired_claims"
                retired.mkdir(exist_ok=True)
                claim.rename(retired / (prior["token"] + "_" + uuid.uuid4().hex))
            try:
                receipt = reconcile(item)
            except Exception as exc:
                _write_new(
                    directory / "blocked.json",
                    {
                        "id": item["id"],
                        "error": f"reconciliation: {type(exc).__name__}: {exc}",
                        "recorded_at": time.time(),
                    },
                )
                return False
            if receipt is not None:
                _write_new(
                    completed,
                    {
                        "id": item["id"],
                        "kind": item["kind"],
                        "source": item["row"]["config_hash"],
                        "reconciled_by_allocation": owner["job_id"],
                        "receipt": str(receipt),
                        "receipt_sha256": sha(receipt),
                        "evidence_sha256": _evidence_digests(item),
                        "reconciled": True,
                        "recorded_at": time.time(),
                    },
                )
                return False
            claim.mkdir()
            _write_new(claim / "owner.json", owner)
            (claim / "ready").touch()
            return True
        finally:
            self._unguard(guard)

    def finish(self, item, owner, receipt=None, error=None):
        directory = self._row_dir(item)
        guard = self._guard(directory)
        if guard is None:
            raise ValueError("transition guard is occupied")
        try:
            claim = directory / "claim"
            if not claim.exists() or _read(claim / "owner.json") != owner:
                raise ValueError("claim owner changed before finish")
            if receipt is not None:
                _write_new(
                    directory / "completed.json",
                    {
                        "id": item["id"],
                        "kind": item["kind"],
                        "source": item["row"]["config_hash"],
                        "producing_allocation": owner["job_id"],
                        "attempt_token": owner["token"],
                        "receipt": str(receipt),
                        "receipt_sha256": sha(receipt),
                        "evidence_sha256": _evidence_digests(item),
                        "recorded_at": time.time(),
                    },
                )
            elif error is not None:
                _write_new(
                    directory / "blocked.json",
                    {
                        "id": item["id"],
                        "error": error,
                        "recorded_at": time.time(),
                    },
                )
            retired = directory / "retired_claims"
            retired.mkdir(exist_ok=True)
            claim.rename(retired / (owner["token"] + "_" + uuid.uuid4().hex))
        finally:
            self._unguard(guard)

    def state(self, item):
        directory = self._row_dir(item)
        if (directory / "blocked.json").exists():
            return "blocked"
        if (directory / "completed.json").exists():
            record = _read(directory / "completed.json")
            if (
                record["id"] != item["id"]
                or record["source"] != item["row"]["config_hash"]
                or sha(record["receipt"]) != record["receipt_sha256"]
                or record["evidence_sha256"] != _evidence_digests(item)
            ):
                raise ValueError("completed ledger or validation receipt was changed")
            return "complete"
        return "pending"

    def review_retry(self, item, commit, decision_path):
        """Requeue one blocked unfinished run only with a source-bound decision."""
        directory = self._row_dir(item)
        guard = self._guard(directory)
        if guard is None:
            raise ValueError("transition guard is occupied")
        try:
            blocked = directory / "blocked.json"
            if (
                not blocked.is_file()
                or (directory / "claim").exists()
                or (directory / "completed.json").exists()
            ):
                raise ValueError("row is not a safely blocked unfinished run")
            decision = _read(decision_path)
            if (
                decision.get("approved") is not True
                or decision.get("expected_commit") != commit
                or decision.get("item_id") != item["id"]
                or decision.get("blocked_sha256") != sha(blocked)
                or not decision.get("decision_reference")
            ):
                raise ValueError("explicit matching recovery decision required")
            if not validate_resume(item, commit):
                raise ValueError("recovery requires an existing valid checkpoint")
            retries = directory / "retry_decisions"
            retries.mkdir(exist_ok=True)
            _write_new(
                retries / (uuid.uuid4().hex + ".json"),
                {
                    "item_id": item["id"],
                    "decision": str(Path(decision_path).resolve()),
                    "decision_sha256": sha(decision_path),
                    "recorded_at": time.time(),
                },
            )
            retired = directory / "retired_blocks"
            retired.mkdir(exist_ok=True)
            blocked.rename(retired / (uuid.uuid4().hex + ".json"))
        finally:
            self._unguard(guard)


def _metadata_matches(run, item, commit):
    metadata = _read(Path(run) / "run_metadata.json")
    if (
        metadata.get("code") != {"commit": commit, "dirty": False}
        or metadata.get("config_hash") != item["row"]["config_hash"]
        or metadata.get("identity", {}).get("architecture")
        != item["row"]["architecture"]
        or metadata.get("identity", {}).get("environment") != item["row"]["environment"]
        or metadata.get("identity", {}).get("seed") != item["row"]["seed"]
    ):
        raise ValueError("checkpoint source/config/logical identity mismatch")
    launches = metadata.get("launches")
    if not launches or any(
        launch.get("commit") != commit
        or launch.get("dirty") is not False
        or launch.get("device", {}).get("platform") != "gpu"
        or launch.get("device", {}).get("device_kind") != GPU_KIND
        or launch.get("device", {}).get("device_count") != 1
        or launch.get("runtime", {}).get("jax") != "0.4.34"
        or launch.get("runtime", {}).get("jaxlib") != "0.4.34"
        or launch.get("runtime", {}).get("matmul_precision") != "tensorfloat32"
        for launch in launches
    ):
        raise ValueError("checkpoint launches lack pinned single-A100 provenance")
    return metadata


def _checkpoint_valid(run, item=None, commit=None):
    from experiments.exp12.state import latest_state_dir, load_meta

    state = latest_state_dir(Path(run) / "state")
    if (
        item is not None
        and item["kind"] == "exp1"
        and (Path(run) / "fork" / "FORK_READY").exists()
    ):
        plan = _natural_fork(item, commit)
        if state is None or load_meta(state)["interaction_step"] < plan["fork_step"]:
            state = latest_state_dir(Path(run) / "fork" / "state")
    if state is None:
        raise ValueError("unfinished run has no committed checkpoint")
    meta = load_meta(state)
    if type(meta.get("interaction_step")) is not int:
        raise ValueError("checkpoint has invalid interaction counter")
    return state


def validate_resume(item, commit):
    run = Path(item["row"]["checkpoint_dir"])
    if not run.exists():
        return False
    if (run / "DONE").exists():
        raise ValueError("DONE must be reconciled, never restarted")
    _metadata_matches(run, item, commit)
    state = _checkpoint_valid(run, item, commit)
    from experiments.exp12.state import load_agent_tree
    from scripts.exp12_launch_plan import validate_replay_normalization

    validate_replay_normalization(state)
    import jax

    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        tree = load_agent_tree(state)
    if not {"actor", "critic", "target_critic", "rng"} <= tree.keys():
        raise ValueError("resume checkpoint lacks essential agent state")
    return True


def _receipt_path(queue_root, item):
    directory = Path(queue_root) / "validation" / item["kind"] / item["id"]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (uuid.uuid4().hex + ".json")


def validate_parent(item, commit, queue_root):
    run = Path(item["row"]["checkpoint_dir"])
    if not (run / "DONE").exists():
        return None
    _metadata_matches(run, item, commit)
    receipt = _receipt_path(queue_root, item)
    import jax

    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        result = postvalidate(
            run, commit, receipt, extra_overrides=item["row"].get("release_overrides", ())
        )
    if (
        result["engineering_verdict"] != "COMPLETE"
        or result["scientific_verdict"] != "CANONICAL_CERTIFICATION_REQUIRED"
    ):
        raise ValueError(
            f"parent DONE failed complete-state validation: {result['errors']}"
        )
    return receipt


def _natural_fork(parent_item, commit):
    """Check the existing natural-trigger rule; never manufacture eligibility."""
    from experiments.exp1 import compose_config
    from experiments.exp12 import fork
    from experiments.exp12.state import latest_state_dir, load_meta
    from scripts.positive_control import _validate_natural_fork

    row = parent_item["row"]
    parent = Path(row["checkpoint_dir"])
    if not (parent / "fork" / "FORK_READY").exists():
        return None
    metadata = _metadata_matches(parent, parent_item, commit)
    if (
        metadata["identity"].get("run_role") != "confirmatory"
        or metadata["identity"].get("experiment") != "exp1"
        or (parent / "fork" / "FORK_READY").read_text().strip() != row["run_key"]
    ):
        raise ValueError(
            "fork marker/parent role is incompatible with natural production eligibility"
        )
    cfg = compose_config(str(MAIN / "configs"), "base_exp12", row["overrides"])
    state = latest_state_dir(parent / "fork" / "state")
    meta = load_meta(state)
    plan = fork.read_fork(parent)
    _validate_natural_fork(cfg, plan, meta)
    original = fork.load_npz(parent / "fork" / "check1_pre.npz")
    control = fork.load_npz(parent / "fork" / "check1_control.npz")
    if not fork.check1(original, control, control, 0.0, False)["pass"]:
        raise ValueError("fork control fails bit-exact Check 1")
    recorded = _read(parent / "fork" / "fork.json")["device"]
    if recorded.get("device_kind") != GPU_KIND or recorded.get("platform") != "gpu":
        raise ValueError("fork is not from the approved A100")
    return plan


def _arm_item(parent_item, commit, injection_m):
    from generate_manifest import exp12_paths

    parent_row = parent_item["row"]
    fork_file = Path(parent_row["checkpoint_dir"]) / "fork" / "fork.json"
    fork_sha256 = sha(fork_file)
    fork_id = _digest(
        {
            "source_id": parent_item["id"],
            "source_commit": commit,
            "fork_sha256": fork_sha256,
        }
    )
    root = Path(parent_row["checkpoint_dir"]).parents[3]
    _, arm_dir = exp12_paths(
        str(root),
        parent_row["architecture"],
        parent_row["environment"],
        parent_row["seed"],
    )
    # The checkpoint root is campaign_root/checkpoints, not campaign_root itself.
    overrides = parent_row["overrides"] + [
        f"fork.source={parent_row['checkpoint_dir']}",
        "fork.arm=injected",
        f"injection.m={injection_m}",
    ]
    command = list(parent_row["command"])
    command[command.index("exp1")] = "exp2_arm"
    command[command.index(parent_row["checkpoint_dir"])] = arm_dir
    for override in overrides[len(parent_row["overrides"]) :]:
        command += ["--overrides", override]
    arm_row = {
        **{
            key: parent_row[key]
            for key in ("architecture", "environment", "seed", "group", "budget")
        },
        "run_key": parent_row["run_key"],
        "source_id": parent_item["id"],
        "source_dir": parent_row["checkpoint_dir"],
        "fork_id": fork_id,
        "fork_sha256": fork_sha256,
        "checkpoint_dir": arm_dir,
        "overrides": overrides,
        "config_hash": config_hash(overrides),
        "command": command,
        "injection_m": injection_m,
    }
    return {
        "kind": "exp2_injected",
        "row": arm_row,
        "id": identity(
            "exp2_injected",
            arm_row,
            commit,
            source_id=parent_item["id"],
            injection_m=injection_m,
        ),
    }


def discover_arms(queue_root, queue, injection_m, parents=None):
    """Idempotently publish eligible arms while parents continue training.

    parents: the effective (release-applied) parent items; defaults to the queue rows."""
    from generate_manifest import EXP12_FORKING

    parents = queue["parents"] if parents is None else parents
    directory = Path(queue_root) / "arms"
    directory.mkdir(exist_ok=True)
    existing = {}
    parents_by_id = {parent["id"]: parent for parent in parents}
    for path in directory.glob("*.json"):
        if path.is_symlink():
            raise ValueError("arm manifest cannot be a symlink")
        arm = _read(path)
        if (
            path.stem != arm.get("id")
            or arm.get("kind") != "exp2_injected"
            or arm["row"].get("injection_m") != injection_m
        ):
            raise ValueError("arm manifest identity/m is incompatible")
        source_id = arm["row"]["source_id"]
        if source_id not in parents_by_id:
            raise ValueError("arm manifest has no approved source parent")
        fork_file = Path(arm["row"]["source_dir"]) / "fork" / "fork.json"
        if sha(fork_file) != arm["row"]["fork_sha256"]:
            raise ValueError("published fork header changed after arm discovery")
        digest = sha(path)
        if path in _VERIFIED_ARM_FILES:
            if _VERIFIED_ARM_FILES[path] != digest:
                raise ValueError("published arm manifest changed")
        elif _arm_item(parents_by_id[source_id], queue["commit"], injection_m) != arm:
            raise ValueError("arm manifest differs from canonical source/configuration")
        _VERIFIED_ARM_FILES[path] = digest
        if source_id in existing and existing[source_id] != arm:
            raise ValueError("one parent has conflicting injected arm manifests")
        existing[source_id] = arm
    for parent in parents:
        if parent["row"]["architecture"] not in EXP12_FORKING:
            continue
        if parent["id"] in existing:
            continue
        if (
            Path(queue_root) / "arm_discovery_errors" / (parent["id"] + ".json")
        ).exists():
            continue
        if not (Path(parent["row"]["checkpoint_dir"]) / "fork" / "FORK_READY").exists():
            continue
        try:
            _natural_fork(parent, queue["commit"])
        except Exception as exc:
            errors = Path(queue_root) / "arm_discovery_errors"
            errors.mkdir(exist_ok=True)
            record = errors / (parent["id"] + ".json")
            if not record.exists():
                try:
                    _write_new(
                        record,
                        {
                            "source_id": parent["id"],
                            "error": f"{type(exc).__name__}: {exc}",
                            "recorded_at": time.time(),
                        },
                    )
                except FileExistsError:
                    pass
            continue
        arm = _arm_item(parent, queue["commit"], injection_m)
        target = directory / (arm["id"] + ".json")
        if target.exists():
            if _read(target) != arm:
                raise ValueError("existing arm manifest differs from natural fork")
            continue
        try:
            _write_new(target, arm)
        except FileExistsError:
            if _read(target) != arm:
                raise ValueError("concurrent arm writer published conflicting identity")
        _VERIFIED_ARM_FILES[target] = sha(target)
    return sorted(
        (_read(path) for path in directory.glob("*.json")),
        key=lambda arm: arm["id"],
    )


def validate_arm(item, commit, queue_root):
    import jax
    import numpy as np
    from experiments.exp12.state import latest_state_dir, load_agent_tree, load_meta
    from scripts.exp12_launch_plan import validate_replay_normalization

    row = item["row"]
    run = Path(row["checkpoint_dir"])
    if not (run / "DONE").exists():
        return None
    metadata = _metadata_matches(run, item, commit)
    if metadata["identity"]["experiment"] != "exp2_arm":
        raise ValueError("injected arm metadata has the wrong experiment")
    parent = Path(row["source_dir"])
    if sha(parent / "fork" / "fork.json") != row["fork_sha256"]:
        raise ValueError("injected arm source fork changed")
    source = _read(parent / "fork" / "fork.json")
    done = _read(run / "DONE")
    if (
        done.get("arm") != "injected"
        or done.get("interaction_step") != source["arm_end_step"]
    ):
        raise ValueError("injected arm has an incomplete or wrong endpoint")
    state_dir = latest_state_dir(run / "state")
    state = load_meta(state_dir)
    if state["interaction_step"] != source["arm_end_step"]:
        raise ValueError("injected arm checkpoint is not terminal")
    required = {
        "update_step",
        "update_counter",
        "numpy_rng_state",
        "python_rng_state",
        "train_env",
        "eval_env",
        "observations",
        "timestep",
        "meters",
        "metrics_rows",
        "eval_rows",
        "extra_state",
        "agent_window_buffers",
    }
    if not required <= state.keys():
        raise ValueError("injected arm checkpoint lacks complete state")
    validate_replay_normalization(state_dir)
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        tree = load_agent_tree(state_dir)
    if not {
        "rng",
        "actor",
        "critic",
        "target_critic",
        "temperature",
    } <= tree.keys() or any(
        not np.isfinite(np.asarray(value)).all()
        for value in jax.tree_util.tree_leaves(tree)
    ):
        raise ValueError("injected arm agent state is incomplete/nonfinite")
    cfg = metadata["resolved_config"]
    if (
        cfg["fork"]["source"] != row["source_dir"]
        or cfg["fork"]["arm"] != "injected"
        or cfg["injection"]["m"] != row["injection_m"]
    ):
        raise ValueError("injected arm source/parameter differs from queue")
    results = Path(cfg["results_root"]) / "exp12" / "exp2" / row["run_key"]
    check1 = _read(results / "check1_injected.json")
    if check1.get("pass") is not True:
        raise ValueError("injected Check 1 did not pass")
    if not (results / "check2.json").is_file():
        raise ValueError("injected Check 2 report is missing")
    arm_results = results / "arm_injected"
    for name in ("checks.csv", "eval_episodes.csv", "metrics.csv"):
        if (
            not (arm_results / name).is_file()
            or not (arm_results / name).stat().st_size
        ):
            raise ValueError(f"injected result {name} is missing")
    import pandas as pd

    evals = pd.read_csv(arm_results / "eval_episodes.csv")
    expected_indices = set(
        range(source["horizon_steps"] // source["eval_every_steps"] + 1)
    )
    if (
        set(evals["eval_index"]) != expected_indices
        or set(evals["arm"]) != {"injected"}
        or any(
            count != source["eval_episodes"]
            for count in evals.groupby("eval_index").size()
        )
        or any(
            row.interaction_step
            != source["fork_step"] + row.eval_index * source["eval_every_steps"]
            for row in evals.itertuples()
        )
    ):
        raise ValueError("injected evaluation schedule is incomplete or mismatched")
    checks = pd.read_csv(arm_results / "checks.csv")
    if checks.empty or set(checks["arm"]) != {"injected"}:
        raise ValueError("injected probe checks are incomplete")
    metrics = pd.read_csv(arm_results / "metrics.csv")
    if metrics.empty or set(metrics["arm"]) != {"injected"}:
        raise ValueError("injected metrics are incomplete")
    receipt = _receipt_path(queue_root, item)
    _write_new(
        receipt,
        {
            "engineering_verdict": "COMPLETE",
            "scientific_verdict": "CANONICAL_CERTIFICATION_REQUIRED",
            "id": item["id"],
            "source_id": row["source_id"],
            "source_commit": commit,
            "m": row["injection_m"],
            "endpoint": source["arm_end_step"],
        },
    )
    return receipt


def validate_done(item, commit, queue_root):
    if item["kind"] == "exp1":
        return validate_parent(item, commit, queue_root)
    return validate_arm(item, commit, queue_root)


def gpu_snapshot(token):
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(token),
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    parts = [part.strip() for part in result.stdout.strip().split(",")]
    if len(parts) != 3 or parts[0] != GPU_KIND:
        raise ValueError(f"GPU {token} is not the approved A100: {parts}")
    return int(parts[1]), int(parts[2])


def available_host_gb():
    available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    limit, used = Path("/sys/fs/cgroup/memory.max"), Path(
        "/sys/fs/cgroup/memory.current"
    )
    if limit.is_file() and used.is_file():
        raw_limit = limit.read_text().strip()
        if raw_limit != "max":
            available = min(available, int(raw_limit) - int(used.read_text().strip()))
    return max(0, available) / 1024**3


def checkpoint_progress(item):
    pointer = Path(item["row"]["checkpoint_dir"]) / "state" / "LATEST"
    if not pointer.is_file():
        return None
    name = pointer.read_text().strip()
    if not re.fullmatch(r"step_[0-9]{9}_[0-9a-f]{8}", name):
        raise ValueError("invalid checkpoint progress pointer")
    return int(name.split("_")[1]), pointer.stat().st_mtime


def update_rates(active):
    """Observe real checkpoint progress; absence of two points blocks packing."""
    for job in active:
        point = checkpoint_progress(job["item"])
        if point is None or point == job.get("point"):
            continue
        previous = job.get("point")
        job["point"] = point
        if previous and point[0] > previous[0] and point[1] > previous[1]:
            job["rate"] = (point[0] - previous[0]) / (point[1] - previous[1])


def can_admit(
    item,
    token,
    active,
    admission,
    resources,
    deadline,
    snapshot,
    host_free,
    validating=(),
):
    """Check nominal packing AND live CPU/RAM/VRAM/throughput before each start."""
    row = item["row"]
    arch = row["architecture"]
    profile = admission["architectures"][arch]
    if (
        time.time() + 60 * (estimated_minutes(item, profile) + admission["stop_margin_minutes"])
        >= deadline
    ):
        return False
    if any(job["item"]["id"] == item["id"] for job in active):
        return False
    occupied = list(active) + list(validating)
    if (
        sum(job["profile"]["cpus"] for job in occupied) + profile["cpus"]
        > resources["cpus_per_node"]
    ):
        return False
    host_reservation = sum(job["profile"]["host_ram_gb"] for job in occupied)
    if (
        host_reservation + profile["host_ram_gb"] + admission["host_headroom_gb"]
        > resources["memory_gb_per_node"]
    ):
        return False
    if host_free < profile["host_ram_gb"] + admission["host_headroom_gb"]:
        return False
    same_gpu = [job for job in active if job["gpu"] == token]
    if len(same_gpu) >= admission.get("max_processes_per_gpu", 1):
        return False
    if (
        sum(WEIGHTS[job["item"]["row"]["architecture"]] for job in same_gpu)
        + WEIGHTS[arch]
        > 1 + 1e-12
    ):
        return False
    if (
        sum(job["item"]["row"]["architecture"] == arch for job in same_gpu)
        >= CEILINGS[arch]
    ):
        return False
    # Checkpoint-based rates are slow to become available, by design. Starting
    # extra processes before actual progress is measurable would assume scaling.
    if any(
        job.get("rate") is None
        or job["rate"] < job["profile"]["min_interactions_per_second"]
        for job in same_gpu
    ):
        return False
    total_mb, used_mb = snapshot
    estimated_mb = 1024 * sum(job["profile"]["gpu_memory_gb"] for job in same_gpu)
    if (
        max(used_mb, estimated_mb)
        + 1024 * (profile["gpu_memory_gb"] + admission["gpu_headroom_gb"])
        > total_mb
    ):
        return False
    return True


def _actual_allocation(approval, job_description, visible):
    from scripts.exp12_launch_plan import allocation_matches

    resources = approval["resources"]
    per_node = {
        "cpus": resources["cpus_per_node"],
        "memory_gb": resources["memory_gb_per_node"],
        "wall_minutes": resources["wall_minutes"],
    }
    allocation_matches(per_node, os.environ, job_description)
    if (
        len(visible) != 4
        or int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) != 4
        or int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != resources["nodes"]
        or len(os.sched_getaffinity(0)) < resources["cpus_per_node"]
    ):
        raise ValueError(
            "actual node/GPU allocation differs from approved A/B resources"
        )
    match = re.search(r"\bEndTime=(\S+)", job_description)
    if not match or match[1] in ("Unknown", "N/A"):
        raise ValueError("Slurm allocation has no verifiable end time")
    return datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S").timestamp()


def _source_preflight(queue, approval, queue_root, job_id):
    from utils.run_metadata import code_version

    require_pinned_stack()
    if sys.version_info[:3] != (3, 12, 13):
        raise ValueError("pinned Python 3.12.13 is required")
    if code_version() != {"commit": queue["commit"], "dirty": False}:
        raise ValueError("scheduler checkout differs from clean approved source")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=MAIN
    ).strip():
        raise ValueError("scheduler checkout contains changed or untracked files")
    import jax
    import jaxlib

    if (
        jax.__version__ != "0.4.34"
        or jaxlib.__version__ != "0.4.34"
        or jax.default_backend() != "gpu"
        or len(jax.devices()) != 4
        or not jax.local_devices(backend="cpu")
    ):
        raise ValueError("pinned CPU+CUDA backend policy is unavailable")
    hb_python = approval["resources"]["humanoid_bench_python"]
    if not Path(hb_python).is_file():
        raise ValueError("approved HumanoidBench interpreter is unavailable")
    reports = (
        Path(queue_root)
        / "preflight"
        / job_id
        / socket.gethostname()
        / uuid.uuid4().hex
    )
    reports.mkdir(parents=True)
    main_report, hb_report = (
        reports / "main_environment.json",
        reports / "hb_environment.json",
    )
    for executable, report, extra in (
        (sys.executable, main_report, []),
        (hb_python, hb_report, ["--main", str(main_report)]),
    ):
        command = [
            executable,
            "scripts/check_exp12_hb_environment.py",
            "--gpu-model",
            GPU_KIND,
            *extra,
            "--out",
            str(report),
        ]
        with (reports / (report.stem + ".log")).open("xb") as log:
            subprocess.run(
                command,
                cwd=MAIN,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


def _owner(job_id):
    return {
        "job_id": job_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "token": uuid.uuid4().hex,
        "claimed_at": time.time(),
    }


def _start(item, token, approval, queue_root, ledger, job_id):
    owner = _owner(job_id)
    commit = approval["expected_commit"]
    if not ledger.claim(
        item,
        owner,
        lambda row: validate_done(row, commit, queue_root),
    ):
        return None
    try:
        resumed = validate_resume(item, commit)
        attempt = (
            Path(queue_root) / "attempts" / item["kind"] / item["id"] / owner["token"]
        )
        attempt.mkdir(parents=True)
        _write_new(
            attempt / "start.json",
            {
                "id": item["id"],
                "kind": item["kind"],
                "job_id": job_id,
                "host": owner["host"],
                "gpu": str(token),
                "resumed": resumed,
                "commit": commit,
                "started_at": time.time(),
            },
        )
        command = list(item["row"]["command"])
        if item["row"]["group"] == "humanoid_bench":
            command[0] = approval["resources"]["humanoid_bench_python"]
            if not Path(command[0]).is_file():
                raise ValueError("approved HumanoidBench interpreter is unavailable")
        else:
            command[0] = sys.executable
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(token)
        if item["row"]["group"] == "humanoid_bench":
            env["MUJOCO_GL"] = "egl"
            env["PYOPENGL_PLATFORM"] = "egl"
        env["EXP12_JAX_CACHE_DIR"] = str(attempt / "jax_cache")
        env["WANDB_DIR"] = str(attempt)
        logfile = (attempt / "training.log").open("xb")
        try:
            process = subprocess.Popen(
                command,
                cwd=MAIN,
                env=env,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            logfile.close()
        return {
            "item": item,
            "gpu": token,
            "owner": owner,
            "attempt": attempt,
            "process": process,
            "profile": approval["admission"]["architectures"][
                item["row"]["architecture"]
            ],
            "point": checkpoint_progress(item),
            "rate": None,
        }
    except Exception as exc:
        ledger.finish(item, owner, error=f"startup: {type(exc).__name__}: {exc}")
        return None


def _finish(job, ledger, commit, queue_root):
    process, item, attempt = job["process"], job["item"], job["attempt"]
    code = process.poll()
    if code is None:
        return False
    _write_new(attempt / "exit.json", {"returncode": code, "ended_at": time.time()})
    if code:
        ledger.finish(item, job["owner"], error=f"exit {code}; inspect {attempt}")
        return True
    try:
        receipt = validate_done(item, commit, queue_root)
        if receipt is None:
            raise ValueError("exit zero without validated terminal DONE")
    except Exception as exc:
        ledger.finish(
            item, job["owner"], error=f"validation: {type(exc).__name__}: {exc}"
        )
        return True
    ledger.finish(item, job["owner"], receipt=receipt)
    return True


def _exp2_m(approval, state):
    """The frozen m in force: allocation approval or Exp2 release; never two values."""
    values = set()
    if approval.get("exp2", {}).get("approved") is True:
        values.add(approval["exp2"]["injection_m"])
    if state["exp2"] is not None:
        values.add(state["exp2"]["injection_m"])
    if len(values) > 1:
        raise ValueError("allocation approval and Exp2 release name different m")
    return values.pop() if values else None


def run(queue_root, approval_path, allocation):
    queue_root, approval_path = (
        Path(queue_root).resolve(),
        Path(approval_path).resolve(),
    )
    if sha(approval_path) != os.environ.get("EXP12_APPROVAL_SHA256"):
        raise ValueError("allocation approval changed after submission")
    queue = load_queue(queue_root)
    if os.environ.get("EXPECTED_COMMIT") != queue["commit"]:
        raise ValueError("submission expected commit differs from queue source")
    approval = _read(approval_path)
    resources, admission = validate_approval(approval, queue_root, allocation)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("dynamic scheduler requires a real Slurm allocation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    description = subprocess.check_output(
        ["scontrol", "show", "job", "--oneliner", job_id], text=True
    )
    deadline = _actual_allocation(approval, description, visible)
    for token in visible:
        gpu_snapshot(token)
    _source_preflight(queue, approval, queue_root, job_id)
    ledger = Ledger(queue_root)
    # Optional: keep polling for releases while nothing is eligible (owner-approved
    # GPU idling traded for not re-queueing); 0 = exit when no work is eligible.
    idle_wait = admission.get("idle_wait_minutes", 0)
    if type(idle_wait) is not int or idle_wait < 0:
        raise ValueError("idle_wait_minutes must be a non-negative integer")
    idle_since = None
    active = []
    validating = {}
    discovered_arms = []
    discovered_once = False
    discovery_future = None
    final_discovery_requested = False
    final_discovery_done = False
    next_discovery = 0.0
    seen_this_allocation = set()
    stop_at = deadline - admission["stop_margin_minutes"] * 60
    with (
        ThreadPoolExecutor(max_workers=4) as validator,
        ThreadPoolExecutor(max_workers=1) as discoverer,
    ):
        while time.time() < stop_at or active or validating:
            parent_plan = _read(queue["plan"])
            if (
                sha(approval_path) != os.environ["EXP12_APPROVAL_SHA256"]
                or sha(queue_root / "queue.json") != approval["queue_sha256"]
                or sha(queue["plan"]) != queue["plan_sha256"]
                or sha(parent_plan["approval"]) != parent_plan["approval_sha256"]
            ):
                raise ValueError(
                    "source-bound approval, queue or plan changed during allocation"
                )
            for job in list(active):
                if job["process"].poll() is not None:
                    active.remove(job)
                    future = validator.submit(
                        _finish, job, ledger, queue["commit"], queue_root
                    )
                    validating[future] = job
            for future in list(validating):
                if future.done():
                    future.result()
                    del validating[future]
            if time.time() >= stop_at:
                if not active and not validating:
                    break
                time.sleep(POLL_SECONDS)
                continue
            update_rates(active)
            state = release_state(queue_root, queue)
            parents = effective_parents(queue, state)
            injection_m = _exp2_m(approval, state)
            if injection_m is not None:
                if discovery_future is not None and discovery_future.done():
                    discovered_arms = discovery_future.result()
                    discovered_once = True
                    discovery_future = None
                    if final_discovery_requested:
                        final_discovery_done = True
                parent_work_remaining = any(
                    ledger.state(parent) == "pending" for parent in parents
                )
                if parent_work_remaining:
                    final_discovery_requested = False
                    final_discovery_done = False
                if discovery_future is None and (
                    (parent_work_remaining and time.time() >= next_discovery)
                    or not discovered_once
                    or (not parent_work_remaining and not final_discovery_done)
                ):
                    discovery_future = discoverer.submit(
                        discover_arms, queue_root, queue, injection_m, parents
                    )
                    next_discovery = time.time() + POLL_SECONDS
                    if not parent_work_remaining:
                        final_discovery_requested = True
            items = discovered_arms + parents
            pending = sorted(
                (
                    item
                    for item in items
                    if item["id"] not in seen_this_allocation
                    and not run_blockers(item, state, injection_m is not None)
                    and ledger.state(item) == "pending"
                ),
                key=lambda item: priority(item, admission),
            )
            started = False
            for token in visible:
                snapshot = gpu_snapshot(token)
                for item in pending:
                    if not can_admit(
                        item,
                        token,
                        active,
                        admission,
                        resources,
                        deadline,
                        snapshot,
                        available_host_gb(),
                        validating.values(),
                    ):
                        continue
                    job = _start(item, token, approval, queue_root, ledger, job_id)
                    if job is not None:
                        seen_this_allocation.add(item["id"])
                        active.append(job)
                        pending.remove(item)
                        started = True
                        break
            if not active and not pending and not validating:
                if discovery_future is not None:
                    time.sleep(1)
                    continue
                idle_since = idle_since or time.time()
                if time.time() - idle_since >= 60 * idle_wait:
                    break
            else:
                idle_since = None
            time.sleep(1 if started else POLL_SECONDS)
    return 0


def queue_status(queue_root):
    """Read-only census: eligibility, ledger state and blockers of every run."""
    queue_root = Path(queue_root).resolve()
    queue = load_queue(queue_root)
    state = release_state(queue_root, queue)
    parents = effective_parents(queue, state)
    approval = _plan_approval(queue)
    exp2_m = state["exp2"]["injection_m"] if state["exp2"] else None
    ledger = Ledger(queue_root)
    arms = [_read(p) for p in sorted((queue_root / "arms").glob("*.json"))] if (
        queue_root / "arms"
    ).is_dir() else []
    rows, counts = [], {}
    for item in parents + arms:
        blockers = run_blockers(item, state, exp2_m is not None)
        status = ledger.state(item)
        if status == "pending" and (ledger._row_dir(item) / "claim").exists():
            status = "claimed"
        if status == "pending":
            status = "blocked_by_prerequisite" if blockers else "eligible"
        counts[(item["kind"], status)] = counts.get((item["kind"], status), 0) + 1
        rows.append(
            {
                "id": item["id"],
                "kind": item["kind"],
                "run_key": item["row"]["run_key"],
                "architecture": item["row"]["architecture"],
                "environment": item["row"]["environment"],
                "status": status,
                "blockers": blockers,
            }
        )
    forks_ready = sum(
        (Path(p["row"]["checkpoint_dir"]) / "fork" / "FORK_READY").exists() for p in parents
    )
    return {
        "commit": queue["commit"],
        "release_mode": approval.get("release_mode", "all_gates"),
        "releases": state["releases"],
        "a2_staged_identity": state["a2"],
        "identity_cases_satisfied": sorted(f"{a}/{s}" for a, s in state["identity"]),
        "exp2_injection_m": exp2_m,
        "forks_ready": forks_ready,
        "counts": {f"{k}:{v}": n for (k, v), n in sorted(counts.items())},
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--plan", required=True)
    prepare_cmd.add_argument("--queue", required=True)
    command = sub.add_parser("allocation-command")
    command.add_argument("--queue", required=True)
    command.add_argument("--approval", required=True)
    command.add_argument("--allocation", choices=tuple(ALLOCATIONS), required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--queue", required=True)
    worker.add_argument("--approval", required=True)
    worker.add_argument("--allocation", choices=tuple(ALLOCATIONS), required=True)
    retry = sub.add_parser("review-retry")
    retry.add_argument("--queue", required=True)
    retry.add_argument("--id", required=True)
    retry.add_argument("--decision", required=True)
    release = sub.add_parser("release")
    release.add_argument("--queue", required=True)
    release.add_argument("--decision", required=True)
    status = sub.add_parser("status")
    status.add_argument("--queue", required=True)
    status.add_argument("--rows", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args.plan, args.queue)
    elif args.action == "allocation-command":
        print(allocation_command(args.queue, args.approval, args.allocation))
    elif args.action == "release":
        print(publish_release(args.queue, args.decision))
    elif args.action == "status":
        report = queue_status(args.queue)
        if not args.rows:
            report.pop("rows")
        print(json.dumps(report, indent=2))
    elif args.action == "review-retry":
        queue = load_queue(args.queue)
        items = effective_parents(queue, release_state(args.queue, queue)) + [
            _read(path) for path in (Path(args.queue) / "arms").glob("*.json")
        ]
        matches = [item for item in items if item["id"] == args.id]
        if len(matches) != 1:
            raise ValueError("retry identity is missing or ambiguous")
        Ledger(args.queue).review_retry(matches[0], queue["commit"], args.decision)
    else:
        return run(args.queue, args.approval, args.allocation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
