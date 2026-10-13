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
    cells,
    config_hash,
    postvalidate,
    require_pinned_stack,
    sha,
    submission_command,
)

WEIGHTS = {"D2W512": 1 / 7, "D4W1024": 1 / 4, "D4W1536": 1 / 2}
CEILINGS = {"D2W512": 7, "D4W1024": 4, "D4W1536": 2}
ALLOCATIONS = {"A": (1, 30 * 60), "B": (2, 48 * 60)}
# Every environment group of the grid (generate_manifest.EXP12_BUDGETS), dmc_medium included.
GROUPS = ("dmc_hard", "dmc_medium", "myosuite_simba", "humanoid_bench")
# squeue/sacct states whose processes may still run; anything else listed has ended.
SLURM_LIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED",
                     "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SIGNALING", "STAGE_OUT", "STOPPED"}
STALE_GUARD_SECONDS = 600
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
    # allocation_command redirects into these; sbatch never sees a missing directory.
    (queue_root / "slurm").mkdir()
    for name in ALLOCATIONS:
        (queue_root / "submissions" / name).mkdir(parents=True)
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


def validate_approval(approval, queue_root, allocation):
    queue = load_queue(queue_root)
    expected_nodes, expected_minutes = ALLOCATIONS[allocation]
    if (
        approval.get("approved") is not True
        or approval.get("allocation") != allocation
        or approval.get("expected_commit") != queue["commit"]
        or approval.get("queue_sha256") != sha(Path(queue_root) / "queue.json")
        or not approval.get("decision_reference")
    ):
        raise ValueError("explicit allocation/source/queue approval is missing")
    resources = approval.get("resources", {})
    if (
        resources.get("nodes") != expected_nodes
        or resources.get("gpus_per_node") != 4
        or resources.get("wall_minutes") != expected_minutes
    ):
        raise ValueError("allocation must be exactly A=4 A100/30h or B=8 A100/48h")
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
        if approval.get("exp2", {}).get("approved") is True and arch != "D2W512":
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
        control = _read(exp2["positive_control_evidence"])
        if (
            control.get("status") != "m_chosen"
            or control.get("chosen_m") != exp2["injection_m"]
            or control.get("source_code") != {"commit": queue["commit"], "dirty": False}
            or control.get("allow_any_setting") is not False
            or control.get("trigger_check", {}).get("forced") is not False
        ):
            raise ValueError("positive control does not qualify this final source/m")
    hb_python = resources.get("humanoid_bench_python")
    if not isinstance(hb_python, str) or not Path(hb_python).is_absolute():
        raise ValueError("approved absolute HumanoidBench Python path required")
    return resources, admission


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


def _job_states(text):
    return {line.strip().split()[0].rstrip("+") for line in text.splitlines() if line.strip()}


def slurm_alive(job):
    """True while the owning allocation may still run processes.

    squeue lists ended jobs for MinJobAge with terminal states and afterwards
    rejects the id ("Invalid job id"); both mean ended. Any other squeue failure
    falls back to sacct; if neither answers, liveness is unknown and recovery is
    refused (RuntimeError), never assumed."""
    result = subprocess.run(
        ["squeue", "-h", "-j", job, "-o", "%T"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode == 0:
        states = _job_states(result.stdout)
        if not states:
            return False
        return bool(states & SLURM_LIVE_STATES) or not states <= _TERMINAL_STATES
    if "invalid job id" in (result.stderr + result.stdout).lower():
        return False
    accounting = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", job, "-o", "State"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if accounting.returncode or not accounting.stdout.strip():
        raise RuntimeError("squeue and sacct failed; claim recovery refused")
    states = _job_states(accounting.stdout)
    return bool(states & SLURM_LIVE_STATES) or not states <= _TERMINAL_STATES


_TERMINAL_STATES = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
                    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "SPECIAL_EXIT", "TIMEOUT"}


_WARNED_GUARDS = set()
_CLAIM_BACKOFF = {}
CLAIM_BACKOFF_SECONDS = 300
# A transition (claim/finish) holds a row's guard for well under a second; claim's
# reconciliation can hash evidence under it for a few seconds. finish waits this long.
FINISH_GUARD_TIMEOUT_SECONDS = 120
FINISH_GUARD_POLL_SECONDS = (0.05, 2.0)  # first wait, maximum wait (doubling)
FINISH_RETRY_SECONDS = 60  # worker loop: wait before re-attempting an unrecorded finish


class LedgerContention(ValueError):
    """finish could not take the row's guard in time; nothing was recorded and the claim is kept."""


def _warn_if_stale_guard(guard, item):
    """Transitions last well under a second; an old guard is an orphan from a killed worker.
    It is never removed automatically (its owner may be alive); it is reported for review."""
    try:
        age = time.time() - guard.stat().st_mtime
    except FileNotFoundError:
        return
    if age > STALE_GUARD_SECONDS and item["id"] not in _WARNED_GUARDS:
        _WARNED_GUARDS.add(item["id"])
        print(f"[scheduler] WARNING: transition guard {guard} is {age:.0f} s old; row "
              f"{item['row'].get('run_key', item['id'])} stays pending until it is reviewed",
              file=sys.stderr, flush=True)


class Ledger:
    def __init__(self, root, query=slurm_alive, finish_timeout=FINISH_GUARD_TIMEOUT_SECONDS):
        self.root = Path(root).resolve() / "ledger"
        self.query = query
        self.finish_timeout = finish_timeout

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

    def _wait_for_guard(self, directory, item):
        """Bounded retry: another worker's claim attempt may hold the guard briefly. Never
        removes or overrides a guard; on timeout raises LedgerContention having written nothing."""
        deadline = time.monotonic() + self.finish_timeout
        wait, ceiling = FINISH_GUARD_POLL_SECONDS
        while True:
            guard = self._guard(directory)
            if guard is not None:
                return guard
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _warn_if_stale_guard(directory / "transition_guard", item)
                raise LedgerContention(
                    f"transition guard of {item['id']} stayed occupied for "
                    f"{self.finish_timeout} s; nothing recorded, claim kept"
                )
            time.sleep(min(wait, remaining))
            wait = min(2 * wait, ceiling)

    def claim(self, item, owner, reconcile):
        """Atomic cross-allocation claim. Never infer completion from DONE alone."""
        directory = self._row_dir(item)
        directory.mkdir(parents=True, exist_ok=True)
        guard = self._guard(directory)
        if guard is None:
            _warn_if_stale_guard(directory / "transition_guard", item)
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
        guard = self._wait_for_guard(directory, item)
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
            # Raises unless the run is resumable from a valid checkpoint or, for an Exp1
            # parent interrupted before its first checkpoint, restartable from scratch.
            if not validate_resume(item, commit) and item["kind"] != "exp1":
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
    from experiments.exp12.state import latest_state_dir as _latest

    if (
        item["kind"] == "exp1"
        and _latest(run / "state") is None
        and not (run / "fork" / "FORK_READY").exists()
    ):
        # Interrupted before its first committed checkpoint: exp1 restarts from scratch
        # deterministically (tests KillMatrixTest.test_killed_before_the_first_checkpoint_*).
        return False
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
        result = postvalidate(run, commit, receipt)
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


def discover_arms(queue_root, queue, injection_m):
    """Idempotently publish eligible arms while parents continue training."""
    from generate_manifest import EXP12_FORKING

    directory = Path(queue_root) / "arms"
    directory.mkdir(exist_ok=True)
    existing = {}
    parents_by_id = {parent["id"]: parent for parent in queue["parents"]}
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
    for parent in queue["parents"]:
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
    group = row["group"]
    estimates = (
        profile["estimated_arm_minutes_by_group"]
        if item["kind"] == "exp2_injected"
        else profile["estimated_minutes_by_group"]
    )
    if (
        time.time() + 60 * (estimates[group] + admission["stop_margin_minutes"])
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
    if _CLAIM_BACKOFF.get(item["id"], 0) > time.time():
        return None
    try:
        claimed = ledger.claim(item, owner, lambda row: validate_done(row, commit, queue_root))
    except Exception as exc:  # one unrecoverable row must not stop the allocation's worker
        _CLAIM_BACKOFF[item["id"]] = time.time() + CLAIM_BACKOFF_SECONDS
        print(f"[scheduler] WARNING: claim of {item['row'].get('run_key', item['id'])} refused: "
              f"{type(exc).__name__}: {exc}; retrying in {CLAIM_BACKOFF_SECONDS} s",
              file=sys.stderr, flush=True)
        return None
    if not claimed:
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
        try:
            ledger.finish(item, owner, error=f"startup: {type(exc).__name__}: {exc}")
        except LedgerContention as contention:
            # Fail closed: the claim stays with this live allocation, so no other worker
            # starts the row; once this allocation ends it is recovered by reconciliation.
            print(f"[scheduler] WARNING: startup failure of {item['id']} not recorded: "
                  f"{contention}", file=sys.stderr, flush=True)
        return None


def _finish(job, ledger, commit, queue_root):
    """Record a finished process once. The outcome is decided (and exit.json written) only on
    the first call; a call repeated after LedgerContention records that same outcome."""
    process, item, attempt = job["process"], job["item"], job["attempt"]
    if "outcome" not in job:
        code = process.poll()
        if code is None:
            return False
        _write_new(attempt / "exit.json", {"returncode": code, "ended_at": time.time()})
        if code:
            job["outcome"] = {"error": f"exit {code}; inspect {attempt}"}
        else:
            try:
                receipt = validate_done(item, commit, queue_root)
                if receipt is None:
                    raise ValueError("exit zero without validated terminal DONE")
                job["outcome"] = {"receipt": receipt}
            except Exception as exc:
                job["outcome"] = {"error": f"validation: {type(exc).__name__}: {exc}"}
    ledger.finish(item, job["owner"], **job["outcome"])
    return True


def _collect_finished(validating, unrecorded, now):
    """Collect completed _finish futures. LedgerContention keeps the job for a later attempt
    (nothing was recorded); any other error still stops the worker (fail closed)."""
    for future in list(validating):
        if not future.done():
            continue
        job = validating.pop(future)
        try:
            future.result()
        except LedgerContention as exc:
            job["retry_finish_at"] = now + FINISH_RETRY_SECONDS
            unrecorded.append(job)
            print(f"[scheduler] WARNING: {exc}; retrying in {FINISH_RETRY_SECONDS} s",
                  file=sys.stderr, flush=True)


def _abandon_unrecorded(unrecorded):
    """At the allocation's end: leave the claims (their owner, this allocation, then ends and
    a later allocation reconciles them) and keep a reviewable record of each outcome."""
    for job in unrecorded:
        outcome = {k: str(v) for k, v in job.get("outcome", {}).items()}
        _write_new(job["attempt"] / "finish_unrecorded.json",
                   {"id": job["item"]["id"], "outcome": outcome, "recorded_at": time.time()})
        print(f"[scheduler] ERROR: outcome of {job['item']['id']} could not be recorded before "
              f"the allocation ended; claim left for reconciliation", file=sys.stderr, flush=True)


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
    active = []
    validating = {}
    unrecorded = []  # finished processes whose ledger update met guard contention
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
        while time.time() < stop_at or active or validating or unrecorded:
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
            for job in [j for j in unrecorded if j["retry_finish_at"] <= time.time()]:
                unrecorded.remove(job)
                validating[validator.submit(_finish, job, ledger, queue["commit"], queue_root)] = job
            _collect_finished(validating, unrecorded, time.time())
            if time.time() >= stop_at:
                if not active and not validating:
                    if unrecorded and time.time() >= deadline - 60:
                        _abandon_unrecorded(unrecorded)
                        break
                    if not unrecorded:
                        break
                time.sleep(POLL_SECONDS)
                continue
            update_rates(active)
            if approval.get("exp2", {}).get("approved") is True:
                if discovery_future is not None and discovery_future.done():
                    discovered_arms = discovery_future.result()
                    discovered_once = True
                    discovery_future = None
                    if final_discovery_requested:
                        final_discovery_done = True
                parent_work_remaining = any(
                    ledger.state(parent) == "pending" for parent in queue["parents"]
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
                        discover_arms,
                        queue_root,
                        queue,
                        approval["exp2"]["injection_m"],
                    )
                    next_discovery = time.time() + POLL_SECONDS
                    if not parent_work_remaining:
                        final_discovery_requested = True
            items = discovered_arms + queue["parents"]
            pending = [
                item
                for item in items
                if item["id"] not in seen_this_allocation
                and ledger.state(item) == "pending"
            ]
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
            if not active and not pending and not validating and not unrecorded:
                if discovery_future is not None:
                    time.sleep(1)
                    continue
                break
            time.sleep(1 if started else POLL_SECONDS)
    return 0


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
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args.plan, args.queue)
    elif args.action == "allocation-command":
        print(allocation_command(args.queue, args.approval, args.allocation))
    elif args.action == "review-retry":
        queue = load_queue(args.queue)
        items = queue["parents"] + [
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
