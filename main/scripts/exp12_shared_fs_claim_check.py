#!/usr/bin/env python3
"""CPU-only, two-host check of the dynamic scheduler's shared-filesystem primitives.

Diagnostic only: it never imports JAX, never touches a GPU, never runs training and never
writes outside its own new --root. Two workers, started on two *different* hosts that mount
the same filesystem (two Delta login nodes, or two CPU-partition tasks), race on many
disposable rows with the real ``Ledger.claim`` and ``_write_new``, and on bare hard-link
creation. ``verify`` then checks, for every row, that exactly one worker won, that the
published bytes and inode belong to the winner, that no transition guard or temporary file
is left behind, and that recovering an ended owner preserves the old claim. A handoff phase
then has each host finish rows it owns while the other keeps trying to claim them; a stale
cross-host view of ``completed.json`` would let the second host claim a finished row.

It replaces only the shared-mount part of the two-node GPU preflight. GPU/CPU binding is
already checked fail-closed when each production allocation starts (``_actual_allocation``,
``_source_preflight``); multi-node ``srun``/``scontrol`` behaviour is not covered here.
"""

import argparse
import json
import os
import socket
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp12_dynamic_scheduler import Ledger, _read, _write_new  # noqa: E402

DEFAULT_ITEMS = 500
READY_TIMEOUT_SECONDS = 600
START_DELAY_SECONDS = 3.0
CLOCK_SKEW_WARN_SECONDS = 60.0  # report only: the scheduler's stale-guard warning is at 600 s
DUMMY_JOB = "1"  # a well-formed Slurm id for the disposable owners; liveness is injected
HANDOFF_FRACTION = 5  # one handoff row per this many race rows
HANDOFF_TIMEOUT_SECONDS = 300


def _item(root, index):
    return {
        "id": f"claimcheck_{index:04d}",
        "kind": "exp1",
        "row": {"config_hash": "diagnostic_only", "checkpoint_dir": str(root / "no_run")},
    }


def _handoff_item(root, index):
    """A row one host claims and finishes while the other keeps trying to claim it."""
    run = root / "handoff_runs" / f"{index:04d}"
    return {
        "id": f"handoff_{index:04d}",
        "kind": "exp1",
        "row": {"config_hash": "diagnostic_only", "checkpoint_dir": str(run)},
    }


def _prepare_handoff(root, index, token):
    run = Path(_handoff_item(root, index)["row"]["checkpoint_dir"])
    (run / "state").mkdir(parents=True)
    for name in ("DONE", "run_metadata.json", "state/LATEST"):
        (run / name).write_text(token)
    receipt = run / "receipt.json"
    receipt.write_text(json.dumps({"token": token}))
    return receipt


def _wait_for(path, until, message):
    while not path.exists():
        if time.monotonic() >= until:
            raise TimeoutError(message)
        time.sleep(0.05)


def _mount(path):
    """Mount point, filesystem type and options of the filesystem holding ``path``."""
    path = os.path.realpath(path)
    best = None
    try:
        lines = Path("/proc/self/mounts").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            continue
        point = fields[1].replace("\\040", " ")
        if (path == point or path.startswith(point.rstrip("/") + "/")) and (
            best is None or len(point) > len(best["mount_point"])
        ):
            best = {"device": fields[0], "mount_point": point, "fstype": fields[2],
                    "options": fields[3]}
    return best


def _clock(root, rank):
    """This host's clock against the filesystem's timestamp for a file it just created."""
    probe = root / f"clock_{rank}_{uuid.uuid4().hex}"
    before = time.time()
    probe.write_text("")
    after = time.time()
    mtime = probe.stat().st_mtime
    probe.unlink()
    return {"host_time": after, "fs_minus_host_seconds": mtime - (before + after) / 2}


def worker(root, rank, items=DEFAULT_ITEMS, host=None, ready_timeout=READY_TIMEOUT_SECONDS):
    root = Path(root)
    host = host or socket.gethostname()
    if rank not in (0, 1):
        raise ValueError("rank must be 0 or 1")
    if sys.modules.get("jax") is not None:
        raise RuntimeError("the shared-filesystem check must not import JAX")
    owner_token = uuid.uuid4().hex
    _write_new(root / f"ready_{rank}.json", {
        "rank": rank, "host": host, "pid": os.getpid(), "token": owner_token,
        "python": sys.version.split()[0], "mount": _mount(root), "clock": _clock(root, rank),
        "items": items,
    })
    until = time.monotonic() + ready_timeout
    while not (root / f"ready_{1 - rank}.json").exists():
        if time.monotonic() >= until:
            raise TimeoutError("the other host never reached the shared filesystem")
        time.sleep(0.2)
    other = _read(root / f"ready_{1 - rank}.json")
    if other["items"] != items:
        raise ValueError("the two workers were started with different --items")
    # Handoff rows this host owns are claimed before the start, while nobody else touches them.
    handoffs = max(1, items // HANDOFF_FRACTION)
    owner = {"job_id": DUMMY_JOB, "token": owner_token, "host": host, "pid": os.getpid(), "rank": rank}
    ledger = Ledger(root, query=lambda _: True)  # both owners are alive throughout
    receipts = {}
    for index in range(rank, handoffs, 2):
        receipts[index] = _prepare_handoff(root, index, owner_token)
        if not ledger.claim(_handoff_item(root, index), owner, lambda _: None):
            raise RuntimeError(f"could not pre-claim handoff row {index}")
    _write_new(root / f"handoff_ready_{rank}.json", {"rows": sorted(receipts)})
    # Rank 0 publishes one common start time; both spin to it so the races overlap.
    if rank == 0:
        _write_new(root / "start.json", {"start_at": time.time() + START_DELAY_SECONDS})
    while not (root / "start.json").exists():
        if time.monotonic() >= until:
            raise TimeoutError("rank 0 never published the start time")
        time.sleep(0.05)
    start_at = _read(root / "start.json")["start_at"]
    while time.time() < start_at:
        time.sleep(0.001)
    rows = []
    started_at = time.time()
    links = root / "links"
    links.mkdir(exist_ok=True)
    for index in range(items):
        source = links / f"source_{index:04d}_{rank}"
        source.write_text(owner_token)
        try:
            os.link(source, links / f"winner_{index:04d}")
            linked = True
        except FileExistsError:
            linked = False
        claimed = ledger.claim(_item(root, index), owner, lambda _: None)
        rows.append({"index": index, "linked": linked, "claimed": claimed})
    finished_at = time.time()
    handoff = _handoff(root, rank, handoffs, ledger, owner, receipts, until)
    _write_new(root / f"result_{rank}.json", {
        "rank": rank, "host": host, "token": owner_token, "rows": rows,
        "started_at": started_at, "finished_at": finished_at, "handoff": handoff,
    })


def _handoff(root, rank, handoffs, ledger, owner, receipts, until):
    """This host finishes its own rows (completed.json, claim retired) while it keeps trying to
    claim the other host's rows. Under a consistent view each attempt sees a live claim, an
    occupied guard or completed.json; a stale cache could let it claim a finished row."""
    _write_new(root / f"handoff_go_{rank}.json", {})
    _wait_for(root / f"handoff_go_{1 - rank}.json", until, "the other host never reached the handoff")
    deadline = time.monotonic() + HANDOFF_TIMEOUT_SECONDS
    mine = sorted(receipts)
    theirs = [i for i in range(1 - rank, handoffs, 2)]
    pending = set(theirs)
    wrong, errors, attempts, finish_conflicts = [], [], 0, 0
    while mine or pending:
        if mine:
            try:
                ledger.finish(_handoff_item(root, mine[0]), owner, receipt=receipts[mine[0]])
                mine.pop(0)
            except ValueError as exc:
                # The other host holds this row's guard for an instant. The production scheduler
                # does not retry here (its worker loop re-raises); the check counts and retries.
                if "transition guard is occupied" not in str(exc):
                    raise
                finish_conflicts += 1
        for index in sorted(pending):
            item = _handoff_item(root, index)
            attempts += 1
            try:
                if ledger.claim(item, owner, lambda _: None):
                    wrong.append(index)
                    pending.discard(index)
                    continue
            except Exception as exc:  # the production worker backs off on these
                errors.append(f"{index}: {type(exc).__name__}: {exc}")
            if (ledger._row_dir(item) / "completed.json").exists():
                pending.discard(index)
        if time.monotonic() >= deadline:
            raise TimeoutError("the other host's handoff rows never completed")
    return {"finished": sorted(receipts), "wrongly_claimed": wrong, "errors": errors,
            "attempts": attempts, "finish_guard_conflicts": finish_conflicts}


def verify(root, require_distinct_hosts=True):
    root = Path(root)
    ready = [_read(root / f"ready_{rank}.json") for rank in (0, 1)]
    results = [_read(root / f"result_{rank}.json") for rank in (0, 1)]
    failures, warnings = [], []
    hosts = [r["host"] for r in results]
    if require_distinct_hosts and hosts[0] == hosts[1]:
        failures.append(f"both workers ran on the same host {hosts[0]}; cross-host behaviour is untested")
    items = ready[0]["items"]
    if any(len(r["rows"]) != items for r in results):
        failures.append("a worker did not record every row")
    tokens = {r["rank"]: r["token"] for r in results}
    ledger = Ledger(root, query=lambda _: True)
    for index in range(items):
        linked = [r["rank"] for r in results if r["rows"][index]["linked"]]
        claimed = [r["rank"] for r in results if r["rows"][index]["claimed"]]
        if len(linked) != 1:
            failures.append(f"row {index}: {len(linked)} hard-link winners")
        else:
            winner = root / "links" / f"winner_{index:04d}"
            source = root / "links" / f"source_{index:04d}_{linked[0]}"
            if winner.read_text() != tokens[linked[0]] or (
                winner.stat().st_dev, winner.stat().st_ino
            ) != (source.stat().st_dev, source.stat().st_ino):
                failures.append(f"row {index}: hard link does not point at its winner's file")
        if len(claimed) != 1:
            failures.append(f"row {index}: {len(claimed)} ledger claim winners")
        else:
            row = ledger._row_dir(_item(root, index))
            claim = row / "claim"
            if not (claim / "ready").is_file():
                failures.append(f"row {index}: winning claim is not marked ready")
            elif _read(claim / "owner.json")["token"] != tokens[claimed[0]]:
                failures.append(f"row {index}: claim owner differs from the reported winner")
            if (row / "transition_guard").exists():
                failures.append(f"row {index}: a transition guard was left behind")
            if (row / "blocked.json").exists() or (row / "completed.json").exists():
                failures.append(f"row {index}: unexpected blocked/completed record")
    handoffs = max(1, items // HANDOFF_FRACTION)
    for r in results:
        handoff = r["handoff"]
        if handoff["wrongly_claimed"]:
            failures.append(f"host {r['host']} claimed finished rows {handoff['wrongly_claimed']}")
        if handoff["errors"]:
            warnings.append(f"host {r['host']}: {len(handoff['errors'])} claim attempts raised "
                            f"(stale view; production backs off): {handoff['errors'][:3]}")
    for index in range(handoffs):
        row = ledger._row_dir(_handoff_item(root, index))
        completed = row / "completed.json"
        if not completed.is_file():
            failures.append(f"handoff row {index}: not completed")
            continue
        if _read(completed)["attempt_token"] != tokens[index % 2] or (row / "claim").exists():
            failures.append(f"handoff row {index}: completed by the wrong owner or claim not retired")
    leftovers = [str(p) for p in root.rglob("*.tmp")]
    if leftovers:
        failures.append(f"temporary files left behind: {leftovers[:5]}")
    # Recovery of an ended owner, on the shared mount, for one disposable row.
    recovery = {"job_id": DUMMY_JOB, "token": uuid.uuid4().hex, "host": socket.gethostname(),
                "pid": os.getpid()}
    first = _item(root, 0)
    if not Ledger(root, query=lambda _: False).claim(first, recovery, lambda _: None):
        failures.append("recovery of an ended owner's claim failed on the shared mount")
    else:
        row = ledger._row_dir(first)
        retired = list((row / "retired_claims").iterdir())
        if len(retired) != 1 or not (retired[0] / "owner.json").is_file():
            failures.append("the ended owner's claim was not preserved under retired_claims")
        if _read(row / "claim" / "owner.json")["token"] != recovery["token"]:
            failures.append("the recovered claim does not belong to the recovering owner")
    # The races only test contention if the two workers' loops actually overlapped in time.
    overlap = min(r["finished_at"] for r in results) - max(r["started_at"] for r in results)
    if overlap <= 0:
        failures.append("the two workers' race loops did not overlap; contention was not tested")
    mounts = [r.get("mount") for r in ready]
    if None in mounts:
        warnings.append("mount information unavailable on a host")
    elif (mounts[0]["fstype"], mounts[0]["device"]) != (mounts[1]["fstype"], mounts[1]["device"]):
        failures.append("the two hosts mount the root from different filesystems")
    elif mounts[0]["options"] != mounts[1]["options"]:
        warnings.append("the two hosts use different mount options")
    skews = [abs(r["clock"]["fs_minus_host_seconds"]) for r in ready]
    if max(skews) > CLOCK_SKEW_WARN_SECONDS:
        warnings.append(f"host/filesystem clock offset up to {max(skews):.1f} s")
    report = {
        "verdict": "FAIL" if failures else "PASS",
        "scope": "cpu_shared_filesystem_claims",
        "hosts": hosts,
        "items": items,
        "race_overlap_seconds": overlap,
        "handoff_rows": handoffs,
        "handoff_attempts": [r["handoff"]["attempts"] for r in results],
        "finish_guard_conflicts": [r["handoff"]["finish_guard_conflicts"] for r in results],
        "mounts": mounts,
        "fs_minus_host_seconds": [r["clock"]["fs_minus_host_seconds"] for r in ready],
        "warnings": warnings,
        "failures": failures,
        "not_covered": [
            "GPU/CPU binding (checked when each production allocation starts)",
            "multi-node srun task placement and scontrol parsing of a two-node allocation",
            "compute-node mount options, unless both workers ran on compute nodes",
        ],
    }
    _write_new(root / "verdict.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=("worker", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=None,
                        help="0 or 1 (default: SLURM_PROCID)")
    parser.add_argument("--items", type=int, default=DEFAULT_ITEMS)
    args = parser.parse_args()
    if not args.root.is_absolute() or not args.root.is_dir():
        parser.error("root must be an existing absolute directory created for this check")
    if args.items < 1:
        parser.error("--items must be positive")
    if args.role == "worker":
        rank = args.rank if args.rank is not None else int(os.environ["SLURM_PROCID"])
        worker(args.root, rank, args.items)
    else:
        report = verify(args.root)
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
