"""Bounded read-only in-wave bank gate; never creates a bank or submits a job."""

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


class BankGateError(ValueError):
    """A producer/bank cannot safely release its consumers."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.asarray(value)
    return hashlib.sha256(
        json.dumps([value.dtype.str, list(value.shape)]).encode() + value.tobytes()
    ).hexdigest()


def verify_bank(path, expected, pool_size, normalized):
    """Verify committed payload and producer normalization identity, without fits."""
    path = Path(path)
    receipt = json.loads((path / "receipt.json").read_text())
    if set(receipt["files"]) != {"bank.npz", "teachers.msgpack"}:
        raise BankGateError("incomplete bank payload")
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise BankGateError("bank source/runtime/teacher/environment identity mismatch")
    for name, value in receipt["files"].items():
        if (path / name).is_symlink() or sha(path / name) != value:
            raise BankGateError("bank file hash mismatch")
    with np.load(path / "bank.npz", allow_pickle=False) as data:
        for name, value in receipt["array_hashes"].items():
            if array_hash(data[name]) != value:
                raise BankGateError("bank array hash mismatch")
        if set(receipt["array_hashes"]) != {
            "raw_observations",
            "observations",
            "actions",
            "replay_indices",
            "base",
        }:
            raise BankGateError("incomplete array provenance")
        for name in ("observations", "actions", "base"):
            if data[name].dtype != np.float32 or not np.isfinite(data[name]).all():
                raise BankGateError("finite float32 bank required")
        if (
            data["base"].shape != (8, pool_size)
            or data["observations"].ndim != 2
            or data["actions"].ndim != 2
            or len(data["observations"]) != pool_size
            or len(data["actions"]) != pool_size
            or data["raw_observations"].shape != data["observations"].shape
            or data["replay_indices"].shape != (pool_size,)
            or data["replay_indices"].dtype.kind not in "iu"
            or np.any(data["replay_indices"] < 0)
            or np.any(data["replay_indices"] >= receipt["num_filled"])
            or data["keys"].shape != (8, 2)
        ):
            raise BankGateError("bank pool/target/index shape mismatch")
        if normalized:
            rms = receipt["rms"]
            if not isinstance(rms, dict) or set(rms) != {
                "mean",
                "var",
                "count",
                "epsilon",
            }:
                raise BankGateError("missing frozen normalization identity")
            for name, value in rms.items():
                if data["rms_" + name].tolist() != value:
                    raise BankGateError("normalizer receipt/array identity mismatch")
            if np.any(data["rms_var"] < 0) or not float(data["rms_epsilon"]) > 0:
                raise BankGateError("invalid normalizer")
        elif receipt["rms"] is not None:
            raise BankGateError("unexpected normalization")
    return sha(path / "receipt.json")


def wait_for_bank(
    path,
    *,
    expected,
    pool_size,
    normalized,
    timeout_seconds,
    allocation_deadline_unix,
    producer_status,
    audit_dir,
    _monotonic=time.monotonic,
    _wall=time.time,
    _sleep=time.sleep,
):
    """Wait before trainer construction/seeding; count waits inside allocation cap.

    Delta infrastructure must atomically publish a source/environment/seed-bound
    producer status and report producer death. Missing status is never permission:
    absent bank/status still times out. Existing invalid/partial banks fail now.
    Audit attempts are distinct and retained, including failures. Private clock
    injection supports bounded CPU tests; production uses real monotonic time.
    """
    started = _monotonic()
    audit = {"status": "STARTING", "bank": str(path), "expected": expected}
    directory = Path(audit_dir)
    if not directory.is_absolute() or not Path(path).is_absolute():
        raise BankGateError("absolute bank and audit paths required")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"attempt_{time.time_ns()}_{os.getpid()}.json"
    try:
        remaining = float(allocation_deadline_unix) - _wall()
        if (
            not math.isfinite(remaining)
            or not 0 < remaining <= 21600
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(float(timeout_seconds))
            or not 0 < float(timeout_seconds) <= remaining
            or not Path(producer_status).is_absolute()
        ):
            raise BankGateError(
                "explicit finite timeout within six-hour allocation required"
            )
        deadline = started + float(timeout_seconds)
        audit.update(
            timeout_seconds=timeout_seconds,
            allocation_deadline_unix=allocation_deadline_unix,
        )
        while True:
            if _monotonic() >= deadline or _wall() >= allocation_deadline_unix:
                raise BankGateError("bounded bank readiness timeout")
            status_path = Path(producer_status)
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if any(
                    status.get(k) != expected[k]
                    for k in ("source", "environment", "seed")
                ):
                    raise BankGateError("producer status identity mismatch")
                state = status.get("status")
                if state not in {
                    "STARTING",
                    "RUNNING",
                    "BANK_READY",
                    "COMPLETED",
                    "FAILED",
                }:
                    raise BankGateError("unknown producer status")
                if state == "FAILED":
                    raise BankGateError("producer failed; consumer refused")
                if state == "COMPLETED" and not Path(path).exists():
                    raise BankGateError("producer completed without committed bank")
            if Path(path).exists():
                receipt_sha = verify_bank(path, expected, pool_size, normalized)
                if _monotonic() >= deadline or _wall() >= allocation_deadline_unix:
                    raise BankGateError("deadline exceeded while verifying bank")
                if status_path.exists():
                    latest = json.loads(status_path.read_text())
                    if latest.get("status") == "FAILED":
                        raise BankGateError("producer failed during verification")
                    if any(
                        latest.get(k) != expected[k]
                        for k in ("source", "environment", "seed")
                    ):
                        raise BankGateError(
                            "producer identity changed during verification"
                        )
                audit.update(status="BANK_VERIFIED", receipt_sha256=receipt_sha)
                return receipt_sha
            _sleep(max(0.0, min(0.25, deadline - _monotonic())))
    except BaseException as exc:
        audit.update(status="FAILED", error=repr(exc))
        raise
    finally:
        audit["wait_seconds"] = _monotonic() - started
        target.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
