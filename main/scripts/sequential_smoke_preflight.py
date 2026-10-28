"""Read-only two-bank staging gate. Never authorizes or submits a GPU job.

Run on a CPU/controller after BOTH complete producer jobs have exited. This is
not a CPU bank generator: bank bytes come only from the unmodified GPU producers.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = "844e3c0cc24998b25c3d1e667bcbfe056b1b336c"
CELLS = {
    "dog-small": ("dog-run", "D2W512", True),
    "dog-medium": ("dog-run", "D4W1024", False),
    "dog-large": ("dog-run", "D4W1536", False),
    "myo-small": ("myo-pen-twirl", "D2W512", True),
    "myo-large": ("myo-pen-twirl", "D4W1536", False),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path, expected):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("absolute, non-symlink committed directory required")
    if not expected or sha(path / "receipt.json") != expected:
        raise ValueError("missing or mismatched pinned receipt SHA256")
    receipt = json.loads((path / "receipt.json").read_text())
    if not isinstance(receipt.get("files"), dict):
        raise ValueError("missing file hash manifest")
    for name, value in receipt["files"].items():
        file = path / name
        if Path(name).name != name or file.is_symlink() or sha(file) != value:
            raise ValueError("sealed file hash mismatch")
    return receipt


def array_hash(value):
    value = np.asarray(value)
    return hashlib.sha256(
        json.dumps([value.dtype.str, list(value.shape)]).encode() + value.tobytes()
    ).hexdigest()


def inspect_banks(plan):
    """Verify immutable expected receipts; success grants no launch approval."""
    if plan["base_sha"] != BASE or any(
        plan.get(flag) is not False for flag in ("approved", "executable", "submitted")
    ):
        raise ValueError("review-only plan and approved base required")
    cells = plan["jobs"]
    actual = {
        row["id"]: (row["environment"], row["architecture"], row["bank_producer"])
        for row in cells
    }
    if len(cells) != 5 or actual != CELLS or any(row["seed"] != 0 for row in cells):
        raise ValueError("exact documented five seed-zero cells required")
    source, execution = plan["source_sha"], plan["expected_execution"]
    if not source or not execution or not plan["teacher_definition"]:
        raise ValueError("owner/source/runtime/teacher bindings are unresolved")
    if set(plan["banks"]) != {"dog-run", "myo-pen-twirl"}:
        raise ValueError("both environment banks required")
    evidence = {}
    for environment, block in plan["banks"].items():
        bank = sealed(block["path"], block["receipt_sha256"])
        required = {
            "environment": environment,
            "seed": 0,
            "source": source,
            "teacher_architecture": plan["teacher_architecture"],
            "teacher_definition": plan["teacher_definition"],
            "scale": plan["target_scale"],
            "teacher_eval_chunk": plan["teacher_eval_chunk"],
            "warmup": plan["warmup"],
            "num_filled": plan["warmup"],
            "execution": execution,
        }
        if any(bank.get(k) != v for k, v in required.items()):
            raise ValueError("bank source/teacher/warmup/runtime identity mismatch")
        if set(bank["files"]) != {"bank.npz", "teachers.msgpack"}:
            raise ValueError("incomplete bank payload")
        with np.load(Path(block["path"]) / "bank.npz", allow_pickle=False) as data:
            if data["base"].shape != (8, plan["pool_size"]):
                raise ValueError("eight-target pool shape mismatch")
            for name, expected in bank["array_hashes"].items():
                if array_hash(data[name]) != expected:
                    raise ValueError("bank numerical array hash mismatch")
            for name in ("observations", "actions", "base"):
                if data[name].dtype != np.float32 or not np.isfinite(data[name]).all():
                    raise ValueError("bank requires finite float32 arrays")
        identity = sealed(block["identity_path"], block["identity_receipt_sha256"])[
            "identity"
        ]
        alt, cfg = identity["alternative"], identity["config"]
        if (
            identity["git_commit"] != source
            or identity["execution"] != execution
            or alt["source"] != source
            or alt["environment"] != environment
            or alt["seed"] != 0
            or alt["architecture"] != "D2W512"
            or alt["bank_producer"] is not True
            or alt["bank"] != block["path"]
            or cfg["num_interaction_steps"] != plan["endpoint"]
        ):
            raise ValueError("producer identity mismatch")
        done_path = Path(block["done_path"])
        if (
            not done_path.is_absolute()
            or done_path.is_symlink()
            or not block["done_sha256"]
        ):
            raise ValueError("pinned producer completion required")
        if sha(done_path) != block["done_sha256"]:
            raise ValueError("producer completion hash mismatch")
        done = json.loads(done_path.read_text())
        if (
            done["mode"] != "alternative_only"
            or done["interaction_step"] != plan["endpoint"]
        ):
            raise ValueError("producer has not completed its prescribed trajectory")
        evidence[environment] = {
            "receipt_sha256": block["receipt_sha256"],
            "done_sha256": block["done_sha256"],
        }
    return {
        "status": "BOTH_BANKS_VERIFIED_ONLY",
        "approved": False,
        "executable": False,
        "submitted": False,
        "submission_allowed": False,
        "banks": evidence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    try:
        result = inspect_banks(json.loads(Path(args.plan).read_text()))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
