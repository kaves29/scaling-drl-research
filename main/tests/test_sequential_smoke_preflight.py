"""Synthetic CPU bank receipts test staging integrity, not smoke qualification."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from diagnostics import sequential_probe as probe
from scripts import sequential_smoke_preflight as gate
from tests.test_sequential_probe import Tiny


@pytest.fixture
def plan(tmp_path):
    result = {
        "base_sha": gate.BASE,
        "source_sha": "0" * 40,
        "approved": False,
        "executable": False,
        "submitted": False,
        "teacher_architecture": "synthetic-tiny",
        "teacher_definition": repr(Tiny()),
        "target_scale": 1e5,
        "teacher_eval_chunk": 2,
        "pool_size": 4,
        "warmup": 2,
        "endpoint": 8,
        "banks": {},
        "jobs": [
            dict(id=k, environment=e, architecture=a, bank_producer=p, seed=0)
            for k, (e, a, p) in gate.CELLS.items()
        ],
    }
    for environment in ("dog-run", "myo-pen-twirl"):
        bank = tmp_path / environment / "bank"
        receipt = probe.create_bank(
            bank,
            observations=np.array([[0, 1], [1, 0]], np.float32),
            actions=np.array([[0], [1]], np.float32),
            num_filled=2,
            warmup=2,
            rms=None,
            environment=environment,
            seed=0,
            teacher=Tiny(),
            teacher_architecture=result["teacher_architecture"],
            pool_size=4,
            teacher_chunk=2,
            source=result["source_sha"],
        )
        result["expected_execution"] = receipt["execution"]
        identity = bank.parent / "identity"
        identity.mkdir()
        probe.seal(
            identity,
            {
                "identity": {
                    "git_commit": result["source_sha"],
                    "execution": receipt["execution"],
                    "config": {"num_interaction_steps": 8},
                    "alternative": dict(
                        source=result["source_sha"],
                        environment=environment,
                        seed=0,
                        architecture="D2W512",
                        bank_producer=True,
                        bank=str(bank),
                    ),
                }
            },
        )
        done = bank.parent / "ALTERNATIVE_DONE.json"
        done.write_text(json.dumps(dict(mode="alternative_only", interaction_step=8)))
        result["banks"][environment] = dict(
            path=str(bank),
            receipt_sha256=gate.sha(bank / "receipt.json"),
            identity_path=str(identity),
            identity_receipt_sha256=gate.sha(identity / "receipt.json"),
            done_path=str(done),
            done_sha256=gate.sha(done),
        )
    return result


def test_both_real_cpu_banks_pass_without_mutation_or_launch(plan):
    files = [
        p
        for block in plan["banks"].values()
        for p in Path(block["path"]).parent.rglob("*")
        if p.is_file()
    ]
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    result = gate.inspect_banks(plan)
    assert result["status"] == "BOTH_BANKS_VERIFIED_ONLY"
    assert not any(result[k] for k in ("approved", "executable", "submitted"))
    assert result["submission_allowed"] is False
    assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in files}


@pytest.mark.parametrize(
    "change",
    [
        "missing_bank",
        "missing_done",
        "tampered_bank",
        "unpinned_receipt",
        "wrong_source",
        "wrong_teacher",
        "wrong_runtime",
        "incomplete_producer",
        "wrong_seed",
        "missing_myo",
        "extra_cell",
        "launch_flag",
    ],
)
def test_stage_gate_refuses_incomplete_or_mismatched_evidence(plan, change):
    plan = deepcopy(plan)
    block = plan["banks"]["myo-pen-twirl"]
    if change == "missing_bank":
        Path(block["path"], "receipt.json").unlink()
    elif change == "missing_done":
        Path(block["done_path"]).unlink()
    elif change == "tampered_bank":
        with Path(block["path"], "bank.npz").open("ab") as f:
            f.write(b"corruption")
    elif change == "unpinned_receipt":
        block["receipt_sha256"] = None
    elif change == "wrong_source":
        plan["source_sha"] = "1" * 40
    elif change == "wrong_teacher":
        plan["teacher_definition"] = "other teacher"
    elif change == "wrong_runtime":
        plan["expected_execution"]["backend"] = "unqualified"
    elif change == "incomplete_producer":
        Path(block["done_path"]).write_text(
            json.dumps(dict(mode="alternative_only", interaction_step=4))
        )
        block["done_sha256"] = gate.sha(block["done_path"])
    elif change == "wrong_seed":
        plan["jobs"][0]["seed"] = 1
    elif change == "missing_myo":
        del plan["banks"]["myo-pen-twirl"]
    elif change == "extra_cell":
        plan["jobs"].append(dict(plan["jobs"][0]))
    elif change == "launch_flag":
        plan["executable"] = True
    with pytest.raises((ValueError, OSError)):
        gate.inspect_banks(plan)


def test_pending_review_plan_refuses_before_model_or_job_execution(tmp_path):
    path = Path(__file__).resolve().parents[1] / "docs/sequential_smoke_review.json"
    result = subprocess.run(
        [sys.executable, str(Path(gate.__file__)), "--plan", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "BLOCKED" in result.stderr
    assert list(tmp_path.iterdir()) == []
