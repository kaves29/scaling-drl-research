"""Revised six-cell diagnostic and bounded gate: CPU engineering checks only."""

from copy import deepcopy
import inspect
import json
from pathlib import Path
import random
import shutil
import time
from unittest.mock import patch

import numpy as np
import pytest

from diagnostics import sequential_bank_gate as gate
from diagnostics import sequential_probe as s
from experiments import exp1
from scripts import sequential_six_plan as plan_module
from tests.test_sequential_probe import package as make_package, run


@pytest.fixture
def package(tmp_path):
    return make_package.__wrapped__(tmp_path)


def review_plan():
    return json.loads(
        (
            Path(plan_module.__file__).parents[1] / "docs/sequential_six_review.json"
        ).read_text()
    )


def assert_even_outputs(output):
    trajectory = s.load_json(output / "trajectory.json")
    assert [r["K"] for r in trajectory["phases"]] == [2, 4, 6, 8]
    assert trajectory["fitted_AB_targets"] == list(range(1, 9))
    previous = None
    fits = 0
    for k in range(1, 9):
        phase = output / f"phase_{k:02d}"
        receipt = s.verify(phase)
        reported = k % 2 == 0
        assert receipt["reported"] == reported
        assert (receipt["summary"] is not None) == reported
        assert (phase / "C.npz").exists() == reported
        arms = "ABC" if reported else "AB"
        assert receipt["optimizer_initializations"] == {a: 1 for a in arms}
        arrays = {}
        for arm in arms:
            with np.load(phase / f"{arm}.npz") as data:
                arrays[arm] = {name: data[name].copy() for name in data.files}
            assert arrays[arm]["losses"].shape == (2000,)
            for name in ("targets", "indices"):
                np.testing.assert_array_equal(arrays[arm][name], arrays["A"][name])
            if arm in "AB" and previous:
                assert (
                    receipt["initial_parameter_hashes"][arm]
                    == previous["final_parameter_hashes"][arm]
                )
            fits += 1
        previous = receipt
    assert fits == 20


def test_even_schedule_same_AB_trajectory_and_only_four_fresh_C_fits(package, tmp_path):
    original = s._fit_literal
    with patch.object(s, "_fit_literal", wraps=original) as fitted:
        run(package, tmp_path / "even", schedule=s.EVEN_MILESTONES)
        assert fitted.call_count == 20
    assert_even_outputs(tmp_path / "even")
    run(package, tmp_path / "all")
    for k in range(1, 9):
        for arm in "AB":
            with np.load(
                tmp_path / "even" / f"phase_{k:02d}" / f"{arm}.npz"
            ) as a, np.load(tmp_path / "all" / f"phase_{k:02d}" / f"{arm}.npz") as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
    with patch.object(s, "_fit_literal", side_effect=AssertionError("committed refit")):
        run(package, tmp_path / "even", schedule=s.EVEN_MILESTONES)


@pytest.mark.parametrize("damage", ["C_on_odd", "K", "optimizer", "array"])
def test_even_resume_rejects_corruption(package, tmp_path, damage):
    output = tmp_path / "out"
    run(package, output, schedule=s.EVEN_MILESTONES)
    phase = output / "phase_01"
    if damage == "C_on_odd":
        shutil.copy2(phase / "A.npz", phase / "C.npz")
    elif damage == "array":
        with (phase / "A.npz").open("ab") as stream:
            stream.write(b"tampered")
    else:
        receipt = s.load_json(phase / "receipt.json")
        if damage == "K":
            receipt["K"] = 2
        else:
            receipt["optimizer_initializations"]["C"] = 1
        s.write_json(phase / "receipt.json", receipt)
    with pytest.raises(ValueError):
        run(package, output, schedule=s.EVEN_MILESTONES)


def test_even_mutation_guard_catches_skipped_AB_phase_and_extra_C(package, tmp_path):
    source = inspect.getsource(s.evaluate)
    changed = source.replace("for k in range(8):", "for k in range(1, 8, 2):")
    namespace = dict(vars(s))
    exec(compile(changed, s.__file__, "exec"), namespace)
    with patch.object(s, "evaluate", namespace["evaluate"]):
        run(package, tmp_path / "skipped", schedule=s.EVEN_MILESTONES)
        with pytest.raises((AssertionError, OSError)):
            assert_even_outputs(tmp_path / "skipped")
    with patch.object(s, "phase_arms", return_value="ABC"):
        run(package, tmp_path / "extra", schedule=s.EVEN_MILESTONES)
    with pytest.raises(AssertionError):
        assert_even_outputs(tmp_path / "extra")


class Clock:
    def __init__(self, callback=None):
        self.seconds = 0.0
        self.callback = callback

    def mono(self):
        return self.seconds

    def wall(self):
        return 1000.0 + self.seconds

    def sleep(self, seconds):
        self.seconds += seconds
        if self.callback:
            callback, self.callback = self.callback, None
            callback()


def bank_args(package, tmp_path, clock):
    bank = package[0]
    record = s.verify(bank)
    expected = {
        k: record[k]
        for k in (
            "source",
            "environment",
            "seed",
            "teacher_architecture",
            "teacher_definition",
            "scale",
            "teacher_eval_chunk",
            "warmup",
            "num_filled",
            "execution",
        )
    }
    status = tmp_path / "producer.json"
    status.write_text(
        json.dumps(
            {
                **{k: expected[k] for k in ("source", "environment", "seed")},
                "status": "RUNNING",
            }
        )
    )
    return dict(
        expected=expected,
        pool_size=4,
        normalized=True,
        timeout_seconds=2,
        allocation_deadline_unix=1010,
        producer_status=str(status),
        audit_dir=tmp_path / "gate",
        _monotonic=clock.mono,
        _wall=clock.wall,
        _sleep=clock.sleep,
    )


def test_in_wave_wait_success_is_readonly_and_does_not_draw_training_rng(
    package, tmp_path
):
    destination = tmp_path / "published"
    clock = Clock(lambda: shutil.copytree(package[0], destination))
    args = bank_args(package, tmp_path, clock)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    before = {p.name: p.read_bytes() for p in package[0].iterdir()}
    with patch.object(np.random, "default_rng", side_effect=AssertionError("RNG draw")):
        result = gate.wait_for_bank(destination, **args)
    assert result == gate.sha(package[0] / "receipt.json")
    assert 0 < clock.seconds <= 2
    assert before == {p.name: p.read_bytes() for p in destination.iterdir()}
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert random.getstate() == python_state
    assert (
        json.loads(next((tmp_path / "gate").glob("*.json")).read_text())["status"]
        == "BANK_VERIFIED"
    )


@pytest.mark.parametrize(
    "failure",
    [
        "failed",
        "timeout",
        "partial",
        "invalid_hash",
        "wrong_teacher",
        "wrong_rms",
        "wrong_rms_values",
        "wrong_status",
        "runtime",
        "missing_after_completion",
        "over_cap",
    ],
)
def test_in_wave_gate_fails_closed(package, tmp_path, failure):
    clock = Clock()
    args = bank_args(package, tmp_path, clock)
    bank = tmp_path / "candidate"
    if failure not in ("timeout", "missing_after_completion"):
        shutil.copytree(package[0], bank)
    status = Path(args["producer_status"])
    if failure in ("failed", "missing_after_completion"):
        data = json.loads(status.read_text())
        data["status"] = "FAILED" if failure == "failed" else "COMPLETED"
        status.write_text(json.dumps(data))
    elif failure == "partial":
        (bank / "receipt.json").unlink()
    elif failure == "invalid_hash":
        with (bank / "bank.npz").open("ab") as stream:
            stream.write(b"bad")
    elif failure == "wrong_teacher":
        args["expected"]["teacher_definition"] = "wrong"
    elif failure == "wrong_rms":
        args["normalized"] = False
    elif failure == "wrong_rms_values":
        record = s.load_json(bank / "receipt.json")
        record["rms"]["count"] += 1
        s.write_json(bank / "receipt.json", record)
    elif failure == "wrong_status":
        record = json.loads(status.read_text())
        record["seed"] += 1
        status.write_text(json.dumps(record))
    elif failure == "runtime":
        args["expected"]["execution"] = {"backend": "unqualified"}
    elif failure == "over_cap":
        args["allocation_deadline_unix"] = 1000 + 21601
    with pytest.raises((ValueError, OSError)):
        gate.wait_for_bank(bank, **args)
    assert clock.seconds <= 2
    assert next((tmp_path / "gate").glob("*.json")).exists()


def test_six_cells_request_configs_and_budget(tmp_path):
    plan = review_plan()
    plan_module.validate_design(plan)
    assert sum(r["bank_producer"] for r in plan["runs"]) == 2
    for gpu in (0, 1):
        assert {r["architecture"] for r in plan["runs"] if r["gpu"] == gpu} == {
            "D2W512",
            "D4W1024",
            "D4W1536",
        }
    for row in plan["runs"]:
        request = plan_module.request_for(
            plan,
            row["id"],
            main=str(Path(plan_module.__file__).parents[1]),
            root=str(tmp_path),
            project="CPU-config-test",
            source_sha="0" * 40,
            checkpoint_interval=10000,
            bank_timeout_seconds=2,
            allocation_deadline_unix=time.time() + 10,
        )
        args = request["args"]
        cfg = exp1.compose_config(
            args["config_path"], args["config_name"], args["overrides"]
        )
        assert (
            cfg.seed == 0
            and cfg.run_role == "dev"
            and cfg.num_interaction_steps == 500000
        )
        assert cfg.actor_hidden_dim == 128 and cfg.actor_num_blocks == 1
        assert cfg.updates_per_interaction_step == 2 and cfg.buffer.min_length == 5000
        assert cfg.probe.steps == 2000 and cfg.probe.pool_size == 25600
        assert cfg.evaluation_per_interaction_step == 50000 and not cfg.fork.enabled
        assert cfg.agent.gamma == (0.99 if row["environment"] == "dog-run" else 0.95)
        assert (
            cfg.agent.critic_learning_rate == 1e-4
            and cfg.agent.critic_weight_decay == 0.01
        )
        assert request["alternative"]["teacher"] == plan_module.TEACHER
        assert request["alternative"]["schedule"] == s.EVEN_MILESTONES
        assert ("bank_wait" in request["alternative"]) != row["bank_producer"]
    assert 2000 * (8 + 8 + 4) == plan["diagnostic"]["updates_per_observation"] == 40000
    assert 40000 * 11 == 440000 and 440000 * 6 == 2640000


@pytest.mark.parametrize(
    "change",
    ["assignment", "missing_cell", "budget", "teacher", "cap", "wave", "approval"],
)
def test_six_plan_rejects_changes(change):
    plan = deepcopy(review_plan())
    if change == "assignment":
        plan["runs"][0]["gpu"] = 1
    elif change == "missing_cell":
        plan["runs"].pop()
    elif change == "budget":
        plan["diagnostic"]["steps_per_target"] = 1000
    elif change == "teacher":
        plan["diagnostic"]["teacher"]["hidden_dim"] = 4
    elif change == "cap":
        plan["requested_allocation"]["walltime_seconds"] = 21601
    elif change == "wave":
        plan["requested_allocation"]["waves"] = 2
    elif change == "approval":
        plan["approved"] = True
    with pytest.raises(ValueError):
        plan_module.validate_design(plan)


def test_real_even_consumer_gate_and_resume_preserve_complete_training_state(tmp_path):
    from contextlib import ExitStack
    from diagnostics import sequential_runner as runner
    from experiments.exp12.state import latest_state_dir, state_differences
    from tests.exp12_helpers import patch_wandb
    from tests.test_sequential_runner import request

    producer_args, producer = request(tmp_path / "producer")
    args, consumer = request(tmp_path / "consumer", ["critic_hidden_dim=8"])
    baseline_args, _ = request(tmp_path / "baseline", ["critic_hidden_dim=8"])
    producer["schedule"] = consumer["schedule"] = s.EVEN_MILESTONES
    consumer.update(architecture="D1W8", bank=producer["bank"], bank_producer=False)
    status = tmp_path / "producer-status.json"
    status.write_text(
        json.dumps(
            dict(
                status="RUNNING",
                source=producer["source"],
                environment=producer["environment"],
                seed=producer["seed"],
            )
        )
    )
    consumer["bank_wait"] = dict(
        timeout_seconds=10,
        allocation_deadline_unix=time.time() + 600,
        producer_status=str(status),
    )
    with ExitStack() as stack:
        for patcher in patch_wandb():
            stack.callback(patcher.stop)
        stack.enter_context(patch("wandb.finish"))
        exp1.run(baseline_args)
        stack.enter_context(
            patch.object(exp1, "RunProbes", side_effect=AssertionError("no RunProbes"))
        )
        runner.run(producer_args, producer)
        before = {p.name: p.read_bytes() for p in Path(producer["bank"]).iterdir()}
        original = s._fit_literal
        calls = []

        def interrupted(*a, **kw):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("even phase interruption")
            return original(*a, **kw)

        with patch.object(s, "_fit_literal", side_effect=interrupted):
            with pytest.raises(RuntimeError, match="even phase interruption"):
                runner.run(args, consumer)
            phase = Path(consumer["output"]) / "checkpoint_000000002/phase_01"
            committed = {
                p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in phase.iterdir()
            }
            runner.run(args, consumer)
        assert len(calls) == 61
        assert committed == {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in phase.iterdir()
        }
        assert before == {
            p.name: p.read_bytes() for p in Path(producer["bank"]).iterdir()
        }
        assert (
            state_differences(
                latest_state_dir(Path(args["checkpoint_dir"]) / "state"),
                latest_state_dir(Path(baseline_args["checkpoint_dir"]) / "state"),
            )
            == []
        )
        for step in (2, 4, 8):
            assert_even_outputs(Path(consumer["output"]) / f"checkpoint_{step:09d}")
        identity = s.verify(Path(args["checkpoint_dir"]) / "alternative_identity")[
            "identity"
        ]
        assert identity["alternative"]["bank_receipt_sha256"] == gate.sha(
            Path(producer["bank"]) / "receipt.json"
        )


@pytest.mark.parametrize("race", ["deadline", "producer_failure"])
def test_verification_cannot_release_late_or_failed_producer(package, tmp_path, race):
    clock = Clock()
    args = bank_args(package, tmp_path, clock)
    original = gate.verify_bank

    def verified(*a, **kw):
        result = original(*a, **kw)
        if race == "deadline":
            clock.seconds = 3
        else:
            status = Path(args["producer_status"])
            value = json.loads(status.read_text())
            value["status"] = "FAILED"
            status.write_text(json.dumps(value))
        return result

    with patch.object(gate, "verify_bank", side_effect=verified):
        with pytest.raises(ValueError):
            gate.wait_for_bank(package[0], **args)
