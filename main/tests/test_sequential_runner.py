"""Tiny real SAC/Exp12Trainer lifecycle checks, CPU only; no campaign runs."""

import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from diagnostics import sequential_probe as s
from diagnostics import sequential_runner as runner
from experiments import exp1
from experiments.exp12.state import latest_state_dir, load_meta, state_differences
from tests.exp12_helpers import CONFIG_PATH, patch_wandb, tiny_overrides


@pytest.fixture(autouse=True)
def offline_logger(monkeypatch):
    monkeypatch.setenv("EXP12_JAX_CACHE_DIR", "off")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    patchers = patch_wandb()
    finish = patch("wandb.finish")
    finish.start()
    yield
    finish.stop()
    for item in reversed(patchers):
        item.stop()


def request(root, extra=()):
    overrides = tiny_overrides(
        seed=1,
        steps=8,
        extra=[
            "actor_hidden_dim=4",
            "critic_hidden_dim=4",
            "buffer.min_length=2",
            "buffer.max_length=16",
            "buffer.sample_batch_size=2",
            "evaluation_per_interaction_step=4",
            "logging_per_interaction_step=4",
            "env.max_episode_steps=4",
            "probe.steps=2000",
            "probe.pool_size=4",
            "probe.batch_size=2",
            "probe.eval_chunk=2",
            "fork.enabled=false",
            "probe.enabled=false",
            "diagnostics.kl_reference_size=4",
            *extra,
        ],
    )
    unique = {x.split("=", 1)[0]: x for x in overrides}
    args = dict(
        config_path=CONFIG_PATH,
        config_name="base_exp12_ten",
        overrides=list(unique.values()),
        checkpoint_dir=str(root),
        checkpoint_interval=4,
        checkpoint_start_frac=0.0,
    )
    alternative = dict(
        enabled=True,
        bank=str(root / "bank"),
        reference=str(root / "reference"),
        output=str(root / "alternative"),
        teacher=dict(
            block_type="residual", num_blocks=1, hidden_dim=4, dtype="float32"
        ),
        teacher_architecture="CPU-test-fixed-D1W4",
        architecture="D1W4",
        environment="hopper-hop",
        seed=1,
        source="synthetic-CPU-lifecycle",
        bank_producer=True,
    )
    return args, alternative


def test_default_delegates_exactly_without_alternative_work():
    args = {"unchanged": object()}
    with patch.object(exp1, "run", return_value="default") as original, patch.object(
        runner.probe,
        "precision_contract",
        side_effect=AssertionError("alternative work"),
    ):
        assert runner.run(args) == "default"
        assert runner.run(args, {"enabled": False}) == "default"
        assert original.call_args_list[0].args == (args,)
        assert original.call_count == 2


def test_real_default_still_constructs_runprobes(tmp_path):
    args, _ = request(tmp_path / "default", ["probe.enabled=true", "probe.checks=2"])
    with patch.object(
        exp1, "RunProbes", side_effect=RuntimeError("approved probe constructed")
    ) as original:
        with pytest.raises(RuntimeError, match="approved probe constructed"):
            runner.run(args)
        assert original.call_count == 1


@pytest.mark.parametrize("fail_at", [None, 4, 28])
def test_real_training_lifecycle_and_phase_resume_are_bit_exact(tmp_path, fail_at):
    args, alternative = request(
        tmp_path / "alternative_run", ["probe.enabled=true"] if fail_at is None else []
    )
    baseline_args, _ = request(tmp_path / "baseline")
    exp1.run(baseline_args)
    expected = latest_state_dir(Path(baseline_args["checkpoint_dir"]) / "state")
    original_fit, original_observe = s._fit_literal, runner.Observer.observe
    calls, checks = [], []

    def observed(observer, trainer):
        before = trainer.save(tmp_path / "isolation_before")
        result = original_observe(observer, trainer)
        after = trainer.save(tmp_path / "isolation_after")
        assert state_differences(before, after) == []
        checks.append((trainer.interaction_step, trainer.update_step))
        return result

    def fit(*a, **kw):
        calls.append(1)
        if fail_at and len(calls) == fail_at:
            raise RuntimeError("deliberate phase interruption")
        return original_fit(*a, **kw)

    root = Path(args["checkpoint_dir"])
    with patch.object(
        exp1, "RunProbes", side_effect=AssertionError("must not construct")
    ), patch.object(runner.Observer, "observe", observed), patch.object(
        s, "_fit_literal", side_effect=fit
    ):
        if fail_at:
            with pytest.raises(RuntimeError, match="deliberate phase interruption"):
                runner.run(args, alternative)
            step = 2 if fail_at == 4 else 4
            phase = root / "alternative" / f"checkpoint_{step:09d}" / "phase_01"
            committed = {
                p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in phase.iterdir()
            }
            latest = latest_state_dir(root / "state")
            assert load_meta(latest)["interaction_step"] == (1 if fail_at == 4 else 4)
            runner.run(args, alternative)
            assert committed == {
                p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in phase.iterdir()
            }
            assert len(calls) == 73
        else:
            runner.run(args, alternative)
            assert len(calls) == 72
    actual = latest_state_dir(root / "state")
    assert state_differences(expected, actual) == []
    assert (2, 0) in checks
    assert (4, 6) in checks and (8, 14) in checks
    meta = load_meta(actual)
    assert {r["interaction_step"] for r in meta["eval_rows"]} == {0, 4, 8}
    assert meta["update_step"] == 14
    assert not (root / "probes").exists()
    assert "probe_records" not in meta["extra_state"]
    assert not (root / "fresh_critic").exists()
    assert (
        json.loads((root / "ALTERNATIVE_DONE.json").read_text())["mode"]
        == "alternative_only"
    )
    for step in (2, 4, 8):
        out = root / "alternative" / f"checkpoint_{step:09d}"
        assert len(json.loads((out / "trajectory.json").read_text())["phases"]) == 8
        for k in range(1, 9):
            receipt = s.verify(out / f"phase_{k:02d}")
            assert receipt["updates_per_arm"] == 2000
            for arm in "ABC":
                with np.load(out / f"phase_{k:02d}" / f"{arm}.npz") as arrays:
                    assert s.array_hash(arrays["indices"]) == receipt["indices_sha256"]
                    assert s.array_hash(arrays["targets"]) == receipt["target_sha256"]


@pytest.mark.parametrize(
    "override",
    [
        "fork.enabled=true",
        "agent.critic_use_cdq=true",
        "agent.mixed_precision=true",
        "critic_sparsity=0.5",
        "probe.steps=20",
        "updates_per_interaction_step=0.5",
        "evaluation_per_interaction_step=2",
    ],
)
def test_rejects_unsupported_runner_before_trainer_creation(tmp_path, override):
    args, alternative = request(tmp_path / "rejected", [override])
    with patch(
        "experiments.exp12.trainer.Exp12Trainer",
        side_effect=AssertionError("no trainer"),
    ):
        with pytest.raises(ValueError):
            runner.run(args, alternative)
    assert not Path(args["checkpoint_dir"]).exists()


def test_rejects_missing_owner_teacher_choice(tmp_path):
    args, alternative = request(tmp_path / "rejected")
    del alternative["teacher"]
    with pytest.raises(ValueError, match="teacher"):
        runner.run(args, alternative)
