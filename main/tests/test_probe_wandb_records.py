"""Probe telemetry transports the existing scientific record without refitting."""

import importlib.util
import json
import os
import pickle
import random
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import wandb

from experiments.exp12 import run_probes
from tests.exp12_helpers import compose, tiny_overrides


def owner(tmp_path, rounds=10):
    # A baseline module can be supplied to demonstrate pre-fix failures.
    cls = run_probes.RunProbes
    if os.environ.get("PROBE_LOGGING_BASELINE"):
        spec = importlib.util.spec_from_file_location(
            "baseline_run_probes", os.environ["PROBE_LOGGING_BASELINE"]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = module.RunProbes
    cfg = compose(tiny_overrides(extra=[f"probe.rounds={rounds}"]))
    trainer = SimpleNamespace(cfg=cfg, extra_state={}, interaction_step=20)
    return cls(trainer, str(tmp_path))


def results(rounds):
    return {
        "fresh": {"score": np.linspace(0.2, 0.4, rounds, dtype=np.float32)},
        "current": {"score": np.linspace(0.1, 0.2, rounds, dtype=np.float32)},
    }


@pytest.mark.parametrize("rounds", [5, 10])
def test_paired_log_contains_exact_stored_aggregates_and_rounds(tmp_path, rounds):
    with mock.patch.object(wandb, "run", object()), mock.patch.object(
        wandb, "define_metric"
    ) as define, mock.patch.object(wandb, "log") as log:
        probes = owner(tmp_path, rounds)
        before = results(rounds)
        snapshot = pickle.dumps(before)
        row = probes.record_check(1, before)
        define.assert_called_once_with("probe/*", step_metric="probe/interaction_step")
        log.assert_called_once()
        expected = {f"probe/{key}": value for key, value in row.items()}
        expected["probe/env_step"] = 40
        assert log.call_args.args[0] == expected
        assert pickle.dumps(before) == snapshot
        assert probes.records == [row]


def test_fresh_capture_logs_its_record_once_without_extra_probe(tmp_path):
    with mock.patch.object(wandb, "run", object()), mock.patch.object(
        wandb, "define_metric"
    ), mock.patch.object(wandb, "log") as log, mock.patch.object(
        run_probes.orbax.checkpoint, "PyTreeCheckpointer"
    ):
        probes = owner(tmp_path)
        probes.trainer._sac_agent = SimpleNamespace(
            critic=SimpleNamespace(params={"weight": np.array([1.0], np.float32)})
        )
        with mock.patch.object(probes, "_probe", return_value=results(10)) as probe:
            probes.capture_fresh(probes.trainer)
            probes.capture_fresh(probes.trainer)
        probe.assert_called_once()
        log.assert_called_once()
        row = probes.records[0]
        assert log.call_args.args[0]["probe/score_fresh_iqm"] == row["score_fresh_iqm"]
        assert log.call_args.args[0]["probe/check_index"] == 0
        assert "probe/score_current_iqm" not in log.call_args.args[0]


def test_no_active_wandb_run_keeps_local_scientific_record(tmp_path):
    with mock.patch.object(wandb, "run", None), mock.patch.object(
        wandb, "define_metric"
    ) as define, mock.patch.object(wandb, "log") as log:
        probes = owner(tmp_path)
        row = probes.record_check(1, results(10))
        define.assert_not_called()
        log.assert_not_called()
        assert row == probes.records[0]
        assert row["valid"]


def test_offline_sdk_history_contains_checkpoint_aggregates_and_preserves_rng(tmp_path):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    with mock.patch.dict(os.environ, {"WANDB_SILENT": "true"}):
        run = wandb.init(
            project="echocritic-probe-transport-test",
            mode="offline",
            dir=str(tmp_path),
            settings=wandb.Settings(disable_git=True, disable_code=True, console="off"),
        )
        try:
            probes = owner(tmp_path / "probe")
            before_np = pickle.dumps(np.random.get_state())
            before_py = pickle.dumps(random.getstate())
            first = probes.record_check(1, results(10))
            probes.trainer.interaction_step = 40
            second = probes.record_check(2, results(10))
            assert pickle.dumps(np.random.get_state()) == before_np
            assert pickle.dumps(random.getstate()) == before_py
            data_path = Path(run.dir).parent / f"run-{run.id}.wandb"
        finally:
            run.finish()
    store = DataStore()
    store.open_for_scan(str(data_path))
    rows = []
    while (chunk := store.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(chunk)
        if record.WhichOneof("record_type") == "history":
            rows.append(
                {
                    item.key or "/".join(item.nested_key): json.loads(item.value_json)
                    for item in record.history.item
                }
            )
    assert len(rows) == 2
    for actual, expected in zip(rows, [first, second]):
        for key in [
            "interaction_step",
            "check_index",
            "score_fresh_iqm",
            "score_current_iqm",
            "loss_iqm",
        ]:
            assert actual[f"probe/{key}"] == expected[key]
