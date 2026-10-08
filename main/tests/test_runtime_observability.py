"""CPU regressions for opt-in diagnostic tracing, not scientific calibration."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jax.numpy as jnp

from experiments.exp12.runtime_trace import RuntimeTrace


class RuntimeObservabilityTest(unittest.TestCase):
    def events(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_pre_call_sync_failure_has_flushed_begin_and_matching_error(self):
        obj = SimpleNamespace(call=mock.Mock())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.jsonl"
            with self.assertRaisesRegex(RuntimeError, "sync failed"):
                with RuntimeTrace(path, synchronize=True) as trace:
                    trace.wrap(obj, "call", "operation")

                    def fail(_):
                        self.assertEqual(self.events(path)[-1]["stage"], "synchronization")
                        raise RuntimeError("sync failed")

                    with mock.patch("jax.block_until_ready", side_effect=fail):
                        obj.call()
            obj.call.assert_not_called()
            ends = [e for e in self.events(path) if e["event"] == "end"]
            self.assertEqual([(e["stage"], e["error"]) for e in ends[-2:]],
                             [("synchronization", "RuntimeError"), ("operation", "RuntimeError")])
            self.assertEqual(ends[-1]["phase"], "before_sync")

    def test_pending_transfer_and_replay_preserve_metrics_and_add_no_waits(self):
        from experiments.exp12.diagnostics import ActorDiagnostics, DiagnosticPendingUpdateMetrics
        # Use the real logger's row aggregation rather than an invented metric convention.
        from scale_rl.common.logger import WandbTrainerLogger
        def make():
            log = WandbTrainerLogger.__new__(WandbTrainerLogger)
            log.reset()
            cfg = SimpleNamespace(diagnostics=SimpleNamespace(enabled=True, kl_reference_size=2,
                                   saturation_threshold=.99), seed=1, logging_per_interaction_step=2000)
            pending = DiagnosticPendingUpdateMetrics(log, 2, ActorDiagnostics(cfg))
            for step in (0, 2):
                pending.add(step, {"train/actor_grad_cosine": jnp.array([.1, .2]),
                                   "train/actor_gnorm": jnp.array([1., 2.])})
            return log, pending

        plain_log, plain = make()
        plain.flush()
        traced_log, traced = make()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.jsonl"
            with RuntimeTrace(path, synchronize=True) as trace:
                trace.install()
                with mock.patch("jax.block_until_ready", side_effect=AssertionError("extra wait")):
                    traced.flush()
            events = self.events(path)
        self.assertEqual(plain_log.average_meter_dict.averages(), traced_log.average_meter_dict.averages())
        self.assertEqual(plain._diagnostics.state(), traced._diagnostics.state())
        self.assertEqual(traced._pending, [])
        transfers = [e for e in events if e.get("stage") == "metric_device_get" and e["event"] == "begin"]
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0]["array_bytes"], 32)
        self.assertEqual(transfers[0]["array_leaves"], 4)
        stages = {"diagnostic_metric_flush", "metric_device_get", "metric_replay"}
        self.assertEqual(
            [(event["event"], event["stage"]) for event in events if event.get("stage") in stages],
            [("begin", "diagnostic_metric_flush"), ("begin", "metric_device_get"),
             ("end", "metric_device_get"), ("begin", "metric_replay"),
             ("end", "metric_replay"), ("end", "diagnostic_metric_flush")],
        )

    def test_boundary_detail_is_limited_to_training_window(self):
        with tempfile.TemporaryDirectory() as root:
            trace = RuntimeTrace(Path(root) / "trace", boundary_detail=True)
            trace.trainer = SimpleNamespace(cfg=SimpleNamespace(logging_per_interaction_step=2000), interaction_step=5900)
            self.assertFalse(trace.near_logging_boundary("sample_actions"))
            trace.training = True
            for step, expected in ((5900, False), (5901, True), (6000, True), (6001, False)):
                trace.trainer.interaction_step = step
                self.assertEqual(trace.near_logging_boundary("update_many"), expected)
            self.assertFalse(trace.near_logging_boundary("eval_env_step"))

    def test_post_call_sync_failure_reports_phase_and_preserves_call(self):
        obj = SimpleNamespace(call=mock.Mock(return_value=17))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace"
            with RuntimeTrace(path, synchronize=True) as trace:
                trace.wrap(obj, "call", "operation")
                with mock.patch("jax.block_until_ready", side_effect=[None, RuntimeError("post-sync")]):
                    with self.assertRaisesRegex(RuntimeError, "post-sync"):
                        obj.call()
            obj.call.assert_called_once_with()
            end = [e for e in self.events(path) if e.get("stage") == "operation"][-1]
            self.assertEqual(end["phase"], "after_sync")
            self.assertEqual(end["error"], "RuntimeError")

    def test_metric_payload_sizes_use_real_flush_and_original_row_order(self):
        from experiments.exp12.diagnostics import DiagnosticPendingUpdateMetrics
        from scale_rl.common.logger import WandbTrainerLogger
        from experiments.exp12.diagnostics import ActorDiagnostics
        cfg = SimpleNamespace(diagnostics=SimpleNamespace(enabled=True, kl_reference_size=2,
                              saturation_threshold=.99), seed=1, logging_per_interaction_step=2000)
        for groups in (0, 1, 1001, 4096):
            with self.subTest(groups=groups):
                log = WandbTrainerLogger.__new__(WandbTrainerLogger)
                log.reset()
                pending = DiagnosticPendingUpdateMetrics(log, 2, ActorDiagnostics(cfg))
                for index in range(groups):
                    pending.add(2 * index, {"train/actor_grad_cosine": jnp.array([.1, .2]),
                                           "train/actor_gnorm": jnp.array([index, index + 1.])})
                meta = RuntimeTrace.payload_metadata([info for _, info in pending._pending])
                self.assertEqual(meta["array_leaves"], 2 * groups)
                self.assertEqual(meta["array_bytes"], 16 * groups)
                pending.flush()
                self.assertEqual(pending._pending, [])
                self.assertEqual(pending._diagnostics.gnorm,
                                 [value for index in range(groups) for value in (float(index), float(index + 1))])

    def test_transfer_failure_is_recorded_without_consuming_pending_rows(self):
        from experiments.exp12.diagnostics import DiagnosticPendingUpdateMetrics
        pending = DiagnosticPendingUpdateMetrics(SimpleNamespace(), 1, SimpleNamespace())
        pending.add(0, {"value": jnp.array([1.])})
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace"
            with mock.patch("jax.device_get", side_effect=OSError("copy failed")):
                with RuntimeTrace(path) as trace:
                    trace.install()
                    with self.assertRaisesRegex(OSError, "copy failed"):
                        pending.flush()
                    self.assertEqual(trace.flush_depth, 0)
            self.assertEqual(len(pending._pending), 1)
            errors = [e for e in self.events(path) if e.get("error") == "OSError"]
            self.assertEqual([e["stage"] for e in errors], ["metric_device_get", "diagnostic_metric_flush"])

    def test_slow_and_failed_trace_writes_are_not_hidden(self):
        class Writer:
            def __init__(self, fail=False):
                self.fail, self.flushed = fail, False
            def write(self, _):
                if self.fail:
                    raise OSError("disk full")
                time.sleep(.01)
            def flush(self):
                self.flushed = True
        trace = RuntimeTrace("unused")
        trace.file = Writer()
        start = time.perf_counter()
        trace.emit("begin", stage="write")
        self.assertGreaterEqual(time.perf_counter() - start, .01)
        self.assertTrue(trace.file.flushed)
        trace.file = Writer(fail=True)
        with self.assertRaisesRegex(OSError, "disk full"):
            trace.emit("begin", stage="write")

    def test_wrapper_captures_stacks_and_preserves_entrypoint_error(self):
        script = Path(__file__).resolve().parents[1] / "scripts/trace_exp12_runtime.py"
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            entry = root / "blocked_entry.py"
            entry.write_text("import time\ntime.sleep(.4)\nraise ValueError('entry failed')\n")
            trace = root / "trace.jsonl"
            result = subprocess.run([sys.executable, str(script), "--out", str(trace),
                                     "--stack-after", ".1", "--", str(entry)],
                                    env={**os.environ, "JAX_PLATFORMS": "cpu"},
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("entry failed", result.stderr)
            events = self.events(trace)
            self.assertEqual(events[0]["event"], "wrapper_start")
            self.assertEqual(events[-1]["error"], "ValueError")
            self.assertIn("blocked_entry.py", trace.with_suffix(".stacks.log").read_text())
            before = trace.read_bytes()
            retry = subprocess.run([sys.executable, str(script), "--out", str(trace), "--", str(entry)],
                                   capture_output=True, text=True, timeout=10)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(trace.read_bytes(), before)

    def test_startup_import_failure_is_flushed_before_jax_trace_import(self):
        path = Path(__file__).resolve().parents[1] / "scripts/trace_exp12_runtime.py"
        spec = importlib.util.spec_from_file_location("trace_wrapper_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as root:
            trace = Path(root) / "trace.jsonl"
            with mock.patch.object(module, "configure_hardware_env"), \
                 mock.patch.object(module, "__import__", side_effect=ImportError("startup failed"), create=True), \
                 mock.patch.object(sys, "argv", [str(path), "--out", str(trace), "--", "unused.py"]):
                with self.assertRaisesRegex(ImportError, "startup failed"):
                    module.main()
            events = self.events(trace)
            self.assertEqual(events[-2]["stage"], "runtime_import")
            self.assertEqual(events[-1]["error"], "ImportError")

    def test_wrapper_rejects_invalid_watchdog_and_preserves_existing_stack_file(self):
        path = Path(__file__).resolve().parents[1] / "scripts/trace_exp12_runtime.py"
        spec = importlib.util.spec_from_file_location("trace_wrapper_guards", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as root:
            trace = Path(root) / "trace.jsonl"
            for value in ("0", "-1", "nan", "inf"):
                with self.subTest(value=value), mock.patch.object(sys, "argv", [str(path), "--out", str(trace),
                                         "--stack-after", value, "--", "unused.py"]):
                    with self.assertRaises(SystemExit):
                        module.main()
                self.assertFalse(trace.exists())
            stacks = trace.with_suffix(".stacks.log")
            stacks.write_text("existing evidence")
            with mock.patch.object(sys, "argv", [str(path), "--out", str(trace), "--stack-after", "60", "--", "unused.py"]):
                with self.assertRaises(FileExistsError):
                    module.main()
            self.assertEqual(stacks.read_text(), "existing evidence")

    def test_full_trace_analyzer_exposes_unmatched_and_malformed_records(self):
        path = Path(__file__).resolve().parents[1] / "scripts/analyze_runtime_trace.py"
        spec = importlib.util.spec_from_file_location("trace_analysis", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as root:
            trace = Path(root) / "trace"
            trace.write_text('\n'.join(json.dumps(e) for e in (
                dict(event="begin", stage="parent", monotonic_time=1),
                dict(event="begin", stage="child", monotonic_time=2),
                dict(event="end", stage="child", monotonic_time=3))) + '\n')
            result = module.analyze(trace)
            self.assertEqual(result["records"], 3)
            self.assertEqual(result["unmatched_begins"], [("parent", 1)])
            with trace.open("a") as stream:
                stream.write(json.dumps(dict(event="end", stage="suppressed", monotonic_time=4,
                                             begin_recorded=False, error="RuntimeError")) + '\n')
            self.assertEqual(module.analyze(trace)["mismatched_ends"], [])
            trace.write_text(trace.read_text() + '{"truncated":')
            with self.assertRaises(json.JSONDecodeError):
                module.analyze(trace)
            trace = Path(root) / "sampled_error"
            obj = SimpleNamespace(call=mock.Mock(side_effect=[1, 2, 3, RuntimeError("fourth call")]))
            with RuntimeTrace(trace) as observer:
                observer.wrap(obj, "call", "sampled", coarse=False)
                for _ in range(3):
                    obj.call()
                with self.assertRaisesRegex(RuntimeError, "fourth call"):
                    obj.call()
            self.assertFalse(self.events(trace)[-2]["begin_recorded"])
            self.assertEqual(module.analyze(trace)["mismatched_ends"], [])
            self.assertEqual(module.analyze(trace)["unmatched_begins"], [])


if __name__ == "__main__":
    unittest.main()
