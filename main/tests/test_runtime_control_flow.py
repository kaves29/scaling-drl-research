"""Expected fork/stop signals must not invalidate completed diagnostic traces."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments import exp1
from experiments.exp12 import fork
from experiments.exp12.runtime_trace import RuntimeTrace
from experiments.exp12.trainer import Exp12Trainer


class RuntimeControlFlowTest(unittest.TestCase):
    def signals(self):
        return (exp1.ForkNow({"fork_step": 100}), fork.ValidationDone())

    def events(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_expected_signals_are_annotations_with_the_same_exception(self):
        for signal in self.signals():
            with self.subTest(
                signal=type(signal).__name__
            ), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "trace.jsonl"
                with mock.patch.object(Exp12Trainer, "train", side_effect=signal):
                    with RuntimeTrace(path) as trace:
                        trace.install()
                        with mock.patch.object(trace, "progress"):
                            with self.assertRaises(type(signal)) as raised:
                                Exp12Trainer.train(
                                    SimpleNamespace(interaction_step=0), 1
                                )
                            self.assertIs(raised.exception, signal)
                events = self.events(path)
                exit_event = next(row for row in events if row["event"] == "train_exit")
                self.assertEqual(exit_event["control_flow"], type(signal).__name__)
                self.assertIsNone(exit_event["error"])
                self.assertEqual(events[-1]["event"], "summary")
                self.assertTrue(all(row.get("error") is None for row in events))

    def test_uncaught_signal_still_fails_the_outer_trace(self):
        for signal in self.signals():
            with self.subTest(
                signal=type(signal).__name__
            ), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "trace.jsonl"
                with mock.patch.object(Exp12Trainer, "train", side_effect=signal):
                    with self.assertRaises(type(signal)):
                        with RuntimeTrace(path) as trace:
                            trace.install()
                            with mock.patch.object(trace, "progress"):
                                Exp12Trainer.train(
                                    SimpleNamespace(interaction_step=0), 1
                                )
                self.assertEqual(self.events(path)[-1]["error"], type(signal).__name__)

    def test_progress_failure_during_transition_is_not_hidden(self):
        signal = exp1.ForkNow({"fork_step": 100})
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.jsonl"
            with mock.patch.object(Exp12Trainer, "train", side_effect=signal):
                with RuntimeTrace(path) as trace:
                    trace.install()
                    with mock.patch.object(
                        trace, "progress", side_effect=ValueError("progress failed")
                    ):
                        with self.assertRaises(exp1.ForkNow) as raised:
                            Exp12Trainer.train(SimpleNamespace(interaction_step=0), 1)
                        self.assertIs(raised.exception, signal)
            events = self.events(path)
            self.assertTrue(any(row.get("error") == "ValueError" for row in events))

    def test_real_parent_and_identity_satisfy_existing_trace_validation(self):
        from scripts.compare_identity_fork import compare
        from tests.exp12_helpers import patch_wandb
        from tests.test_exp12_fork import fork_overrides, run_arm, run_exp1

        patchers = patch_wandb()
        self.addCleanup(lambda: [p.stop() for p in patchers])
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            overrides = fork_overrides(
                str(root / "results"),
                extra=[
                    "seed=101",
                    "testing.force_trigger_check=2",
                    "fork.identity_snapshot_steps=5",
                    "testing.stop_after_identity_snapshot=true",
                ],
            )
            parent, identity = root / "parent", root / "identity"
            for name, action, transitions in (
                (
                    "parent",
                    lambda: run_exp1(str(parent), overrides),
                    ["ForkNow", "ValidationDone"],
                ),
                (
                    "identity",
                    lambda: run_arm(str(identity), overrides, str(parent), "identity"),
                    ["ValidationDone"],
                ),
            ):
                path = root / (name + ".jsonl")
                with RuntimeTrace(path, synchronize=True) as trace:
                    trace.install()
                    action()
                events = self.events(path)
                self.assertEqual(events[-1]["event"], "summary")
                self.assertTrue(all(row.get("error") is None for row in events))
                self.assertEqual(
                    [row["control_flow"] for row in events if "control_flow" in row],
                    transitions,
                )
            result = compare(parent, identity)
            self.assertTrue(result["pass"], result)
            self.assertEqual(result["differences"], [])
            self.assertEqual(
                (
                    result["fork_step"],
                    result["interaction_step"],
                    result["identity_interaction_step"],
                ),
                (40, 45, 45),
            )


if __name__ == "__main__":
    unittest.main()
