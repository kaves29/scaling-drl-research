"""CPU checks of live runner observation, lifecycle, and mutation isolation."""

import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from analysis.campaign.provenance import DiagnosticSession
from experiments.exp12 import instrumentation as recording
from tests.mutation_support import StableSource


class InstrumentationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def assert_checkpoint_receipt(self, run_dir, receipt):
        import pickle
        import numpy as np
        import jax
        from experiments.exp12.state import latest_state_dir, load_agent_tree, load_meta

        state = latest_state_dir(run_dir / "state")
        tree, meta = load_agent_tree(state), load_meta(state)
        self.assertEqual(receipt["checkpoint"], str(state))
        self.assertEqual(receipt["interaction_step"], meta["interaction_step"])
        self.assertEqual(receipt["optimizer_update"], meta["update_step"])
        for name in ("actor", "critic"):
            stored_count = sum(
                np.asarray(leaf).size
                for leaf in jax.tree_util.tree_leaves(tree[name]["params"])
            )
            self.assertEqual(receipt["parameter_counts"][name], stored_count)
        with (state / "buffer_meta.pkl").open("rb") as stream:
            buffer_meta = pickle.load(stream)
        self.assertEqual(
            receipt["replay"]["populated_transitions"], buffer_meta["num_in_buffer"]
        )
        self.assertEqual(
            receipt["source"]["commit"],
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parents[2]),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            ).strip(),
        )

    def test_actual_diagnostic_rejects_campaign_before_writes_for_both_argument_forms(
        self,
    ):
        from scripts import exp96_fixed_task_diagnostics as module

        campaign = self.root / "frozen_campaign"
        campaign.mkdir()
        sentinel = campaign / "historical.txt"
        sentinel.write_text("original evidence")
        output = campaign / "sidecars"
        args = SimpleNamespace(campaign=str(campaign), plan="must-not-be-read")
        for keyword in (False, True):
            with self.subTest(keyword=keyword), mock.patch.dict(
                os.environ, {recording.ENVIRONMENT_VARIABLE: str(output)}
            ), mock.patch.object(
                module,
                "load_plan",
                side_effect=AssertionError("must reject before preflight"),
            ) as preflight:
                with self.assertRaisesRegex(ValueError, "frozen campaign"):
                    if keyword:
                        module.execute(args=args)
                    else:
                        module.execute(args)
                preflight.assert_not_called()
                self.assertFalse(output.exists())
                self.assertEqual(list(campaign.iterdir()), [sentinel])
                self.assertEqual(sentinel.read_text(), "original evidence")

    def test_observation_arrays_nonfinite_and_media_are_explicit(self):
        import jax.numpy as jnp
        import numpy as np

        output = self.root / "receipts"
        with DiagnosticSession(
            output, Path(__file__).resolve().parents[2], {}
        ) as session:
            session.record_observation(
                {
                    "q": jnp.asarray([0.25, 0.5]),
                    "loss": np.float32(0.125),
                    "missing_numeric": float("nan"),
                    "media": object(),
                },
                4,
                2,
                "state",
            )
        event = json.loads((output / "measurement_000000.json").read_text())
        self.assertEqual(event["metrics"]["q"], [0.25, 0.5])
        self.assertEqual(event["metrics"]["loss"], 0.125)
        self.assertEqual(event["metrics"]["missing_numeric"], {"nonfinite": "nan"})
        self.assertEqual(
            event["metrics"]["media"], {"unserialized_type": "builtins.object"}
        )

    def test_disabled_entrypoint_does_not_capture_or_change_return(self):
        sentinel = object()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "analysis.campaign.provenance.capture",
            side_effect=AssertionError("must not capture"),
        ):

            @recording.entrypoint("diagnostic")
            def run():
                return sentinel

            self.assertIs(run(), sentinel)

    def test_enabled_diagnostic_rows_preserve_values_unknown_counters_and_partial(self):
        output = self.root / "receipts"
        rows = [
            {
                "task_id": "numeric-task-hash",
                "steps": 4,
                "status": "PARTIAL",
                "measurements": [0.25, 0.125],
            }
        ]
        runner = SimpleNamespace(
            args=SimpleNamespace(campaign=str(self.root / "campaign")),
            rows=rows,
            inputs=[{"source": "frozen"}],
            plan={"mode": "test"},
            plan_sha256="plan-hash",
            optimizer_provenance=[{"learning_rate": 1e-4}],
        )

        @recording.entrypoint("diagnostic")
        def run():
            recording.diagnostic_state(runner)
            return 3

        with mock.patch.dict(os.environ, {recording.ENVIRONMENT_VARIABLE: str(output)}):
            self.assertEqual(run(), 3)
        terminal = json.loads((output / "terminal.json").read_text())
        measurement = json.loads((output / "measurement_000000.json").read_text())
        self.assertEqual(terminal["status"], "partial")
        self.assertEqual(measurement["metrics"]["rows"], rows)
        self.assertIsNone(measurement["optimizer_update"])
        self.assertIn("seed", terminal["missing_fields"])
        self.assertEqual(
            measurement["missing_counters"], ["interaction_step", "optimizer_update"]
        )

    def test_original_failure_survives_terminal_io_failure(self):
        output = self.root / "receipts"
        error = RuntimeError("primary")
        with self.assertRaisesRegex(RuntimeError, "primary") as raised:
            with DiagnosticSession(output, Path(__file__).resolve().parents[2], {}):
                (output / "terminal.json").write_text("existing")
                raise error
        self.assertIs(raised.exception, error)
        self.assertIn("Terminal receipt", error.__notes__[0])

    def test_failed_entrypoint_is_recorded_and_context_is_reset(self):
        output = self.root / "receipts"

        @recording.entrypoint("diagnostic")
        def run():
            raise ValueError("preflight failed")

        with mock.patch.dict(os.environ, {recording.ENVIRONMENT_VARIABLE: str(output)}):
            with self.assertRaisesRegex(ValueError, "preflight failed"):
                run()
            with self.assertRaises(FileExistsError):
                run()
        self.assertIsNone(recording._active.get())
        terminal = json.loads((output / "terminal.json").read_text())
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["measurement_count"], 0)

    def test_rejects_source_checkpoint_and_campaign_outputs(self):
        @recording.entrypoint("training")
        def train(args):
            self.fail("must reject before training")

        checkpoint = self.root / "checkpoint"
        for output in (
            Path(__file__).resolve().parents[2] / "receipts",
            checkpoint / "receipts",
        ):
            with mock.patch.dict(
                os.environ, {recording.ENVIRONMENT_VARIABLE: str(output)}
            ):
                with self.assertRaisesRegex(ValueError, "outside source/checkpoints"):
                    train({"checkpoint_dir": str(checkpoint)})
        output = self.root / "campaign" / "receipts"

        @recording.entrypoint("diagnostic")
        def diagnostic(args):
            self.fail("must reject before creating receipts")

        with mock.patch.dict(os.environ, {recording.ENVIRONMENT_VARIABLE: str(output)}):
            with self.assertRaisesRegex(ValueError, "frozen campaign"):
                diagnostic(SimpleNamespace(campaign=str(output.parent)))
        self.assertFalse(output.exists())

    def test_live_diagnostic_entrypoint_and_save_record_partial_results(self):
        from contextlib import ExitStack
        import time
        from scripts import exp96_fixed_task_diagnostics as module

        output = self.root / "sidecars"
        args = SimpleNamespace(
            root=str(self.root / "diagnostic"),
            campaign=str(self.root / "campaign"),
            commit="a" * 40,
            cap_seconds=3000,
            plan=str(
                Path(__file__).resolve().parents[1]
                / "configs/diagnostics/reconstruction.json"
            ),
        )
        runner = module.Investigation.__new__(module.Investigation)
        runner.args, runner.root, runner.start = args, Path(args.root), time.monotonic()
        runner.rows, runner.inputs, runner.timing = [], [], []
        runner.optimizer_provenance, runner.optimizer_sha256 = [
            {"optimizer": {"learning_rate": 1e-4}}
        ], "optimizer-hash"
        runner.plan, runner.plan_sha256 = {}, "plan-hash"

        def execute():
            runner.rows.append(
                {
                    "task_id": "original-task",
                    "steps": 1000,
                    "status": "PARTIAL",
                    "measurements": [{"step": 0, "mse": 2.5}],
                }
            )
            runner.save()
            return False

        runner.run = execute
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(
                    os.environ,
                    {recording.ENVIRONMENT_VARIABLE: str(output), "SLURM_JOB_ID": "42"},
                )
            )
            # capture uses the same subprocess module as the diagnostic preflight.
            from analysis.campaign import provenance

            stack.enter_context(
                mock.patch.object(
                    provenance,
                    "capture",
                    side_effect=lambda repo, cfg, **fields: {
                        "config": cfg,
                        "missing_fields": ["seed"],
                        **fields,
                    },
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module.argparse.ArgumentParser, "parse_args", return_value=args
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module.subprocess, "check_output", side_effect=[args.commit, ""]
                )
            )
            stack.enter_context(
                mock.patch.object(module.jax, "default_backend", return_value="gpu")
            )
            stack.enter_context(
                mock.patch.object(
                    module.jax,
                    "devices",
                    return_value=[
                        SimpleNamespace(
                            device_kind="NVIDIA A100-SXM4-40GB", memory_stats=lambda: {}
                        )
                    ],
                )
            )
            stack.enter_context(
                mock.patch(
                    "experiments.exp12.precision.runtime_info",
                    return_value={
                        "platform": "cpu",
                        "matmul_precision": "tensorfloat32",
                    },
                )
            )
            stack.enter_context(mock.patch.object(module, "set_matmul_precision"))
            stack.enter_context(
                mock.patch.object(
                    module,
                    "runtime_info",
                    return_value={
                        "jax_cuda12_plugin": "0.4.34",
                        "jax_cuda12_pjrt": "0.4.34",
                    },
                )
            )
            stack.enter_context(
                mock.patch.object(module, "Investigation", return_value=runner)
            )
            stack.enter_context(mock.patch.object(module.threading, "Thread"))
            self.assertEqual(module.main(), 3)
        self.assertEqual(
            json.loads((output / "terminal.json").read_text())["status"], "partial"
        )
        measurement = json.loads((output / "measurement_000000.json").read_text())
        self.assertEqual(measurement["metrics"]["rows"], runner.rows)
        self.assertEqual(
            json.loads((runner.root / "results.json").read_text())["rows"], runner.rows
        )

    def test_actual_exp2_enabled_disabled_injected_restores_are_identical(self):
        from tests.exp12_helpers import patch_wandb, state_differences
        from tests.test_exp12_fork import run_exp1, run_arm, fork_overrides
        from experiments.exp12.state import latest_state_dir

        for p in patch_wandb():
            self.addCleanup(p.stop)
        source, disabled, enabled = (
            self.root / n for n in ("parent", "arm_off", "arm_on")
        )
        overrides = fork_overrides(str(self.root / "results"))
        with mock.patch.dict(os.environ):
            os.environ.pop(recording.ENVIRONMENT_VARIABLE, None)
            run_exp1(str(source), overrides)
            run_arm(str(disabled), overrides, str(source), "injected")
        output = self.root / "sidecars"
        with mock.patch.dict(os.environ, {recording.ENVIRONMENT_VARIABLE: str(output)}):
            run_arm(str(enabled), overrides, str(source), "injected")
        self.assertEqual(
            state_differences(
                latest_state_dir(disabled / "state"),
                latest_state_dir(enabled / "state"),
            ),
            [],
        )
        terminal = json.loads((output / "terminal.json").read_text())
        self.assertEqual(terminal["status"], "complete")
        self.assert_checkpoint_receipt(enabled, terminal)
        self.assertIn(
            "injected_optimizer", terminal["optimizer"]["critic"]["transformation"]
        )
        self.assertIn("injection", terminal["architectures"])
        measurements = [
            json.loads(p.read_text()) for p in sorted(output.glob("measurement_*.json"))
        ]
        self.assertEqual(measurements[0]["metrics"], {"event": "checkpoint_restored"})
        self.assertGreater(
            measurements[-1]["optimizer_update"], measurements[0]["optimizer_update"]
        )

    def test_actual_exp1_enabled_disabled_checkpoints_are_identical(self):
        import numpy as np
        from tests.exp12_helpers import patch_wandb, state_differences, tiny_overrides
        from tests.test_exp12_fork import run_exp1
        from experiments.exp12.state import latest_state_dir

        patchers = patch_wandb()
        for p in patchers:
            self.addCleanup(p.stop)
        overrides = tiny_overrides(
            steps=100,
            extra=[
                "fork.enabled=false",
                "probe.checks=2",
                "env.max_episode_steps=20",
                f"results_root={self.root / 'results'}",
            ],
        )
        disabled, enabled = self.root / "disabled", self.root / "enabled"
        with mock.patch.dict(os.environ):
            os.environ.pop(recording.ENVIRONMENT_VARIABLE, None)
            with mock.patch(
                "analysis.campaign.provenance.capture",
                side_effect=AssertionError("disabled capture"),
            ):
                run_exp1(str(disabled), overrides)
        output = self.root / "sidecars"
        with mock.patch.dict(os.environ, {recording.ENVIRONMENT_VARIABLE: str(output)}):
            run_exp1(str(enabled), overrides)
        self.assertEqual(
            state_differences(
                latest_state_dir(disabled / "state"),
                latest_state_dir(enabled / "state"),
            ),
            [],
        )
        terminal = json.loads((output / "terminal.json").read_text())
        self.assertEqual(terminal["status"], "complete")
        self.assert_checkpoint_receipt(enabled, terminal)
        self.assertEqual(terminal["interaction_step"], 100)
        self.assertGreater(terminal["optimizer_update"], 0)
        self.assertEqual(terminal["optimizer"]["critic"]["learning_rate"], 1e-4)
        self.assertEqual(terminal["optimizer"]["critic"]["weight_decay"], 0.01)
        self.assertGreater(terminal["parameter_counts"]["critic"], 0)
        self.assertEqual(terminal["seed"], 1)
        self.assertFalse(terminal["missing_fields"])
        self.assertNotIn("populated_transitions", terminal["config"]["buffer"])
        np.testing.assert_equal(terminal["config"]["probe"]["steps"], 20)


class MutationHarnessTest(unittest.TestCase):
    def test_guard_detects_commit_and_tracked_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:

            def git(*args):
                subprocess.run(
                    ["git", "-C", tmp, *args], check=True, capture_output=True
                )

            git("init")
            git("config", "user.email", "test@example.invalid")
            git("config", "user.name", "Test")
            path = Path(tmp) / "source.py"
            path.write_text("original")
            git("add", ".")
            git("commit", "-m", "initial")
            guard = StableSource(tmp)
            guard.check()
            path.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                guard.check()
            path.write_text("original")
            guard.check()
            git("commit", "--allow-empty", "-m", "new revision")
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                guard.check()

    def test_source_revision_change_causes_the_reported_restored_setup_failure(self):
        from experiments.exp2_arm import check_matches_parent
        from tests.exp12_helpers import compose, tiny_overrides

        cfg = compose(tiny_overrides())
        parent_code = {"commit": "a" * 40, "dirty": False}
        parent = {"code": parent_code}
        with mock.patch(
            "experiments.exp2_arm.load_run_metadata", return_value=parent
        ), mock.patch(
            "experiments.exp2_arm.code_version",
            return_value={"commit": "b" * 40, "dirty": False},
        ):
            with self.assertRaisesRegex(
                ValueError, "arm source commit/dirty state differs"
            ):
                check_matches_parent(cfg, Path("/unused"))

    def test_twin_fixture_cleans_up_even_when_setup_fails(self):
        from tests.test_exp12_twin_critic import TwinForkEndToEndTest
        import wandb

        before, before_run = wandb.init, wandb.run

        def fail_after_wandb_init(*args, **kwargs):
            wandb.init()
            raise RuntimeError("setup failed")

        suite = unittest.defaultTestLoader.loadTestsFromName(
            "test_runs_are_twin_and_record_the_critic_count", TwinForkEndToEndTest
        )
        with mock.patch(
            "tests.test_exp12_twin_critic.run_exp1",
            side_effect=fail_after_wandb_init,
        ):
            result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        self.assertEqual(len(result.errors), 1)
        self.assertIs(wandb.init, before)
        self.assertIs(wandb.run, before_run)
        self.assertFalse(Path(TwinForkEndToEndTest.tmp).exists())
