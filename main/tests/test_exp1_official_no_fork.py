"""Official Exp1 keeps scaled critics on the ordinary training path."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from dotmap import DotMap

from experiments import exp1


class OfficialExp1NoForkTest(unittest.TestCase):
    def test_official_run_rejects_existing_exp2_fork(self):
        cfg = DotMap({"official_exp1_version": "no-hb-v1", "seed": 1})
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            for name in ("set_matmul_precision", "configure_compilation_cache"):
                stack.enter_context(mock.patch.object(exp1, name))
            stack.enter_context(mock.patch.object(exp1, "compose_config", return_value=cfg))
            stack.enter_context(mock.patch.object(exp1, "latest_state_dir", return_value=None))
            stack.enter_context(mock.patch.object(exp1.fork, "is_ready", return_value=True))
            with self.assertRaisesRegex(ValueError, "cannot resume an Exp2 fork/control state"):
                exp1.run({"checkpoint_dir": str(Path(tmp) / "run"),
                          "config_name": "base_exp1_official", "config_path": tmp,
                          "overrides": []})

    def test_large_run_ignores_fork_trigger_and_finishes_normal_budget(self):
        class Trainer:
            instances = []

            def __init__(self, _cfg, _run_dir):
                self.extra_state = {}
                self.metrics_rows = []
                self.interaction_step = 0
                self.train_end = None
                self.instances.append(self)

            def start(self):
                pass

            def train(self, end, after_step, before_first_update):
                self.train_end = end
                for step in range(1, end + 1):
                    self.interaction_step = step
                    after_step(self)

            def save(self, _path):
                pass

            def close(self):
                pass

        class Probes:
            f_star = {"interaction_step": 2, "check_index": 1}
            records = []
            dir = "probe-output"
            fresh_path = "fresh-critic"

            def __init__(self, _trainer, _run_dir):
                pass

            def maybe_check(self, _trainer):
                pass

            def capture_fresh(self, *_args):
                pass

            def write_csv(self):
                pass

        cfg = DotMap({
            "official_exp1_version": "no-hb-v1", "seed": 1,
            "num_interaction_steps": 3, "num_env_steps": 6, "action_repeat": 2,
            "run_role": "confirmatory", "results_root": "/tmp/results",
            "probe": {"enabled": True, "checks": 50},
            "fork": {"enabled": True, "architectures": ["D4W1024", "D4W1536"]},
        })
        identity = SimpleNamespace(run_key="exp1-D4W1536-dog-run-seed1",
                                   architecture="D4W1536", environment="dog-run", seed=1)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            run_dir = Path(tmp) / "run"
            args = {"checkpoint_dir": str(run_dir), "config_name": "base_exp1_official",
                    "config_path": tmp, "overrides": [], "checkpoint_start_frac": 0.0,
                    "checkpoint_interval": 100}
            for name in ("set_matmul_precision", "configure_compilation_cache", "record_metadata"):
                stack.enter_context(mock.patch.object(exp1, name))
            stack.enter_context(mock.patch.object(exp1, "compose_config", return_value=cfg))
            stack.enter_context(mock.patch.object(exp1, "run_identity", return_value=identity))
            stack.enter_context(mock.patch.object(exp1, "code_version", return_value={"commit": "test"}))
            stack.enter_context(mock.patch.object(exp1, "latest_state_dir", return_value=None))
            stack.enter_context(mock.patch.object(exp1.fork, "is_ready", return_value=False))
            stack.enter_context(mock.patch.object(exp1.fork, "check_validation_flags"))
            stack.enter_context(mock.patch.object(exp1, "Exp12Trainer", Trainer))
            stack.enter_context(mock.patch.object(exp1, "RunProbes", Probes))
            write_fork = stack.enter_context(mock.patch.object(exp1.fork, "write_fork"))
            control_eval = stack.enter_context(mock.patch.object(exp1.fork, "post_fork_eval"))
            fork_plan = stack.enter_context(mock.patch.object(exp1.fork, "fork_plan"))
            ledger = stack.enter_context(mock.patch.object(exp1.ledger, "write_run"))
            exp2_ledger = stack.enter_context(mock.patch.object(exp1.exp2_ledger, "write_arm"))

            exp1.run(args)

            self.assertEqual(len(Trainer.instances), 1)
            self.assertEqual(Trainer.instances[0].train_end, 3)
            self.assertTrue((run_dir / exp1.DONE_MARKER).exists())
            self.assertIsNone(ledger.call_args.kwargs["fork_step"])
            write_fork.assert_not_called()
            fork_plan.assert_not_called()
            control_eval.assert_not_called()
            exp2_ledger.assert_not_called()


if __name__ == "__main__":
    unittest.main()
