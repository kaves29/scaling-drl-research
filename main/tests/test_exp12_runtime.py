"""Runtime regressions that preserve checkpoint values and continuation."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import numpy as np
import orbax.checkpoint

from tests.test_exp12_diagnostics import _agent, _batches


class RestoreRuntimeTest(unittest.TestCase):
    def test_one_payload_restore_with_and_without_reference(self):
        original = orbax.checkpoint.PyTreeCheckpointer.restore
        for trained in (False, True):
            with self.subTest(trained=trained), tempfile.TemporaryDirectory() as root:
                source = _agent()
                if trained:
                    source.update_many(0, _batches(2), 30)
                source.save_checkpoint(root)
                restored = _agent()
                with mock.patch.object(
                    orbax.checkpoint.PyTreeCheckpointer,
                    "restore",
                    autospec=True,
                    side_effect=original,
                ) as restore:
                    restored.load_checkpoint(root)
                self.assertEqual(restore.call_count, 1)
                self.assert_tree_equal(self.state(source), self.state(restored))
                left = source.update_many(2 if trained else 0, _batches(2), 30)
                right = restored.update_many(2 if trained else 0, _batches(2), 30)
                self.assert_tree_equal(left, right)
                self.assert_tree_equal(self.state(source), self.state(restored))

    def state(self, agent):
        agent = getattr(agent, "agent", agent)
        return (
            agent._rng,
            agent.actor.params,
            agent.critic.params,
            agent._target_critic.params,
            agent.actor.opt_state,
            agent.critic.opt_state,
            agent._temperature.params,
            agent._temperature.opt_state,
            agent._actor.update_step,
            agent._critic.update_step,
            agent._target_critic.update_step,
            agent._temperature.update_step,
            agent.churn_ref_batch,
        )

    def assert_tree_equal(self, left, right):
        a, ta = jax.tree_util.tree_flatten(left)
        b, tb = jax.tree_util.tree_flatten(right)
        self.assertEqual(ta, tb)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


class RuntimeTraceTest(unittest.TestCase):
    def test_instrumentation_preserves_complete_state(self):
        from tests.exp12_helpers import compose, patch_wandb, tiny_overrides
        from experiments.exp12.run_probes import RunProbes
        from experiments.exp12.runtime_trace import RuntimeTrace
        from experiments.exp12.state import latest_state_dir, state_differences
        from experiments.exp12.trainer import Exp12Trainer

        patchers = patch_wandb()
        self.addCleanup(lambda: [p.stop() for p in patchers])
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)

            def run(name):
                cfg = compose(
                    tiny_overrides(steps=400, extra=["env.max_episode_steps=20"])
                )
                np.random.seed(cfg.seed)
                random.seed(cfg.seed)
                t = Exp12Trainer(cfg, str(root / name))
                t.start()
                probes = RunProbes(t, str(root / name))
                t.train(
                    42,
                    after_step=probes.maybe_check,
                    before_first_update=probes.capture_fresh,
                )
                t.save(root / name / "state")
                t.close()
                return latest_state_dir(root / name / "state")

            plain = run("plain")
            trace_path = root / "trace.jsonl"
            with RuntimeTrace(trace_path, synchronize=True, progress_every=10) as trace:
                trace.install()
                traced = run("traced")
            self.assertEqual(state_differences(plain, traced), [])
            self.assertEqual(
                (plain / "buffer_meta.pkl").read_bytes(),
                (traced / "buffer_meta.pkl").read_bytes(),
            )
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            progress = [e for e in events if e["event"] == "progress"][-1]
            self.assertEqual(
                (
                    progress["interaction_step"],
                    progress["update_step"],
                    progress["last_check"],
                ),
                (42, 66, 2),
            )
            totals = events[-1]["totals"]
            self.assertEqual(totals["update_many"]["calls"], 33)
            self.assertEqual(totals["probe_round"]["calls"], 15)
            self.assertEqual(totals["train_env_step"]["calls"], 42)

    def test_trace_preserves_exception_and_refuses_overwrite(self):
        from experiments.exp12.runtime_trace import RuntimeTrace

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.jsonl"
            with self.assertRaisesRegex(ValueError, "original"):
                with RuntimeTrace(path):
                    raise ValueError("original")
            self.assertEqual(
                json.loads(path.read_text().splitlines()[-1])["error"], "ValueError"
            )
            with self.assertRaises(FileExistsError):
                with RuntimeTrace(path):
                    pass
