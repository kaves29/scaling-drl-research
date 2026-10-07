"""Runtime regressions that preserve checkpoint values and continuation."""

import json
import pickle
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jax
import numpy as np
import orbax.checkpoint

from tests.test_exp12_diagnostics import _agent, _batches


class RestoreRuntimeTest(unittest.TestCase):
    def test_twin_restore_uses_saved_reference_shape(self):
        from tests.test_exp12_twin_critic import _twin_agent

        original = orbax.checkpoint.PyTreeCheckpointer.restore
        with tempfile.TemporaryDirectory() as root:
            source = _twin_agent()
            source.update_many(0, _batches(2), 30)
            source.churn_ref_batch = jax.tree_util.tree_map(
                lambda x: x[:3], source.churn_ref_batch
            )
            source.save_checkpoint(root)
            restored = _twin_agent()
            with mock.patch.object(
                orbax.checkpoint.PyTreeCheckpointer,
                "restore",
                autospec=True,
                side_effect=original,
            ) as restore:
                restored.load_checkpoint(root)
            self.assertEqual(restore.call_count, 1)
            self.assert_tree_equal(self.state(source), self.state(restored))
            left = source.update_many(2, _batches(2), 30)
            right = restored.update_many(2, _batches(2), 30)
            self.assert_tree_equal(left, right)
            self.assert_tree_equal(self.state(source), self.state(restored))

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
    def test_progress_failure_preserves_training_exception(self):
        from experiments.exp12.runtime_trace import RuntimeTrace
        from experiments.exp12.trainer import Exp12Trainer

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.jsonl"
            with mock.patch.object(
                Exp12Trainer, "train", side_effect=RuntimeError("training failed")
            ):
                with RuntimeTrace(path) as trace:
                    trace.install()
                    with mock.patch.object(
                        trace, "progress", side_effect=ValueError("timing failed")
                    ):
                        with self.assertRaisesRegex(RuntimeError, "training failed"):
                            Exp12Trainer.train(SimpleNamespace(interaction_step=0), 1)
            events = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertTrue(any(x["event"] == "progress_error" for x in events))
            self.assertEqual(
                next(x for x in events if x["event"] == "train_exit")["error"],
                "RuntimeError",
            )

    def test_backend_startup_failure_is_flushed(self):
        from experiments.exp12.runtime_trace import RuntimeTrace

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.jsonl"

            def fail():
                events = [json.loads(x) for x in path.read_text().splitlines()]
                self.assertEqual(events[-1]["stage"], "backend_init")
                self.assertEqual(events[-1]["event"], "begin")
                raise RuntimeError("backend failed")

            with mock.patch("jax.devices", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "backend failed"):
                    with RuntimeTrace(path):
                        pass
            events = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(events[-1]["error"], "RuntimeError")

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
                t.inject("last", int(cfg.seed))
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
            self.assertEqual(totals["inject"]["calls"], 1)
            self.assertEqual(totals["probe_init"]["calls"], 1)

    def test_cache_outcomes_and_post_fork_progress(self):
        from experiments.exp12.runtime_trace import RuntimeTrace

        cache = SimpleNamespace(
            read=mock.Mock(side_effect=[(None, None), (object(), 1)])
        )
        trainer = SimpleNamespace(
            interaction_step=2,
            update_step=4,
            eval_rows=[{}],
            cfg=SimpleNamespace(action_repeat=2, num_train_envs=1),
            extra_state={"post_fork_evals": [{}, {}]},
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.jsonl"
            with RuntimeTrace(path) as trace:
                trace.wrap(cache, "read", "cache_read")
                self.assertEqual(cache.read(), (None, None))
                self.assertIsNotNone(cache.read()[0])
                trace.progress(trainer)
            events = [json.loads(x) for x in path.read_text().splitlines()]
            ends = [
                x for x in events if x["event"] == "end" and x["stage"] == "cache_read"
            ]
            self.assertEqual([x["cache_hit"] for x in ends], [False, True])
            progress = next(x for x in events if x["event"] == "progress")
            self.assertEqual(progress["post_fork_evaluation_episodes"], 2)
            self.assertEqual(cache.read.call_count, 2)

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


class CompleteStateComparisonTest(unittest.TestCase):
    def test_replay_metadata_and_state_keys_are_compared(self):
        from tests.exp12_helpers import compose, patch_wandb, tiny_overrides
        from experiments.exp12.state import state_differences
        from experiments.exp12.trainer import Exp12Trainer

        patchers = patch_wandb()
        self.addCleanup(lambda: [p.stop() for p in patchers])
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cfg = compose(tiny_overrides(extra=["env.max_episode_steps=20"]))
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)
            trainer = Exp12Trainer(cfg, str(root / "run"))
            self.addCleanup(trainer.close)
            trainer.start()
            trainer.train(15)
            state = trainer.save(root / "state")
            other = root / "copy"
            shutil.copytree(state, other)
            self.assertEqual(state_differences(state, other), [])
            path = other / "buffer_meta.pkl"
            original = path.read_bytes()
            for key in ("current_idx", "num_in_buffer", "n_step_transitions"):
                with self.subTest(key=key):
                    meta = pickle.loads(original)
                    meta[key] = (
                        [{"reward": np.ones(1)}]
                        if key == "n_step_transitions"
                        else meta[key] + 1
                    )
                    path.write_bytes(pickle.dumps(meta))
                    self.assertIn(f"buffer_meta:{key}", state_differences(state, other))
            path.write_bytes(original)
            with mock.patch(
                "experiments.exp12.state.load_agent_tree",
                side_effect=[{"left": np.ones(1)}, {"right": np.ones(1)}],
            ):
                self.assertIn("agent:structure", state_differences(state, other))
            path = other / "meta.pkl"
            meta = pickle.loads(path.read_bytes())
            meta["unexpected_state"] = 1
            path.write_bytes(pickle.dumps(meta))
            self.assertIn("meta:unexpected_state", state_differences(state, other))
            path = other / "obs_rms.pkl"
            rms = pickle.loads(path.read_bytes())
            rms["unexpected_state"] = 1
            path.write_bytes(pickle.dumps(rms))
            self.assertIn("obs_rms:unexpected_state", state_differences(state, other))
            path = other / "buffer.npz"
            with np.load(path) as data:
                arrays = {k: data[k] for k in data.files}
            np.savez(path, **arrays, unexpected_state=np.ones(1))
            self.assertIn("buffer:unexpected_state", state_differences(state, other))
            self.assertEqual(
                set(state_differences(state, other)),
                set(state_differences(other, state)),
            )
