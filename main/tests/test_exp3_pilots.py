"""Independent CPU-only tiny engineering fixtures; no scientific pilot budgets."""

import copy
import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from experiments.exp12.state import state_differences
from experiments.exp12.trainer import Exp12Trainer
from experiments.exp1 import record_metadata
from experiments.exp3.artifacts import (
    Artifact,
    core,
    digest,
    inspect_artifact,
    require_matched,
    state_fingerprint,
    tree_equal,
)
from experiments.exp3.guidance import (
    assert_fork_equivalence,
    intervention_gradient,
    measure_chunk,
    optimizer_step,
    substitution,
)
from experiments.exp3.passive import PassivePair, assert_replay_equal, replay_batch
from experiments.exp3.protocol import manifest, validate_protocol
from experiments.exp3.runner import array_file, make_passive, run, target_checkpoint
from experiments.exp3.streams import (
    Benchmark,
    StreamReader,
    StreamWriter,
    record_training,
)
from experiments.exp3.target_policy import TargetPair, fit_target, targets
from scale_rl.agents.sac.sac_update import update_actor, update_critic
from tests.exp12_helpers import compose, patch_wandb, tiny_overrides
from tests.test_exp12_diagnostics import _agent, _batches


def batch():
    return {k: jnp.asarray(v[0, :3]) for k, v in _batches(1).items()}


class GuidanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.u = core(_agent(seed=1))
        cls.i = core(_agent(seed=2))

    def test_per_state_sac_gradient_matches_independent_loss_and_update(self):
        a, u, i, b, key = (
            self.u.actor,
            self.u.critic,
            self.i.critic,
            batch(),
            jax.random.PRNGKey(5),
        )
        alpha = self.u.temperature()
        values, grads = measure_chunk(a, u, i, b["observation"], key, alpha, False)

        def ref(p):
            d = a.apply({"params": p}, observations=b["observation"])
            act = d.sample(seed=key)
            return (
                alpha * d.log_prob(act)
                - u(observations=b["observation"], actions=act).reshape(-1)
            ).mean()

        expected = jax.grad(ref)(a.params)
        for x, y in zip(
            jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(grads["u"])
        ):
            np.testing.assert_allclose(
                x, y, rtol=2e-5, atol=1e-6
            )  # CPU oracle tolerance, never a pilot criterion
        d = a(observations=b["observation"])
        np.testing.assert_array_equal(values["action"], d.sample(seed=key))
        updated = optimizer_step(a, expected)
        ordinary = update_actor(key, a, u, self.u.temperature, b, False)[0]
        self.assertTrue(tree_equal(updated, ordinary))

    def test_full_substitution_and_sanity_reuse_exact_actor_gradient(self):
        b, key, a = batch(), jax.random.PRNGKey(2), self.u.actor
        for mode, c in (("sanity", self.u.critic), ("full", self.i.critic)):
            g = intervention_gradient(
                a,
                self.u.critic,
                self.i.critic,
                b["observation"],
                key,
                self.u.temperature(),
                False,
                mode,
                "error",
            )
            actual = optimizer_step(a, g)
            ref = update_actor(key, a, c, self.u.temperature, b, False)[0]
            self.assertTrue(tree_equal(actual, ref))

    def test_direction_and_magnitude_and_zero_are_explicit(self):
        u, i = jnp.array([[3.0, 4.0]]), jnp.array([[0.0, 10.0]])
        np.testing.assert_array_equal(
            substitution(u, i, "direction", "error"), [[0, 5]]
        )
        np.testing.assert_array_equal(
            substitution(u, i, "magnitude", "error"), [[6, 8]]
        )
        with self.assertRaisesRegex(ValueError, "undefined"):
            substitution(u, jnp.zeros_like(i), "direction", "error")
        np.testing.assert_array_equal(
            substitution(u, jnp.zeros_like(i), "direction", "keep_baseline"), u
        )

    def test_parameter_space_substitution_is_explicit_and_keeps_entropy(self):
        from jax.flatten_util import ravel_pytree

        obs, key = batch()["observation"], jax.random.PRNGKey(4)
        values, _ = measure_chunk(
            self.u.actor, self.u.critic, self.i.critic, obs, key, 0.01, False
        )
        expected = (
            values["sac_parameter_gradient_u"]
            - values["critic_parameter_gradient_u"]
            + substitution(
                values["critic_parameter_gradient_u"],
                values["critic_parameter_gradient_i"],
                "direction",
                "error",
            )
        ).mean(0)
        result = intervention_gradient(
            self.u.actor,
            self.u.critic,
            self.i.critic,
            obs,
            key,
            0.01,
            False,
            "direction",
            "error",
            space="parameter_per_state",
        )
        np.testing.assert_array_equal(ravel_pytree(result)[0], expected)
        for mode, critic in (("sanity", self.u.critic), ("full", self.i.critic)):
            gradient = intervention_gradient(
                self.u.actor,
                self.u.critic,
                self.i.critic,
                obs,
                key,
                self.u.temperature(),
                False,
                mode,
                "error",
                space="parameter_per_state",
            )
            actual = optimizer_step(self.u.actor, gradient)
            reference = update_actor(
                key,
                self.u.actor,
                critic,
                self.u.temperature,
                {"observation": obs},
                False,
            )[0]
            self.assertTrue(tree_equal(actual, reference))

    def test_artifact_specific_critic_normalization_keeps_common_actions(self):
        obs, key = batch()["observation"], jax.random.PRNGKey(9)
        norms = {"u": obs * 2, "i": obs * 3}
        values, _ = measure_chunk(
            self.u.actor, self.u.critic, self.i.critic, obs, key, 0.01, False, norms
        )
        np.testing.assert_array_equal(values["critic_observation_u"], norms["u"])
        np.testing.assert_array_equal(
            values["q_u"],
            self.u.critic(observations=norms["u"], actions=values["action"]),
        )
        np.testing.assert_array_equal(
            values["q_i"],
            self.i.critic(observations=norms["i"], actions=values["action"]),
        )

    def test_no_mutation_rng_or_optimizer_during_measurement(self):
        a, key = self.u.actor, jax.random.PRNGKey(7)
        before = (copy.deepcopy(a), np.asarray(self.u._rng).copy())
        x = measure_chunk(
            a, self.u.critic, self.i.critic, batch()["observation"], key, 0.01, False
        )[0]
        y = measure_chunk(
            a, self.u.critic, self.i.critic, batch()["observation"], key, 0.01, False
        )[0]
        self.assertTrue(tree_equal(x, y))
        self.assertTrue(tree_equal(a, before[0]))
        np.testing.assert_array_equal(self.u._rng, before[1])

    def test_twin_actor_loss_uses_min_not_mean(self):
        g = core(_agent(extra=["agent.critic_use_cdq=true"]))
        b, key = batch(), jax.random.PRNGKey(3)
        out, grad = measure_chunk(
            g.actor, g.critic, g.critic, b["observation"], key, g.temperature(), True
        )
        a = out["action"]
        ref = jax.grad(
            lambda actions: jnp.minimum(
                *g.critic(observations=b["observation"], actions=actions)
            ).sum()
        )(a)
        np.testing.assert_array_equal(out["dq_da_u"], ref)
        np.testing.assert_array_equal(out["q_u"].shape, (2, 3, 1))
        self.assertTrue(tree_equal(grad["u"], grad["i"]))

    def test_fork_optimizer_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fork"):
            assert_fork_equivalence(self.u, self.i)


class TargetTest(unittest.TestCase):
    def test_single_and_twin_targets_and_ordinary_critic_fit(self):
        for twin in (False, True):
            g = core(_agent(extra=[f"agent.critic_use_cdq={str(twin).lower()}"]))
            b, key = batch(), jax.random.PRNGKey(3)
            d = g.actor(observations=b["next_observation"])
            noise = jax.random.normal(key, d.distribution.mean().shape)
            target, values = targets(
                g.actor, g._target_critic, b, noise, g.temperature(), 0.99, 1, twin
            )
            # Independent NumPy arithmetic with clipped target networks and terminal masking.
            q = np.asarray(
                g._target_critic(
                    observations=b["next_observation"], actions=values["action"]
                )
            )
            if twin:
                q = np.minimum(q[0], q[1])
            expected = np.asarray(b["reward"]) + np.float32(0.99) * (
                1 - np.asarray(b["terminated"])
            ) * q.reshape(-1)
            expected -= (
                np.float32(0.99)
                * (1 - np.asarray(b["terminated"]))
                * float(g.temperature())
                * np.asarray(values["log_prob"])
            )
            np.testing.assert_allclose(target, expected, rtol=2e-6, atol=2e-6)
            fit = fit_target(g.critic, b, target)[0]

            def ref_loss(p):
                pred = g.critic.apply(
                    {"params": p}, observations=b["observation"], actions=b["action"]
                )
                losses = (
                    (
                        (pred[0].reshape(-1) - target) ** 2
                        + (pred[1].reshape(-1) - target) ** 2
                    ).mean()
                    if twin
                    else ((pred.reshape(-1) - target) ** 2).mean()
                )
                return losses, {}

            ref = jax.jit(lambda c: c.apply_gradient(ref_loss)[0])(g.critic)
            self.assertTrue(tree_equal(fit, ref))

    def test_shared_noise_terminal_mask_frozen_actor_evaluator_and_alpha(self):
        g = core(_agent())
        p = TargetPair(
            g.critic,
            g.actor,
            g.actor,
            g._target_critic,
            g.temperature(),
            0.99,
            1,
            False,
            jax.random.PRNGKey(1),
        )
        b = batch()
        b["terminated"] = jnp.ones_like(b["terminated"])
        old = copy.deepcopy((p.actors, p.evaluator, p.alpha))
        result = p.update(b)
        np.testing.assert_array_equal(result["u"]["target"], b["reward"])
        np.testing.assert_array_equal(result["u"]["action"], result["i"]["action"])
        self.assertTrue(tree_equal(p.critics["u"], p.critics["i"]))
        self.assertTrue(tree_equal(old, (p.actors, p.evaluator, p.alpha)))

    def test_target_state_is_serializable_without_optimizer_closures(self):
        g = core(_agent())
        p = TargetPair(
            g.critic,
            g.actor,
            g.actor,
            g._target_critic,
            0.01,
            0.99,
            1,
            False,
            jax.random.PRNGKey(1),
        )
        p.update(batch())
        with tempfile.TemporaryDirectory() as d:
            target_checkpoint(Path(d), {"f": p}, np.random.RandomState(1))
            saved = pickle.load(open(Path(d) / "target_state.pkl", "rb"))
            self.assertEqual(saved["pairs"]["f"]["update_step"], 1)

    def test_saturated_actions_keep_preimage_and_finite_log_density(self):
        g = core(_agent())
        b = batch()
        noise = jnp.full_like(
            g.actor(observations=b["next_observation"]).distribution.mean(), 100000.0
        )
        target, info = targets(
            g.actor, g._target_critic, b, noise, 0.01, 0.99, 1, False
        )
        self.assertTrue(np.isfinite(np.asarray(info["log_prob"])).all())
        self.assertTrue(np.isfinite(np.asarray(target)).all())


class FixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.patchers = patch_wandb()
        cfg = compose(
            tiny_overrides(
                steps=40,
                extra=[
                    "run_role=dev",
                    "diagnostics.kl_reference_size=4",
                    "actor_grad_cosine_every=10000",
                ],
            )
        )
        t = Exp12Trainer(cfg, str(cls.root / "source"))
        t.start()
        t.train(20)
        record_metadata(cfg, t.run_dir, False)
        cls.metadata = t.run_dir / "run_metadata.json"
        cls.fstate = t.save(cls.root / "fork_state")
        from experiments.exp12.fork import sample_panel

        cls.fork_panel = cls.root / "fork_panel.npz"
        np.savez(cls.fork_panel, **sample_panel(t, 1, 1))
        cls.fork = {"state": str(cls.fstate), "metadata": str(cls.metadata)}
        cls.provenance = {
            "start_step": 20,
            "fork_metadata_sha256": digest(cls.metadata),
            "config_hash": json.loads(cls.metadata.read_text())["config_hash"],
            "arm": "untreated",
            "fork_state_sha256": state_fingerprint(cls.fstate),
            "injection": None,
        }
        plan = {
            "fork_step": 20,
            "run_key": json.loads(cls.metadata.read_text())["identity"]["run_key"],
        }
        t.extra_state["fork"] = plan
        writer = StreamWriter(cls.root / "stream_u", cls.provenance, 2)
        record_training(t, 24, writer)
        cls.ustate = t.save(cls.root / "u_state")
        t.restore(cls.fstate)
        t.inject("last", 11)
        t.extra_state["fork"] = plan
        writer = StreamWriter(
            cls.root / "stream_i",
            {
                **cls.provenance,
                "arm": "injected",
                "injection": {"m": "last", "seed": 11},
            },
            2,
        )
        record_training(t, 24, writer)
        cls.istate = t.save(cls.root / "i_state")
        t.close()
        cls.artifacts = {
            "fork": cls.fork,
            "untreated": {"state": str(cls.ustate), "metadata": str(cls.metadata)},
            "injected": {"state": str(cls.istate), "metadata": str(cls.metadata)},
        }

    @classmethod
    def tearDownClass(cls):
        for p in cls.patchers:
            p.stop()
        cls.temp.cleanup()

    def spec(self, pilot):
        common = {
            "schema_version": 1,
            "run_role": "dev",
            "approved": True,
            "pilot": pilot,
            "chunk_size": 2,
            "rng_seed": 9,
            "artifacts": self.artifacts,
        }
        if pilot == 1:
            return {
                **common,
                "panel_size": 3,
                "normalization": "common_fork",
                "alpha_source": "fork",
                "intervention_space": "action",
                "interventions": ["sanity", "full"],
                "zero_signal_policy": "error",
                "outcome_eval_episodes": 1,
            }
        if pilot == 2:
            return {
                **common,
                "arrivals": 4,
                "updates_per_arrival": 2,
                "stage_b": False,
                "evaluation_every": 4,
                "evaluation_episodes": 1,
                "checkpoint_every": 2,
                "normalization": "source_statistics",
                "injection_m": "last",
                "injection_seed": 11,
                "streams": {"u": str(self.root / "stream_u")},
                "fork_panel": str(self.fork_panel),
            }
        return {
            **common,
            "enabled": True,
            "updates": 2,
            "batch_size": 3,
            "panel_size": 3,
            "alpha_source": "fork",
            "critic_source": "untreated",
            "evaluator_source": "fork",
            "alternative_evaluator": "injected",
            "checkpoint_every": 1,
            "normalization": "common_fork",
        }

    def test_complete_checkpoint_exact_actor_critic_target_optimizer_rng_normalizer(
        self,
    ):
        a = Artifact(**self.fork)
        from experiments.exp12.state import load_agent_tree

        saved = load_agent_tree(self.fstate)
        g = core(a.agent)
        # Orbax optimizer dictionaries differ from runtime OptState tuples.
        # Compare in the SAME serialization representation, byte for byte.
        with tempfile.TemporaryDirectory() as d:
            a.agent.save_checkpoint(d)
            self.assertTrue(tree_equal(saved, load_agent_tree(Path(d))))
        np.testing.assert_array_equal(saved["rng"], g._rng)
        rms = pickle.load(open(self.fstate / "obs_rms.pkl", "rb"))
        for k in rms:
            self.assertTrue(tree_equal(rms[k], getattr(a.agent.obs_rms, k)))

    def test_missing_artifact_components_are_not_reconstructed(self):
        for name in (
            "obs_rms.pkl",
            "buffer_meta.pkl",
            "buffer.npz",
            "meta.pkl",
            "agent_ckpt",
        ):
            with tempfile.TemporaryDirectory() as d:
                target = Path(d) / "state"
                shutil.copytree(self.fstate, target)
                p = target / name
                shutil.rmtree(p) if p.is_dir() else p.unlink()
                with self.assertRaisesRegex(ValueError, "missing checkpoint component"):
                    inspect_artifact(target, self.metadata)

    def test_provenance_fingerprint_and_checkpoint_pairing_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            value = json.loads(self.metadata.read_text())
            value["resolved_config"]["seed"] += 1
            p = Path(d) / "metadata.json"
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                inspect_artifact(self.fstate, p)
        with self.assertRaisesRegex(ValueError, "steps"):
            require_matched(Artifact(**self.fork), Artifact(self.ustate, self.metadata))

    def test_stream_order_roundtrip_integrity_and_prefix(self):
        reader = StreamReader(self.root / "stream_u")
        rows = list(reader)
        self.assertEqual([r["step"] for r in rows], [21, 22, 23, 24])
        self.assertEqual([r["updates"] for r in rows], [2] * 4)
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "stream"
            shutil.copytree(reader.root, target)
            chunk = target / reader.manifest["chunks"][0]["name"]
            with open(chunk, "ab") as f:
                f.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                list(StreamReader(target))
        with tempfile.TemporaryDirectory() as d:
            writer = StreamWriter(Path(d) / "s", self.provenance, 2)
            with self.assertRaisesRegex(ValueError, "order"):
                writer.append(22, rows[0]["transition"], 2, rows[0]["normalization"])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                StreamReader(writer.root)

    def test_passive_shared_sampling_no_training_experience_and_resume_exact(self):
        reader = StreamReader(self.root / "stream_u")
        rows = list(reader)
        spec = self.spec(2)
        p = make_passive(spec, reader)
        with mock.patch.object(
            p.learners["u"].agent,
            "sample_actions",
            side_effect=AssertionError("active learner"),
        ), mock.patch.object(
            p.learners["i"].agent,
            "sample_actions",
            side_effect=AssertionError("active learner"),
        ):
            p.deliver(rows[0], 2)
            self.assertEqual(p.last_indices.shape, (2, 8))
            assert_replay_equal(*(v.buffer for v in p.learners.values()))
        with tempfile.TemporaryDirectory() as d:
            p.save(d)
            q = make_passive(spec, reader)
            q.restore(d)
            for row in rows[1:]:
                p.deliver(row, 2)
                q.deliver(row, 2)
            for label in ("u", "i"):
                a, b = core(p.learners[label].agent), core(q.learners[label].agent)
                self.assertTrue(
                    tree_equal(
                        (a._actor, a._critic, a._target_critic, a._temperature, a._rng),
                        (b._actor, b._critic, b._target_critic, b._temperature, b._rng),
                    )
                )
                assert_replay_equal(p.learners[label].buffer, q.learners[label].buffer)
            np.testing.assert_array_equal(p.rng.get_state()[1], q.rng.get_state()[1])

    def test_passive_mismatched_timing_replay_rng_and_stream_refused(self):
        reader = StreamReader(self.root / "stream_u")
        pair = make_passive(self.spec(2), reader)
        row = next(iter(reader))
        with self.assertRaisesRegex(ValueError, "timing"):
            pair.deliver(row, 1)
        pair.learners["i"].buffer._actions[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "replay"):
            assert_replay_equal(*(v.buffer for v in pair.learners.values()))
        pair = make_passive(self.spec(2), reader)
        with tempfile.TemporaryDirectory() as d:
            pair.save(d)
            q = make_passive(self.spec(2), reader)
            q.stream_fingerprint = "different"
            with self.assertRaisesRegex(ValueError, "stream_fingerprint"):
                q.restore(d)

    def test_stream_recording_preserves_complete_source_state(self):
        cfg = compose(
            tiny_overrides(
                steps=40,
                extra=[
                    "run_role=dev",
                    "diagnostics.kl_reference_size=4",
                    "actor_grad_cosine_every=10000",
                ],
            )
        )
        with tempfile.TemporaryDirectory() as d:
            t = Exp12Trainer(cfg, str(Path(d) / "run"))
            try:
                t.restore(self.fstate)
                t.extra_state["fork"] = {
                    "fork_step": 20,
                    "run_key": json.loads(self.metadata.read_text())["identity"][
                        "run_key"
                    ],
                }
                t.train(24)
                saved = t.save(Path(d) / "state")
                self.assertEqual(state_differences(saved, self.ustate), [])
            finally:
                t.close()

    def test_all_three_end_to_end_cpu_pilots(self):
        with tempfile.TemporaryDirectory() as d:
            for pilot in (1, 2, 3):
                out = Path(d) / str(pilot)
                report = run(self.spec(pilot), out)
                self.assertTrue((out / "DONE").exists())
                self.assertTrue((out / "SHA256SUMS.json").exists())
                self.assertTrue(report)

    def test_passive_end_to_end_restart_matches_uninterrupted_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            spec = self.spec(2)
            run(spec, root / "reference")
            ordinary = PassivePair.deliver

            def interrupted(pair, record, budget):
                if pair.cursor == 2:
                    raise RuntimeError("simulated interruption")
                return ordinary(pair, record, budget)

            with mock.patch.object(PassivePair, "deliver", interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    run(spec, root / "resumed")
            run(spec, root / "resumed", resume=True)
            from experiments.exp12.state import load_agent_tree

            for label in ("u", "i"):
                states = []
                for mode in ("reference", "resumed"):
                    c = root / mode / "source_u/checkpoints"
                    states.append(
                        c
                        / json.loads((c / "LATEST.json").read_text())["directory"]
                        / label
                    )
                self.assertTrue(
                    tree_equal(load_agent_tree(states[0]), load_agent_tree(states[1]))
                )
                with np.load(states[0] / "buffer.npz") as a, np.load(
                    states[1] / "buffer.npz"
                ) as b:
                    self.assertTrue(tree_equal(dict(a), dict(b)))
                for name in ("buffer_meta.pkl", "obs_rms.pkl", "windows.pkl"):
                    with open(states[0] / name, "rb") as a, open(
                        states[1] / name, "rb"
                    ) as b:
                        self.assertTrue(tree_equal(pickle.load(a), pickle.load(b)))

    def test_target_end_to_end_restart_matches_uninterrupted_and_alternative_evaluator(
        self,
    ):
        import experiments.exp3.runner as runner

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            spec = self.spec(3)
            run(spec, root / "reference")
            ordinary = runner.target_checkpoint

            def interrupted(out, pairs, sampler):
                ordinary(out, pairs, sampler)
                raise RuntimeError("simulated interruption")

            with mock.patch.object(runner, "target_checkpoint", interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    run(spec, root / "resumed")
            run(spec, root / "resumed", resume=True)
            with open(root / "reference/target_state.pkl", "rb") as f:
                expected = pickle.load(f)
            with open(root / "resumed/target_state.pkl", "rb") as f:
                actual = pickle.load(f)
            self.assertTrue(tree_equal(expected, actual))

    def test_source_instrumentation_benchmark_has_no_trajectory_changes(self):
        cfg = compose(
            tiny_overrides(
                steps=40,
                extra=[
                    "run_role=dev",
                    "diagnostics.kl_reference_size=4",
                    "actor_grad_cosine_every=10000",
                ],
            )
        )
        with tempfile.TemporaryDirectory() as d:
            t = Exp12Trainer(cfg, str(Path(d) / "run"))
            try:
                t.restore(self.fstate)
                t.extra_state["fork"] = {
                    "fork_step": 20,
                    "run_key": json.loads(self.metadata.read_text())["identity"][
                        "run_key"
                    ],
                }
                bench = Benchmark()
                writer = StreamWriter(Path(d) / "stream", self.provenance, 2)
                record_training(t, 24, writer, benchmark=bench)
                saved = t.save(Path(d) / "state")
                self.assertEqual(state_differences(saved, self.ustate), [])
                self.assertEqual(bench.values["environment_step"]["count"], 4)
                self.assertEqual(bench.values["sac_update"]["count"], 4)
            finally:
                t.close()

    def test_twin_passive_injection_and_matched_replay_update(self):
        cfg = compose(
            tiny_overrides(
                steps=40,
                extra=[
                    "run_role=dev",
                    "agent.critic_use_cdq=true",
                    "diagnostics.kl_reference_size=4",
                    "actor_grad_cosine_every=10000",
                ],
            )
        )
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            t = Exp12Trainer(cfg, str(root / "source"))
            try:
                t.start()
                t.train(20)
                record_metadata(cfg, t.run_dir, False)
                state = t.save(root / "fork")
                metadata = t.run_dir / "run_metadata.json"
                u, i = Artifact(state, metadata), Artifact(state, metadata)
                i.inject("last", 11)
                p = PassivePair(
                    u, i, u.meta["numpy_rng_state"], "source_statistics", "twin-fixture"
                )
                writer = StreamWriter(root / "stream", {"start_step": 20}, 1)
                record_training(t, 21, writer)
                p.deliver(next(iter(StreamReader(writer.root))), 2)
                self.assertEqual(
                    np.asarray(
                        core(i.agent).critic(
                            observations=jnp.asarray(
                                i.agent._normalize(i.buffer._observations[:2])
                            ),
                            actions=jnp.asarray(i.buffer._actions[:2]),
                        )
                    ).shape,
                    (2, 2, 1),
                )
                self.assertEqual(p.update_step, u.meta["update_step"] + 2)
                assert_replay_equal(u.buffer, i.buffer)
            finally:
                t.close()

    def test_missing_fork_lineage_is_rejected(self):
        u, i, f = (
            Artifact(**self.artifacts[k]) for k in ("untreated", "injected", "fork")
        )
        u.meta["extra_state"].pop("fork")
        with self.assertRaisesRegex(ValueError, "fork lineage"):
            require_matched(u, i, f)

    def test_matched_interaction_with_mismatched_update_count_is_rejected(self):
        u, i = (Artifact(**self.artifacts[k]) for k in ("untreated", "injected"))
        i.meta["update_step"] += 1
        with self.assertRaisesRegex(ValueError, "update counters"):
            require_matched(u, i)

    def test_process_local_source_recorder_restart_is_exact_and_restores_original_loop(
        self,
    ):
        from experiments.exp3.recording import record_exp2_scope

        cfg = compose(
            tiny_overrides(
                steps=40,
                extra=[
                    "run_role=dev",
                    "diagnostics.kl_reference_size=4",
                    "actor_grad_cosine_every=10000",
                ],
            )
        )
        original = Exp12Trainer.train
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            t = Exp12Trainer(cfg, str(root / "source"))
            t.run_dir.mkdir()
            shutil.copyfile(self.metadata, t.run_dir / "run_metadata.json")
            froot = t.run_dir / "fork/state"
            froot.mkdir(parents=True)
            shutil.copytree(self.fstate, froot / self.fstate.name)
            (froot / "LATEST").write_text(self.fstate.name)
            try:
                t.restore(self.fstate)
                t.extra_state["fork"] = {
                    "fork_step": 20,
                    "run_key": json.loads(self.metadata.read_text())["identity"][
                        "run_key"
                    ],
                }
                saved = []

                def stop_at_checkpoint(trainer):
                    if trainer.interaction_step == 22:
                        # Routine root: the recorder publishes arrivals only with routine saves (C3).
                        saved.append(trainer.save(trainer.run_dir / "state"))
                        raise RuntimeError("simulated source interruption")

                with self.assertRaisesRegex(RuntimeError, "simulated source"):
                    with record_exp2_scope(root / "stream", 10):
                        t.train(24, after_step=stop_at_checkpoint)
                self.assertIs(Exp12Trainer.train, original)
                self.assertEqual(
                    StreamReader(root / "stream", require_complete=False).manifest[
                        "count"
                    ],
                    2,
                )
                t.restore(saved[0])
                with record_exp2_scope(root / "stream", 10, resume=True):
                    t.train(24)
                self.assertIs(Exp12Trainer.train, original)
                self.assertEqual(
                    [r["step"] for r in StreamReader(root / "stream")], [21, 22, 23, 24]
                )
                final = t.save(root / "final")
                self.assertEqual(state_differences(final, self.ustate), [])
            finally:
                t.close()

    def test_missing_optimizer_metadata_and_source_rng_mismatch_rejected(self):
        import orbax.checkpoint

        original = orbax.checkpoint.PyTreeCheckpointer.metadata

        def without_optimizer(checkpointer, path):
            data = original(checkpointer, path)
            data["actor"].pop("opt_state")
            return data

        with mock.patch.object(
            orbax.checkpoint.PyTreeCheckpointer, "metadata", without_optimizer
        ):
            with self.assertRaisesRegex(ValueError, "optimizer"):
                inspect_artifact(self.fstate, self.metadata)
        u, i = Artifact(**self.fork), Artifact(**self.fork)
        core(i.agent)._rng = jax.random.PRNGKey(555)
        with self.assertRaisesRegex(ValueError, "RNG"):
            PassivePair(u, i, u.meta["numpy_rng_state"], "source_statistics", "fixture")

    def test_stage_b_requires_independent_injected_source(self):
        s = self.spec(2)
        s["stage_b"] = True
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(KeyError):
                run(s, Path(d) / "out")

    def test_stage_b_all_four_passive_arms_and_source_assignment(self):
        s = self.spec(2)
        s["stage_b"] = True
        s["streams"]["i"] = str(self.root / "stream_i")
        with tempfile.TemporaryDirectory() as d:
            report = run(s, Path(d) / "out")
            self.assertEqual(set(report["sources"]), {"u", "i"})
            self.assertTrue(all(row["passive"] for row in report["sources"].values()))
        with self.assertRaisesRegex(ValueError, "arm assignment"):
            make_passive(s, StreamReader(self.root / "stream_u"), source="i")

    def test_immutable_output_replay_detects_dtype_and_signed_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.npz"
            array_file(p, {"x": np.array([0.0], np.float32)})
            for x in (np.array([-0.0], np.float32), np.array([0.0], np.float64)):
                with self.assertRaisesRegex(ValueError, "differs"):
                    array_file(p, {"x": x})

    def test_no_scientific_defaults_or_unapproved_protocol(self):
        s = self.spec(1)
        s["panel_size"] = None
        with self.assertRaisesRegex(ValueError, "explicit"):
            validate_protocol(s)
        s = self.spec(3)
        s["enabled"] = False
        with self.assertRaisesRegex(ValueError, "opt-in"):
            validate_protocol(s)
        s = self.spec(2)
        s["approved"] = False
        with self.assertRaisesRegex(ValueError, "approved"):
            validate_protocol(s)

    def test_myo_selection_is_not_invented_and_manifest_has_20_unique_cells(self):
        import yaml

        cfg = Path(__file__).resolve().parents[1] / "configs/exp3/pilots.yaml"
        with self.assertRaisesRegex(ValueError, "TWO"):
            manifest(cfg, self.root)
        values = yaml.safe_load(cfg.read_text())
        values["environments"]["myosuite"] = [
            "myo-key-turn",
            "myo-reach",
        ]  # TEST selection, not production
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.yaml"
            p.write_text(yaml.safe_dump(values))
            m = manifest(p, self.root)
            self.assertEqual(len(m["cells"]), 20)
            self.assertEqual(
                len({(c["environment"], c["seed"]) for c in m["cells"]}), 20
            )


if __name__ == "__main__":
    unittest.main()
