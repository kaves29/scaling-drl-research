"""Independent adversarial review tests for the Exp3 pilots (CPU, tiny fixtures).

Tolerances below are CPU oracle tolerances for alternative computation graphs, never
pilot criteria. Run from main/: python -m unittest tests.test_exp3_review -v
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from experiments.exp3.artifacts import Artifact, core, require_matched, tree_equal
from experiments.exp3.guidance import intervention_gradient, measure_chunk
from experiments.exp3.passive import PassivePair, assert_replay_equal
from experiments.exp3.streams import StreamReader
from experiments.exp3.target_policy import targets
from tests.test_exp12_diagnostics import _agent, _batches
from experiments.exp3.runner import make_passive
from tests import test_exp3_pilots as pilots_tests


def obs(n=6):
    return jnp.asarray(_batches(1)["observation"][0, :n])


def flat(tree):
    return np.asarray(ravel_pytree(tree)[0])


class GuidanceReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.u, cls.i = core(_agent(seed=1)), core(_agent(seed=2))

    def test_matched_base_noise_across_different_actors(self):
        """Pilot1 reuses one chunk key for every actor: the Gaussian base noise must be identical."""
        o, key = obs(), jax.random.PRNGKey(3)
        eps, actions = [], []
        for agent in (self.u, self.i):
            values, _ = measure_chunk(agent.actor, self.u.critic, self.i.critic, o, key, 0.01, False)
            dist = agent.actor(observations=o)
            np.testing.assert_array_equal(values["action"], dist.sample(seed=key))
            base = dist.distribution
            eps.append(np.asarray((base.sample(seed=key) - base.mean()) / base.stddev()))
            actions.append(np.asarray(values["action"]))
        self.assertFalse(np.allclose(actions[0], actions[1]))  # genuinely different actors
        np.testing.assert_allclose(eps[0], eps[1], rtol=0, atol=1e-3)  # (mean + std*z - mean)/std roundoff

    def test_per_state_gradients_do_not_depend_on_other_states(self):
        o, key = obs(), jax.random.PRNGKey(4)
        a, _ = measure_chunk(self.u.actor, self.u.critic, self.i.critic, o, key, 0.01, False)
        b, _ = measure_chunk(self.u.actor, self.u.critic, self.i.critic, o.at[1:].multiply(-3.0), key, 0.01, False)
        for k in ("sac_parameter_gradient_u", "critic_parameter_gradient_i", "dq_da_u"):
            np.testing.assert_allclose(a[k][0], b[k][0], rtol=1e-5, atol=1e-7, err_msg=k)

    def test_panel_mean_of_per_state_gradients_equals_ordinary_gradient(self):
        o, key = obs(), jax.random.PRNGKey(5)
        values, grads = measure_chunk(self.u.actor, self.u.critic, self.i.critic, o, key, 0.01, False)
        for label in ("u", "i"):
            np.testing.assert_allclose(values[f"sac_parameter_gradient_{label}"].mean(0), flat(grads[label]),
                                       rtol=1e-5, atol=1e-7)

    def test_substitutions_reduce_to_ordinary_sac_gradient_when_signals_coincide(self):
        """With U == I, direction/magnitude must be the ordinary SAC gradient, via BOTH implemented spaces."""
        o, key, alpha = obs(), jax.random.PRNGKey(6), self.u.temperature()
        args = (self.u.actor, self.u.critic, self.u.critic, o, key, alpha, False)
        sanity = flat(intervention_gradient(*args, "sanity", "error"))
        for space in ("action", "parameter_per_state"):
            for mode in ("direction", "magnitude"):
                g = flat(intervention_gradient(*args, mode, "error", space=space))
                np.testing.assert_allclose(g, sanity, rtol=1e-4, atol=1e-6, err_msg=f"{space}/{mode}")

    def test_action_space_direction_changes_only_the_critic_term(self):
        """direction(U,I) - sanity must equal the critic-term difference J^T(gU - gI'), entropy untouched."""
        o, key, alpha = obs(), jax.random.PRNGKey(7), self.u.temperature()
        args = (self.u.actor, self.u.critic, self.i.critic, o, key, alpha, False)
        d0 = flat(intervention_gradient(*args, "direction", "error"))
        d1 = flat(intervention_gradient(self.u.actor, self.u.critic, self.i.critic, o, key, alpha * 3, False,
                                        "direction", "error"))
        s0 = flat(intervention_gradient(*args, "sanity", "error"))
        s1 = flat(intervention_gradient(self.u.actor, self.u.critic, self.i.critic, o, key, alpha * 3, False,
                                        "sanity", "error"))
        # Changing alpha changes both by the same entropy gradient.
        np.testing.assert_allclose(d1 - d0, s1 - s0, rtol=1e-4, atol=1e-6)


class TargetReviewTest(unittest.TestCase):
    def test_frozen_target_log_density_matches_ordinary_sac_for_unsaturated_draws(self):
        agent = core(_agent(seed=1))
        b = {k: jnp.asarray(v[0, :6]) for k, v in _batches(1).items()}
        noise = jax.random.normal(jax.random.PRNGKey(8), (6, b["action"].shape[-1]))
        t, info = targets(agent.actor, agent._target_critic, b, noise, 0.1, 0.99, 1, False)
        dist = agent.actor(observations=b["next_observation"])
        rows = np.asarray((jnp.abs(info["action"]) < 0.99).all(-1))
        self.assertGreater(rows.sum(), 0)
        np.testing.assert_allclose(np.asarray(info["log_prob"])[rows],
                                   np.asarray(dist.log_prob(info["action"]))[rows], rtol=1e-3, atol=1e-3)
        q = agent._target_critic(observations=b["next_observation"], actions=info["action"]).reshape(-1)
        expected = b["reward"] + 0.99 * (1 - b["terminated"]) * (q - 0.1 * info["log_prob"])
        np.testing.assert_allclose(t, expected, rtol=1e-5, atol=1e-5)


class PassiveReviewTest(unittest.TestCase):
    """Built on Codex's FixtureTest artifacts (fork at 20, active U and I continued to 24)."""

    @classmethod
    def setUpClass(cls):
        pilots_tests.FixtureTest.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        pilots_tests.FixtureTest.tearDownClass.__func__(cls)

    def pair(self, inject):
        u, i = Artifact(**self.fork), Artifact(**self.fork)
        if inject:
            i.inject("last", 11)
        reader = StreamReader(self.root / "stream_u")
        return PassivePair(u, i, u.meta["numpy_rng_state"], "source_statistics", reader.fingerprint), reader

    def test_isolation_without_injection_learners_stay_bit_identical(self):
        """The only asymmetry between passive arms must be the injection itself."""
        p, reader = self.pair(inject=False)
        for row in reader:
            p.deliver(row, 2)
        a, b = (core(v.agent) for v in p.learners.values())
        self.assertTrue(tree_equal((a._actor, a._critic, a._target_critic, a._temperature, a._rng),
                                   (b._actor, b._critic, b._target_critic, b._temperature, b._rng)))
        assert_replay_equal(*(v.buffer for v in p.learners.values()))

    def test_injection_changes_critic_and_propagates_to_actor(self):
        p, reader = self.pair(inject=True)
        for row in reader:
            p.deliver(row, 2)
        a, b = (core(v.agent) for v in p.learners.values())
        self.assertFalse(tree_equal(a._actor, b._actor))
        self.assertTrue(tree_equal(a._rng, b._rng))

    def test_passive_replay_and_normalization_reproduce_active_source(self):
        """Arrival order/content and recorded normalization must rebuild the active U replay exactly."""
        p, reader = self.pair(inject=False)
        for row in reader:
            p.deliver(row, 2)
        active = Artifact(**self.artifacts["untreated"])
        assert_replay_equal(p.learners["u"].buffer, active.buffer)
        self.assertTrue(tree_equal(vars(p.learners["u"].agent.obs_rms), vars(active.agent.obs_rms)))

    def test_passive_fidelity_when_action_key_consumption_is_emulated(self):
        """Positive control: emulating the one key split per sample_actions call, the passive U learner
        must reproduce the ACTIVE U learner bit-for-bit. Without it, only the JAX key stream differs."""
        active = core(Artifact(**self.artifacts["untreated"]).agent)
        for emulate in (True, False):
            p, reader = self.pair(inject=False)
            for row in reader:
                if emulate:
                    for v in p.learners.values():
                        core(v.agent)._rng = jax.random.split(core(v.agent)._rng)[0]
                p.deliver(row, 2)
            a = core(p.learners["u"].agent)
            same = tree_equal((a._actor, a._critic, a._target_critic, a._temperature),
                              (active._actor, active._critic, active._target_critic, active._temperature))
            self.assertEqual(same, emulate, f"emulate={emulate}")

    def test_stage_b_passive_injection_must_match_the_injected_source_arm(self):
        """Before review, a passive I learner with another m/seed than the stream's arm was accepted."""
        reader = StreamReader(self.root / "stream_i")
        spec = pilots_tests.FixtureTest.spec(self, 2)
        make_passive(spec, reader, source="i")  # the stream's actual injection: last/11
        for change in ({"injection_seed": 12}, {"injection_m": "all"}):
            with self.assertRaisesRegex(ValueError, "injection differs"):
                make_passive({**spec, **change}, reader, source="i")

    def test_recorder_resumes_after_a_crash_between_checkpoints(self):
        """A crash after stream chunks were published past the last routine save must not strand the
        stream ahead of the restored checkpoint (production jobs resume across Slurm segments)."""
        import json
        import shutil
        import tempfile
        from pathlib import Path

        from experiments.exp12.state import state_differences
        from experiments.exp12.trainer import Exp12Trainer
        from experiments.exp3.recording import record_exp2_scope
        from tests.exp12_helpers import compose, tiny_overrides

        cfg = compose(tiny_overrides(steps=40, extra=["run_role=dev", "diagnostics.kl_reference_size=4",
                                                      "actor_grad_cosine_every=10000"]))
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
                t.extra_state["fork"] = {"fork_step": 20,
                                         "run_key": json.loads(self.metadata.read_text())["identity"]["run_key"]}

                def save_then_crash(trainer):
                    if trainer.interaction_step == 22:
                        trainer.save(trainer.run_dir / "state")
                    if trainer.interaction_step == 23:
                        raise RuntimeError("simulated timeout between checkpoints")

                with self.assertRaisesRegex(RuntimeError, "simulated timeout"):
                    with record_exp2_scope(root / "stream", 1):  # chunk_size 1: publish eagerly
                        t.train(24, after_step=save_then_crash)
                from experiments.exp12.state import latest_state_dir
                t.restore(latest_state_dir(t.run_dir / "state"))
                with record_exp2_scope(root / "stream", 1, resume=True):
                    t.train(24)
                    t.save(t.run_dir / "state")
                self.assertEqual([r["step"] for r in StreamReader(root / "stream")], [21, 22, 23, 24])
                self.assertEqual(state_differences(t.save(root / "final"), self.ustate), [])
            finally:
                t.close()

    def test_fork_time_pilot1_needs_an_injected_fork_step_artifact_that_exp2_does_not_retain(self):
        """require_matched demands injection provenance in a saved I checkpoint at the same step as U.

        exp2_arm does write a post-injection save at the fork step, but the next routine save deletes
        it; retention (tests.test_exp3_capture) keeps it."""
        f = Artifact(**self.fork)
        with self.assertRaisesRegex(ValueError, "injection provenance"):
            require_matched(Artifact(**self.fork), Artifact(**self.fork), f)


if __name__ == "__main__":
    unittest.main()
