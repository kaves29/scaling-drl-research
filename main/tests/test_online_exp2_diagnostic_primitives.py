"""Behavioral checks for the revised online diagnostic primitives."""
import unittest
import numpy as np

from experiments.exp12.online_exp2_diagnostic import ObservationBuffer, ActorUpdateSchedule, reference_roles, diagnostic_actor_step, DiagnosticActorBranch, paired_returns, regime_contrasts, combine_pilot_analyses
from tests import test_online_exp2 as existing_tests
from scale_rl.agents.sac.sac_update import update_actor
import jax
import jax.numpy as jnp
from scale_rl.envs.wrappers.repeat_action import RepeatAction
from tests.test_online_exp2_driver import TinyEnv
from experiments.exp12.online_exp2_pilot import _success, _executed_repeats
from gymnasium.vector import SyncVectorEnv


class DiagnosticPrimitivesTest(unittest.TestCase):
    def test_terminal_success_reads_vector_final_info(self):
        self.assertEqual(_success({"final_info": [{"success": 1.}]}), 1.)
        self.assertEqual(_success({"success": [0.], "final_info": [{"success": 1.}]}), 1.)
        self.assertEqual(_success({}), 0.)

    def test_action_repeat_reports_actual_simulator_steps_at_terminal(self):
        env = RepeatAction(TinyEnv(), action_repeat=4)
        env.reset()
        env.step(np.zeros(2, np.float32))
        self.assertEqual(env.last_executed_repeats, 4)
        env.step(np.zeros(2, np.float32))
        self.assertEqual(env.last_executed_repeats, 1)
        env.close()
        vector = SyncVectorEnv([lambda: RepeatAction(TinyEnv(), action_repeat=4)])
        vector.reset()
        vector.step(np.zeros((1, 2), np.float32))
        self.assertEqual(_executed_repeats(vector), 4)
        vector.step(np.zeros((1, 2), np.float32))
        self.assertEqual(_executed_repeats(vector), 1)
        vector.close()

    def test_identical_critics_and_histories_produce_identical_branch_updates(self):
        existing_tests.OnlineExp2Test.setUpClass()
        fixture = existing_tests.OnlineExp2Test
        branches = [DiagnosticActorBranch(
            fixture.actor, fixture.shadows['small'].critic, fixture.temperature,
            seed=33, batch_size=4, updates_per_interaction=1.5,
            critic_use_cdq=False) for _ in range(2)]
        for value in (1., 2., 3.):
            observations = np.full((1, 3), value, np.float32)
            actions = [branch.act(observations) for branch in branches]
            np.testing.assert_array_equal(actions[0], actions[1])
            updates = [branch.observe_and_update(observations[0]) for branch in branches]
            self.assertEqual(updates[0]['sample_indices'], updates[1]['sample_indices'])
        self.assertTrue(existing_tests.equal_tree(branches[0].actor.params, branches[1].actor.params))
        self.assertTrue(existing_tests.equal_tree(branches[0].actor.opt_state, branches[1].actor.opt_state))

    def test_cross_regime_analysis_requires_same_environment_seed_and_schedule(self):
        small = {'reference_regime': 'reference_small', 'environment': 'dog-run',
                 'training_seed': 1, 'horizons': {5000: {'checkpoint_means': {2: 1., 3: 3.}}}}
        large = {'reference_regime': 'reference_large', 'environment': 'dog-run',
                 'training_seed': 1, 'horizons': {5000: {'checkpoint_means': {2: 8., 3: 12.}}}}
        self.assertEqual(combine_pilot_analyses(small, large)[5000]['sensitivity'], 8.)
        large['environment'] = 'other'
        with self.assertRaises(ValueError):
            combine_pilot_analyses(small, large)

    def test_paired_return_and_equal_regime_contrasts(self):
        rows = paired_returns([10., 20.], [12., 18.], [13., 16.])
        self.assertEqual(rows[0], {'baseline': 10., 'small': 12., 'large': 13.,
                                   'delta_small': 2., 'delta_large': 3., 'advantage_large': 1.})
        self.assertEqual(rows[1]['advantage_large'], -2.)
        result = regime_contrasts({1: 1., 2: 3.}, {1: 8., 2: 12.})
        self.assertEqual(result, {'small_reference': 2., 'large_reference': 10.,
                                  'sensitivity': 8., 'equal_weight_mean': 6.})
        with self.assertRaises(ValueError):
            paired_returns([1.], [2., 3.], [4.])
        with self.assertRaises(ValueError):
            regime_contrasts({1: np.nan}, {1: 1.})
        with self.assertRaises(ValueError):
            regime_contrasts({1: 1., 2: 3.}, {1: 10.})

    def test_branch_resumes_sampling_and_actor_updates(self):
        existing_tests.OnlineExp2Test.setUpClass()
        fixture = existing_tests.OnlineExp2Test
        branch = DiagnosticActorBranch(fixture.actor, fixture.shadows['small'].critic,
                                       fixture.temperature, seed=12, batch_size=4,
                                       updates_per_interaction=1.5, critic_use_cdq=False)
        branch.observe_and_update(np.ones(3, np.float32))
        state = branch.state_dict()
        clone = DiagnosticActorBranch.from_state_dict(state, fixture.shadows['small'].critic,
                                                      fixture.temperature)
        for value in (2., 3.):
            a = branch.observe_and_update(np.full(3, value, np.float32))
            b = clone.observe_and_update(np.full(3, value, np.float32))
            self.assertEqual(a['sample_indices'], b['sample_indices'])
        self.assertEqual(branch.interactions, 3)
        self.assertEqual(branch.optimizer_updates, 4)
        self.assertTrue(existing_tests.equal_tree(branch.actor.params, clone.actor.params))
        self.assertTrue(existing_tests.equal_tree(branch.actor.opt_state, clone.actor.opt_state))

    def test_actor_step_matches_source_sac_and_keeps_critic_frozen(self):
        existing_tests.OnlineExp2Test.setUpClass()
        actor = existing_tests.OnlineExp2Test.actor
        critic = existing_tests.OnlineExp2Test.shadows['small'].critic
        temperature = existing_tests.OnlineExp2Test.temperature
        observations = jnp.ones((4, 3), dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        before = jax.tree.map(lambda x: np.asarray(x).copy(), critic.params)
        actual, info = diagnostic_actor_step(actor, critic, temperature, observations, key, False)
        expected, _ = update_actor(key, actor, critic, temperature, {'observation': observations}, False)
        self.assertTrue(existing_tests.equal_tree(actual.params, expected.params))
        self.assertTrue(existing_tests.equal_tree(actual.opt_state, expected.opt_state))
        self.assertTrue(existing_tests.equal_tree(critic.params, before))
        self.assertGreater(float(info['parameter_update_norm']), 0.)
        altered_critic = critic.replace(params=jax.tree.map(lambda x: x * 1.1, critic.params))
        _, other_info = diagnostic_actor_step(actor, altered_critic,
                                         temperature, observations, key, False)
        self.assertNotEqual(float(info['train/actor_gnorm']),
                            float(other_info['train/actor_gnorm']))

    def test_reference_roles_are_explicit(self):
        self.assertEqual(reference_roles('reference_small'), ((2, 512), (2, 512), (4, 1536)))
        self.assertEqual(reference_roles('reference_large'), ((4, 1536), (2, 512), (4, 1536)))
        with self.assertRaises(ValueError):
            reference_roles('unknown')

    def test_sampling_replay_and_snapshot_are_branch_local(self):
        a = ObservationBuffer(seed=7)
        b = ObservationBuffer(seed=7)
        with self.assertRaises(ValueError):
            a.sample(2)
        a.append(np.array([1., 2.]))
        a.append(np.array([3., 4.]))
        b.append(np.array([9., 8.]))
        state = a.state_dict()
        observations, indices = a.sample(20)
        restored = ObservationBuffer.from_state_dict(state)
        restored_observations, restored_indices = restored.sample(20)
        np.testing.assert_array_equal(indices, restored_indices)
        np.testing.assert_array_equal(observations, restored_observations)
        self.assertEqual(len(b), 1)
        self.assertTrue(set(indices.tolist()).issubset({0, 1}))
        self.assertEqual(len(observations), 20)  # replacement permits a batch larger than history
        self.assertTrue(set(map(tuple, observations)).issubset({(1., 2.), (3., 4.)}))

    def test_schedule_retains_fractional_carry_and_resume(self):
        schedule = ActorUpdateSchedule(1.5, carry=0.25)
        self.assertEqual(schedule.advance(), 1)
        self.assertEqual(schedule.advance(), 2)
        restored = ActorUpdateSchedule.from_state_dict(schedule.state_dict())
        self.assertEqual([schedule.advance() for _ in range(4)],
                         [restored.advance() for _ in range(4)])

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            ActorUpdateSchedule(-1)
        buffer = ObservationBuffer(seed=3)
        with self.assertRaises(ValueError):
            buffer.append(np.array([np.nan]))
        buffer.append(np.array([1.]))
        with self.assertRaises(ValueError):
            buffer.sample(0)


if __name__ == '__main__':
    unittest.main()
