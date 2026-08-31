import unittest

import numpy as np

from experiments.angle_2a.agent_runner import ProbeCapture, TrainedAgentHandle
from experiments.angle_2a.config import RoleArchitecture
from experiments.angle_2a.probes import (
    Probe,
    compute_diagonal_errors,
    evaluate_both_critics,
    run_monte_carlo_rollouts,
    sample_probes,
)


class _FakeAgent:
    """Deterministic fake: Q(s,a) = q_value_fn(s,a); actor always emits a fixed action."""

    def __init__(self, q_value_fn, fixed_action):
        self.q_value_fn = q_value_fn
        self.fixed_action = np.asarray(fixed_action, dtype=np.float32)

    def get_q_value(self, observations, actions):
        return np.array([self.q_value_fn(s, a) for s, a in zip(observations, actions)])

    def sample_actions(self, interaction_step, prev_timestep, training):
        return np.stack([self.fixed_action])


class _DummyPhysics:
    """No-op-ish stand-in so the real restore_env_state() (which every probe
    rollout genuinely calls) has something to call get_state/set_state/
    forward on. Also resets the owning fake env's episode-step counter on
    set_state, mirroring what a real "reset to state s" must do: multiple
    probes/rollouts reuse the same underlying env instance (exactly as
    Angle 2A's real single_env reuse does), so each restore must behave like
    a fresh episode start, not accumulate steps across probes."""

    def __init__(self, owner):
        self._owner = owner

    def get_state(self):
        return np.zeros(1)

    def set_state(self, state):
        self._owner._steps_taken = 0

    def forward(self):
        pass


class _FakeEnv:
    """Deterministic single-step-then-done env: reward = sum(action), episode
    ends after exactly `episode_len` steps since the last restore."""

    def __init__(self, episode_len=1, reward_fn=None):
        self.episode_len = episode_len
        self.reward_fn = reward_fn or (lambda action: float(np.sum(action)))
        self._steps_taken = 0
        self.physics = _DummyPhysics(self)

    @property
    def unwrapped(self):
        return self

    def step(self, action):
        self._steps_taken += 1
        reward = self.reward_fn(action)
        done = self._steps_taken >= self.episode_len
        next_obs = np.zeros(2, dtype=np.float32)
        return next_obs, reward, done, False, {}


def _fake_handle(role, q_value_fn, fixed_action, episode_len=1, reward_fn=None, capacity=20):
    agent = _FakeAgent(q_value_fn, fixed_action)
    env = _FakeEnv(episode_len=episode_len, reward_fn=reward_fn)
    probe_capture = ProbeCapture(capacity=capacity, observation_shape=(2,), action_shape=(1,))
    for i in range(capacity):
        probe_capture.add(i, np.array([float(i), 0.0], dtype=np.float32), np.array([float(i)], dtype=np.float32), {"physics_state": np.zeros(1), "elapsed_steps": None})

    return TrainedAgentHandle(
        role=role,
        architecture_label="TestArch",
        architecture=RoleArchitecture(role="x", critic_num_blocks=2, critic_hidden_dim=512),
        agent=agent,
        buffer=None,
        train_env=None,
        eval_env=None,
        single_env=env,
        stop_step=capacity,
        probe_capture=probe_capture,
    )


class TestProbeSampling(unittest.TestCase):
    def test_exactly_ten_probes_per_source(self):
        D = _fake_handle("D", lambda s, a: 0.0, [0.0])
        R = _fake_handle("R", lambda s, a: 0.0, [0.0])
        rng = np.random.default_rng(0)

        probes = sample_probes("m1", D, R, num_probes_per_source=10, rng=rng)

        self.assertEqual(len(probes), 20)
        self.assertEqual(sum(1 for p in probes if p.source == "D"), 10)
        self.assertEqual(sum(1 for p in probes if p.source == "R"), 10)

    def test_source_labels_correct_and_states_actions_preserved(self):
        D = _fake_handle("D", lambda s, a: 0.0, [0.0])
        R = _fake_handle("R", lambda s, a: 0.0, [0.0])
        rng = np.random.default_rng(1)

        probes = sample_probes("m1", D, R, num_probes_per_source=5, rng=rng)

        for p in probes:
            self.assertIn(p.source, ("D", "R"))
            # state[0] and action[0] were constructed identically (== the
            # sampled buffer index) in _fake_handle, so this also verifies
            # state/action integrity survives sampling.
            self.assertEqual(p.state[0], p.action[0])

    def test_probes_never_mix_sources_buffers(self):
        # D and R capacities/contents are disjoint value ranges (D: 0..9, R: 100..109)
        # to make cross-contamination detectable.
        D = _fake_handle("D", lambda s, a: 0.0, [0.0], capacity=10)
        R_agent = _FakeAgent(lambda s, a: 0.0, [0.0])
        R_env = _FakeEnv()
        R_capture = ProbeCapture(capacity=10, observation_shape=(2,), action_shape=(1,))
        for i in range(10):
            R_capture.add(i, np.array([float(100 + i), 0.0], dtype=np.float32), np.array([float(100 + i)], dtype=np.float32), {"physics_state": np.zeros(1), "elapsed_steps": None})
        R = TrainedAgentHandle(
            role="R", architecture_label="Ref", architecture=RoleArchitecture("reference", 2, 512),
            agent=R_agent, buffer=None, train_env=None, eval_env=None, single_env=R_env,
            stop_step=10, probe_capture=R_capture,
        )

        rng = np.random.default_rng(2)
        probes = sample_probes("m1", D, R, num_probes_per_source=10, rng=rng)

        for p in probes:
            if p.source == "D":
                self.assertLess(p.state[0], 100)
            else:
                self.assertGreaterEqual(p.state[0], 100)

    def test_insufficient_transitions_raises_instead_of_sampling_with_replacement(self):
        D = _fake_handle("D", lambda s, a: 0.0, [0.0], capacity=3)
        R = _fake_handle("R", lambda s, a: 0.0, [0.0], capacity=3)
        rng = np.random.default_rng(3)

        with self.assertRaises(ValueError):
            sample_probes("m1", D, R, num_probes_per_source=10, rng=rng)


class TestCriticEvaluation(unittest.TestCase):
    def test_both_critics_evaluated_on_every_probe(self):
        D = _fake_handle("D", lambda s, a: 10.0 + s[0], [0.0])
        R = _fake_handle("R", lambda s, a: 20.0 + s[0], [0.0])
        rng = np.random.default_rng(4)
        probes = sample_probes("m1", D, R, num_probes_per_source=3, rng=rng)

        evaluate_both_critics(probes, D, R)

        for p in probes:
            self.assertIsNotNone(p.q_d)
            self.assertIsNotNone(p.q_r)
            self.assertAlmostEqual(p.q_d, 10.0 + p.state[0])
            self.assertAlmostEqual(p.q_r, 20.0 + p.state[0])

    def test_off_diagonal_values_are_retained_not_discarded(self):
        D = _fake_handle("D", lambda s, a: 1.0, [0.0])
        R = _fake_handle("R", lambda s, a: 2.0, [0.0])
        rng = np.random.default_rng(5)
        probes = sample_probes("m1", D, R, num_probes_per_source=2, rng=rng)
        evaluate_both_critics(probes, D, R)

        for p in probes:
            self.assertIsNotNone(p.q_other)
            if p.source == "D":
                self.assertEqual(p.q_source, p.q_d)
                self.assertEqual(p.q_other, p.q_r)
            else:
                self.assertEqual(p.q_source, p.q_r)
                self.assertEqual(p.q_other, p.q_d)


class TestMonteCarloRollouts(unittest.TestCase):
    def test_exactly_fifteen_rollouts_per_probe(self):
        D = _fake_handle("D", lambda s, a: 0.0, [1.0], episode_len=1)
        R = _fake_handle("R", lambda s, a: 0.0, [1.0], episode_len=1)
        rng = np.random.default_rng(6)
        probes = sample_probes("m1", D, R, num_probes_per_source=2, rng=rng)

        run_monte_carlo_rollouts(probes, D, R, num_rollouts=15, gamma=0.99, max_rollout_steps=10)

        for p in probes:
            self.assertEqual(len(p.mc_rollout_returns), 15)

    def test_forced_first_action_is_used(self):
        # reward_fn returns the action value itself; episode ends after the
        # first (forced) step, so mc_return must equal the probe's own action.
        D = _fake_handle("D", lambda s, a: 0.0, [999.0], episode_len=1, reward_fn=lambda a: float(a[0]))
        R = _fake_handle("R", lambda s, a: 0.0, [999.0], episode_len=1, reward_fn=lambda a: float(a[0]))
        rng = np.random.default_rng(7)
        probes = sample_probes("m1", D, R, num_probes_per_source=3, rng=rng)

        run_monte_carlo_rollouts(probes, D, R, num_rollouts=5, gamma=0.99, max_rollout_steps=10)

        for p in probes:
            self.assertAlmostEqual(p.mc_return, float(p.action[0]))

    def test_continuation_uses_source_agents_actor_only(self):
        # episode continues for 2 steps: reward at t=0 is the forced action's
        # value (discarded here via reward_fn ignoring it - use identity), at
        # t=1 reward equals whichever actor's fixed_action produced it. D's
        # actor always emits 111, R's actor always emits 222; if a probe's
        # rollout ever used the wrong actor, its second-step reward would
        # betray that.
        D = _fake_handle("D", lambda s, a: 0.0, [111.0], episode_len=2, reward_fn=lambda a: float(a[0]))
        R = _fake_handle("R", lambda s, a: 0.0, [222.0], episode_len=2, reward_fn=lambda a: float(a[0]))
        rng = np.random.default_rng(8)
        probes = sample_probes("m1", D, R, num_probes_per_source=2, rng=rng)

        run_monte_carlo_rollouts(probes, D, R, num_rollouts=1, gamma=1.0, max_rollout_steps=10)

        for p in probes:
            # total_return = forced_action_reward (t=0) + continuation_reward (t=1)
            expected_continuation = 111.0 if p.source == "D" else 222.0
            self.assertAlmostEqual(p.mc_return, float(p.action[0]) + expected_continuation)


class TestDiagonalErrors(unittest.TestCase):
    def test_error_calculation_matches_synthetic_values(self):
        probe_d = Probe(probe_id="p1", source="D", state=np.zeros(2), action=np.zeros(1), q_d=5.0, q_r=100.0, mc_return=3.0)
        probe_r = Probe(probe_id="p2", source="R", state=np.zeros(2), action=np.zeros(1), q_d=100.0, q_r=7.0, mc_return=10.0)

        compute_diagonal_errors([probe_d, probe_r])

        self.assertAlmostEqual(probe_d.diagonal_error, abs(5.0 - 3.0))
        self.assertAlmostEqual(probe_r.diagonal_error, abs(7.0 - 10.0))

    def test_error_never_crosses_actor_critic_pairing(self):
        # D-source probe: E_D must use q_d and mc_return (assumed computed
        # from D's actor), NEVER q_r.
        probe = Probe(probe_id="p1", source="D", state=np.zeros(2), action=np.zeros(1), q_d=1.0, q_r=999.0, mc_return=1.0)
        compute_diagonal_errors([probe])
        self.assertAlmostEqual(probe.diagonal_error, 0.0)

    def test_missing_prerequisites_raises(self):
        probe = Probe(probe_id="p1", source="D", state=np.zeros(2), action=np.zeros(1))
        with self.assertRaises(ValueError):
            compute_diagonal_errors([probe])


if __name__ == "__main__":
    unittest.main()
