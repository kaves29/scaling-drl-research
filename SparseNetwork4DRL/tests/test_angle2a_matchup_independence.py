"""Verifies the cross-matchup independence invariants from the Angle 2A spec:
four genuinely separate agents per seed, Matchup 2's reference agent is never
Matchup 1's, and nothing is shared between D and R within a matchup.

Trains are mocked out (real SAC/dm_control training is exercised by the
config/probes/env_state/storage/onset_lookup test modules and by manual
end-to-end runs - see the Angle 2A deliverables notes) so this module can
focus purely on the object-identity bookkeeping that a future refactor could
otherwise silently break.
"""

import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf

from experiments.angle_2a import matchup as matchup_module
from experiments.angle_2a.agent_runner import ProbeCapture, TrainedAgentHandle
from experiments.angle_2a.config import RoleArchitecture

CLOSED_HANDLES = []


class _FakeAgent:
    def __init__(self, tag):
        self.tag = tag

    def get_q_value(self, observations, actions):
        return np.zeros(len(observations))

    def sample_actions(self, interaction_step, prev_timestep, training):
        return np.zeros((1, 1), dtype=np.float32)

    def save_checkpoint(self, checkpoint_dir):
        # run_matchup() now persists a frozen-agent snapshot for every role
        # (see experiments/angle_2a/storage.py:save_frozen_agent_snapshot,
        # added for Angle 2B) - real training/checkpointing is still
        # deliberately mocked out here per this module's docstring, so this
        # only needs to satisfy the call, not produce a loadable checkpoint.
        pass


class _FakeEnv:
    class _Physics:
        def get_state(self):
            return np.zeros(1)

        def set_state(self, s):
            pass

        def forward(self):
            pass

    def __init__(self):
        self.physics = self._Physics()

    @property
    def unwrapped(self):
        return self

    def step(self, action):
        return np.zeros(2, dtype=np.float32), 0.0, True, False, {}


class _FakeVectorEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _make_fake_handle(role, architecture, architecture_label, stop_step):
    capacity = 12
    probe_capture = ProbeCapture(capacity=capacity, observation_shape=(2,), action_shape=(1,))
    for i in range(capacity):
        probe_capture.add(
            i,
            np.array([float(i), 0.0], dtype=np.float32),
            np.array([float(i)], dtype=np.float32),
            {"physics_state": np.zeros(1), "elapsed_steps": None},
        )

    handle = TrainedAgentHandle(
        role=role,
        architecture_label=architecture_label,
        architecture=architecture,
        agent=_FakeAgent(tag=f"{role}-{architecture_label}-{id(object())}"),
        buffer=object(),  # a unique sentinel per call; identity is what we check
        train_env=_FakeVectorEnv(),
        eval_env=_FakeVectorEnv(),
        single_env=_FakeEnv(),
        stop_step=stop_step,
        probe_capture=probe_capture,
    )
    return handle


def _fake_train_agent_to_step(role, architecture, architecture_label, base_cfg, stop_step, seed_context):
    handle = _make_fake_handle(role, architecture, architecture_label, stop_step)
    CLOSED_HANDLES.append(handle)
    return handle


class _FakeBaseCfg:
    gamma = 0.99
    # run_matchup() now also resolves each role's agent config to persist
    # alongside its checkpoint (see build_role_agent_cfg, used for Angle 2B -
    # experiments/angle_2a/storage.py:save_frozen_agent_snapshot). This must
    # be a real OmegaConf DictConfig, not a plain object, since
    # build_role_agent_cfg calls OmegaConf.set_struct/to_container on it.
    agent = OmegaConf.create({"critic_num_blocks": 2, "critic_hidden_dim": 512, "critic_use_cdq": True})

    class env:
        max_episode_steps = 100


class TestMatchupIndependence(unittest.TestCase):
    def setUp(self):
        CLOSED_HANDLES.clear()
        self.tmpdir = tempfile.mkdtemp()
        self.patcher = mock.patch.object(matchup_module, "train_agent_to_step", side_effect=_fake_train_agent_to_step)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, matchup_name, scaled_arch, scaled_label, **kwargs):
        reference = RoleArchitecture(role="reference", critic_num_blocks=2, critic_hidden_dim=512)
        return matchup_module.run_matchup(
            matchup_name=matchup_name,
            scaled_architecture=scaled_arch,
            scaled_architecture_label=scaled_label,
            reference_architecture=reference,
            reference_architecture_label="D2W512",
            onset_step=kwargs.pop("onset_step", 10),
            onset_source_run_key="fake_run_key",
            base_cfg=_FakeBaseCfg(),
            seed=42,
            environment="reacher-hard",
            experiment_name="angle_2_a",
            num_probes_per_source=3,
            num_mc_rollouts=2,
            output_root=self.tmpdir,
            wandb_enabled=False,
            **kwargs,
        )

    def test_four_independent_agents_across_two_matchups(self):
        scaled_a = RoleArchitecture(role="scaled_a", critic_num_blocks=5, critic_hidden_dim=768)
        scaled_b = RoleArchitecture(role="scaled_b", critic_num_blocks=7, critic_hidden_dim=1024)

        self._run("matchup_1", scaled_a, "D5W768")
        self._run("matchup_2", scaled_b, "D7W1024")

        # train_agent_to_step must have been called exactly 4 times (D+R for
        # each of the 2 matchups), each producing a distinct object.
        self.assertEqual(len(CLOSED_HANDLES), 4)
        agent_ids = {id(h.agent) for h in CLOSED_HANDLES}
        buffer_ids = {id(h.buffer) for h in CLOSED_HANDLES}
        self.assertEqual(len(agent_ids), 4, "all four agents must be distinct objects")
        self.assertEqual(len(buffer_ids), 4, "all four replay buffers must be distinct objects")

    def test_matchup_2_reference_is_not_matchup_1_reference(self):
        scaled_a = RoleArchitecture(role="scaled_a", critic_num_blocks=5, critic_hidden_dim=768)
        scaled_b = RoleArchitecture(role="scaled_b", critic_num_blocks=7, critic_hidden_dim=1024)

        self._run("matchup_1", scaled_a, "D5W768")
        r1 = CLOSED_HANDLES[1]  # (D, R) order per _run's two train_agent_to_step calls
        self._run("matchup_2", scaled_b, "D7W1024")
        r2 = CLOSED_HANDLES[3]

        self.assertEqual(r1.role, "R")
        self.assertEqual(r2.role, "R")
        self.assertIsNot(r1.agent, r2.agent)
        self.assertIsNot(r1.buffer, r2.buffer)
        self.assertIsNot(r1, r2)

    def test_d_and_r_within_a_matchup_do_not_share_state(self):
        scaled_a = RoleArchitecture(role="scaled_a", critic_num_blocks=5, critic_hidden_dim=768)
        self._run("matchup_1", scaled_a, "D5W768")

        D, R = CLOSED_HANDLES[0], CLOSED_HANDLES[1]
        self.assertIsNot(D.agent, R.agent)
        self.assertIsNot(D.buffer, R.buffer)
        self.assertIsNot(D.single_env, R.single_env)
        self.assertIsNot(D.probe_capture, R.probe_capture)

    def test_all_agents_are_closed_even_on_downstream_failure(self):
        scaled_a = RoleArchitecture(role="scaled_a", critic_num_blocks=5, critic_hidden_dim=768)

        with mock.patch.object(matchup_module, "sample_probes", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self._run("matchup_1", scaled_a, "D5W768")

        D, R = CLOSED_HANDLES[0], CLOSED_HANDLES[1]
        self.assertTrue(D.train_env.closed)
        self.assertTrue(D.eval_env.closed)
        self.assertTrue(R.train_env.closed)
        self.assertTrue(R.eval_env.closed)


class TestMatchupWithSharedReferenceHandle(unittest.TestCase):
    """run_matchup(reference_handle=...) contract: used for the two real
    matchups so a second reference training run is never created (see
    experiments/angle_2_a.py + agent_runner.train_reference_agent_with_snapshots).
    The null-baseline path (reference_handle=None, exercised by the class
    above) must remain completely unaffected."""

    def setUp(self):
        CLOSED_HANDLES.clear()
        self.tmpdir = tempfile.mkdtemp()
        self.patcher = mock.patch.object(matchup_module, "train_agent_to_step", side_effect=_fake_train_agent_to_step)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, matchup_name, scaled_arch, scaled_label, reference_handle, onset_step, reference_run_key=None):
        reference = RoleArchitecture(role="reference", critic_num_blocks=2, critic_hidden_dim=512)
        return matchup_module.run_matchup(
            matchup_name=matchup_name,
            scaled_architecture=scaled_arch,
            scaled_architecture_label=scaled_label,
            reference_architecture=reference,
            reference_architecture_label="D2W512",
            onset_step=onset_step,
            onset_source_run_key="fake_run_key",
            base_cfg=_FakeBaseCfg(),
            seed=42,
            environment="reacher-hard",
            experiment_name="angle_2_a",
            num_probes_per_source=3,
            num_mc_rollouts=2,
            output_root=self.tmpdir,
            wandb_enabled=False,
            reference_handle=reference_handle,
            reference_run_key=reference_run_key,
        )

    def test_supplying_reference_handle_skips_training_a_second_reference(self):
        shared_r = _make_fake_handle("R", RoleArchitecture("reference", 2, 512), "D2W512", stop_step=120_000)
        scaled_a = RoleArchitecture(role="scaled_a", critic_num_blocks=5, critic_hidden_dim=768)

        self._run("matchup_1", scaled_a, "D5W768", reference_handle=shared_r, onset_step=120_000, reference_run_key="ref_key")

        # train_agent_to_step must only have been called once (for D) - never
        # for R, since a reference_handle was supplied.
        self.assertEqual(len(CLOSED_HANDLES), 1)
        self.assertEqual(CLOSED_HANDLES[0].role, "D")

    def test_both_matchups_use_the_same_shared_reference_object(self):
        shared_r = _make_fake_handle("R", RoleArchitecture("reference", 2, 512), "D2W512", stop_step=180_000)
        scaled_a = RoleArchitecture(role="scaled_a", critic_num_blocks=5, critic_hidden_dim=768)
        scaled_b = RoleArchitecture(role="scaled_b", critic_num_blocks=7, critic_hidden_dim=1024)

        result_1 = self._run("matchup_1", scaled_a, "D5W768", reference_handle=shared_r, onset_step=120_000, reference_run_key="ref_key")
        result_2 = self._run("matchup_2", scaled_b, "D7W1024", reference_handle=shared_r, onset_step=180_000, reference_run_key="ref_key")

        # only D was trained fresh in each call (2 total); R was never
        # (re)trained via train_agent_to_step in either call.
        self.assertEqual(len(CLOSED_HANDLES), 2)
        self.assertTrue(all(h.role == "D" for h in CLOSED_HANDLES))

        self.assertEqual(result_1.run_metadata["reference_run_key"], "ref_key")
        self.assertEqual(result_2.run_metadata["reference_run_key"], "ref_key")
        self.assertEqual(result_1.run_metadata["reference_run_key"], result_2.run_metadata["reference_run_key"])

    def test_matchup_1_and_matchup_2_record_their_own_snapshot_step(self):
        shared_r = _make_fake_handle("R", RoleArchitecture("reference", 2, 512), "D2W512", stop_step=180_000)
        scaled_a = RoleArchitecture(role="scaled_a", critic_num_blocks=5, critic_hidden_dim=768)
        scaled_b = RoleArchitecture(role="scaled_b", critic_num_blocks=7, critic_hidden_dim=1024)

        result_1 = self._run("matchup_1", scaled_a, "D5W768", reference_handle=shared_r, onset_step=120_000, reference_run_key="ref_key")
        result_2 = self._run("matchup_2", scaled_b, "D7W1024", reference_handle=shared_r, onset_step=180_000, reference_run_key="ref_key")

        self.assertEqual(result_1.run_metadata["reference_snapshot_step"], 120_000)
        self.assertEqual(result_2.run_metadata["reference_snapshot_step"], 180_000)
        self.assertNotEqual(
            result_1.run_metadata["reference_snapshot_step"],
            result_2.run_metadata["reference_snapshot_step"],
        )

    def test_run_matchup_never_closes_a_supplied_reference_handle(self):
        shared_r = _make_fake_handle("R", RoleArchitecture("reference", 2, 512), "D2W512", stop_step=120_000)
        scaled_a = RoleArchitecture(role="scaled_a", critic_num_blocks=5, critic_hidden_dim=768)

        self._run("matchup_1", scaled_a, "D5W768", reference_handle=shared_r, onset_step=120_000, reference_run_key="ref_key")

        self.assertFalse(shared_r.train_env.closed, "run_matchup must not close a shared reference_handle it does not own")
        self.assertFalse(shared_r.eval_env.closed)

    def test_reference_handle_survives_a_downstream_failure_in_the_first_matchup(self):
        shared_r = _make_fake_handle("R", RoleArchitecture("reference", 2, 512), "D2W512", stop_step=180_000)
        scaled_a = RoleArchitecture(role="scaled_a", critic_num_blocks=5, critic_hidden_dim=768)

        with mock.patch.object(matchup_module, "sample_probes", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self._run("matchup_1", scaled_a, "D5W768", reference_handle=shared_r, onset_step=120_000, reference_run_key="ref_key")

        # even though matchup_1 failed after D was trained, the shared
        # reference must still be usable (not closed) for matchup_2.
        self.assertFalse(shared_r.train_env.closed)
        self.assertFalse(shared_r.eval_env.closed)

    def test_null_baseline_path_is_unaffected_no_reference_handle(self):
        # reference_handle=None (the default / null-baseline path) must
        # still train and close its own fresh R, exactly as before.
        scaled_a = RoleArchitecture(role="scaled_a", critic_num_blocks=5, critic_hidden_dim=768)
        result = self._run("null_matchup_1", scaled_a, "null_D2W512", reference_handle=None, onset_step=120_000)

        self.assertEqual(len(CLOSED_HANDLES), 2)  # D and a freshly trained R
        self.assertCountEqual([h.role for h in CLOSED_HANDLES], ["D", "R"])
        self.assertNotIn("reference_run_key", result.run_metadata)


if __name__ == "__main__":
    unittest.main()
