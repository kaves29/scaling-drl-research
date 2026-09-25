import os
import shutil
import tempfile
import unittest

import numpy as np

from experiments.angle_2a.probes import Probe
from experiments.angle_2a.storage import load_matchup_result, save_frozen_agent_snapshot, save_matchup_result


def _make_probe(probe_id, source, seed):
    rng = np.random.default_rng(seed)
    return Probe(
        probe_id=probe_id,
        source=source,
        state=rng.normal(size=4).astype(np.float32),
        action=rng.normal(size=2).astype(np.float32),
        q_d=float(rng.normal()),
        q_r=float(rng.normal()),
        mc_rollout_returns=list(rng.normal(size=15)),
        mc_return=float(rng.normal()),
        diagonal_error=float(abs(rng.normal())),
    )


class TestAngle2AStorage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_round_trip_preserves_metadata_and_probe_scalars(self):
        probes = [_make_probe(f"p{i}", "D" if i < 3 else "R", seed=i) for i in range(6)]
        run_metadata = {
            "experiment": "angle_2_a",
            "matchup": "matchup_1",
            "scaled_architecture": "D5W768",
            "reference_architecture": "D2W512",
            "scaled_onset_step": 12345,
        }

        save_matchup_result(
            environment="reacher-hard", seed=7, matchup_name="matchup_1",
            run_metadata=run_metadata, probes=probes, root=self.tmpdir,
        )
        loaded_metadata, probes_df, arrays = load_matchup_result(
            environment="reacher-hard", seed=7, matchup_name="matchup_1", root=self.tmpdir,
        )

        self.assertEqual(loaded_metadata["scaled_architecture"], "D5W768")
        self.assertEqual(loaded_metadata["scaled_onset_step"], 12345)
        self.assertEqual(loaded_metadata["num_probes"], 6)

        self.assertEqual(len(probes_df), 6)
        self.assertCountEqual(probes_df["probe_id"].tolist(), [p.probe_id for p in probes])
        for p in probes:
            row = probes_df[probes_df["probe_id"] == p.probe_id].iloc[0]
            self.assertEqual(row["source"], p.source)
            self.assertAlmostEqual(row["q_d"], p.q_d, places=5)
            self.assertAlmostEqual(row["q_r"], p.q_r, places=5)
            self.assertAlmostEqual(row["diagonal_error"], p.diagonal_error, places=5)
            self.assertEqual(row["num_rollouts"], 15)

    def test_round_trip_preserves_high_dimensional_arrays_exactly(self):
        probes = [_make_probe(f"p{i}", "D", seed=i) for i in range(3)]
        save_matchup_result(
            environment="cheetah-run", seed=1, matchup_name="matchup_2",
            run_metadata={}, probes=probes, root=self.tmpdir,
        )
        _metadata, probes_df, arrays = load_matchup_result(
            environment="cheetah-run", seed=1, matchup_name="matchup_2", root=self.tmpdir,
        )

        self.assertEqual(arrays["state"].shape, (3, 4))
        self.assertEqual(arrays["action"].shape, (3, 2))
        self.assertEqual(arrays["rollout_returns"].shape, (3, 15))

        # order in the npz's probe_id array matches the order probes were saved
        for i, p in enumerate(probes):
            self.assertEqual(arrays["probe_id"][i], p.probe_id)
            np.testing.assert_allclose(arrays["state"][i], p.state, rtol=1e-5)
            np.testing.assert_allclose(arrays["action"][i], p.action, rtol=1e-5)
            np.testing.assert_allclose(arrays["rollout_returns"][i], p.mc_rollout_returns, rtol=1e-5)

    def test_does_not_reproduce_a_previous_matchups_data(self):
        probes_1 = [_make_probe("only_in_matchup_1", "D", seed=1)]
        probes_2 = [_make_probe("only_in_matchup_2", "D", seed=2)]

        save_matchup_result(environment="env", seed=1, matchup_name="matchup_1", run_metadata={}, probes=probes_1, root=self.tmpdir)
        save_matchup_result(environment="env", seed=1, matchup_name="matchup_2", run_metadata={}, probes=probes_2, root=self.tmpdir)

        _, df1, _ = load_matchup_result(environment="env", seed=1, matchup_name="matchup_1", root=self.tmpdir)
        _, df2, _ = load_matchup_result(environment="env", seed=1, matchup_name="matchup_2", root=self.tmpdir)

        self.assertEqual(df1["probe_id"].tolist(), ["only_in_matchup_1"])
        self.assertEqual(df2["probe_id"].tolist(), ["only_in_matchup_2"])


class _FakeAgent:
    """Records the exact path agent.save_checkpoint() was called with,
    without needing a real (JAX-dependent) SACAgent - this test is only
    about save_frozen_agent_snapshot's own path-resolution logic, which is
    identical regardless of what kind of agent it's given."""

    def __init__(self):
        self.save_checkpoint_called_with = None

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        self.save_checkpoint_called_with = checkpoint_dir


class _FakeProbeCapture:
    def __len__(self):
        return 0


class TestSaveFrozenAgentSnapshotCheckpointDirIsAlwaysAbsolute(unittest.TestCase):
    """Regression coverage for the 2026-09-21/23 incident: agent.save_checkpoint()
    -> Orbax requires an absolute path, and every real caller of this
    function (angle_2_a.py's Phase 2, angle_1.py's save_probe_capture_snapshot)
    happened to pass a relative `root`. Fixed via .resolve() inside
    save_frozen_agent_snapshot itself, so it protects every caller
    regardless of what root they pass - this test exercises exactly that
    guarantee, with a relative root and a relative cwd (chdir'd into the
    tmpdir), the scenario that actually crashed in production."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_relative_root_still_produces_an_absolute_save_checkpoint_call(self):
        agent = _FakeAgent()
        save_frozen_agent_snapshot(
            environment="dog-run", seed=1, matchup_name="matchup_1", role="D",
            agent=agent, probe_capture=_FakeProbeCapture(), agent_cfg={"critic_use_cdq": False},
            root="results/angle_2a",  # relative, exactly as angle_2_a.py used to hardcode it
        )
        self.assertIsNotNone(agent.save_checkpoint_called_with)
        self.assertTrue(
            os.path.isabs(agent.save_checkpoint_called_with),
            f"save_checkpoint was called with a relative path: {agent.save_checkpoint_called_with!r}",
        )

    def test_already_absolute_root_is_left_correct(self):
        agent = _FakeAgent()
        abs_root = os.path.join(self.tmpdir, "results", "angle_2a")
        save_frozen_agent_snapshot(
            environment="dog-run", seed=1, matchup_name="matchup_1", role="D",
            agent=agent, probe_capture=_FakeProbeCapture(), agent_cfg={"critic_use_cdq": False},
            root=abs_root,
        )
        self.assertTrue(os.path.isabs(agent.save_checkpoint_called_with))
        self.assertTrue(agent.save_checkpoint_called_with.startswith(abs_root))


if __name__ == "__main__":
    unittest.main()
