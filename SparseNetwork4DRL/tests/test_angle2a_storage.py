import shutil
import tempfile
import unittest

import numpy as np

from experiments.angle_2a.probes import Probe
from experiments.angle_2a.storage import load_matchup_result, save_matchup_result


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


if __name__ == "__main__":
    unittest.main()
