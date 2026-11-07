"""Expose silent marker omissions without accepting payloads or mutating runs."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_exp12_campaign_inventory import inventory


class CampaignInventoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.commit = "a" * 40
        self.parent = self.root / "exp1/D4W1024/dog-run/seed_1"
        self.key = "exp1_D4W1024_dog-run_seed1"

    def complete(self, step=500000):
        self.parent.mkdir(parents=True)
        (self.parent / "DONE").write_text(json.dumps({"interaction_step": step}))
        (self.parent / "run_metadata.json").write_text(
            json.dumps(
                dict(
                    identity=dict(run_key=self.key, run_role="confirmatory"),
                    code=dict(commit=self.commit, dirty=False),
                    launches=[dict(commit=self.commit, dirty=False)],
                )
            )
        )
        state = self.parent / f"state/step_{step:09d}_1234abcd"
        state.mkdir(parents=True)
        (state.parent / "LATEST").write_text(state.name)
        for name in (
            "meta.pkl",
            "buffer.npz",
            "buffer_meta.pkl",
            "obs_rms.pkl",
            "agent_ckpt/data",
        ):
            p = state / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"structural fixture; intentionally not a valid payload")
        return state

    def row(self):
        return next(
            r
            for r in inventory(self.root, self.commit)["parents"]
            if r["run_key"] == self.key
        )

    def test_independent_full_population_and_budgets(self):
        report = inventory(self.root, self.commit)
        self.assertEqual(report["parent_count"], 195)
        self.assertEqual(len({r["run_key"] for r in report["parents"]}), 195)
        self.assertEqual(report["counts"], {"fresh": 195})
        self.assertEqual(report["qualification_status"], "not_assessed")

    def test_structure_never_certifies_the_payload(self):
        self.complete()
        row = self.row()
        self.assertEqual(row["status"], "complete_marker_structure_only")
        self.assertEqual(row["payload_integrity"], "not_assessed")

    def test_malformed_done_preserved_and_blocked(self):
        self.complete()
        path = self.parent / "DONE"
        path.write_bytes(b"invalid marker")
        before = path.read_bytes()
        from generate_manifest import classify_exp12

        self.assertEqual(classify_exp12(str(self.parent)), "done")
        self.assertEqual(self.row()["status"], "blocked")
        self.assertEqual(path.read_bytes(), before)

    def test_stale_done_cannot_silently_suppress_a_parent(self):
        self.complete(499999)
        self.assertIn("DONE terminal step", self.row()["errors"][0])

    def test_latest_escape_and_missing_payload_are_blocked(self):
        state = self.complete()
        (state / "buffer.npz").unlink()
        self.assertIn("missing buffer.npz", self.row()["errors"][0])
        (state.parent / "LATEST").write_text("../../outside")
        self.assertIn("LATEST", self.row()["errors"][0])

    def test_mixed_resume_source_is_blocked(self):
        self.complete()
        p = self.parent / "run_metadata.json"
        meta = json.loads(p.read_text())
        meta["launches"][0]["commit"] = "b" * 40
        p.write_text(json.dumps(meta))
        self.assertIn("resume launch source", self.row()["errors"][0])
        meta["launches"][0]["commit"] = self.commit
        meta["code"]["dirty"] = 0
        p.write_text(json.dumps(meta))
        self.assertIn("metadata source revision", self.row()["errors"][0])

    def test_claims_are_not_queried_or_deleted(self):
        self.complete()
        claim = self.parent / "claim"
        claim.mkdir()
        (claim / "owner.json").write_text('{"slurm_job_id":"123"}')
        self.assertIn("not queried", self.row()["ownership_note"])
        self.assertTrue(claim.is_dir())
        self.assertEqual((claim / "owner.json").read_text(), '{"slurm_job_id":"123"}')

    def test_ready_marker_without_fork_header_is_blocked(self):
        self.complete()
        fork = self.parent / "fork"
        fork.mkdir()
        (fork / "FORK_READY").write_text(self.key)
        self.assertEqual(self.row()["status"], "blocked")

    def test_done_with_an_unready_fork_is_blocked(self):
        self.complete()
        fork = self.parent / "fork"
        fork.mkdir()
        (fork / "fork.json").write_text("{}")
        self.assertIn("unready fork", self.row()["errors"][0])

    def test_late_control_horizon_matches_the_approved_marker(self):
        self.complete(600000)
        fork = self.parent / "fork"
        fork.mkdir()
        (fork / "FORK_READY").write_text(self.key)
        (fork / "fork.json").write_text(
            json.dumps(
                dict(
                    run_key=self.key,
                    fork_step=475000,
                    fork_check_index=19,
                    num_interaction_steps=500000,
                    horizon_steps=125000,
                    arm_end_step=600000,
                    control_end_step=600000,
                    device=dict(device_kind="A100"),
                )
            )
        )
        self.assertEqual(self.row()["status"], "complete_marker_structure_only")

    def test_missing_launch_history_is_blocked(self):
        self.complete()
        path = self.parent / "run_metadata.json"
        meta = json.loads(path.read_text())
        meta["launches"] = []
        path.write_text(json.dumps(meta))
        self.assertIn("missing resume launch", self.row()["errors"][0])

    def test_file_root_is_not_misreported_as195_fresh_runs(self):
        root = self.root / "not-directory"
        root.write_text("existing data")
        with self.assertRaisesRegex(ValueError, "not a directory"):
            inventory(root, self.commit)

    def test_parent_aliases_do_not_duplicate_identity(self):
        self.complete()
        (self.parent.parent / "seed_2").symlink_to(
            self.parent, target_is_directory=True
        )
        report = inventory(self.root, self.commit)
        row = next(
            r
            for r in report["parents"]
            if r["run_key"].endswith("D4W1024_dog-run_seed2")
        )
        self.assertEqual(row["status"], "blocked")
        self.assertIn("aliases", row["errors"][0])


if __name__ == "__main__":
    unittest.main()
