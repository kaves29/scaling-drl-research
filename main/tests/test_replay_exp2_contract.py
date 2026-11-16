"""Admission rejects mistaken parent identities before loading any tensors."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp12.replay_exp2_contract import SOURCE, admit_parent, checkpoints


class ContractTests(unittest.TestCase):
    def test_budget_schedule_exact_endpoint_and_unique(self):
        for total in (15, 16, 990000, 990002):
            points = checkpoints(total)
            self.assertEqual(len(set(points)), 15)
            self.assertEqual(points[-1], total)
            self.assertGreater(points[0], 0)
            self.assertLessEqual(max(b-a for a,b in zip(points,points[1:])) -
                                 min(b-a for a,b in zip(points,points[1:])), 1)
        for total in (0, 14, 15.5, True):
            with self.assertRaises(ValueError): checkpoints(total)

    def _parent(self, root):
        row = dict(environment="dog-run", seed=1, architecture="D2W512",
                   source_commit=SOURCE, done_step=500000, state="step_000500000_abcd",
                   config_hash="exact-config")
        run = root / "D2W512/dog-run/seed_1"
        state = run / "state" / row["state"]
        (state / "agent_ckpt").mkdir(parents=True)
        for name in ("meta.pkl", "buffer.npz", "buffer_meta.pkl", "obs_rms.pkl"):
            (state/name).write_bytes(b"fixture")
        (run / "DONE").write_text(json.dumps({"interaction_step":500000}))
        (run / "state/LATEST").write_text(row["state"])
        (run / "run_metadata.json").write_text(json.dumps({
            "identity": {k:row[k] for k in ("architecture","environment","seed")},
            "code":{"commit":SOURCE},"config_hash":row["config_hash"]}))
        manifest = root/"manifest.json"
        manifest.write_text(json.dumps({"runs":[row]}))
        (root/"verification.json").write_text(json.dumps({
            "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "all_files_ok":True}))
        return manifest,run,row

    def test_admit_and_reject_wrong_regime_or_changed_parent(self):
        identity=dict(environment="dog-run",seed=1,reference_regime="reference_small")
        with tempfile.TemporaryDirectory() as tmp:
            manifest,run,row=self._parent(Path(tmp))
            state,parent=admit_parent(manifest,identity)
            self.assertEqual(state.name,row["state"])
            self.assertEqual(parent["source_commit"],SOURCE)
            with self.assertRaises(ValueError):
                admit_parent(manifest,{**identity,"reference_regime":"reference_large"})
            (run/"DONE").write_text(json.dumps({"interaction_step":1000}))
            with self.assertRaisesRegex(ValueError,"DONE"):
                admit_parent(manifest,identity)

    def test_missing_verification_or_incomplete_payload_rejected(self):
        identity=dict(environment="dog-run",seed=1,reference_regime="reference_small")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,run,row=self._parent(root)
            (root/"verification.json").unlink()
            with self.assertRaisesRegex(ValueError,"verification receipt"):
                admit_parent(manifest,identity)
        with tempfile.TemporaryDirectory() as tmp:
            manifest,run,row=self._parent(Path(tmp))
            (run/"state"/row["state"]/"buffer.npz").unlink()
            with self.assertRaisesRegex(ValueError,"missing parent payload"):
                admit_parent(manifest,identity)


if __name__ == "__main__": unittest.main()

class OfflineLoopTests(unittest.TestCase):
    def test_offline_loop_does_not_collect_reference_experience(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import numpy as np
        from experiments.exp12.replay_exp2_pilot import ReplayExp2Pilot
        with tempfile.TemporaryDirectory() as tmp:
            runner = object.__new__(ReplayExp2Pilot)
            runner.root = Path(tmp)
            runner.update_budget = 3
            runner.checks = [1,2,3]
            runner.parent = {"source_commit": SOURCE}
            runner.fresh_done = True
            trainer = SimpleNamespace(update_step=0, extra_state={},
                                      _attach_logger=Mock(), close=Mock())
            trainer.buffer = Mock()
            trainer.buffer.sample.return_value = ({"reward":np.array([1.])},np.array([0]))
            runner.trainer=trainer
            runner._check_resource_cap=Mock()
            runner._shadow_updates=Mock()
            runner._checkpoint=Mock()
            runner.save=Mock()
            runner._pilot_analysis=Mock(return_value={})
            with patch("experiments.exp12.online_exp2_pilot._reference_state",return_value={"actor":"fixed"}), \
                 patch("experiments.exp12.replay_exp2_pilot.wandb.run",None):
                runner.run()
            self.assertEqual(trainer.update_step,3)
            self.assertEqual(runner._shadow_updates.call_count,3)
            self.assertEqual(runner._checkpoint.call_count,3)
            self.assertEqual(trainer.buffer.sample.call_count,3)
            trainer.buffer.add.assert_not_called()
            trainer.close.assert_called_once()
            self.assertEqual(json.loads((runner.root/"analysis.json").read_text())["checkpoint_axis"],"critic_updates")
