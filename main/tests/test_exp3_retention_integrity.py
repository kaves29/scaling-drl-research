"""Regression coverage for the E1-E3 retention integrity fixes (CPU).

Run from main/: python -m unittest tests.test_exp3_retention_integrity -v
"""

import base64
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.exp12.trainer import Exp12Trainer
from experiments.exp3 import recording, retention
from experiments.exp3.recording import CaptureIncomplete, finalize_capture, record_exp2_scope
from experiments.exp3.streams import StreamReader
from tests import test_exp3_capture_review as fixtures
from tests import test_exp3_pilots as pilots
from tests import test_exp3_retention_adversarial as adversarial
from tests.exp12_helpers import compose, tiny_overrides


def markers(root, kind):
    return sorted(Path(root).glob(f"{kind}_*.json"))


def edit_sharding(state, change):
    path = Path(state) / "agent_ckpt" / "_sharding"
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


class CheckpointValidationTest(unittest.TestCase):
    """Invalid Orbax metadata is never published nor certified equivalent; valid
    reserialization still is."""

    @classmethod
    def setUpClass(cls):
        fixtures.ResaveRetentionTest.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        fixtures.ResaveRetentionTest.tearDownClass.__func__(cls)

    def clone(self, name):
        return Path(shutil.copytree(self.first, self.tmp / "clones" / name))

    def corruptions(self):
        first_key = None

        def drop_one(record):
            record.pop(next(iter(record)))

        def wrong_type(record):
            nonlocal first_key
            first_key = next(iter(record))
            record[first_key] = json.dumps({"sharding_type": "NamedSharding", "device_str": "TFRT_CPU_0"})

        def absent_device(record):
            for k in record:
                record[k] = json.dumps({"sharding_type": "SingleDeviceSharding", "device_str": "cuda:7"})

        def bad_key(record):
            record[base64.b64encode(b"not.a.leaf").decode()] = next(iter(record.values()))

        return {
            "invalid_json": ("invalid Orbax sharding", lambda s: (s / "agent_ckpt/_sharding").write_text("{oops")),
            "missing_leaf": ("does not cover", lambda s: edit_sharding(s, drop_one)),
            "extra_leaf": ("does not cover", lambda s: edit_sharding(s, bad_key)),
            "incompatible_type": ("incompatible Orbax sharding", lambda s: edit_sharding(s, wrong_type)),
            "unavailable_device": ("unavailable here", lambda s: edit_sharding(s, absent_device)),
            "corrupt_metadata": ("invalid Orbax checkpoint metadata",
                                 lambda s: (s / "agent_ckpt/_METADATA").write_text("{oops")),
        }

    def test_valid_checkpoints_validate_with_device_backed_restore(self):
        for state in (self.first, self.resaved):
            v = retention.validate_checkpoint(state)
            self.assertEqual(v["restore"], "device-backed with saved sharding")

    def test_invalid_incoming_checkpoint_is_never_published(self):
        for name, (message, corrupt) in self.corruptions().items():
            with self.subTest(name):
                bad = self.clone("publish_" + name)
                corrupt(bad)
                keep = self.tmp / ("publish_keep_" + name)
                with self.assertRaisesRegex(ValueError, message):
                    retention.retain_state(bad, keep, 20, "omit")
                self.assertFalse(list(keep.glob("step_*")))

    def test_invalid_resave_is_never_certified_equivalent(self):
        for name, (message, corrupt) in self.corruptions().items():
            for mode in ("omit", "retain"):
                with self.subTest(name=name, mode=mode):
                    keep = self.tmp / f"resave_keep_{name}_{mode}"
                    target = retention.retain_state(self.first, keep, 20, mode)
                    bad = self.clone(f"resave_{name}_{mode}")
                    corrupt(bad)
                    with self.assertRaisesRegex(ValueError, message):
                        retention.retain_state(bad, keep, 20, mode)
                    self.assertEqual(markers(keep, "EQUIVALENT"), [])
                    retention.read_snapshot(target)  # the original stays usable

    def test_legitimate_reserialization_is_still_equivalent_and_validated(self):
        for mode in ("omit", "retain"):
            with self.subTest(mode):
                keep = self.tmp / f"legit_{mode}"
                target = retention.retain_state(self.first, keep, 20, mode)
                retention.retain_state(self.resaved, keep, 20, mode)
                self.assertEqual(len(markers(keep, "EQUIVALENT")), 1)
                self.assertEqual(markers(keep, "CONFLICT"), [])
                record = retention.read_snapshot(target)
                self.assertEqual(record["validation"]["orbax_metadata"], "valid")

    def test_corrupt_existing_snapshot_blocks_both_acceptance_paths(self):
        for incoming in (self.first, self.resaved):  # fast path, logical fallback
            with self.subTest(str(incoming)):
                keep = self.tmp / f"existing_{incoming.name}"
                # Snapshots hard-link their source: retain a copy so tampering
                # cannot reach the shared fixture.
                source = self.clone(f"existing_source_{incoming.name}")
                target = retention.retain_state(source, keep, 20, "omit")
                (target / "meta.pkl").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "existing retained snapshot failed integrity"):
                    retention.retain_state(incoming, keep, 20, "omit")
                self.assertEqual(markers(keep, "EQUIVALENT"), [])

    def test_retention_report_states_explicit_reasons(self):
        keep = self.tmp / "report"
        retention.retain_state(self.first, keep, 20, "omit")
        self.assertEqual(retention.retention_report(keep, [20])["verdict"], "COMPLETE")
        report = retention.retention_report(keep, [20, 40])
        self.assertEqual(report["verdict"], "FAILED")
        self.assertIn("step 40: required snapshot missing", report["reasons"])
        retention.record_failure(keep, 20, self.first, ValueError("synthetic"))
        self.assertTrue(any(r.startswith("FAILED record") for r in retention.retention_report(keep, [20])["reasons"]))


class RecordedRetentionFailureTest(unittest.TestCase):
    """A retention failure inside a live recorded source neither stops the source nor
    yields a COMPLETE capture; the reason is explicit."""

    @classmethod
    def setUpClass(cls):
        pilots.FixtureTest.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        pilots.FixtureTest.tearDownClass.__func__(cls)

    def test_failed_retention_is_recorded_and_capture_is_incomplete(self):
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
            keep = root / "keep"
            original = retention.validate_checkpoint

            def fail_at_22(state):
                from experiments.exp12.state import load_meta

                if load_meta(state)["interaction_step"] == 22:
                    raise ValueError("synthetic validation failure")
                return original(state)

            try:
                t.restore(self.fstate)
                t.extra_state["fork"] = {"fork_step": 20, "arm_end_step": 24, "run_key": json.loads(
                    self.metadata.read_text())["identity"]["run_key"]}

                def save_routine(trainer):
                    if trainer.interaction_step in (22, 24):
                        trainer.save(trainer.run_dir / "state")

                with mock.patch.object(retention, "validate_checkpoint", fail_at_22):
                    with self.assertRaises(CaptureIncomplete) as caught:
                        with record_exp2_scope(root / "stream", 4,
                                               retain={"root": str(keep), "until_step": 24, "replay": "omit"}):
                            t.train(24, after_step=save_routine)
                report = caught.exception.report
                self.assertEqual(report["stream_verdict"], "COMPLETE")
                self.assertEqual(report["retention"]["verdict"], "FAILED")
                self.assertTrue(any("step 22: required snapshot missing" == r for r in report["retention"]["reasons"]))
                self.assertTrue(any(r.startswith("FAILED record") for r in report["retention"]["reasons"]))
                self.assertEqual(report["capture_verdict"], "INCOMPLETE")
                self.assertTrue(StreamReader(root / "stream").manifest["complete"])  # stream itself sound
                self.assertEqual(json.loads((root / "stream/capture_report.json").read_text()), report)
                self.assertTrue((keep / "step_000000024").is_dir())
                self.assertEqual(t.interaction_step, 24)  # the source ran to completion
            finally:
                t.close()


class FinalizationIntegrityTest(unittest.TestCase):
    """Real Exp1 fork + Exp2 arm (Codex fixture): death after DONE before sealing."""

    @classmethod
    def setUpClass(cls):
        adversarial.RealFinalizationSafetyTest.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        adversarial.RealFinalizationSafetyTest.tearDownClass.__func__(cls)

    setUp = adversarial.RealFinalizationSafetyTest.setUp
    tearDown = adversarial.RealFinalizationSafetyTest.tearDown

    def test_missing_intermediate_snapshot_fails_with_explicit_reason(self):
        steps = sorted(p for p in self.keep_copy.glob("step_*") if p != self.snapshot)
        shutil.rmtree(steps[1])
        with self.assertRaises(CaptureIncomplete) as caught:
            finalize_capture(self.stream_copy, self.arm)
        self.assertIn(f"step {int(steps[1].name[5:])}: required snapshot missing",
                      caught.exception.report["retention"]["reasons"])

    def test_fork_step_injected_snapshot_is_required(self):
        manifest = StreamReader(self.stream_copy, require_complete=False).manifest
        fork_step = manifest["provenance"]["start_step"]
        shutil.rmtree(self.keep_copy / f"step_{fork_step:09d}")
        with self.assertRaisesRegex(CaptureIncomplete, f"step {fork_step}: required snapshot missing"):
            finalize_capture(self.stream_copy, self.arm)

    def test_interrupted_finalization_resumes_to_the_same_verdict(self):
        with mock.patch.object(recording, "capture_report", side_effect=RuntimeError("death in finalize")):
            with self.assertRaisesRegex(RuntimeError, "death in finalize"):
                finalize_capture(self.stream_copy, self.arm)
        resumed = finalize_capture(self.stream_copy, self.arm)
        self.assertEqual(resumed["capture_verdict"], "COMPLETE")
        self.assertEqual(resumed, finalize_capture(self.stream_copy, self.arm))
        self.assertEqual(markers(self.keep_copy, "FAILED") + markers(self.keep_copy, "CONFLICT"), [])

    def test_failures_are_repeatable_not_cleared_by_rerun(self):
        (self.snapshot / "agent_ckpt/_sharding").write_text("corrupt")
        for _ in range(2):
            with self.assertRaises(CaptureIncomplete) as caught:
                finalize_capture(self.stream_copy, self.arm)
            self.assertEqual(caught.exception.report["capture_verdict"], "INCOMPLETE")
            self.assertEqual(caught.exception.report["stream_verdict"], "COMPLETE")


class HealthyResumptionTest(unittest.TestCase):
    """Resumed real arm (final save, death before DONE, resume): COMPLETE, no markers."""

    def test_resumed_capture_report_is_complete(self):
        try:
            fixtures.FinalSaveDeathTest.setUpClass()
            case = fixtures.FinalSaveDeathTest("test_resumed_final_save_keeps_the_arm_end_snapshot_usable")
            case.test_resumed_final_save_keeps_the_arm_end_snapshot_usable()
            base = fixtures.FinalSaveDeathTest.tmp
            report = json.loads((base / "stream_i" / "capture_report.json").read_text())
            self.assertEqual(report["capture_verdict"], "COMPLETE")
            self.assertEqual(report["retention"]["reasons"], [])
            self.assertEqual(len(markers(base / "snap_i", "EQUIVALENT")), 1)
            self.assertEqual(markers(base / "snap_i", "FAILED") + markers(base / "snap_i", "CONFLICT"), [])
        finally:
            fixtures.FinalSaveDeathTest.tearDownClass()


if __name__ == "__main__":
    unittest.main()
