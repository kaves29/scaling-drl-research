"""Independent adversarial review of codex/exp3-capture-fixes @ 853b6a5 (CPU).

Run from main/: python -m unittest tests.test_exp3_capture_review -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.exp12 import fork
from experiments.exp12.state import latest_state_dir
from experiments.exp12.trainer import Exp12Trainer
from experiments.exp3.artifacts import Artifact
from experiments.exp3.recording import record_exp2_scope
from experiments.exp3.retention import read_snapshot, retain_state
from experiments.exp3.streams import StreamReader
from tests.exp12_helpers import compose, patch_wandb, tiny_overrides
from tests.test_exp12_fork import fork_overrides, run_arm, run_exp1


def records(root, kind):
    return sorted(Path(root).glob(f"{kind}_*.json"))


class ResaveRetentionTest(unittest.TestCase):
    """Orbax/pickle bytes differ between two saves of the same logical state."""

    @classmethod
    def setUpClass(cls):
        cls.patchers = patch_wandb()
        cls.tmp = Path(tempfile.mkdtemp())
        cfg = compose(tiny_overrides(steps=40, extra=["run_role=dev", "diagnostics.kl_reference_size=4",
                                                      "actor_grad_cosine_every=10000"]))
        t = Exp12Trainer(cfg, str(cls.tmp / "src"))
        t.start()
        t.train(20)
        cls.first = t.save(cls.tmp / "first")
        t.train(21)
        cls.later = t.save(cls.tmp / "later")  # a genuinely different state
        t.close()
        resumed = Exp12Trainer(cfg, str(cls.tmp / "src2"))
        resumed.restore(cls.first)
        cls.resaved = resumed.save(cls.tmp / "resaved")  # same logical state, new bytes
        resumed.close()

    @classmethod
    def tearDownClass(cls):
        for p in cls.patchers:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_resave_bytes_differ_but_logical_state_is_identical(self):
        from experiments.exp12.state import state_differences
        from experiments.exp3.artifacts import digest

        self.assertEqual(state_differences(self.first, self.resaved), [])
        byte_equal = all(digest(p) == digest(self.resaved / p.relative_to(self.first))
                         for p in self.first.rglob("*") if p.is_file())
        self.assertFalse(byte_equal)  # the premise of byte-hash conflict detection is false

    def test_logically_identical_resave_is_not_a_conflict(self):
        for mode in ("omit", "retain"):
            with self.subTest(mode):
                root = self.tmp / f"keep_{mode}"
                target = retain_state(self.first, root, 20, mode)
                retain_state(self.resaved, root, 20, mode)
                self.assertEqual(records(root, "CONFLICT"), [])
                self.assertEqual(len(records(root, "EQUIVALENT")), 1)
                read_snapshot(target)  # still usable

    def test_genuinely_different_resave_is_still_a_conflict(self):
        for mode in ("omit", "retain"):
            with self.subTest(mode):
                root = self.tmp / f"conflict_{mode}"
                target = retain_state(self.first, root, 20, mode)
                retain_state(self.later, root, 20, mode)
                conflicts = records(root, "CONFLICT")
                self.assertEqual(len(conflicts), 1)
                self.assertTrue(json.loads(conflicts[0].read_text())["logical_differences"])
                with self.assertRaisesRegex(ValueError, "conflicting"):
                    read_snapshot(target)


class FinalSaveDeathTest(unittest.TestCase):
    """Real entry points: an injected arm dies after its final routine save, before DONE."""

    @classmethod
    def setUpClass(cls):
        cls.patchers = patch_wandb()
        cls.tmp = Path(tempfile.mkdtemp())
        cls.overrides = fork_overrides(str(cls.tmp / "results"))
        cls.parent = cls.tmp / "parent"
        run_exp1(str(cls.parent), cls.overrides, interval=20)

    @classmethod
    def tearDownClass(cls):
        for p in cls.patchers:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_resumed_final_save_keeps_the_arm_end_snapshot_usable(self):
        import experiments.exp2_arm as exp2_arm

        arm, stream, snaps = self.tmp / "arm", self.tmp / "stream_i", self.tmp / "snap_i"
        retain = {"root": str(snaps), "until_step": "arm_end", "replay": "omit"}
        original = exp2_arm.atomic_write_text

        def die_at_done(path, text):
            if Path(path).name == exp2_arm.DONE_MARKER:
                raise RuntimeError("simulated death between final save and DONE")
            return original(path, text)

        with mock.patch.object(exp2_arm, "atomic_write_text", die_at_done):
            with self.assertRaisesRegex(RuntimeError, "simulated death"):
                with record_exp2_scope(stream, 16, retain=retain):
                    run_arm(str(arm), self.overrides, str(self.parent), "injected", interval=20)
        self.assertFalse((arm / exp2_arm.DONE_MARKER).exists())
        with record_exp2_scope(stream, 16, resume=True, retain=retain):
            run_arm(str(arm), self.overrides, str(self.parent), "injected", interval=20)
        self.assertTrue((arm / exp2_arm.DONE_MARKER).exists())
        plan = fork.read_fork(self.parent)
        end = snaps / f"step_{plan['arm_end_step']:09d}"
        self.assertEqual(records(snaps, "CONFLICT"), [])
        read_snapshot(end)
        Artifact(end, arm / "run_metadata.json", load_replay=False)
        reader = StreamReader(stream)  # sealed and complete
        self.assertEqual(reader.manifest["count"], plan["arm_end_step"] - plan["fork_step"])
        self.assertEqual(latest_state_dir(arm / "state").name.split("_")[1],
                         f"{plan['arm_end_step']:09d}")


if __name__ == "__main__":
    unittest.main()
