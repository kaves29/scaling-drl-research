"""Read-only pilot inspection on real tiny Exp1 runs: no trigger, natural fork, forced fork."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp12_helpers import patch_wandb, tiny_overrides  # noqa: E402
from test_exp12_fork import fork_overrides, run_exp1  # noqa: E402

from scripts import exp12_pilot_inspect  # noqa: E402


def tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        digest.update(str(path.relative_to(root)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def natural_overrides(results, threshold):
    return tiny_overrides("hopper-hop", "dmc_medium", seed=102, steps=400, extra=[
        "run_role=dev", "fork.architectures=[D1W8]", "env.max_episode_steps=40",
        "fork.eval_episodes=1", f"trigger.null_threshold={threshold}", f"results_root={results}"])


class PilotInspectTest(unittest.TestCase):
    def setUp(self):
        self.patchers = patch_wandb()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for p in self.patchers:
            self.addCleanup(p.stop)

    def run_and_inspect(self, overrides):
        root = Path(self.tmp.name) / "pilot"
        run_exp1(str(root / "checkpoints" / "run"), overrides)
        before = tree_digest(root)
        out = Path(self.tmp.name) / "report.json"
        with self.assertRaises(SystemExit):
            exp12_pilot_inspect.main(["--root", str(root), "--out", str(root / "inside.json")])
        exp12_pilot_inspect.main(["--root", str(root), "--out", str(out)])
        self.assertEqual(tree_digest(root), before)  # read-only
        return json.loads(out.read_text())

    def test_no_natural_trigger_prescribes_stop_and_consult(self):
        report = self.run_and_inspect(natural_overrides(Path(self.tmp.name) / "r", 1e9))
        self.assertEqual(report["verdict"], "NO_NATURAL_TRIGGER")
        self.assertTrue(report["horizon"]["complete"])
        self.assertEqual(report["trigger"]["fired_checks"], [])
        self.assertIsNone(report["trigger"]["recomputed_f_star"])
        self.assertFalse(report["fork"]["fork_dir_exists"])
        self.assertIn("stop and consult", report["next_step"])

    def test_natural_trigger_yields_a_valid_fork_and_exact_statistics(self):
        report = self.run_and_inspect(natural_overrides(Path(self.tmp.name) / "r", -1e9))
        self.assertEqual(report["verdict"], "NATURAL_FORK_AVAILABLE", report["problems"])
        self.assertEqual(report["trigger"]["recomputed_f_star"]["check_index"], 2)
        self.assertEqual(report["trigger"]["recorded_f_star"]["check_index"], 2)
        self.assertTrue(report["fork"]["natural_fork_valid"])
        self.assertEqual(report["fork"]["plan"]["fork_check_index"], 2)
        self.assertTrue(report["log_corroboration"]["files_scanned"] >= 0)

    def test_forced_fork_is_never_eligible(self):
        report = self.run_and_inspect(fork_overrides(Path(self.tmp.name) / "r"))
        self.assertEqual(report["verdict"], "FORCED_FORK_NOT_ELIGIBLE")
        self.assertTrue(report["fork"]["fork_ready"])
        self.assertTrue(report["trigger"]["forced_checks"])


if __name__ == "__main__":
    unittest.main()
