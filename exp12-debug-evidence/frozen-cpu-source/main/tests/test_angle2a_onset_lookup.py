import os
import shutil
import tempfile
import unittest

from experiments.angle_2a.errors import Angle2AOnsetLookupError
from experiments.angle_2a.onset_lookup import lookup_critic_degradation_onset
from utils.onset_ledger import WandbIdentity, log_onset_event


class _FakeRun:
    class _Summary(dict):
        def update(self, d):
            dict.update(self, d)

    def __init__(self, name, id_):
        self.name = name
        self.id = id_
        self.summary = self._Summary()


def make_row(run_key, arch, env, seed, status="success", onset_step=1000, exact_run_name=None, wandb_run_id=None):
    return {
        "run_key": run_key,
        "exact_run_name": exact_run_name or run_key,
        "wandb_run_id": wandb_run_id or f"wid_{run_key}",
        "architecture": arch,
        "environment": env,
        "seed": seed,
        "critic_degradation_onset_step": onset_step,
        "critic_degradation_method": "td_variance_p95_sustained_v1",
        "propagation_onset_step": None,
        "propagation_method": None,
        "propagation_lag": None,
        "status": status,
        "detection_notes": "test row",
    }


class TestAngle2AOnsetLookup(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ledger_root = os.path.join(self.tmpdir, "ledgers")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _log(self, arch, env, seed, **kwargs):
        run_key = f"angle1_{arch}_{env}_seed{seed}"
        row = make_row(run_key, arch, env, seed, **kwargs)
        identity = WandbIdentity(run_obj=_FakeRun(row["exact_run_name"], row["wandb_run_id"]))
        log_onset_event("angle_1", arch, row, identity=identity, root=self.ledger_root, mirror_to_wandb=False)

    def test_correct_onset_selected_for_5x768(self):
        self._log("D5W768", "reacher-hard", seed=3, onset_step=4000)
        self._log("D7W1024", "reacher-hard", seed=3, onset_step=9000)

        result = lookup_critic_degradation_onset(
            architecture="D5W768", environment="reacher-hard", seed=3, ledger_root=self.ledger_root
        )
        self.assertEqual(result.onset_step, 4000)

    def test_correct_onset_selected_for_7x1024_is_independent_from_5x768(self):
        self._log("D5W768", "reacher-hard", seed=3, onset_step=4000)
        self._log("D7W1024", "reacher-hard", seed=3, onset_step=9000)

        result_a = lookup_critic_degradation_onset(
            architecture="D5W768", environment="reacher-hard", seed=3, ledger_root=self.ledger_root
        )
        result_b = lookup_critic_degradation_onset(
            architecture="D7W1024", environment="reacher-hard", seed=3, ledger_root=self.ledger_root
        )

        self.assertNotEqual(result_a.onset_step, result_b.onset_step)
        # explicitly guard against the prohibited "average the two onsets" shortcut
        self.assertNotEqual(result_a.onset_step, (4000 + 9000) / 2)
        self.assertNotEqual(result_b.onset_step, (4000 + 9000) / 2)

    def test_lookup_is_seed_specific(self):
        self._log("D5W768", "reacher-hard", seed=1, onset_step=1000)
        self._log("D5W768", "reacher-hard", seed=2, onset_step=5000)

        result_seed1 = lookup_critic_degradation_onset(
            architecture="D5W768", environment="reacher-hard", seed=1, ledger_root=self.ledger_root
        )
        result_seed2 = lookup_critic_degradation_onset(
            architecture="D5W768", environment="reacher-hard", seed=2, ledger_root=self.ledger_root
        )

        self.assertEqual(result_seed1.onset_step, 1000)
        self.assertEqual(result_seed2.onset_step, 5000)

    def test_missing_entry_raises(self):
        with self.assertRaises(Angle2AOnsetLookupError):
            lookup_critic_degradation_onset(
                architecture="D5W768", environment="reacher-hard", seed=99, ledger_root=self.ledger_root
            )

    def test_no_onset_detected_status_is_rejected(self):
        self._log("D5W768", "reacher-hard", seed=3, status="no_onset_detected", onset_step=None)
        with self.assertRaises(Angle2AOnsetLookupError):
            lookup_critic_degradation_onset(
                architecture="D5W768", environment="reacher-hard", seed=3, ledger_root=self.ledger_root
            )

    def test_needs_manual_review_status_is_rejected(self):
        self._log("D5W768", "reacher-hard", seed=3, status="needs_manual_review", onset_step=None)
        with self.assertRaises(Angle2AOnsetLookupError):
            lookup_critic_degradation_onset(
                architecture="D5W768", environment="reacher-hard", seed=3, ledger_root=self.ledger_root
            )

    def test_ambiguous_is_no_longer_a_valid_status_at_all(self):
        # "ambiguous" was removed from the methodology entirely (not just
        # from onset_lookup's accepted set) - the ledger itself must refuse
        # to persist a row carrying it, before onset_lookup ever runs.
        from utils.onset_ledger import LedgerSchemaError

        run_key = "angle1_D5W768_reacher-hard_seed3"
        row = make_row(run_key, "D5W768", "reacher-hard", seed=3, status="ambiguous", onset_step=None)
        identity = WandbIdentity(run_obj=_FakeRun(row["exact_run_name"], row["wandb_run_id"]))
        with self.assertRaises(LedgerSchemaError):
            log_onset_event("angle_1", "D5W768", row, identity=identity, root=self.ledger_root, mirror_to_wandb=False)

    def test_wrong_environment_is_not_matched(self):
        self._log("D5W768", "cheetah-run", seed=3, onset_step=1234)
        with self.assertRaises(Angle2AOnsetLookupError):
            lookup_critic_degradation_onset(
                architecture="D5W768", environment="reacher-hard", seed=3, ledger_root=self.ledger_root
            )

    def test_never_falls_back_to_another_seed(self):
        self._log("D5W768", "reacher-hard", seed=1, onset_step=1000)
        # seed=2 has no entry at all; must raise, never silently reuse seed=1's onset
        with self.assertRaises(Angle2AOnsetLookupError):
            lookup_critic_degradation_onset(
                architecture="D5W768", environment="reacher-hard", seed=2, ledger_root=self.ledger_root
            )


if __name__ == "__main__":
    unittest.main()
