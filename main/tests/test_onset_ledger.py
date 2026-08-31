import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.onset_ledger import (
    REQUIRED_COLUMNS,
    LedgerSchemaError,
    RunIdentityValidationError,
    WandbIdentity,
    load_onset_ledger,
    log_onset_event,
)


class _FakeWandbRun:
    def __init__(self, name, id_):
        self.name = name
        self.id = id_
        self.summary = {}

    class _Summary(dict):
        def update(self, d):
            dict.update(self, d)

    def __post_init__(self):
        pass


def make_fake_run(name, id_):
    run = _FakeWandbRun(name, id_)
    run.summary = _FakeWandbRun._Summary()
    return run


def make_row(run_key, exact_run_name, wandb_run_id, architecture="D2W512",
             environment="reacher-hard", seed=1, status="success",
             degrad_step=1000, prop_step=None):
    return {
        "run_key": run_key,
        "exact_run_name": exact_run_name,
        "wandb_run_id": wandb_run_id,
        "architecture": architecture,
        "environment": environment,
        "seed": seed,
        "critic_degradation_onset_step": degrad_step,
        "critic_degradation_method": "td_variance_p95_sustained_v1",
        "propagation_onset_step": prop_step,
        "propagation_method": "actor_grad_cosine_p95_windowed_v1" if prop_step is not None else None,
        "propagation_lag": (prop_step - degrad_step) if prop_step is not None and degrad_step is not None else None,
        "status": status,
        "detection_notes": "unit test row",
    }


class TestOnsetLedger(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = os.path.join(self.tmpdir, "ledgers")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_ledger_and_architecture_dir_are_created(self):
        row = make_row("angle1_D2W512_reacher-hard_seed1", "run-abc", "id1")
        identity = WandbIdentity(run_obj=make_fake_run("run-abc", "id1"))

        path = log_onset_event("angle_1", "D2W512", row, identity=identity, root=self.root, mirror_to_wandb=False)

        self.assertTrue(path.exists())
        self.assertEqual(path, Path(self.root) / "angle_1" / "architectures" / "D2W512" / "onset_events.csv")
        df = pd.read_csv(path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["run_key"], "angle1_D2W512_reacher-hard_seed1")
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, df.columns)

    def test_inserting_a_second_distinct_run_appends_row(self):
        identity1 = WandbIdentity(run_obj=make_fake_run("run-1", "id1"))
        identity2 = WandbIdentity(run_obj=make_fake_run("run-2", "id2"))

        log_onset_event("angle_1", "D2W512", make_row("run_key_1", "run-1", "id1"),
                         identity=identity1, root=self.root, mirror_to_wandb=False)
        log_onset_event("angle_1", "D2W512", make_row("run_key_2", "run-2", "id2"),
                         identity=identity2, root=self.root, mirror_to_wandb=False)

        df = load_onset_ledger("angle_1", "D2W512", root=self.root)
        self.assertEqual(len(df), 2)
        self.assertCountEqual(df["run_key"].tolist(), ["run_key_1", "run_key_2"])

    def test_upsert_updates_existing_run_key_without_duplicating(self):
        identity = WandbIdentity(run_obj=make_fake_run("run-1", "id1"))

        log_onset_event(
            "angle_1", "D2W512",
            make_row("run_key_1", "run-1", "id1", status="no_onset_detected", degrad_step=None),
            identity=identity, root=self.root, mirror_to_wandb=False,
        )
        log_onset_event(
            "angle_1", "D2W512",
            make_row("run_key_1", "run-1", "id1", status="success", degrad_step=5000),
            identity=identity, root=self.root, mirror_to_wandb=False,
        )

        df = load_onset_ledger("angle_1", "D2W512", root=self.root)
        self.assertEqual(len(df), 1, "upsert must not create a duplicate row for the same run_key")
        self.assertEqual(df.iloc[0]["status"], "success")
        self.assertEqual(df.iloc[0]["critic_degradation_onset_step"], 5000)

    def test_missing_required_column_in_row_raises(self):
        identity = WandbIdentity(run_obj=make_fake_run("run-1", "id1"))
        bad_row = make_row("run_key_1", "run-1", "id1")
        del bad_row["status"]

        with self.assertRaises(LedgerSchemaError):
            log_onset_event("angle_1", "D2W512", bad_row, identity=identity, root=self.root, mirror_to_wandb=False)

    def test_run_identity_mismatch_raises_and_does_not_write(self):
        # row claims a different exact_run_name than the live WandB run
        identity = WandbIdentity(run_obj=make_fake_run("actual-run-name", "id1"))
        row = make_row("run_key_1", "claimed-wrong-name", "id1")

        path = Path(self.root) / "angle_1" / "architectures" / "D2W512" / "onset_events.csv"
        with self.assertRaises(RunIdentityValidationError):
            log_onset_event("angle_1", "D2W512", row, identity=identity, root=self.root, mirror_to_wandb=False)

        self.assertFalse(path.exists(), "a mismatched identity must never produce a ledger write")

    def test_unverifiable_identity_downgrades_status_instead_of_crashing(self):
        # identity=None => cannot verify => must not raise, must downgrade status
        row = make_row("run_key_1", "run-1", "id1", status="success")
        path = log_onset_event("angle_1", "D2W512", row, identity=None, root=self.root, mirror_to_wandb=False)

        df = pd.read_csv(path)
        self.assertEqual(df.iloc[0]["status"], "needs_manual_review")

    def test_malformed_existing_csv_is_not_silently_overwritten(self):
        path = Path(self.root) / "angle_1" / "architectures" / "D2W512" / "onset_events.csv"
        path.parent.mkdir(parents=True)
        path.write_text("this,is,not\na,valid,,,, ledger\"\"\"")

        identity = WandbIdentity(run_obj=make_fake_run("run-1", "id1"))
        row = make_row("run_key_1", "run-1", "id1")

        original_bytes = path.read_bytes()
        try:
            log_onset_event("angle_1", "D2W512", row, identity=identity, root=self.root, mirror_to_wandb=False)
            wrote_ok = True
        except LedgerSchemaError:
            wrote_ok = False

        if not wrote_ok:
            self.assertEqual(path.read_bytes(), original_bytes, "malformed ledger must be left untouched on failure")

    def test_missing_required_column_in_existing_ledger_raises_on_load(self):
        path = Path(self.root) / "angle_1" / "architectures" / "D2W512" / "onset_events.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame([{"run_key": "x", "status": "success"}]).to_csv(path, index=False)

        with self.assertRaises(LedgerSchemaError):
            load_onset_ledger("angle_1", "D2W512", root=self.root)

    def test_no_leftover_temp_files_after_write(self):
        identity = WandbIdentity(run_obj=make_fake_run("run-1", "id1"))
        log_onset_event("angle_1", "D2W512", make_row("run_key_1", "run-1", "id1"),
                         identity=identity, root=self.root, mirror_to_wandb=False)

        arch_dir = Path(self.root) / "angle_1" / "architectures" / "D2W512"
        leftover_tmp_files = [p for p in arch_dir.iterdir() if p.suffix == ".tmp"]
        self.assertEqual(leftover_tmp_files, [])

    def test_load_onset_ledger_returns_empty_schema_when_nothing_exists(self):
        df = load_onset_ledger("angle_1", "D2W512", root=self.root)
        self.assertEqual(len(df), 0)
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, df.columns)

    def test_load_onset_ledger_without_architecture_concatenates_all(self):
        identity1 = WandbIdentity(run_obj=make_fake_run("run-1", "id1"))
        identity2 = WandbIdentity(run_obj=make_fake_run("run-2", "id2"))
        log_onset_event("angle_1", "D2W512", make_row("run_key_1", "run-1", "id1", architecture="D2W512"),
                         identity=identity1, root=self.root, mirror_to_wandb=False)
        log_onset_event("angle_1", "XXL", make_row("run_key_2", "run-2", "id2", architecture="XXL"),
                         identity=identity2, root=self.root, mirror_to_wandb=False)

        df = load_onset_ledger("angle_1", architecture=None, root=self.root)
        self.assertEqual(len(df), 2)
        self.assertCountEqual(df["architecture"].tolist(), ["D2W512", "XXL"])


if __name__ == "__main__":
    unittest.main()
