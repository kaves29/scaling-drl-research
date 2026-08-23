import os
import shutil
import tempfile
import unittest

import pandas as pd

from analysis.metrics_store import METRIC_COLUMNS, RunIdentity, metrics_path
from analysis.pipeline import run_post_hoc_onset_analysis
from utils.onset_ledger import WandbIdentity, load_onset_ledger


class _FakeRun:
    class _Summary(dict):
        def update(self, d):
            dict.update(self, d)

    def __init__(self, name, id_):
        self.name = name
        self.id = id_
        self.summary = self._Summary()


LOGGING_PER_INTERACTION_STEP = 1000  # matches the 1000-spaced `steps` fixtures below

ONSET_CFG = {
    # raw interaction steps; sustain_window / LOGGING_PER_INTERACTION_STEP == 2
    # recorded points, reproducing the old sustain_window_n=2 test semantics.
    "sustain_window": 2 * LOGGING_PER_INTERACTION_STEP,
    "propagation_window": 5000,
    "baseline_percentile": 95,
    "force_baseline_recompute": False,
    "critic_degradation_method_version": "td_variance_p95_sustained_v1",
    "propagation_method_version": "actor_grad_cosine_p95_windowed_v1",
}


class TestPostHocPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.metrics_root = os.path.join(self.tmpdir, "metrics")
        self.baseline_root = os.path.join(self.tmpdir, "baselines")
        self.ledger_root = os.path.join(self.tmpdir, "ledgers")
        self.baseline_identities = [
            RunIdentity(experiment="angle_1", architecture="D2W512", environment="reacher-hard", seed=s)
            for s in range(1, 6)
        ]

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_metrics(self, identity, steps, td_var, actor_cos):
        path = metrics_path(identity, root=self.metrics_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "interaction_step": steps,
                "env_step": [s * 2 for s in steps],
                "td_error_variance": td_var,
                "actor_grad_cosine": actor_cos,
            },
            columns=METRIC_COLUMNS,
        ).to_csv(path, index=False)

    def _write_flat_baselines(self, steps, td_threshold_seed_values, cos_threshold_seed_values):
        for i, identity in enumerate(self.baseline_identities):
            self._write_metrics(
                identity, steps,
                td_var=[td_threshold_seed_values[i]] * len(steps),
                actor_cos=[cos_threshold_seed_values[i]] * len(steps),
            )

    def test_end_to_end_success_path(self):
        steps = list(range(1000, 11000, 1000))  # 1000..10000
        self._write_flat_baselines(steps, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])

        run_identity = RunIdentity(experiment="angle_1", architecture="D2W512", environment="reacher-hard", seed=42)
        td_var = [0.5, 0.5, 10.0, 10.0, 10.0, 0.5, 0.5, 0.5, 0.5, 0.5]
        actor_cos = [0.0] * 10
        actor_cos[4] = 99.0  # step 5000, inside [3000, 8000] window of onset at step 3000
        self._write_metrics(run_identity, steps, td_var, actor_cos)

        identity = WandbIdentity(run_obj=_FakeRun("angle1-D2W512-reacher-hard-seed42", "wid42"))
        path = run_post_hoc_onset_analysis(
            run_identity=run_identity,
            critic_degradation_enabled=True,
            pathology_prop_enabled=True,
            baseline_identities=self.baseline_identities,
            onset_cfg=ONSET_CFG,
            logging_per_interaction_step=LOGGING_PER_INTERACTION_STEP,
            wandb_identity=identity,
            metrics_root=self.metrics_root,
            baseline_root=self.baseline_root,
            ledger_root=self.ledger_root,
        )

        df = load_onset_ledger("angle_1", "D2W512", root=self.ledger_root)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["run_key"], run_identity.run_key)
        self.assertEqual(row["critic_degradation_onset_step"], 3000)
        self.assertEqual(row["propagation_onset_step"], 5000)
        self.assertEqual(row["propagation_lag"], 2000)
        self.assertEqual(row["status"], "success")
        self.assertTrue(path.exists())

    def test_pathology_prop_without_critic_degradation_flag_is_needs_manual_review(self):
        steps = [1000, 2000, 3000]
        self._write_flat_baselines(steps, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])

        run_identity = RunIdentity(experiment="angle_1", architecture="D2W512", environment="reacher-hard", seed=7)
        self._write_metrics(run_identity, steps, td_var=[0.1, 0.1, 0.1], actor_cos=[99.0, 99.0, 99.0])

        identity = WandbIdentity(run_obj=_FakeRun("run-7", "wid7"))
        run_post_hoc_onset_analysis(
            run_identity=run_identity,
            critic_degradation_enabled=False,
            pathology_prop_enabled=True,
            baseline_identities=self.baseline_identities,
            onset_cfg=ONSET_CFG,
            logging_per_interaction_step=LOGGING_PER_INTERACTION_STEP,
            wandb_identity=identity,
            metrics_root=self.metrics_root,
            baseline_root=self.baseline_root,
            ledger_root=self.ledger_root,
        )

        df = load_onset_ledger("angle_1", "D2W512", root=self.ledger_root)
        row = df[df["run_key"] == run_identity.run_key].iloc[0]
        self.assertEqual(row["status"], "needs_manual_review")
        self.assertTrue(pd.isna(row["propagation_onset_step"]))

    def test_missing_baseline_produces_needs_manual_review_not_crash(self):
        run_identity = RunIdentity(experiment="angle_1", architecture="D2W512", environment="reacher-hard", seed=99)
        self._write_metrics(run_identity, [1000, 2000], td_var=[1.0, 1.0], actor_cos=[1.0, 1.0])
        # no baseline metrics written at all for self.baseline_identities

        identity = WandbIdentity(run_obj=_FakeRun("run-99", "wid99"))
        path = run_post_hoc_onset_analysis(
            run_identity=run_identity,
            critic_degradation_enabled=True,
            pathology_prop_enabled=False,
            baseline_identities=self.baseline_identities,
            onset_cfg=ONSET_CFG,
            logging_per_interaction_step=LOGGING_PER_INTERACTION_STEP,
            wandb_identity=identity,
            metrics_root=self.metrics_root,
            baseline_root=self.baseline_root,
            ledger_root=self.ledger_root,
        )

        df = load_onset_ledger("angle_1", "D2W512", root=self.ledger_root)
        row = df[df["run_key"] == run_identity.run_key].iloc[0]
        self.assertEqual(row["status"], "needs_manual_review")
        self.assertTrue(path.exists())

    def test_no_persisted_metrics_produces_needs_manual_review(self):
        run_identity = RunIdentity(experiment="angle_1", architecture="D2W512", environment="reacher-hard", seed=123)
        identity = WandbIdentity(run_obj=_FakeRun("run-123", "wid123"))

        run_post_hoc_onset_analysis(
            run_identity=run_identity,
            critic_degradation_enabled=True,
            pathology_prop_enabled=False,
            baseline_identities=self.baseline_identities,
            onset_cfg=ONSET_CFG,
            logging_per_interaction_step=LOGGING_PER_INTERACTION_STEP,
            wandb_identity=identity,
            metrics_root=self.metrics_root,
            baseline_root=self.baseline_root,
            ledger_root=self.ledger_root,
        )

        df = load_onset_ledger("angle_1", "D2W512", root=self.ledger_root)
        row = df[df["run_key"] == run_identity.run_key].iloc[0]
        self.assertEqual(row["status"], "needs_manual_review")

    def test_disabled_flags_write_nothing(self):
        run_identity = RunIdentity(experiment="angle_1", architecture="D2W512", environment="reacher-hard", seed=1)
        result = run_post_hoc_onset_analysis(
            run_identity=run_identity,
            critic_degradation_enabled=False,
            pathology_prop_enabled=False,
            baseline_identities=self.baseline_identities,
            onset_cfg=ONSET_CFG,
            logging_per_interaction_step=LOGGING_PER_INTERACTION_STEP,
            wandb_identity=None,
            metrics_root=self.metrics_root,
            baseline_root=self.baseline_root,
            ledger_root=self.ledger_root,
        )
        self.assertIsNone(result)
        df = load_onset_ledger("angle_1", "D2W512", root=self.ledger_root)
        self.assertEqual(len(df), 0)


if __name__ == "__main__":
    unittest.main()
