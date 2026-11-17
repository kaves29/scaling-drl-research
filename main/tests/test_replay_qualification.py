import unittest

from experiments.exp12.replay_exp2_qualification import validate_evidence


class QualificationEvidenceTests(unittest.TestCase):
    def test_requires_all_four_full_branches_and_both_probes(self):
        conditions = {name: {"raw_env_steps": 10000} for name in (
            "small_frozen", "small_learning", "large_frozen", "large_learning")}
        rows = [{"type": "pilot_diagnostic", "checkpoint": 10, "branch_counters": conditions},
                *({"type": "probe", "check": 10, "architecture": arch} for arch in ("small", "large"))]
        validate_evidence(rows, 10, 10000)
        with self.assertRaises(ValueError):
            validate_evidence(rows[:-1], 10, 10000)
        conditions["small_learning"]["raw_env_steps"] = 9999
        with self.assertRaises(ValueError):
            validate_evidence(rows, 10, 10000)

    def test_qualification_uses_distinct_fixed_5000_update_schedule(self):
        from types import SimpleNamespace
        from experiments.exp12.replay_exp2_qualification import configure_qualification
        runner = SimpleNamespace(checks=[66001], probes=[66001], update_budget=990002,
                                 config_hash='pilot-config')
        configure_qualification(runner)
        self.assertEqual(runner.checks, [5000])
        self.assertEqual(runner.probes, [5000])
        self.assertEqual(runner.update_budget, 5000)
        self.assertNotEqual(runner.config_hash, 'pilot-config')

    def test_replay_loop_reaches_exactly_5000_before_qualification_checkpoint(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import numpy as np
        from experiments.exp12.replay_exp2_pilot import ReplayExp2Pilot
        from experiments.exp12.replay_exp2_qualification import configure_qualification

        class AtCheckpoint(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            runner = object.__new__(ReplayExp2Pilot)
            runner.root = Path(tmp)
            runner.update_budget = 990002
            runner.config_hash = 'pilot-config'
            runner.parent = {'source_commit': 'fixture'}
            runner.fresh_done = True
            runner.trainer = SimpleNamespace(update_step=0, extra_state={},
                _attach_logger=Mock(), close=Mock(),
                _sac_agent=SimpleNamespace(actor=SimpleNamespace(params={})))
            runner.trainer.buffer = Mock()
            runner.trainer.buffer.sample.return_value = ({'reward': np.array([1.])}, np.array([0]))
            runner._check_resource_cap = Mock()
            runner._shadow_updates = Mock()
            runner._checkpoint = Mock(side_effect=AtCheckpoint())
            runner.save = Mock()
            configure_qualification(runner)
            with patch('experiments.exp12.online_exp2_pilot._reference_state', return_value={}), \
                 patch('experiments.exp12.replay_exp2_pilot.wandb.run', None):
                with self.assertRaises(AtCheckpoint):
                    runner.run()
            self.assertEqual(runner.trainer.update_step, 5000)
            self.assertEqual(runner._shadow_updates.call_count, 5000)
            runner._checkpoint.assert_called_once_with(runner.trainer)
