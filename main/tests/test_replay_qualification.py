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

    def test_a100_smoke_is_short_and_source_bound_without_changing_pilot_config(self):
        from types import SimpleNamespace
        from omegaconf import OmegaConf
        from experiments.exp12.replay_exp2_qualification import configure_a100_smoke
        cfg = OmegaConf.create({'online_exp2': {'branch_raw_steps': 10000,
            'measurement_raw_steps': [0, 5000, 10000], 'diagnostic_save_every': 250},
            'probe': {'rounds': 10, 'steps': 2000, 'pool_size': 25600,
                      'batch_size': 256, 'eval_chunk': 2560}})
        runner = SimpleNamespace(checks=[66001], probes=[66001], update_budget=990002,
                                 config_hash='pilot-config', cfg=OmegaConf.create(OmegaConf.to_container(cfg)))
        configure_a100_smoke(runner)
        self.assertEqual((runner.update_budget, runner.checks, runner.probes), (100, [100], [100]))
        self.assertEqual(runner.cfg.online_exp2.branch_raw_steps, 20)
        self.assertEqual((runner.cfg.probe.rounds, runner.cfg.probe.steps), (1, 5))
        self.assertNotEqual(runner.config_hash, 'pilot-config')
        self.assertEqual(cfg.online_exp2.branch_raw_steps, 10000)
        self.assertEqual(cfg.probe.steps, 2000)

    def test_short_probe_runs_both_critics_without_full_assay(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from experiments.online_exp2 import OnlineExp2
        from experiments.exp12.replay_exp2_qualification import configure_a100_smoke
        from tests.test_revised_exp2_runner import patches, revised_cfg
        with tempfile.TemporaryDirectory() as tmp, patches():
            runner = OnlineExp2(revised_cfg(), tmp)
            runner.update_budget = 990002
            configure_a100_smoke(runner)
            trainer = runner.trainer
            trainer.start()
            for i in range(8):
                trainer.buffer.add({'observation': np.full((1, 3), i / 10, np.float32),
                    'next_observation': np.full((1, 3), (i + 1) / 10, np.float32),
                    'action': np.zeros((1, 2), np.float32), 'reward': np.array([1.], np.float32),
                    'terminated': np.array([False]), 'truncated': np.array([False])})
            runner._probe(0)
            self.assertEqual({r['architecture'] for r in runner.rows if r['type'] == 'probe'}, {'small', 'large'})
            for name in ('small', 'large'):
                with np.load(Path(tmp) / 'probes' / f'{name}_0000.npz') as data:
                    self.assertEqual(data['fresh_losses'].shape, (1, 5))
            trainer.close()
