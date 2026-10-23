"""Fail-closed diagnostic configuration and numerical engineering contracts."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.exp96_fixed_task_diagnostics import ARCHS, FROZEN, Investigation
from utils.run_metadata import _hash, _strip_locations


class OptimizerBindingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = []
        for env in ('dog-run', 'humanoid-run', 'myo-key-turn', 'myo-pen-twirl'):
            for arch, (depth, width) in ARCHS.items():
                for seed in range(1, 9):
                    cp = self.root / 'checkpoints' / env / arch / str(seed)
                    cp.mkdir(parents=True)
                    cfg = dict(seed=seed, env_name=env, agent=dict(
                        critic_learning_rate=1e-4, critic_weight_decay=.01,
                        critic_block_type='residual', critic_num_blocks=depth,
                        critic_hidden_dim=width, critic_use_cdq=False))
                    row = dict(environment=env, architecture=arch, seed=seed,
                               run_key=f'exp1_{arch}_{env}_seed{seed}',
                               checkpoint_dir=str(cp), config_hash=_hash(_strip_locations(cfg)))
                    meta = dict(code=dict(commit=FROZEN, dirty=False),
                                identity={k: row[k] for k in ('environment','architecture','seed','run_key')},
                                config_hash=row['config_hash'], resolved_config=cfg,
                                launches=[dict(commit=FROZEN, dirty=False)])
                    (cp / 'run_metadata.json').write_text(json.dumps(meta))
                    self.rows.append(row)
        (self.root / 'queue').mkdir()
        self.queue = dict(commit=FROZEN, parents=[dict(row=r) for r in self.rows])
        self.save_queue()
        self.args = SimpleNamespace(root=self.root / 'output', campaign=self.root,
                                    commit='a'*40, cap_seconds=600)

    def save_queue(self):
        (self.root / 'queue/queue.json').write_text(json.dumps(self.queue))

    def edit_metadata(self, edit):
        row = self.rows[0]
        p = Path(row['checkpoint_dir']) / 'run_metadata.json'
        meta = json.loads(p.read_text()); edit(meta)
        p.write_text(json.dumps(meta))

    def test_constructor_uses_archived_optimizer_instead_of_generic_sac(self):
        from scripts import exp96_fixed_task_diagnostics as runner
        original = runner.critic_optimizer
        with mock.patch.object(runner, 'critic_optimizer', wraps=original) as factory:
            investigation = Investigation(self.args)
        cfg = factory.call_args.args[0]
        self.assertEqual(cfg.critic_learning_rate, 1e-4)
        self.assertEqual(cfg.critic_weight_decay, .01)
        self.assertEqual(len(investigation.optimizer_provenance), 96)

    def test_missing_metadata_is_rejected_without_fallback(self):
        (Path(self.rows[0]['checkpoint_dir']) / 'run_metadata.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            Investigation(self.args)

    def test_source_identity_and_config_hash_mismatches_are_rejected(self):
        for mutate in (
            lambda m: m['code'].update(commit='b'*40),
            lambda m: m['identity'].update(seed=99),
            lambda m: m['resolved_config']['agent'].update(critic_learning_rate=.002),
            lambda m: m['launches'][0].update(dirty=True),
        ):
            p = Path(self.rows[0]['checkpoint_dir']) / 'run_metadata.json'
            before = p.read_text()
            self.edit_metadata(mutate)
            with self.assertRaises(ValueError): Investigation(self.args)
            p.write_text(before)

    def test_duplicate_queue_identity_is_rejected(self):
        self.queue['parents'].append(copy.deepcopy(self.queue['parents'][0]))
        self.save_queue()
        with self.assertRaises(ValueError): Investigation(self.args)

    def test_valid_but_conflicting_parent_optimizers_are_rejected(self):
        def change(m):
            m['resolved_config']['agent']['critic_learning_rate'] = .0003
            m['config_hash'] = _hash(_strip_locations(m['resolved_config']))
            self.rows[0]['config_hash'] = m['config_hash']
        self.edit_metadata(change); self.save_queue()
        with self.assertRaises(ValueError): Investigation(self.args)


if __name__ == '__main__':
    unittest.main()
