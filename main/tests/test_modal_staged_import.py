"""The Modal worker re-imports the launcher outside its source checkout."""
import hashlib
import inspect
import json
import os
import runpy
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class StagedImportTest(unittest.TestCase):
    def test_remote_import_never_runs_client_source_setup(self):
        launcher = Path(__file__).parents[1] / 'infra/modal/exp2_modal.py'
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            staged = Path(tmp) / 'exp2_modal.py'
            shutil.copyfile(launcher, staged)
            modal = types.ModuleType('modal')
            modal.is_local = lambda: False
            modal.Image = MagicMock()
            modal.Volume = MagicMock()
            modal.Secret = MagicMock()
            app = MagicMock()
            app.function.side_effect = lambda **kwargs: lambda f: f
            app.local_entrypoint.side_effect = lambda **kwargs: lambda f: f
            modal.App = lambda *args, **kwargs: app
            with patch.dict(sys.modules, {'modal': modal}), patch('subprocess.check_output', side_effect=AssertionError('client git called')):
                module = runpy.run_path(str(staged))
            with patch.dict(os.environ, {'PYOPENGL_PLATFORM': 'glfw'}):
                env = module['worker_environment']()
            self.assertEqual(env['MUJOCO_GL'], 'egl')
            self.assertEqual(env['PYOPENGL_PLATFORM'], 'egl')
            self.assertIn('budget_usd', inspect.signature(module['main']).parameters)
            remote = MagicMock()
            with patch.dict(module['main'].__globals__, {'qualify': remote}):
                module['main']('qualify', 15.0)
            remote.remote.assert_called_once_with(15.0, 5.0)
            training = MagicMock()
            with patch.dict(module['main'].__globals__, {'train': training}):
                module['main']('train', 15.0)
            training.remote.assert_called_once_with(budget_usd=15.0)
            receipt = {'source_commit': 'abc', 'manifest_sha256': 'digest',
                       'gpu_model': 'NVIDIA L4', 'parent_restore': True,
                       'full_branch_horizon': True, 'plasticity_probe': True,
                       'restart': True, 'qualified_workers': 1,
                       'checkpoint': 5000, 'qualification_updates': 5000,
                       'qualification_protocol': 'replay-qualification-5000-v1',
                       'identity': {'environment': 'dog-run', 'seed': 1, 'reference_regime': 'reference_small'},
                       'branch_horizon': 10000,
                       'branch_counters': {f'{a}_{m}': {'raw_env_steps': 10000}
                                           for a in ('small', 'large') for m in ('frozen', 'learning')},
                       'probe_architectures': ['small', 'large'],
                       'interruption': {'checkpoint': 5000, 'raw_env_steps': 5000}}
            self.assertEqual(module['validate_qualification'](receipt, 'abc', 'digest', receipt['identity'], 10000), 1)
            with self.assertRaises(ValueError):
                module['validate_qualification']({**receipt, 'checkpoint': 66001}, 'abc', 'digest', receipt['identity'], 10000)
            with self.assertRaises(ValueError):
                module['validate_qualification']({**receipt, 'qualified_workers': True}, 'abc', 'digest', receipt['identity'], 10000)
            with self.assertRaises(ValueError):
                module['validate_qualification']({**receipt, 'branch_counters': {}}, 'abc', 'digest', receipt['identity'], 10000)
            seconds = module['reservation_seconds'](29.0, 40.0, 5.0)
            self.assertGreaterEqual(seconds, 300)
            self.assertLessEqual(seconds * module['PER_SECOND'], 5.0)
            with self.assertRaises(ValueError):
                module['reservation_seconds'](39.99, 40.0, 5.0)
            root = Path(tmp)
            manifest = root / 'parents/exp1-pilot-parents/manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text('manifest')
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            good = {**receipt, 'manifest_sha256': digest}
            persisted = root / 'results/replay-pilot/l4-qualification-abc.json'
            persisted.parent.mkdir(parents=True)
            def fake_path(path):
                return root / str(path).lstrip('/')
            def qualify_ok(*args):
                persisted.write_text(json.dumps(good))
                return good
            qualify = MagicMock()
            qualify.local.side_effect = qualify_ok
            train = MagicMock()
            with patch.dict(module['qualify_then_train'].__globals__,
                            {'Path': fake_path, 'qualify': qualify, 'train': train,
                             'qualification_context': lambda manifest: (good['identity'], 10000)}), \
                 patch('subprocess.check_output', return_value='abc'):
                module['qualify_then_train'](40.0, 5.0)
            train.local.assert_called_once()
            self.assertEqual(train.local.call_args.kwargs['max_workers'], 1)
            self.assertEqual(train.local.call_args.kwargs['budget_usd'], 40.0)
            self.assertLessEqual(train.local.call_args.kwargs['max_seconds'], 22 * 3600)
            qualify.reset_mock()
            train.reset_mock()
            with patch.dict(module['qualify_then_train'].__globals__,
                            {'Path': fake_path, 'qualify': qualify, 'train': train,
                             'qualification_context': lambda manifest: (good['identity'], 10000)}), \
                 patch('subprocess.check_output', return_value='abc'):
                module['qualify_then_train'](40.0, 5.0)
            qualify.local.assert_not_called()
            train.local.assert_called_once()
            persisted.unlink()
            qualify.local.side_effect = lambda *args: {**good, 'restart': False}
            train.reset_mock()
            with patch.dict(module['qualify_then_train'].__globals__,
                            {'Path': fake_path, 'qualify': qualify, 'train': train,
                             'qualification_context': lambda manifest: (good['identity'], 10000)}), \
                 patch('subprocess.check_output', return_value='abc'):
                with self.assertRaises(ValueError):
                    module['qualify_then_train'](40.0, 5.0)
            train.local.assert_not_called()
