"""The Modal worker re-imports the launcher outside its source checkout."""
import inspect
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
            remote.remote.assert_called_once_with(15.0)
