"""The Modal worker re-imports the launcher outside its source checkout."""
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
                runpy.run_path(str(staged))
