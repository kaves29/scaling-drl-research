import contextlib
import importlib.util
from pathlib import Path
import sys
from unittest import mock

import jax

source, out, mode = map(str, sys.argv[1:])
sys.path.insert(0, source + '/main')
harness = Path('/Users/shouryakaveti/VS_Projects/scaling-drl-runtime-debug/main/scripts/exp12_recording_isolation_qualification.py')
spec = importlib.util.spec_from_file_location('qualification', harness)
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)
from experiments import exp1

args = ['qualification', '--out', out, '--source-commit', qualification.git_head(source),
        '--harness-commit', qualification.git_head(harness.parent), mode]
with (mock.patch.object(sys, 'argv', args),
      mock.patch.object(qualification, 'require_a100_backend'),
      mock.patch.object(exp1, 'set_matmul_precision', side_effect=lambda:
                        jax.config.update('jax_default_matmul_precision', 'tensorfloat32')),
      contextlib.chdir(Path(source) / 'main')):
    raise SystemExit(qualification.main())
