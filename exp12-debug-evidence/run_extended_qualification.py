import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest import mock

repo = Path('/Users/shouryakaveti/VS_Projects/scaling-drl-runtime-debug')
sys.path.insert(0, str(repo / 'main'))
from tests.test_exp12_recording_qualification import RecordingQualificationPrecisionTest
from scripts import exp12_recording_isolation_qualification as qualification
from experiments.exp12 import state
import jax

root = Path('/Users/shouryakaveti/VS_Projects/exp12-debug-evidence/extended-qualification')
root.mkdir(exist_ok=True)
case = RecordingQualificationPrecisionTest()
case.setUp()
summary = {'cpu_only': True, 'backend': jax.default_backend(),
           'head': qualification.git_head(repo),
           'diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=repo)).hexdigest(),
           'harness_sha256': qualification.sha(qualification.__file__), 'runs': {}, 'mutations': {}}
try:
    for precision in ('highest', 'production_tf32'):
        out = root / precision
        if not (out / 'receipt.json').exists():
            assert case.run_harness(out, precision, steps=400, probes=True) == 0
        receipt = json.loads((out / 'receipt.json').read_text())
        assert receipt['backend'] == 'cpu'
        assert all(receipt[arm]['updates'] == 782 for arm in receipt['arms'])
        summary['runs'][precision] = {'verdict': receipt['verdict'], 'steps': receipt['steps'],
                                    'receipt_sha256': qualification.sha(out / 'receipt.json')}
    # Reintroduce the failed-job assertion only in memory, leaving source untouched.
    source = Path(qualification.__file__).read_text()
    original_main = qualification.main
    broken = source.replace('expected_precision = "highest" if precision == "highest" else "tensorfloat32"',
                            'expected_precision = "highest"')
    assert broken != source
    namespace = dict(qualification.__dict__)
    exec(compile(broken, qualification.__file__, 'exec'), namespace)
    broken_main = types.FunctionType(namespace['main'].__code__, qualification.__dict__)
    with mock.patch.object(qualification, 'main', broken_main):
        try:
            case.run_harness(root / 'original-precision-defect', 'production_tf32', steps=12)
        except ValueError as error:
            assert str(error) == 'precision changed during the training run'
            summary['mutations']['original_precision_assertion'] = 'EXPECTED_FAILURE'
        else:
            raise AssertionError('Original defect did not fail')
    # Reintroduce the old full-meter comparison against real saved partial windows.
    with mock.patch.object(qualification, 'checkpoint_differences', side_effect=lambda a, b:
                           state.state_differences(a, b, ignore_meta=qualification.EXCLUDED_META)):
        out = root / 'original-meter-defect'
        assert case.run_harness(out, 'production_tf32', steps=41) == 3
        receipt = json.loads((out / 'receipt.json').read_text())
        assert receipt['verdict'] == 'BLOCKED_DIAGNOSTICS_ASSOCIATED_DIVERGENCE'
        assert 'meta:meters' in receipt['pairs']['on_off']['complete_checkpoint_differences']['ancillary']
        summary['mutations']['original_meter_comparator'] = 'EXPECTED_FAILURE'
    assert case.run_harness(root / 'restored', 'production_tf32', steps=41) == 0
    summary['restored'] = 'PASS'
finally:
    case.doCleanups()
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
