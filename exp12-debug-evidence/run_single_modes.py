import json
from pathlib import Path
import sys
from unittest import mock

repo = Path('/Users/shouryakaveti/VS_Projects/scaling-drl-runtime-debug')
sys.path.insert(0, str(repo / 'main'))
from tests.test_exp12_recording_qualification import RecordingQualificationPrecisionTest
from scripts import exp12_recording_isolation_qualification as qualification
from experiments.exp12.trainer import Exp12Trainer

root = Path('/Users/shouryakaveti/VS_Projects/exp12-debug-evidence/single-modes-corrected')
root.mkdir()
case = RecordingQualificationPrecisionTest()
case.setUp()
original_train = Exp12Trainer.train
calls = []
def observed(self, last_step, after_step=None, before_first_update=None):
    calls.append(after_step)
    return original_train(self, last_step, after_step=after_step,
                          before_first_update=before_first_update)
try:
    for arm, flag in (('on', '--single-on'), ('off', '--single-off')):
        out = root / arm
        with mock.patch.object(Exp12Trainer, 'train', observed):
            assert case.run_harness(out, 'highest', steps=41, extra=[flag]) == 0
        receipt = json.loads((out / 'receipt.json').read_text())
        assert receipt['verdict'] == f'SINGLE_{arm.upper()}_COMPLETE'
        assert receipt['precision'] == 'production_tf32'
        assert receipt['backend'] == 'cpu'
        assert not (out / arm / 'fingerprints.jsonl').exists()
        assert receipt[arm]['status'] == 'COMPLETE'
    assert len(calls) == 2 and all(callback.__module__ == 'experiments.exp1' and
                                 callback.__qualname__ == 'run.<locals>.after_step'
                                 for callback in calls), calls
finally:
    case.doCleanups()
(root / 'summary.json').write_text(json.dumps({'single_on': 'PASS', 'single_off': 'PASS',
     'unhooked': True, 'harness_sha256': qualification.sha(qualification.__file__)}, indent=2))
