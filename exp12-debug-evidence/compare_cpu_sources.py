import json
from pathlib import Path
import sys

repo = Path('/Users/shouryakaveti/VS_Projects/scaling-drl-runtime-debug')
sys.path.insert(0, str(repo / 'main'))
from experiments.exp12.state import latest_state_dir, state_differences
from scripts.exp12_recording_isolation_qualification import checkpoint_differences, agent_deviations, sha

root = Path('/Users/shouryakaveti/VS_Projects/exp12-debug-evidence')
frozen = latest_state_dir(root / 'frozen-on-cpu/on/state')
candidate = latest_state_dir(root / 'candidate-on-cpu/on/state')
true_off = latest_state_dir(root / 'frozen-off-cpu/off/state')
assert all(p is not None for p in (frozen, candidate, true_off))
summary = {'scope': 'CPU only, 400 interactions; no CUDA qualification',
           'candidate_on_vs_frozen_on': state_differences(candidate, frozen, ignore_meta=('wandb_run_id',)),
           'candidate_on_vs_true_off': checkpoint_differences(candidate, true_off),
           'candidate_vs_true_off_agent_deviations': agent_deviations(candidate, true_off),
           'receipt_sha256': {label: sha(root / label / 'receipt.json')
                              for label in ('frozen-on-cpu', 'candidate-on-cpu', 'frozen-off-cpu')}}
(root / 'cpu-source-comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
assert not summary['candidate_on_vs_frozen_on']
assert not summary['candidate_on_vs_true_off']
