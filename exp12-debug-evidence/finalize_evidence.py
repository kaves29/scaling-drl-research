import hashlib
import json
from pathlib import Path
import subprocess
import sys

repo = Path('/Users/shouryakaveti/VS_Projects/scaling-drl-runtime-debug')
root = Path('/Users/shouryakaveti/VS_Projects/exp12-debug-evidence')
sys.path.insert(0, str(repo / 'main'))
from scripts.run_exp12_cpu_validation import fingerprint

summary = json.loads((root / 'final-clean-cpu/summary.json').read_text())
assert summary['success'] and summary['source_unchanged'] and not summary['failed_modules']
assert fingerprint(repo / 'main') == summary['provenance']['source_hash']
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True) == ''
harness = repo / 'main/scripts/exp12_recording_isolation_qualification.py'
receipt = {'base_commit': '723da0d1c0c6f95910468a4aae9c94f1c40f60ed', 'final_commit': commit,
    'branch': 'codex/exp12-qualification-debug', 'push_verified': True,
    'cpu_source_hash': summary['provenance']['source_hash'],
    'harness_sha256': hashlib.sha256(harness.read_bytes()).hexdigest(),
    'tests_run': summary['tests_run'], 'passed': summary['tests_run'] - len(summary['skips']),
    'skipped': len(summary['skips']), 'failed_modules': [], 'gpu_jobs_submitted': 0,
    'frozen_baseline_modified': False,
    'runner_exit_codes': {'final_clean_cpu': 0, 'final_regressions': 0,
        'gnu_tools_regressions': 0, 'extended_mutations': 0, 'panel_mutation_restoration': 0,
        'single_modes_corrected': 0, 'environment_restoration': 0, 'methodology_reference': 0,
        'cpu_source_comparison': 0},
    'external_limits': ['HumanoidBench absent locally (5 test skips, 10 simulator cells unavailable)',
        'Delta CPU allocation rejected: only biqc-delta-gpu account available',
        'Updated A100 qualification outside CPU-only authorization']}
(root / 'final-validation-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
included = []
for name in ('final-clean-cpu', 'final-regressions', 'extended-qualification', 'single-modes-corrected',
             'frozen-on-cpu', 'candidate-on-cpu', 'frozen-off-cpu', 'job22900932'):
    included.extend(p for p in (root / name).rglob('*') if p.is_file())
included.extend(p for p in root.iterdir() if p.is_file() and p.suffix in ('.json', '.log', '.py', '.txt', '.md'))
manifest = []
for path in sorted(set(included)):
    manifest.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + str(path.relative_to(root)))
(root / 'final-evidence.sha256').write_text('\n'.join(manifest) + '\n')
print(json.dumps(receipt, indent=2))
print('Manifest files:', len(manifest))
