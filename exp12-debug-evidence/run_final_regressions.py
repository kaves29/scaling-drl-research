import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

repo = Path('/Users/shouryakaveti/VS_Projects/scaling-drl-runtime-debug')
sys.path.insert(0, str(repo / 'main'))
from scripts import run_exp12_cpu_validation as runner

out = Path('/Users/shouryakaveti/VS_Projects/exp12-debug-evidence/final-regressions')
out.mkdir()
modules = ['tests.test_exp12_dynamic_scheduler', 'tests.test_exp12_min_qualification',
           'tests.test_exp12_review_blockers', 'tests.test_exp12_review_fixes',
           'tests.test_exp12_twin_critic', 'tests.test_exp12_recording_qualification']
source_hash = runner.fingerprint(repo / 'main')
provenance = {'source_hash': source_hash, 'commit': subprocess.check_output(
              ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
              'diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=repo)).hexdigest(),
              'python': sys.executable, 'environment': dict(os.environ), 'modules': modules,
              'file_sha256': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [repo / 'main/scripts/exp12_recording_isolation_qualification.py',
                                        *[repo / ('main/' + m.replace('.', '/') + '.py') for m in modules]]}}
# Record only reproducibility-related environment variables, never credentials.
provenance['environment'] = {k: os.environ.get(k) for k in ('JAX_PLATFORMS', 'JAX_PLATFORM_NAME',
    'MUJOCO_GL', 'EXP12_JAX_CACHE_DIR', 'WANDB_MODE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
runner.save_json(out / 'provenance.json', provenance)
results = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    pending = {pool.submit(runner.run_module, module, out, source_hash, dict(os.environ)): module
               for module in modules}
    for task in concurrent.futures.as_completed(pending):
        module = pending[task]
        results[module] = task.result()
        runner.save_json(out / 'modules.json', results)
        print(json.dumps({'module': module, 'success': results[module]['success']}), flush=True)
runner.save_json(out / 'summary.json', {'success': all(r['success'] for r in results.values()),
    'tests_run': sum(r['tests_run'] for r in results.values()), 'modules': results})
raise SystemExit(0 if all(r['success'] for r in results.values()) else 1)
