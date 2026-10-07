import os, subprocess, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/Users/shouryakaveti/VS_Projects/sparse-ppo-drl-research')
OUT=Path('/tmp/exp12-followup-evidence')
PY=str(ROOT/'.venv/bin/python')
twin='tests.test_exp12_twin_critic.TwinCheck1Test.test_panel_values'
baoding='tests.test_angle2a_env_state_determinism_smoke.TestMyosuiteDeterminism.test_myo_baoding_p1'
jobs=[('twin_original_current_1',ROOT/'main',twin),('twin_original_current_2',ROOT/'main',twin),('baoding_original_current_1',ROOT/'main',baoding),('baoding_original_current_2',ROOT/'main',baoding),('baoding_original_6ffccba',Path('/tmp/exp12-history/6ffccba/main'),baoding),('baoding_original_b554d45',Path('/tmp/exp12-history/b554d45/main'),baoding)]
def run(job):
 name,cwd,test=job
 env={**os.environ,'JAX_PLATFORMS':'cpu','MUJOCO_GL':'disable','EXP12_JAX_CACHE_DIR':'off','PYTHONPATH':f'{cwd}:/tmp/exp12-validation-deps','MPLCONFIGDIR':'/tmp/exp12-mpl'}
 with open(OUT/(name+'.log'),'w') as f:
  p=subprocess.run([PY,'-m','unittest','-v',test],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 result={'name':name,'cwd':str(cwd),'test':test,'exit_code':p.returncode}
 print(json.dumps(result),flush=True)
 return result
with ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(run,jobs))
(OUT/'original_test_results.json').write_text(json.dumps(results,indent=2))
