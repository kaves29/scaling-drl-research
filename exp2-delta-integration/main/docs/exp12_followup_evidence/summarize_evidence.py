import json,csv,hashlib,platform,sys
from pathlib import Path
from importlib.metadata import version
import numpy as np
root=Path('/tmp/exp12-followup-evidence')
b=json.loads((root/'baoding_current_1/baoding.json').read_text())
arrays=np.load(root/'baoding_current_1/baoding_arrays.npz')
for row in b['differences']:
 row['relative_to_second_numpy_desired']=row['abs_diff']/abs(row['observed_second']) if row['observed_second'] else float('inf')
 row['expected_first_hex']=row['expected_first'].hex();row['observed_second_hex']=row['observed_second'].hex()
for row in b['step_summaries']:
 s=row['step'];x=arrays['first_obs'][s].astype(float);y=arrays['second_obs'][s].astype(float);diff=np.abs(x-y)
 rel=np.divide(diff,np.abs(y),out=np.zeros_like(diff),where=y!=0)
 for key,vals in [('max_abs_element',diff),('max_relative_to_second_element',rel)]:
  j=int(np.argmax(vals));row[key]={'index':j,'first':float(x[j]),'second':float(y[j]),'abs_diff':float(diff[j]),'relative_to_first':float(diff[j]/abs(x[j])),'relative_to_second':float(rel[j])}
 row['reward_relative_to_first']=row['reward_abs_diff']/abs(row['reward_first'])
 row['reward_relative_to_second']=row['reward_abs_diff']/abs(row['reward_second'])
(root/'baoding_exact_summary.json').write_text(json.dumps(b,indent=2))
for name,records in [('baoding_differences.csv',b['differences']),('twin_differences.csv',json.loads((root/'twin_current_1/twin.json').read_text())['differences'])]:
 with (root/name).open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
t=np.load(root/'twin_current_1/twin_arrays.npz')
try:np.testing.assert_allclose(t['g_panel'],t['g_eager'],rtol=1e-5,atol=1e-7)
except AssertionError as e:(root/'twin_later_gradient_assertion.log').write_text(str(e)+'\n')
comparisons={}
for family,others,filename in [('baoding',['baoding_current_2','baoding_6ffccba','baoding_b554d45'],'baoding_arrays.npz'),('twin',['twin_current_2','twin_c293b9a'],'twin_arrays.npz')]:
 base=np.load(root/(family+'_current_1')/filename)
 for other in others:
  dat=np.load(root/other/filename);comparisons[other]={k:np.array_equal(base[k],dat[k]) for k in base.files}
metadata={'head':'438a5b049d897019e09407b2a40cf1da565e0d88','python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'packages':{k:version(k) for k in ['jax','jaxlib','flax','optax','orbax-checkpoint','numpy','pandas','mujoco','dm-control','myosuite','rliable']},'exact_array_reproducibility':comparisons}
(root/'metadata.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(b['step_summaries'],indent=2))
print(json.dumps(metadata,indent=2))
