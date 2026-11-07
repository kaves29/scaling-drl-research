import sys,json,hashlib,copy
from pathlib import Path
import numpy as np
from experiments.angle_2a.env_state import capture_env_state,restore_env_state
from scale_rl.envs import create_vec_env
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
v=create_vec_env(env_type='myosuite',env_name='myo-baoding-p1',num_envs=1,seed=0)
e=v.envs[0];e.reset(seed=0)
for _ in range(5):e.step(e.action_space.sample())
base=e.unwrapped
captured=capture_env_state(e,'myosuite');counter=base.counter
actions=[e.action_space.sample() for _ in range(5)]
fields=[];offset=0
for k in base.obs_keys:
 size=np.asarray(base.obs_dict[k]).size;fields.append((k,offset,offset+size));offset+=size

def rollout(reset_counter=False):
 before=base.counter;restore_env_state(e,captured)
 if reset_counter:base.counter=counter
 restored=base.counter
 rows=[]
 for a in actions:
  o,r,t,u,_=e.step(a);rows.append((np.asarray(o).copy(),float(r),bool(t),bool(u),base.counter))
  if t or u:break
 return rows,{'before_restore':before,'after_restore':restored,'after_rollout':base.counter}
first,c1=rollout();second,c2=rollout();fixed1,f1=rollout(True);fixed2,f2=rollout(True)
rows=[];summary=[]
for step,(a,b) in enumerate(zip(first,second)):
 x,y=a[0],b[0];d=np.abs(x.astype(float)-y.astype(float));rel=np.divide(d,np.abs(x),out=np.full_like(d,np.inf),where=x!=0);rel[(x==0)&(y==0)]=0
 for j in np.flatnonzero(x!=y):
  field=next(k for k,start,end in fields if start<=j<end)
  rows.append({'step':step,'index':int(j),'field':field,'expected_first':float(x[j]),'observed_second':float(y[j]),'abs_diff':float(d[j]),'rel_to_first':float(rel[j])})
 summary.append({'step':step,'different':int(np.sum(x!=y)),'max_abs':float(d.max()),'max_rel_to_first':float(rel.max()),'reward_first':a[1],'reward_second':b[1],'reward_abs_diff':abs(a[1]-b[1]),'flags_first':a[2:4],'flags_second':b[2:4]})
fixed_exact=all(np.array_equal(a[0],b[0]) and a[1:4]==b[1:4] for a,b in zip(fixed1,fixed2))
fixed_matches_first=all(np.array_equal(a[0],b[0]) and a[1:4]==b[1:4] for a,b in zip(first,fixed1))
payload={'source_module':sys.modules['experiments.angle_2a.env_state'].__file__,'capture_keys':list(captured),'captured_counter_not_in_state':counter,'first_counters':c1,'second_counters':c2,'diagnostic_counter_reset_counters':[f1,f2],'diagnostic_counter_reset_exact':fixed_exact,'diagnostic_reset_matches_original_first':fixed_matches_first,'obs_fields':fields,'step_summaries':summary,'differences':rows}
(out/'baoding.json').write_text(json.dumps(payload,indent=2));np.savez(out/'baoding_arrays.npz',first_obs=np.array([a[0] for a in first]),second_obs=np.array([a[0] for a in second]),first_rewards=[a[1] for a in first],second_rewards=[a[1] for a in second],actions=actions)
print(json.dumps({k:val for k,val in payload.items() if k not in ('differences','obs_fields')},indent=2));v.close()
