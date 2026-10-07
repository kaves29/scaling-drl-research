import json
from pathlib import Path
import numpy as np
from experiments.exp12.envs import _make_one_env
rows=[]
for task in ['myo-key-turn','myo-pen-twirl','myo-pose-hard','myo-reach','myo-baoding-p1']:
 a=_make_one_env('myosuite',task,0,True,True,2,1.,100)
 b=_make_one_env('myosuite',task,99,True,True,2,1.,100)
 a.reset()
 for _ in range(5):a.step(a.action_space.sample())
 state=a.restore_state();actions=[a.action_space.sample() for _ in range(5)]
 def rollout(env):
  env.restore(state);result=[]
  for action in actions:
   o,r,t,u,_=env.step(action);result.append((np.asarray(o),float(r),bool(t),bool(u)))
   if t or u:break
  return result
 one=rollout(a);two=rollout(b);three=rollout(b)
 exact=lambda x,y:len(x)==len(y) and all(np.array_equal(i[0],j[0]) and i[1:]==j[1:] for i,j in zip(x,y))
 rows.append({'task':task,'first_vs_fresh_restore_exact':exact(one,two),'two_fresh_restores_exact':exact(two,three),'replay_steps':len(one)})
 a.close();b.close()
Path('/tmp/exp12-followup-evidence/exp12_restore.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
