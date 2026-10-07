import json,tempfile,shutil,sys
from pathlib import Path
import numpy as np
import pandas as pd
from analysis import exp1_analysis as e1,exp2_analysis as e2
from experiments.exp12 import ledger,exp2_ledger
from tests.test_exp12_exp2_analysis import _write_run,PLAN
out=Path('/tmp/exp12-followup-evidence');rows=[]
envs=['dog-run','dog-trot','humanoid-run','humanoid-walk','humanoid-stand','swimmer-swimmer15','hopper-hop','myo-key-turn','myo-pen-twirl','myo-pose-hard','myo-reach','h1-run-v0','h1-reach-v0']
runs=pd.DataFrame([{'architecture':a,'environment':e,'seed':s,'final_loss_iqm':float(s)/10+i/100} for i,a in enumerate(['D2W512','D4W1024','D4W1536']) for e in envs for s in range(1,6)])
variants={'complete_195':runs,'missing_seed5_all':runs[runs.seed!=5],'missing_seed5_scaled_only':runs[~((runs.seed==5)&(runs.architecture=='D4W1536'))],'missing_h1reach_all':runs[runs.environment!='h1-reach-v0'],'missing_one_cell':runs.drop(0),'duplicate_identity':pd.concat([runs,runs.iloc[[0]].assign(final_loss_iqm=999)]),'extra_seed6':pd.concat([runs,runs[runs.seed==5].assign(seed=6)]),'extra_environment':pd.concat([runs,runs[runs.environment=='dog-run'].assign(environment='unexpected-task')]),'infinite_endpoint':runs.copy()}
variants['infinite_endpoint'].loc[0,'final_loss_iqm']=np.inf
for name,data in variants.items():
 try:
  x=e1.primary_endpoint(data,reps=20)
  r={'experiment':1,'case':name,'accepted':True,'primary':x.to_dict('records'),'shapes':{a:list(e1.score_matrix(data,a).shape) for a in ['D2W512','D4W1024','D4W1536']}}
 except Exception as exc:r={'experiment':1,'case':name,'accepted':False,'error':str(exc)}
 rows.append(r)
root=Path(tempfile.mkdtemp(prefix='exp12-acceptance-base-'));key=_write_run(root,'D4W1536','dog-run',1)
d=Path(exp2_ledger.run_root(key,root));parent=ledger.ledger_root(root)/key
plan={**PLAN,'num_interaction_steps':400,'horizon_steps':100,'eval_every_steps':4,'eval_episodes':10}
(d/'fork.json').write_text(json.dumps(plan))
for arm in ['control','injected']:
 frame=pd.DataFrame([{'arm':arm,'eval_index':k,'steps_since_fork':4*k,'interaction_step':100+4*k,'episode':ep,'return':100.+(20 if arm=='injected' else 0)+ep,'length':10,'success':0} for k in range(26) for ep in range(10)])
 frame.to_csv(d/f'arm_{arm}/eval_episodes.csv',index=False)
def change_csv(path,fn):fn(pd.read_csv(path)).to_csv(path,index=False)
cases=['complete_26x10','one_episode_per_eval','wrong_eval_indices','misaligned_arm_steps','duplicate_episode_rows','nan_one_episode','inf_episode','missing_evaluation','missing_arm','missing_check1','failed_check1','missing_check2','failed_check2','running_parent','missing_fork_json','missing_entire_fork','self_declared_shorter_plan','D2_fork']
for name in cases:
 work=Path(tempfile.mkdtemp(prefix='exp12-acceptance-'));shutil.copytree(root,work,dirs_exist_ok=True)
 fork=Path(exp2_ledger.run_root(key,work));p=ledger.ledger_root(work)/key
 paths=[fork/f'arm_{a}/eval_episodes.csv' for a in ['control','injected']]
 if name=='one_episode_per_eval':
  for path in paths:change_csv(path,lambda f:f[f.episode==0])
 if name=='wrong_eval_indices':
  for path in paths:change_csv(path,lambda f:f.assign(eval_index=f.eval_index+100))
 if name=='misaligned_arm_steps':change_csv(paths[1],lambda f:f.assign(steps_since_fork=f.steps_since_fork+1,interaction_step=f.interaction_step+1))
 if name=='duplicate_episode_rows':change_csv(paths[1],lambda f:pd.concat([f,f.iloc[[0]].assign(**{'return':999.})]))
 if name in ['nan_one_episode','inf_episode']:
  def bad(f):
   f.loc[0,'return']=np.nan if name=='nan_one_episode' else np.inf
   return f
  change_csv(paths[1],bad)
 if name=='missing_evaluation':change_csv(paths[1],lambda f:f[f.eval_index!=25])
 if name=='missing_arm':shutil.rmtree(fork/'arm_injected')
 if name in ['missing_check1','missing_check2']:(fork/('check1_injected.json' if name=='missing_check1' else 'check2.json')).unlink()
 if name in ['failed_check1','failed_check2']:
  path=fork/('check1_injected.json' if name=='failed_check1' else 'check2.json');obj=json.loads(path.read_text());obj['pass']=False;path.write_text(json.dumps(obj))
 if name=='running_parent':change_csv(p/'run.csv',lambda f:f.assign(status='running'))
 if name=='missing_fork_json':(fork/'fork.json').unlink()
 if name=='missing_entire_fork':shutil.rmtree(fork)
 if name=='self_declared_shorter_plan':
  (fork/'fork.json').write_text(json.dumps(PLAN))
  for path in paths:change_csv(path,lambda f:f[f.eval_index<3])
 if name=='D2_fork':change_csv(p/'run.csv',lambda f:f.assign(architecture='D2W512'))
 try:
  loaded=e2.load(work);f=loaded['forks']
  paired=e2.paired_returns(loaded['evals'],f) if not f.empty else pd.DataFrame()
  r={'experiment':2,'case':name,'load_error':False,'forks':f.to_dict('records'),'paired_rows':len(paired),'nonfinite_differences':int((~np.isfinite(paired.difference)).sum()) if len(paired) else 0,'first_paired':paired.head(1).to_dict('records')}
 except Exception as exc:r={'experiment':2,'case':name,'load_error':True,'error':str(exc)}
 rows.append(r);shutil.rmtree(work)
shutil.rmtree(root)
(out/'analysis_acceptance.json').write_text(json.dumps(rows,indent=2,default=str))
for row in rows:
 print(json.dumps({k:v for k,v in row.items() if k not in ['primary','first_paired','forks']},default=str))
 if row.get('forks'):print('complete=',row['forks'][0]['complete'],'check1=',row['forks'][0]['check1_pass'])
