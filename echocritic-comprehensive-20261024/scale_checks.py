from pathlib import Path
import pandas as pd,numpy as np
from scipy.stats import spearmanr
R=Path(__file__).resolve().parent;D=R/'derived';H=pd.read_pickle(D/'histories.pkl');S=pd.read_csv(D/'seed_endpoints.csv');I=pd.read_csv(D/'run_inventory.csv');rows=[];corr=[]
for _,r in I[I.cohort=='exp96'].iterrows():
 d=H[(H.id==r.id)&(H.fraction>=.6)&(H.fraction<=.7)];rows.append(dict(id=r.id,environment=r.environment,architecture=r.architecture,seed=r.seed,reward=d['train/rew_mean'].median(),q=d['train/q1_mean'].median(),td=d['train/critic_loss'].median(),tdvar=d['train/td_error_var'].median(),actor_grad=d['train/actor_gnorm'].median(),critic_grad=d['train/critic_gnorm'].median() if 'train/critic_gnorm'in d else np.nan))
pd.DataFrame(rows).to_csv(D/'common70_training_scales.csv',index=False)
for env,d in S.groupby('environment'):
 for arch,sub in [('all',d)]+list(d.groupby('architecture')):
  rho,p=spearmanr(sub.common70_deficit,sub.return70);corr.append(dict(environment=env,architecture=arch,n=len(sub),spearman_rho=rho,exploratory_p=p))
pd.DataFrame(corr).to_csv(D/'same_stage_deficit_return_correlations.csv',index=False)
print(pd.DataFrame(rows).groupby(['environment','architecture'])[['reward','q','td','tdvar']].median().to_string());print(pd.DataFrame(corr).to_string(index=False))
