"""Exploratory cross-validation ablations and event-definition sensitivity."""
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;D=R/'derived';L=pd.read_csv(D/'lagged_observations.csv');rng=np.random.default_rng(1428);out=[]
for env,d0 in L.groupby('environment'):
 for scope in ['all_seeds','exclude_pilot_seed1']:
  d=d0[d0.seed!=1].dropna() if scope!='all_seeds' else d0.dropna();Y=d.future_change.to_numpy();errors={}
  for model,extras in [('baseline',[]),('deficit_only',['deficit']),('q_only',['log_abs_q']),('td_only',['log_td_loss']),('combined',['deficit','log_abs_q','log_td_loss'])]:
   X=np.column_stack([d[['fraction','current_return','previous_change']+extras].to_numpy(),pd.get_dummies(d.architecture,dtype=float)]);pred=np.zeros(len(d))
   for seed in d.seed.unique():
    test=d.seed.to_numpy()==seed;tr=~test;mu=X[tr].mean(0);sd=X[tr].std(0);sd[sd==0]=1;Z=np.column_stack([np.ones(len(d)),(X-mu)/sd]);pen=np.eye(Z.shape[1]);pen[0,0]=0;beta=np.linalg.solve(Z[tr].T@Z[tr]+pen,Z[tr].T@Y[tr]);pred[test]=Z[test]@beta
   err=(pred-Y)**2;errors[model]=err
   if model=='baseline':baseline=err
   ss=pd.DataFrame({'seed':d.seed.to_numpy(),'gain':baseline-err}).groupby('seed').gain.mean().values;draw=ss[rng.integers(0,len(ss),(10000,len(ss)))];lo,hi=np.quantile(draw.mean(1),[.025,.975])
   out.append(dict(environment=env,scope=scope,model=model,n_seeds=len(ss),mse=err.mean(),relative_mse_reduction=1-err.mean()/baseline.mean(),seed_mean_gain=ss.mean(),gain_ci_low=lo,gain_ci_high=hi))
pd.DataFrame(out).to_csv(D/'predictive_ablation.csv',index=False)
H=pd.read_pickle(D/'histories.pkl');H=H[H.cohort=='exp96'];I=pd.read_csv(D/'run_inventory.csv');events=[]
for rid,d in H.groupby('id'):
 d=d.sort_values('fraction');meta=I[I.id==rid].iloc[0]
 for metric in ['train/critic_loss','train/q1_mean','train/policy_kl']:
  q=d.dropna(subset=[metric]);t=q.fraction.to_numpy();v=np.abs(q[metric].to_numpy());base=pd.Series(v).rolling(5,min_periods=5).median().shift(1).to_numpy()
  for ratio in [2,3,5]:
   hit=(v>ratio*base)&(base>0)&(t>=.2)
   for persistence in [1,3]:
    on=next((k for k in range(len(v)-persistence+1) if hit[k:k+persistence].all()),None)
    events.append(dict(id=rid,environment=meta.environment,architecture=meta.architecture,seed=meta.seed,metric=metric,ratio=ratio,persistence=persistence,onset_fraction=t[on] if on is not None else np.nan))
pd.DataFrame(events).to_csv(D/'critic_event_sensitivity.csv',index=False)
print('ABLATIONS_COMPLETE')
