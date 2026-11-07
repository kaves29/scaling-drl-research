"""Targeted robustness, event and archived-curve analyses; no fitting performed."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;D=ROOT/'derived';F=ROOT/'figures';rng=np.random.default_rng(6141)
I=pd.read_csv(D/'run_inventory.csv');S=pd.read_csv(D/'seed_endpoints.csv');H=pd.read_pickle(D/'histories.pkl');P=pd.read_csv(D/'probe_trajectories.csv');E=pd.read_csv(D/'paired_effects.csv');EV=pd.read_csv(D/'collapse_events.csv');L=pd.read_csv(D/'lagged_observations.csv');C=I[I.cohort=='exp96'];HC=H[H.cohort=='exp96'];ARCH=['D2W512','D4W1024','D4W1536']
def save(n,rows):pd.DataFrame(rows).to_csv(D/(n+'.csv'),index=False)
def ci(vals):
 x=np.asarray(vals,float);x=x[np.isfinite(x)];draw=x[rng.integers(0,len(x),(10000,len(x)))];return float(x.mean()),*np.quantile(draw.mean(1),[.025,.975])
# Exact cross-cohort observation identity; records are not independent confirmation.
identity=[]
for _,r in I[I.cohort=='pilot'].iterrows():
 c=C[(C.environment==r.environment)&(C.architecture==r.architecture)&(C.seed==r.seed)].iloc[0];a=H[H.id==r.id].set_index('raw_step');b=H[H.id==c.id].set_index('raw_step');fields=[x for x in H if x.startswith(('train/','eval/'))];j=a[fields].join(b[fields],lsuffix='_p',rsuffix='_c');n=same=0
 for f in fields:
  mask=j[f+'_p'].notna()&j[f+'_c'].notna();n+=mask.sum();same+=(j.loc[mask,f+'_p']==j.loc[mask,f+'_c']).sum()
 identity.append(dict(pilot=r.id,campaign=c.id,environment=r.environment,architecture=r.architecture,shared_values=n,exact_matches=same))
save('pilot_campaign_trajectory_identity',identity)
# Probe sign prevalence, onset and reversibility under operational source trigger.
onset=[]
for _,r in C.iterrows():
 p=P[(P.id==r.id)&(P.check>0)].sort_values('check');firing=p['probe/ci_low'].to_numpy()>0;k=p.check.to_numpy();two=[k[i] for i in range(1,len(k)) if firing[i] and firing[i-1] and k[i]==k[i-1]+1]
 onset.append(dict(id=r.id,environment=r.environment,architecture=r.architecture,seed=r.seed,observed_checks=len(p),first_two_positive_check=two[0] if two else np.nan,positive_fraction=np.mean(p['probe/loss_iqm']>0),firing_fraction=firing.mean(),final_sign=np.sign(p['probe/loss_iqm'].iloc[-1]),max_deficit=p['probe/loss_iqm'].max(),min_deficit=p['probe/loss_iqm'].min(),fresh_initial=P[(P.id==r.id)&(P.check==0)]['probe/score_fresh_iqm'].iloc[0],fresh70=p.loc[p.check==35,'probe/score_fresh_iqm'].iloc[0],fresh_last=p['probe/score_fresh_iqm'].iloc[-1]))
O=pd.DataFrame(onset);save('adaptation_onset',O)
# More conservative cluster CIs across shared seed labels for within-run associations.
QA=[]
for env in sorted(C.environment.unique()):
 q=pd.read_csv(D/('q_probe_aligned_'+env+'.csv'))
 for target in ['deficit','fresh']:
  vals=[]
  for rid,d in q.groupby('id'):
   X=np.column_stack([np.ones(len(d)),d.t,d.t**2]);x=d.q-X@np.linalg.lstsq(X,d.q,rcond=None)[0];y=d[target]-X@np.linalg.lstsq(X,d[target],rcond=None)[0]
   if len(d)>5 and np.std(x)>0 and np.std(y)>0:vals.append({'seed':int(d.seed.iloc[0]),'r':float(pearsonr(x,y).statistic)})
  byseed=pd.DataFrame(vals).groupby('seed').r.mean();m,lo,hi=ci(byseed);QA.append(dict(environment=env,target=target,n_runs=len(vals),n_seed_groups=len(byseed),mean_within_run_time_adjusted_r=m,ci_low=lo,ci_high=hi))
save('q_probe_associations_clustered',QA)
# Sensitivity to any one seed and to pilot-overlapping seed1.
robust=[]
for env,ss in S.groupby('environment'):
 for metric in ['return70','auc70','common70_deficit','final_return']:
  for a,b in [('D2W512','D4W1024'),('D2W512','D4W1536')]:
   x=ss[ss.architecture==a].set_index('seed')[metric];y=ss[ss.architecture==b].set_index('seed')[metric];d=(y-x).dropna();loo=[d.drop(seed).mean() for seed in d.index]
   robust.append(dict(environment=env,metric=metric,left=a,right=b,n=len(d),difference=d.mean(),exclude_seed1=d.drop(1,errors='ignore').mean(),leave_one_out_min=min(loo),leave_one_out_max=max(loo)))
save('leave_one_seed_sensitivity',robust)
# Predictive improvement uncertainty clustered by held-out seed; no fitting new RL models.
cv=[]
for env,d in L.groupby('environment'):
 d=d.dropna().copy();Y=d.future_change.to_numpy();preds={};base=['fraction','current_return','previous_change']
 for model,features in [('baseline',base),('critic_augmented',base+['deficit','log_abs_q','log_td_loss'])]:
  X=np.column_stack([d[features].to_numpy(),pd.get_dummies(d.architecture,dtype=float)]);pred=np.full(len(d),np.nan)
  for seed in sorted(d.seed.unique()):
   test=d.seed.to_numpy()==seed;train=~test;mu=X[train].mean(0);sd=X[train].std(0);sd[sd==0]=1;Z=np.column_stack([np.ones(len(d)),(X-mu)/sd]);pen=np.eye(Z.shape[1]);pen[0,0]=0;beta=np.linalg.solve(Z[train].T@Z[train]+pen,Z[train].T@Y[train]);pred[test]=Z[test]@beta
  preds[model]=pred
 d['baseline_squared_error']=(preds['baseline']-Y)**2;d['augmented_squared_error']=(preds['critic_augmented']-Y)**2
 by=d.groupby('seed')[['baseline_squared_error','augmented_squared_error']].mean();diff=by.baseline_squared_error-by.augmented_squared_error;m,lo,hi=ci(diff)
 cv.append(dict(environment=env,n_seed_groups=len(by),mean_seed_MSE_improvement=m,ci_low=lo,ci_high=hi,improved_seed_groups=int((diff>0).sum()),total_seed_groups=len(by)))
 save('prediction_errors_'+env,d[['id','seed','fraction','future_change','baseline_squared_error','augmented_squared_error']])
save('predictive_improvement_clustered',cv)
# Every >=30% post-20%-budget drawdown after reaching half own observed peak.
cases=[]
for _,r in EV[(EV.threshold==.3)&EV.any_event].iterrows():
 t=r.onset_fraction;d=HC[HC.id==r.id].sort_values('fraction');p=P[(P.id==r.id)&(P.probe_fraction<=t-.1+1e-8)];pre=d[(d.fraction<=t-.1+1e-8)&(d.fraction>=t-.2-1e-8)]
 row=r.to_dict();row['preceding_deficit']=p['probe/loss_iqm'].iloc[-1] if len(p) else np.nan
 for metric in ['train/q1_mean','train/critic_loss','train/policy_kl','train/actor_gnorm','train/entropy','train/temperature']:
  row['preceding_'+metric]=pre[metric].median()
 cases.append(row)
save('collapse_case_catalog',cases)
# Collapse-aligned observed returns; each run relative to its own historical peak (descriptive).
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,env in zip(axs,['myo-key-turn','myo-pen-twirl']):
 for r in cases:
  if r['environment']!=env:continue
  d=HC[HC.id==r['id']].dropna(subset=['eval/avg_return']).sort_values('fraction');ax.plot(d.fraction-r['onset_fraction'],d['eval/avg_return']/r['peak_return'],alpha=.5,lw=.8,label=f"{r['architecture']} s{r['seed']}")
 ax.set(title=env,xlabel='Budget fraction relative to first event',ylabel='Return / own observed peak');ax.axvline(0,color='black',lw=.5);ax.legend(fontsize=5)
fig.suptitle('Exploratory event alignment; observed evaluations only; individual runs, no pooled CI');fig.tight_layout();fig.savefig(F/'collapse_aligned.pdf');plt.close(fig)
# One independently archived original full probe; no reconstruction or refitting.
archives=list((ROOT/'raw_archived').glob('*.npz'))
fitrows=[]
if archives:
 p=archives[0];a=np.load(p);(ROOT/'archived_probe_provenance.json').write_text(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'D2W512 myo-pen-twirl seed5 check50 only'},indent=2))
 fig,ax=plt.subplots(figsize=(8,4))
 for arm in ['fresh','current']:
  for k,y in enumerate(a[arm+'_losses']):
   peak=int(y.argmax());recovery=np.flatnonzero(y[peak+1:]<=y[0]);fitrows.append(dict(arm=arm,round=k,initial_minibatch=y[0],peak=y[peak],peak_to_initial=y[peak]/y[0],peak_index=peak,first_recovery=peak+1+recovery[0] if len(recovery) else np.nan,final_minibatch=y[-1],final_pool=a[arm+'_final_loss'][k],baseline=a[arm+'_b'][k],offset=a[arm+'_offset'][k],excess_sum=np.maximum(y-y[0],0).sum()))
   ax.plot(np.arange(2000),y,color='tab:blue' if arm=='fresh' else 'tab:red',alpha=.4,lw=.6,label=arm if k==0 else None)
 ax.set(yscale='log',xlabel='Fitting update index (pre-update minibatch MSE)',ylabel='MSE',title='Original D2W512 pen-twirl seed5 check50; n=1 trained seed, 10 paired rounds');ax.legend();fig.tight_layout();fig.savefig(F/'archived_original_fitting.pdf');plt.close(fig)
 save('archived_original_fit_dynamics',fitrows)
print('DEEP_CHECKS_COMPLETE',len(cases),'collapse cases')
