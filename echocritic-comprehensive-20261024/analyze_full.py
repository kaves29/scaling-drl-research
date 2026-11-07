"""Reproducible CPU exploratory analysis; training seeds are inference units."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
from scipy.stats import trim_mean,spearmanr,pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'raw_api';OUT=ROOT/'derived';FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
RNG=np.random.default_rng(20261024);B=10000
ARCH=['D2W512','D4W1024','D4W1536']
COLORS=dict(zip(ARCH,['#16748d','#b26d12','#a93949']))
def iqm(a,axis=None):
 a=np.asarray(a,dtype=np.float64)
 if axis is None and a.size==10:
  x=np.sort(a);return (x[2]+x[7])/10+np.sum(x[3:7])/5
 return trim_mean(a,.25,axis=axis)
def interval(x,fn='mean'):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 if not len(x):return (np.nan,np.nan,np.nan)
 f={'mean':lambda a,axis=None:np.mean(a,axis=axis),'iqm':iqm}[fn]
 point=float(f(x))
 if len(x)<2:return point,np.nan,np.nan
 draws=x[RNG.integers(0,len(x),(B,len(x)))];return point,*np.quantile(f(draws,axis=1),[.025,.975])
def save(name,rows):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);d.to_csv(OUT/(name+'.csv'),index=False);return d
inventory=[];hist=[];valid=[];crashes=[]
for p in sorted(RAW.glob('*/metadata.json')):
 m=json.loads(p.read_text());c=m['config'];a=c['agent'];probe=c['probe'];rid=m['id'];h=p.with_name('history.jsonl')
 if not h.exists():raise RuntimeError('missing history '+rid)
 rows=[json.loads(s) for s in h.read_text().splitlines() if s.strip()];d=pd.DataFrame(rows);d['id']=rid;d['raw_step']=d['_step'];d['fraction']=d.raw_step/c['num_env_steps']
 ar=f"D{a['critic_num_blocks']}W{a['critic_hidden_dim']}";cohort='pilot' if m['attrs']['commit'].startswith('f861') else 'exp96'
 row=dict(id=rid,name=m['name'],state=m['state'],environment=c['env_name'],architecture=ar,seed=c['seed'],cohort=cohort,source=m['attrs']['commit'],created=m['attrs']['createdAt'],history_rows=len(d),expected_history_rows=m['attrs']['historyLineCount'],last_raw_step=d.raw_step.max(),raw_budget=c['num_env_steps'],fraction=d.fraction.max(),rounds=probe['rounds'],checks=probe['checks'],fit_steps=probe['steps'],critic_lr=a['critic_learning_rate'],critic_decay=a['critic_weight_decay'],actor_params=c.get('actor_num_params'),critic_params=c.get('critic_num_params'),total_params=c.get('num_params'))
 inventory.append(row)
 for k in ('environment','architecture','seed','cohort'):d[k]=row[k]
 hist.append(d)
 valid.append(dict(id=rid,history_count_matches=len(d)==m['attrs']['historyLineCount'],unique_steps=not d.raw_step.duplicated().any(),monotone=d.raw_step.is_monotonic_increasing,nonfinite_numeric=int(np.isinf(d.select_dtypes('number')).sum().sum())))
 if m['state']=='crashed':
  log=p.parent/'files/output.log';tail=log.read_text(errors='replace')[-12000:] if log.exists() else 'NO LOG'
  crashes.append(dict(id=rid,log_tail=tail,classification='unresolved; no crash cause inferred from run state'))
I=save('run_inventory',inventory);save('data_validation',valid);save('crash_log_evidence',crashes)
H=pd.concat(hist,ignore_index=True);H.to_pickle(OUT/'histories.pkl');C=I[I.cohort=='exp96'];HC=H[H.cohort=='exp96']
assert len(I)==103 and len(C)==96
save('duplicate_display_names',I[I.name.duplicated(keep=False)].sort_values('name'))
conditions=C.groupby(['environment','architecture','seed']).size();save('condition_counts',conditions.rename('records').reset_index())
assert conditions.max()==1
# Probe rows use their own counters, not the W&B logging row's earlier timestamp.
P=HC.dropna(subset=['probe/check_index']).copy();P['check']=P['probe/check_index'].astype(int);P['probe_fraction']=P['probe/env_step']/P['environment'].map(C.groupby('environment').raw_budget.first())
P=P.sort_values(['id','check']);assert not P.duplicated(['id','check']).any()
# The logger emits fresh IQM only at check zero; reconstruct later values from all ten rounds.
P['fresh_iqm_reconstructed']=P['probe/score_fresh_iqm'].isna()
for idx,r in P.iterrows():
 if pd.isna(r.get('probe/score_fresh_iqm')):P.loc[idx,'probe/score_fresh_iqm']=iqm([r[f'probe/score_fresh_r{k}'] for k in range(10)])
 P.loc[idx,'derived_current_score_iqm']=iqm([r.get(f'probe/score_current_r{k}',np.nan) for k in range(10)])
pr=[]
for _,r in P.iterrows():
 fresh=np.array([r.get(f'probe/score_fresh_r{k}',np.nan) for k in range(10)],np.float32);cur=np.array([r.get(f'probe/score_current_r{k}',np.nan) for k in range(10)],np.float32)
 err=float(abs(iqm(fresh-cur)-r['probe/loss_iqm'])) if r.check>0 else np.nan
 pr.append(dict(id=r.id,check=r.check,iqm_recompute_error=err))
save('probe_score_validation',pr);save('probe_trajectories',P)
# Seed-level endpoints and partial-run summaries; shared 70% horizon is below every crash.
S=[]
for _,r in C.iterrows():
 d=HC[HC.id==r.id];e=d.dropna(subset=['eval/avg_return']).sort_values('raw_step');x=e.fraction.to_numpy();y=e['eval/avg_return'].to_numpy();p=P[P.id==r.id]
 s={k:r[k] for k in ('id','environment','architecture','seed','state')};s.update(eval_count=len(e),observed_fraction=x.max(),final_return=y[-1] if x.max()==1 else np.nan,last_observed_return=y[-1],mean_return=y.mean(),median_return=np.median(y),temporal_sd=np.std(y),peak_return=y.max(),peak_fraction=x[y.argmax()],observed_auc=np.sum(np.diff(x)*(y[1:]+y[:-1])/2),final_window=e.loc[e.fraction>=.9,'eval/avg_return'].mean() if x.max()==1 else np.nan,auc70=np.sum(np.diff(x[x<=.7])*(y[x<=.7][1:]+y[x<=.7][:-1])/2),return70=float(e.loc[np.isclose(e.fraction,.7),'eval/avg_return'].iloc[0]))
 for check,label in [(1,'early'),(25,'middle'),(35,'common70'),(50,'final')]:
  q=p[p.check==check]
  for metric,short in [('probe/loss_iqm','deficit'),('probe/score_fresh_iqm','fresh_score')]:s[label+'_'+short]=float(q[metric].iloc[0]) if len(q) else np.nan
 s['fresh_score_change70']=float(p.loc[p.check==35,'probe/score_fresh_iqm'].iloc[0]-p.loc[p.check==0,'probe/score_fresh_iqm'].iloc[0])
 s['late_deficit']=p['probe/loss_iqm'].iloc[-1];s['fresh_score_change']=p['probe/score_fresh_iqm'].iloc[-1]-p['probe/score_fresh_iqm'].iloc[0]
 s['late_q_abs']=d.loc[d.fraction>=.7,'train/q1_mean'].abs().median();s['late_td_loss']=d.loc[d.fraction>=.7,'train/critic_loss'].median()
 S.append(s)
S=save('seed_endpoints',S);G=[]
metrics=['final_return','final_window','peak_return','auc70','return70','final_deficit','common70_deficit','fresh_score_change','fresh_score_change70','late_q_abs','late_td_loss']
for (env,arch),sub in S.groupby(['environment','architecture']):
 for metric in metrics:
  vals=sub[metric].dropna().values;n=len(vals)
  point,lo,hi=interval(vals);q,qlo,qhi=interval(vals,'iqm')
  G.append(dict(environment=env,architecture=arch,metric=metric,n=n,mean=point,ci_low=lo,ci_high=hi,iqm=q,iqm_low=qlo,iqm_high=qhi,median=np.median(vals) if n else np.nan,sd=np.std(vals,ddof=1) if n>1 else np.nan))
G=save('architecture_summaries',G);effects=[]
for env,sub in S.groupby('environment'):
 for left,right in itertools.combinations(ARCH,2):
  l=sub[sub.architecture==left].set_index('seed');r=sub[sub.architecture==right].set_index('seed')
  for metric in metrics:
   joined=pd.concat([l[metric].rename('left'),r[metric].rename('right')],axis=1).dropna();delta=(joined.right-joined.left).values
   pt,lo,hi=interval(delta)
   # Exhaustive paired sign permutation for exploratory p-values (n<=8).
   perms=np.array(list(itertools.product([-1,1],repeat=len(delta)))) if len(delta) else np.empty((0,0))
   pval=float(np.mean(np.abs((perms*delta).mean(axis=1))>=abs(pt)-1e-12)) if len(delta) else np.nan
   effects.append(dict(environment=env,left=left,right=right,metric=metric,n=len(delta),right_minus_left=pt,ci_low=lo,ci_high=hi,positive_seeds=int((delta>0).sum()),negative_seeds=int((delta<0).sum()),paired_permutation_p=pval))
E=pd.DataFrame(effects);order=E.paired_permutation_p.sort_values().index;adjusted=[];running=0
for rank,idx in enumerate(order):running=max(running,min(1,E.loc[idx,'paired_permutation_p']*(len(order)-rank)));E.loc[idx,'holm_all_exploratory_p']=running
save('paired_effects',E)
# Every positive-peak drawdown candidate and definition sensitivities, without deleting recovered runs.
events=[]
for _,r in C.iterrows():
 e=HC[HC.id==r.id].dropna(subset=['eval/avg_return']).sort_values('raw_step');y=e['eval/avg_return'].to_numpy();t=e.fraction.to_numpy();peak=np.maximum.accumulate(y);draw=np.divide(peak-y,peak,out=np.zeros_like(y),where=peak>0)
 for threshold in [.2,.3,.5]:
  candidates=np.flatnonzero((draw>=threshold)&(peak>0)&(t>=.2)&(peak>=.5*y.max()));on=int(candidates[0]) if len(candidates) else None
  pers=[k for k in candidates if k+1<len(y) and draw[k+1]>=threshold]
  events.append(dict(id=r.id,environment=r.environment,architecture=r.architecture,seed=r.seed,state=r.state,threshold=threshold,any_event=on is not None,onset_fraction=t[on] if on is not None else np.nan,persistent_two_evals=bool(pers),persistent_onset=t[pers[0]] if pers else np.nan,max_drawdown=float(draw.max()),max_absolute_decline=float((peak-y).max()),final_drawdown=float(draw[-1]),peak_return=float(y.max()),last_return=float(y[-1]),eval_count=len(y)))
EV=save('collapse_events',events)
# Within-run detrending then association: no pooling unadjusted common learning trends.
assoc=[];lagrows=[]
for _,r in C.iterrows():
 d=HC[HC.id==r.id].sort_values('fraction');p=P[P.id==r.id]
 for metric in ['train/q1_mean','train/critic_loss','train/policy_kl','train/actor_gnorm','train/entropy','train/temperature']:
  q=d.dropna(subset=[metric,'eval/avg_return']);xx=q[metric].to_numpy();yy=q['eval/avg_return'].to_numpy();tt=q.fraction.to_numpy()
  if len(xx)>=5:
   X=np.column_stack([np.ones(len(tt)),tt,tt**2]);xr=xx-X@np.linalg.lstsq(X,xx,rcond=None)[0];yr=yy-X@np.linalg.lstsq(X,yy,rcond=None)[0]
   corr=float(pearsonr(xr,yr).statistic) if np.std(xr)>0 and np.std(yr)>0 else np.nan
   assoc.append(dict(id=r.id,environment=r.environment,architecture=r.architecture,seed=r.seed,metric=metric,n=len(q),detrended_r=corr))
 e=d.dropna(subset=['eval/avg_return'])
 for k in range(1,len(e)-1):
  now=e.iloc[k];later=e.iloc[k+1];prev=e.iloc[k-1];pp=p[p.probe_fraction<=now.fraction];recent=d[(d.fraction<=now.fraction)&(d.fraction>=now.fraction-.1)]
  lagrows.append(dict(id=r.id,environment=r.environment,architecture=r.architecture,seed=r.seed,fraction=now.fraction,current_return=now['eval/avg_return'],previous_change=now['eval/avg_return']-prev['eval/avg_return'],future_change=later['eval/avg_return']-now['eval/avg_return'],deficit=pp['probe/loss_iqm'].iloc[-1] if len(pp) else np.nan,log_abs_q=np.log1p(recent['train/q1_mean'].abs().median()),log_td_loss=np.log1p(recent['train/critic_loss'].median())))
A=save('detrended_run_associations',assoc);L=save('lagged_observations',lagrows)
AS=[]
for key,ss in A.groupby(['environment','metric']):
 pt,lo,hi=interval(ss.detrended_r.dropna());AS.append(dict(environment=key[0],metric=key[1],n_runs=len(ss),mean_detrended_r=pt,ci_low=lo,ci_high=hi))
save('association_summaries',AS)
# Leave one training-seed number out across architectures within environment. Baseline vs augmented ridge.
CV=[]
for env,d in L.groupby('environment'):
 d=d.dropna();y=d.future_change.to_numpy();base=['fraction','current_return','previous_change'];extra=['deficit','log_abs_q','log_td_loss']
 for model,features in [('baseline',base),('critic_augmented',base+extra)]:
  pred=np.full(len(d),np.nan);X=d[features].to_numpy();arch=pd.get_dummies(d.architecture,dtype=float).to_numpy();X=np.column_stack([X,arch])
  for seed in sorted(d.seed.unique()):
   test=d.seed.to_numpy()==seed;train=~test;mu=X[train].mean(0);sd=X[train].std(0);sd[sd==0]=1;xt=(X-mu)/sd;xt=np.column_stack([np.ones(len(xt)),xt]);pen=np.eye(xt.shape[1]);pen[0,0]=0
   beta=np.linalg.solve(xt[train].T@xt[train]+pen,xt[train].T@y[train]);pred[test]=xt[test]@beta
  CV.append(dict(environment=env,model=model,n_runs=d.id.nunique(),n_seed_groups=d.seed.nunique(),observations=len(d),mse=float(np.mean((pred-y)**2)),mae=float(np.mean(abs(pred-y))),r=float(pearsonr(pred,y).statistic)))
save('predictive_cross_validation',CV)
# Q/deficit associations with stage/architecture adjustments within environment; repeat rows not independent.
QA=[]
for env,pp in P[P.check>0].groupby('environment'):
 rows=[]
 for rid,p in pp.groupby('id'):
  d=HC[HC.id==rid].sort_values('fraction')
  for _,r in p.iterrows():
   before=d[(d.fraction<=r.probe_fraction)&(d.fraction>=r.probe_fraction-.02)]
   if len(before):rows.append(dict(id=rid,architecture=r.architecture,seed=r.seed,t=r.probe_fraction,q=np.log1p(before['train/q1_mean'].abs().median()),deficit=r['probe/loss_iqm'],fresh=r['probe/score_fresh_iqm']))
 q=pd.DataFrame(rows).dropna();save('q_probe_aligned_'+env,q)
 for target in ['deficit','fresh']:
  # Per-run correlations after removing linear/quadratic time; summarize independent run values descriptively.
  rr=[]
  for rid,d in q.groupby('id'):
   X=np.column_stack([np.ones(len(d)),d.t,d.t**2]);x=d.q-X@np.linalg.lstsq(X,d.q,rcond=None)[0];y=d[target]-X@np.linalg.lstsq(X,d[target],rcond=None)[0]
   if len(d)>5 and np.std(x)>0 and np.std(y)>0:rr.append(float(pearsonr(x,y).statistic))
  pt,lo,hi=interval(rr);QA.append(dict(environment=env,target=target,n_runs=len(rr),mean_within_run_time_adjusted_r=pt,ci_low=lo,ci_high=hi))
save('q_probe_associations',QA)
# All unsmoothed seed trajectories plus exact-step aggregate uncertainty, separate environment axes.
for metric,label in [('eval/avg_return','Episode return (reward units)'),('train/q1_mean','Mean Q (Q units)'),('train/critic_loss','Training TD MSE (squared Q units)'),('train/policy_kl','Policy KL (nats)'),('probe/loss_iqm','Paired adaptation deficit (target MSE units)'),('probe/score_fresh_iqm','Fresh score (target MSE units)')]:
 fig,axs=plt.subplots(2,2,figsize=(12,8));src=P if metric.startswith('probe') else HC;step='probe_fraction' if metric.startswith('probe') else 'fraction'
 for ax,env in zip(axs.flat,sorted(C.environment.unique())):
  for arch in ARCH:
   sub=src[(src.environment==env)&(src.architecture==arch)].dropna(subset=[metric]);n=sub.id.nunique()
   for rid,d in sub.groupby('id'):ax.plot(d[step],d[metric],color=COLORS[arch],alpha=.2,lw=.6)
   means=[];lo=[];hi=[];ts=[]
   for t,d in sub.groupby(step):
    xx=d[metric].to_numpy();pt,l,h=interval(xx);ts.append(t);means.append(pt);lo.append(l);hi.append(h)
   ax.plot(ts,means,color=COLORS[arch],label=f'{arch}; initial n={n}');ax.fill_between(ts,lo,hi,color=COLORS[arch],alpha=.15)
  ax.set(title=env,xlabel='Fraction of raw training budget',ylabel=label);ax.legend(fontsize=7)
  if metric in ('train/critic_loss','train/q1_mean','probe/loss_iqm'):ax.set_yscale('symlog',linthresh=.01)
 fig.suptitle('Individual seeds + observed-check mean; pointwise 95% seed bootstrap; late n declines for D4W1536',fontsize=10);fig.tight_layout();fig.savefig(FIG/(metric.replace('/','_')+'.pdf'));plt.close(fig)
# Every individual return and probe trajectory in one multi-page scientific atlas.
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(FIG/'all_seed_atlas.pdf') as pdf:
 for _,r in C.sort_values(['environment','architecture','seed']).iterrows():
  d=HC[HC.id==r.id];p=P[P.id==r.id];fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True)
  for ax,metric,label in zip(axs,['eval/avg_return','train/q1_mean','train/critic_loss'],['Return','Mean Q','TD MSE']):
   q=d.dropna(subset=[metric]);ax.plot(q.fraction,q[metric],lw=.8);ax.set_ylabel(label)
  axs[-1].set_yscale('symlog');axs[-1].set_xlabel('Raw budget fraction');fig.suptitle(f'{r.environment} {r.architecture} seed {r.seed}: {r.state}; {r.id}');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
print('ANALYSIS_COMPLETE',len(C),'campaign runs',len(P),'probe rows',flush=True)
