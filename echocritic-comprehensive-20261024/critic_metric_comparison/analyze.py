"""Read-only source-verified critic-metric/probe comparison; seed-level inference."""
from pathlib import Path
import subprocess,hashlib,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parent;B=R.parent;D=B/'derived';rng=np.random.default_rng(20261026)
H=pd.read_pickle(D/'histories.pkl');H=H[H.cohort=='exp96'].copy();H['interaction_step']=(H.fraction*500000).round().astype(int)
P=pd.read_csv(B/'initial_reference_comparison/per_seed_check_comparison.csv');colors={'D2W512':'#16748d','D4W1024':'#b26d12','D4W1536':'#a93949'};envs=['dog-run','humanoid-run','myo-key-turn','myo-pen-twirl'];repo='/Users/shouryakaveti/VS_Projects/scaling_rl_research/echocritic-batch2';sha='f597ee9898e4d401912092c336c3e5330cf2fda1'
files=['scale_rl/networks/metrics.py','scale_rl/networks/layers.py','scale_rl/agents/sac/sac_network.py','scale_rl/agents/sac/sac_agent.py','scale_rl/agents/sac/sac_update.py','scale_rl/networks/trainer.py','experiments/exp12/trainer.py'];manifest=[]
for name in files:
 raw=subprocess.check_output(['git','-C',repo,'show',sha+':main/'+name]);dest=R/'source'/name;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(raw);manifest.append({'source':sha,'file':name,'sha256':hashlib.sha256(raw).hexdigest()})
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
metrics={'train/critic_DR0.1':'Dormant fraction τ=0.1 (%)','train/critic_DR0.2':'Dormant fraction τ=0.2 (%)','train/critic_srank':'99%-singular-value count (maximum 256)','train/critic_gnorm':'Critic gradient L2 (training-loss units)','train/critic_fnorm':'Mean encoder feature L2','train/critic_wnorm':'Critic parameter L2','train/critic_loss':'Training TD MSE','train/td_error_var':'Training TD-error variance','train/q1_mean':'Mean critic Q'}
rows=[];coverage=[]
for rid,d in H.groupby('id'):
 row={k:d[k].iloc[0] for k in ['id','environment','architecture','seed']}
 for metric in metrics:
  q=d.dropna(subset=[metric]).sort_values('interaction_step');coverage.append({'id':rid,'metric':metric,'observations':len(q),'first_interaction':q.interaction_step.iloc[0] if len(q) else None,'last_interaction':q.interaction_step.iloc[-1] if len(q) else None})
  row[metric+'_initial']=q[metric].iloc[0] if len(q) else np.nan
  for label,lo,hi in [('early',.1,.2),('common70',.6,.7),('late_observed',.8,1)]:row[metric+'_'+label]=q.loc[q.fraction.between(lo,hi),metric].median()
  row[metric+'_change']=row[metric+'_common70']-row[metric+'_early']
  row[metric+'_ratio']=row[metric+'_common70']/row[metric+'_early'] if row[metric+'_early']>0 else np.nan
  row[metric+'_max']=q[metric].max();row[metric+'_chronic60_70_fraction']=(q.loc[q.fraction.between(.6,.7),metric]>1).mean() if 'DR' in metric else np.nan
 for metric in ['train/critic_DR0.1','train/critic_DR0.2','train/critic_srank']:
  row[metric+'_from_initial']=row[metric+'_common70']-row[metric+'_initial']
 pp=P[P.id==rid];row['anchored_gap70']=pp.loc[pp.check==35,'anchored_deficit'].iloc[0];row['trained_score70']=pp.loc[pp.check==35,'trained_score'].iloc[0]
 rows.append(row)
S=pd.DataFrame(rows);S.to_csv(R/'per_seed_summaries.csv',index=False);pd.DataFrame(coverage).to_csv(R/'metric_coverage.csv',index=False)
def ci(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 if not len(x):return np.nan,np.nan,np.nan
 a=x[rng.integers(0,len(x),(10000,len(x)))];lo,hi=np.quantile(a.mean(1),[.025,.975]);return x.mean(),lo,hi
agg=[]
for (env,arch),d in S.groupby(['environment','architecture']):
 for metric in metrics:
  row={'environment':env,'architecture':arch,'metric':metric,'n_seeds':len(d)}
  for label in ['initial','early','common70','late_observed','change','ratio']:
   m,lo,hi=ci(d[metric+'_'+label]);row[label]=m;row[label+'_ci_low']=lo;row[label+'_ci_high']=hi
  agg.append(row)
G=pd.DataFrame(agg);G.to_csv(R/'architecture_summaries.csv',index=False)
# Derived relative norms for visualization only; preserve original units and original logged values.
for metric in ['train/critic_fnorm','train/critic_wnorm']:
 initial=H.sort_values('interaction_step').dropna(subset=[metric]).groupby('id')[metric].first();H[metric+'_relative_initial']=H[metric]/H.id.map(initial)
# Trajectories, exact observed counters; no interpolation and no smoothers.
series={}
for metric in list(metrics)+['train/critic_fnorm_relative_initial','train/critic_wnorm_relative_initial']:
 for (env,arch),d in H.dropna(subset=[metric]).groupby(['environment','architecture']):
  rr=[]
  for t,q in d.groupby('interaction_step'):
   assert not q.id.duplicated().any();m,lo,hi=ci(q[metric]);rr.append((t,m,lo,hi,len(q)))
  series[(metric,env,arch)]=pd.DataFrame(rr,columns=['step','mean','low','high','n'])
for metric in ['anchored_deficit','trained_score']:
 for (env,arch),d in P.groupby(['environment','architecture']):
  rr=[]
  for t,q in d.groupby('interaction_step'):
   m,lo,hi=ci(q[metric]);rr.append((t,m,lo,hi,len(q)))
  series[(metric,env,arch)]=pd.DataFrame(rr,columns=['step','mean','low','high','n'])
def panel(ax,metric,env,label,log=False):
 for arch,color in colors.items():
  d=series[(metric,env,arch)];ax.plot(d.step/1000,d['mean'],color=color,label=f"{arch} n={d.n.iloc[0]}→{d.n.iloc[-1]}",lw=1.3);ax.fill_between(d.step/1000,d.low,d.high,color=color,alpha=.13)
 if log:ax.set_yscale('log')
 if metric=='anchored_deficit':ax.set_yscale('symlog',linthresh=.02);ax.axhline(0,color='grey',lw=.5)
 ax.set(xlabel='Training interactions (thousands)',ylabel=label);ax.grid(alpha=.1)
for group,ee in [('DMC',envs[:2]),('Myo',envs[2:])]:
 fig,axs=plt.subplots(2,5,figsize=(20,8))
 for i,env in enumerate(ee):
  specs=[('train/critic_DR0.2','Dormant fraction τ=.2 (%)',False),('train/critic_srank','99% singular-value count',False),('train/critic_gnorm','Training gradient L2',True),('anchored_deficit','Initial fresh P − trained P',False),('train/critic_wnorm_relative_initial','Parameter norm / initial norm',False)]
  for j,(metric,label,log) in enumerate(specs):panel(axs[i,j],metric,env,label,log);axs[i,j].set_title(env)
 axs[0,0].legend(fontsize=7);fig.suptitle(group+': independent critic metrics alongside 2,000-update probe\nMeans and pointwise 95% training-seed bootstrap; observed points only; late large-critic n declines from 8 to 6.',fontsize=12);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(R/(group+'_overview.pdf'));fig.savefig(R/(group+'_overview.png'),dpi=150);plt.close(fig)
for metric,label in metrics.items():
 fig,axs=plt.subplots(2,2,figsize=(11,7))
 for ax,env in zip(axs.flat,envs):
  for arch,color in colors.items():
   d=H[(H.environment==env)&(H.architecture==arch)].dropna(subset=[metric]);
   for rid,dd in d.groupby('id'):ax.plot(dd.interaction_step/1000,dd[metric],color=color,alpha=.13,lw=.5)
  panel(ax,metric,env,label,metric in ['train/critic_gnorm','train/critic_loss','train/td_error_var']);ax.set_title(env);ax.legend(fontsize=7)
 fig.suptitle('Individual training seeds + mean with pointwise 95% seed-bootstrap interval',fontsize=10);fig.tight_layout();name=metric.replace('train/','').replace('.','p');fig.savefig(R/(name+'.pdf'));fig.savefig(R/(name+'.png'),dpi=140);plt.close(fig)
# The source rank statistic is batch-limited and not stable rank. No width normalization is introduced.
with PdfPages(R/'all_96_seed_metric_comparisons.pdf') as pdf:
 for (env,arch),dd in H.groupby(['environment','architecture']):
  for subset in [range(1,5),range(5,9)]:
   fig,axs=plt.subplots(4,4,figsize=(14,10))
   for i,seed in enumerate(subset):
    d=dd[dd.seed==seed];pp=P[(P.environment==env)&(P.architecture==arch)&(P.seed==seed)]
    for j,(metric,label) in enumerate([('train/critic_DR0.2','Dormancy τ=.2 (%)'),('train/critic_srank','99% singular-value count'),('train/critic_gnorm','Gradient L2')]):
     q=d.dropna(subset=[metric]);axs[i,j].plot(q.interaction_step/1000,q[metric],lw=.8);axs[i,j].set_ylabel(label);axs[i,j].set_title(f'seed{seed}');
     if j==2:axs[i,j].set_yscale('log')
    axs[i,3].plot(pp.interaction_step/1000,pp.anchored_deficit,lw=.8);axs[i,3].axhline(0,color='grey',lw=.5);axs[i,3].set_yscale('symlog',linthresh=.02);axs[i,3].set_ylabel('Initial P − trained P')
   for ax in axs[-1]:ax.set_xlabel('Interactions (thousands)')
   fig.suptitle(f'{env} {arch}; individual seeds, no pooling checkpoints as independent runs');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
# Align probe checks using their own interaction counters, with no future observations.
aligned=[]
for rid,pp in P.groupby('id'):
 h=H[H.id==rid].sort_values('interaction_step')
 for _,r in pp.iterrows():
  q=h[h.interaction_step<=r.interaction_step]
  row={k:r[k] for k in ['id','environment','architecture','seed','interaction_step','anchored_deficit','trained_score','source_paired_deficit']}
  for metric in ['train/critic_DR0.1','train/critic_DR0.2','train/critic_srank','train/critic_gnorm','train/critic_wnorm']:
   v=q.dropna(subset=[metric]);row[metric]=v[metric].iloc[-1] if len(v) and r.interaction_step-v.interaction_step.iloc[-1]<=2000 else np.nan
  aligned.append(row)
A=pd.DataFrame(aligned);A.to_csv(R/'aligned_probe_metrics.csv',index=False)
corr=[]
for rid,d in A.groupby('id'):
 for m in ['train/critic_DR0.2','train/critic_srank','train/critic_gnorm']:
  q=d.dropna(subset=[m,'anchored_deficit']);x=q[m].to_numpy();y=q.anchored_deficit.to_numpy();t=q.interaction_step.to_numpy()/500000
  if len(q)<6 or np.std(x)==0:continue
  X=np.column_stack([np.ones(len(q)),t,t*t]);xx=x-X@np.linalg.lstsq(X,x,rcond=None)[0];yy=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
  corr.append(dict(id=rid,environment=q.environment.iloc[0],architecture=q.architecture.iloc[0],seed=q.seed.iloc[0],metric=m,n=len(q),spearman_raw=float(spearmanr(x,y).statistic),spearman_time_residual=float(spearmanr(xx,yy).statistic)))
pd.DataFrame(corr).to_csv(R/'within_run_associations.csv',index=False)
# Quantify conflicting signals without imposing scientific pathology thresholds.
checks={'campaign_runs':S.id.nunique(),'metric_rows':len(H),'aligned_probe_checks':len(A),'duplicate_probe_rows':int(A.duplicated(['id','interaction_step']).sum()),'all_DR02_at_least_DR01':bool((H['train/critic_DR0.2'].dropna()>=H.loc[H['train/critic_DR0.2'].notna(),'train/critic_DR0.1']).all()),'all_rank_between_1_and_256':bool(H['train/critic_srank'].dropna().between(1,256).all()),'no_new_fits':True};(R/'VALIDATION.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks));print(G[G.metric.isin(['train/critic_DR0.2','train/critic_srank','train/critic_gnorm'])][['environment','architecture','metric','early','common70','change','change_ci_low','change_ci_high','ratio']].to_string(index=False))
