"""Compare archived trained probe scores with each parent's initial fresh score.
No fitting is run. The anchored contrast is descriptive across different tasks.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
P=pd.read_csv(BASE/'derived/probe_trajectories.csv');rng=np.random.default_rng(20261025)
ARCH=['D2W512','D4W1024','D4W1536'];ENVS=['dog-run','humanoid-run','myo-key-turn','myo-pen-twirl'];COLORS=dict(zip(ARCH,['#16748d','#b26d12','#a93949']))
def iqm10(x):
 x=np.sort(np.asarray(x,np.float64));return (x[2]+x[7])/10+np.sum(x[3:7])/5
rows=[]
for rid,d in P.groupby('id'):
 initial=d[d.check==0].iloc[0];b=iqm10([initial[f'probe/score_fresh_r{k}'] for k in range(10)])
 for _,r in d[d.check>0].sort_values('check').iterrows():
  cur=iqm10([r[f'probe/score_current_r{k}'] for k in range(10)]);fresh=iqm10([r[f'probe/score_fresh_r{k}'] for k in range(10)])
  rows.append(dict(id=rid,environment=r.environment,architecture=r.architecture,seed=int(r.seed),check=int(r.check),interaction_step=int(r['probe/interaction_step']),initial_interaction_step=int(initial['probe/interaction_step']),initial_fresh_score=b,trained_score=cur,contemporaneous_fresh_score=fresh,anchored_deficit=b-cur,score_gap_contemporaneous=fresh-cur,source_paired_deficit=r['probe/loss_iqm'],fresh_reference_drop=b-fresh))
S=pd.DataFrame(rows);S.to_csv(ROOT/'per_seed_check_comparison.csv',index=False)
metrics=['initial_fresh_score','trained_score','contemporaneous_fresh_score','anchored_deficit','score_gap_contemporaneous','source_paired_deficit','fresh_reference_drop']
agg=[]
for (e,a,c),d in S.groupby(['environment','architecture','check']):
 row=dict(environment=e,architecture=a,check=c,interaction_step=d.interaction_step.iloc[0],n_seeds=len(d),seed_ids=','.join(map(str,sorted(d.seed))))
 indices=rng.integers(0,len(d),(10000,len(d)))
 for m in metrics:
  x=d[m].to_numpy();lo,hi=np.quantile(x[indices].mean(1),[.025,.975]);row[m]=x.mean();row[m+'_ci_low']=lo;row[m+'_ci_high']=hi
 agg.append(row)
G=pd.DataFrame(agg);G.to_csv(ROOT/'architecture_trajectories.csv',index=False)
G[G.check.isin([1,25,35,50])].to_csv(ROOT/'selected_checkpoints.csv',index=False)
# Every plotted baseline uses the same observed seeds as its trained-score point.
fig,axs=plt.subplots(4,3,figsize=(14,13),sharex=True)
for i,e in enumerate(ENVS):
 for j,a in enumerate(ARCH):
  ax=axs[i,j];d=G[(G.environment==e)&(G.architecture==a)].sort_values('check');x=d.interaction_step/1000
  for metric,label,color,style in [('trained_score','Trained score','#16748d','-'),('contemporaneous_fresh_score','Fresh at this checkpoint','#d78926','--'),('initial_fresh_score','Initial fresh reference','#555555',':')]:
   ax.plot(x,d[metric],label=label,color=color,ls=style,lw=1.5)
  ax.fill_between(x,d.trained_score_ci_low,d.trained_score_ci_high,color='#16748d',alpha=.15)
  ax.axhline(0,color='grey',lw=.5);ax.set_yscale('symlog',linthresh=.1)
  ax.set(title=f'{e} | {a} | n={d.n_seeds.iloc[0]}→{d.n_seeds.iloc[-1]}',ylabel='Adaptation score P (higher is better)')
  if i==3:ax.set_xlabel('Training interaction steps (thousands)')
  if i==0 and j==0:ax.legend(fontsize=7)
fig.suptitle('Archived 2,000-update probes every 10,000 interactions; initial fresh probe at 5,000\nShading: pointwise 95% bootstrap over training seeds. Initial reference matched to observed seeds at each check.',fontsize=10);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(ROOT/'trained_fresh_initial_scores.pdf');fig.savefig(ROOT/'trained_fresh_initial_scores.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
for ax,e in zip(axs.flat,ENVS):
 for a in ARCH:
  d=G[(G.environment==e)&(G.architecture==a)].sort_values('check');x=d.interaction_step/1000;ax.plot(x,d.anchored_deficit,color=COLORS[a],label=f'{a} (n={d.n_seeds.iloc[0]}→{d.n_seeds.iloc[-1]})');ax.fill_between(x,d.anchored_deficit_ci_low,d.anchored_deficit_ci_high,color=COLORS[a],alpha=.13)
 ax.axhline(0,color='black',lw=.7);ax.set_yscale('symlog',linthresh=.02);ax.set(title=e,xlabel='Training interaction steps (thousands)',ylabel='Initial fresh score − trained score');ax.legend(fontsize=8)
fig.suptitle('Initial-reference comparison: positive = worse adaptation than the initial fresh baseline\nDifferent replay/teacher tasks across time; this is not an isolated plasticity-loss measurement.',fontsize=11);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(ROOT/'anchored_deficit_by_architecture.pdf');fig.savefig(ROOT/'anchored_deficit_by_architecture.png',dpi=170);plt.close(fig)
with PdfPages(ROOT/'all_96_seed_comparisons.pdf') as pdf:
 for (e,a),dd in S.groupby(['environment','architecture']):
  fig,axs=plt.subplots(4,2,figsize=(10,11),sharex=True)
  for ax,seed in zip(axs.flat,range(1,9)):
   d=dd[dd.seed==seed].sort_values('check');x=d.interaction_step/1000
   ax.plot(x,d.trained_score,label='trained');ax.plot(x,d.contemporaneous_fresh_score,'--',label='fresh at check');ax.axhline(d.initial_fresh_score.iloc[0],color='grey',ls=':',label='initial fresh');ax.set_yscale('symlog',linthresh=.1);ax.set(title=f'seed {seed}',ylabel='Score P');ax.set_xlabel('Interactions (thousands)')
  axs[0,0].legend(fontsize=6);fig.suptitle(f'{e}, {a}: individual training seeds; original archived scores');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
# Explicit numerical identity: anchored contrast = contemporary score gap + fresh reference drop.
err=np.max(np.abs(S.anchored_deficit-S.score_gap_contemporaneous-S.fresh_reference_drop))
validation={'training_runs':S.id.nunique(),'checks':len(S),'max_anchored_decomposition_error':float(err),'all_initial_checks_at_5000':bool((S.initial_interaction_step==5000).all()),'all_checks_on_10000_interaction_grid':bool((S.interaction_step==S.check*10000).all()),'score_iqm_definition':'source fractional IQM over ten rounds','no_new_fits':True}
(ROOT/'VALIDATION.json').write_text(json.dumps(validation,indent=2));assert err<1e-12
print(json.dumps(validation));print(G[G.check==35][['environment','architecture','n_seeds','initial_fresh_score','trained_score','contemporaneous_fresh_score','anchored_deficit','anchored_deficit_ci_low','anchored_deficit_ci_high']].to_string(index=False))
