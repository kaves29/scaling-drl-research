from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;D=R/'derived';F=R/'figures';S=pd.read_csv(D/'seed_endpoints.csv');P=pd.read_csv(D/'probe_trajectories.csv');colors={'D2W512':'#16748d','D4W1024':'#b26d12','D4W1536':'#a93949'}
fig,axs=plt.subplots(2,2,figsize=(11,7))
for ax,(env,d) in zip(axs.flat,S.groupby('environment')):
 for a,s in d.groupby('architecture'):
  ax.scatter(s.common70_deficit,s.return70,color=colors[a],label=a+' n=8')
  for _,r in s.iterrows():ax.annotate(str(r.seed),(r.common70_deficit,r.return70),fontsize=6)
 ax.set(title=env,xlabel='Paired adaptation deficit at 70% (target MSE units)',ylabel='Return at 70% (reward units)');ax.legend(fontsize=7)
fig.suptitle('Same-stage probe-policy dissociation; numbers denote independent training seeds');fig.tight_layout();fig.savefig(F/'probe_policy_common70.pdf');fig.savefig(F/'probe_policy_common70.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11,7))
for ax,(env,d) in zip(axs.flat,S.groupby('environment')):
 for i,(a,s) in enumerate(d.groupby('architecture')):
  ax.scatter([i]*len(s),s.fresh_score_change70,color=colors[a],label=a+' n=8');ax.plot([i-.15,i+.15],[s.fresh_score_change70.mean()]*2,color='black')
 ax.set(xticks=[0,1,2],xticklabels=list(colors),title=env,ylabel='Fresh-score change, initial to 70%');ax.axhline(0,color='grey',lw=.7)
fig.suptitle('Fresh parameters never underwent SAC updates; task/checkpoint context changes');fig.tight_layout();fig.savefig(F/'fresh_reference_drift.pdf');plt.close(fig)
# Q versus deficit at common stage, retaining reward-scale separation.
fig,axs=plt.subplots(2,2,figsize=(11,7))
for ax,(env,d) in zip(axs.flat,S.groupby('environment')):
 for a,s in d.groupby('architecture'):ax.scatter(s.late_q_abs,s.common70_deficit,color=colors[a],label=a+' n=8')
 ax.set(xscale='log',title=env,xlabel='Median |Q mean| over observed >=70% (Q units)',ylabel='70% deficit (target MSE units)');ax.legend(fontsize=7)
fig.suptitle('Descriptive scale association; Q window extends after probe and is not a predictor');fig.tight_layout();fig.savefig(F/'q_deficit_scale.pdf');plt.close(fig)
fit=pd.read_csv(D/'archived_original_fit_dynamics.csv')
fig,axs=plt.subplots(1,2,figsize=(10,4))
for arm,d in fit.groupby('arm'):
 axs[0].scatter(d.initial_minibatch,d.final_pool,label=arm)
 axs[1].scatter(d['round'],d.peak_to_initial,label=arm)
axs[0].set(xscale='log',yscale='log',xlabel='Initial pre-update minibatch MSE',ylabel='Final post-fit full-pool MSE')
axs[1].set(yscale='log',xlabel='Paired probe round (not training seed)',ylabel='Peak / initial minibatch loss')
for ax in axs:ax.legend()
fig.suptitle('Archived D2W512 pen-twirl seed5 check50 only; one independent training run');fig.tight_layout();fig.savefig(F/'archived_initial_final_amplification.pdf');plt.close(fig)
