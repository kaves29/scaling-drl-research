"""Read-only pilot analysis. Repetitions are descriptive, never independent seeds."""
from pathlib import Path
import zipfile, io, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import trim_mean
ROOT=Path(__file__).resolve().parent
Z=Path('/Users/shouryakaveti/VS_Projects/echo_pilot_local_partial.zip')
OUT=ROOT/'derived'; OUT.mkdir(exist_ok=True)
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
z=zipfile.ZipFile(Z)
manifest=json.loads(z.read('MANIFEST.json'))
validation=[]
for f in manifest['files']:
 validation.append(dict(path=f['path'],passed=hashlib.sha256(z.read(f['path'])).hexdigest()==f['sha256']))
pd.DataFrame(validation).to_csv(OUT/'manifest_validation.csv',index=False)
(ROOT/'archive_sha256.txt').write_text(hashlib.sha256(Z.read_bytes()).hexdigest()+'  '+str(Z)+'\n')
runs=[]; fits=[]; checks=[]; summary=[]; validations=[]
for n in z.namelist():
 if not n.endswith('/run.csv'):continue
 base=n.rsplit('/',1)[0]; run=pd.read_csv(z.open(n)).iloc[0].to_dict();runs.append(run); key=run['run_key']
 m=pd.read_csv(z.open(base+'/metrics.csv')); c=pd.read_csv(z.open(base+'/checks.csv')); checks.append(c)
 a=np.load(io.BytesIO(z.read(base+'/probe_curves.npz')))
 for name in a.files:
  if not name.endswith('_losses'):continue
  prefix=name[:-7]; check=int(prefix.split('/')[0].split('_')[1]); arm=prefix.split('/')[1]
  for rep,y in enumerate(a[name]):
   peak=int(np.argmax(y)); recovered=np.flatnonzero(y[peak+1:]<=y[0]); recovery=peak+1+int(recovered[0]) if len(recovered) else np.nan
   fits.append(dict(run_key=key,environment=run['environment'],architecture=run['architecture'],check=check,arm=arm,rep=rep,initial=y[0],peak=y[peak],amplification=y[peak]/y[0],peak_index=peak,recovery_index=recovery,final_minibatch=y[-1],final_pool=a[prefix+'_final_loss'][rep],excess_sum=np.maximum(y-y[0],0).sum()))
  validations.append(dict(run_key=key,field=prefix+'_score',max_error=float(np.max(np.abs(a[prefix+'_score']-(a[prefix+'_b']-a[prefix+'_final_loss']))))))
 e=m.dropna(subset=['eval/avg_return']).sort_values('interaction_step'); y=e['eval/avg_return'].to_numpy();t=e.interaction_step.to_numpy();late=y[t>=.9*t.max()]
 summary.append(dict(run_key=key,environment=run['environment'],architecture=run['architecture'],seed=run['seed'],eval_count=len(y),final_return=y[-1],final_window_mean=late.mean(),peak_return=y.max(),peak_step=t[y.argmax()],peak_to_final=y.max()-y[-1],late_to_peak_ratio=late.mean()/y.max(),observed_auc=np.trapezoid(y,t),observed_end=t.max(),initial_fresh=run['initial_fresh_score_iqm'],final_deficit=run['final_loss_iqm']))
 for cols,label,symlog in [(['eval/avg_return'],'return',False),(['train/q1_mean'],'q',True),(['train/critic_loss'],'td_loss',True),(['train/actor_loss','train/policy_kl','train/policy_churn'],'actor',True)]:
  fig,ax=plt.subplots(figsize=(7,3.5))
  for col in cols:
   s=m.dropna(subset=[col]);ax.plot(s.interaction_step,s[col],label=col,lw=.8)
  if symlog:ax.set_yscale('symlog')
  ax.set(xlabel='Interaction steps',ylabel=label,title=key+'; n=1 training seed');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(FIG/(key+'_'+label+'.pdf'));plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,3.5))
 for arm in ['fresh','current']:
  for k in [0,1,10,20]:
   name=f'check_{k:02d}/{arm}_losses'
   if name not in a:continue
   for rep,row in enumerate(a[name]):ax.plot(np.arange(len(row)),row,alpha=.35,lw=.6,label=f'{arm} check {k}' if rep==0 else None)
 ax.set(yscale='log',xlabel='Fitting update index (pre-update minibatch loss)',ylabel='MSE',title=key+'; five repetitions, one seed');ax.legend(fontsize=6);fig.tight_layout();fig.savefig(FIG/(key+'_fitting.pdf'));plt.close(fig)
for name,rows in [('run_inventory',runs),('fit_dynamics',fits),('performance',summary),('score_validation',validations)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
pd.concat(checks).to_csv(OUT/'all_checks.csv',index=False)
f=pd.DataFrame(fits)
f.groupby(['run_key','arm']).agg(median_amplification=('amplification','median'),max_amplification=('amplification','max'),median_final_pool=('final_pool','median'),median_peak_index=('peak_index','median')).to_csv(OUT/'fit_summary.csv')
print(pd.DataFrame(summary).to_string(index=False));print(f.groupby(['run_key','arm']).amplification.median().to_string()); print('Manifest',all(x['passed'] for x in validation))
