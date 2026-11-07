"""Independent arithmetic audits against raw API JSON and archived arrays."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;D=R/'derived';checks=[]
def check(name,passed,detail):checks.append({'check':name,'passed':bool(passed),'detail':detail})
I=pd.read_csv(D/'run_inventory.csv');V=pd.read_csv(D/'data_validation.csv');C=I[I.cohort=='exp96'];S=pd.read_csv(D/'seed_endpoints.csv');P=pd.read_csv(D/'probe_trajectories.csv');Q=pd.read_csv(D/'probe_score_validation.csv')
check('record_counts',len(I)==103 and len(C)==96 and (I.state=='crashed').sum()==8,'103 records, 96 campaign, eight crashed')
check('complete_canonical_grid',len(C.groupby(['environment','architecture','seed']))==96 and set(C.seed)==set(range(1,9)),'4 environments × 3 architectures × seeds1–8')
check('full_history_counts',V.history_count_matches.all(),str(V.history_count_matches.sum())+'/103')
check('step_integrity',V.unique_steps.all() and V.monotone.all(),'No duplicate or reversed logging steps')
check('source_fractional_IQM_reconstruction',Q.iqm_recompute_error.max()==0,str(len(Q.dropna()))+' nonzero-check scores exactly reproduced')
errors=[]
for _,r in C[C.state=='finished'].iterrows():
 raw=[json.loads(x) for x in (R/'raw_api'/r.id/'history.jsonl').read_text().splitlines()];last=next(x['eval/avg_return'] for x in reversed(raw) if 'eval/avg_return'in x);derived=S[S.id==r.id].final_return.iloc[0];errors.append(abs(last-derived))
check('independent_final_return_recalculation',max(errors)<1e-10,'max discrepancy '+str(max(errors)))
identity=pd.read_csv(D/'pilot_campaign_trajectory_identity.csv');check('pilot_overlap_detected',(identity.shared_values==identity.exact_matches).all(),'Seven exact observational overlaps; pilot not pooled')
a=np.load(next((R/'raw_archived').glob('*.npz')));rid=C[(C.environment=='myo-pen-twirl')&(C.architecture=='D2W512')&(C.seed==5)].id.iloc[0];p=P[(P.id==rid)&(P.check==50)].iloc[0]
for arm in ['fresh','current']:
 delta=np.max(np.abs(a[arm+'_score']-np.array([p[f'probe/score_{arm}_r{k}'] for k in range(10)])))
 check('archived_curve_'+arm+'_WandB_binding',delta<1e-7,'max score error '+str(delta))
 check('archived_score_identity_'+arm,np.array_equal(a[arm+'_b']-a[arm+'_final_loss'],a[arm+'_score']),'FP32 baseline minus pool-MSE identity')
manifest=json.loads((R/'raw_api_manifest.json').read_text());check('raw_manifest_integrity',all(hashlib.sha256((R/'raw_api'/f['path']).read_bytes()).hexdigest()==f['sha256'] for f in manifest),str(len(manifest))+' immutable files')
(R/'VALIDATION.json').write_text(json.dumps({'checks':checks,'all_passed':all(x['passed'] for x in checks)},indent=2));print(json.dumps(checks,indent=2));assert all(x['passed'] for x in checks)
