from pathlib import Path
import pandas as pd,numpy as np
R=Path(__file__).resolve().parent;D=R/'derived';P=pd.read_csv(D/'probe_trajectories.csv');rows=[]
for rid,d in P.groupby('id'):
 early=d[d.check==1].iloc[0];late=d[d.check==35].iloc[0];r={k:late[k] for k in ['id','environment','architecture','seed']}
 for arm in ['fresh','current']:
  r[arm+'_early_mean_score']=np.mean([early[f'probe/score_{arm}_r{k}'] for k in range(10)])
  r[arm+'_70_mean_score']=np.mean([late[f'probe/score_{arm}_r{k}'] for k in range(10)])
  r[arm+'_change']=r[arm+'_70_mean_score']-r[arm+'_early_mean_score']
 r['mean_deficit_change']=r['fresh_change']-r['current_change'];rows.append(r)
pd.DataFrame(rows).to_csv(D/'reference_decomposition.csv',index=False)
print(pd.DataFrame(rows).groupby(['environment','architecture'])[['fresh_change','current_change','mean_deficit_change']].mean().to_string())
