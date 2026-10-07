import numpy as np,json
from pathlib import Path
r=Path('/tmp/exp12-followup-evidence/twin_current_1');a=dict(np.load(r/'twin_arrays.npz'));flat=dict(np.load(r/'twin_params.npz'))
params={}
for name,val in flat.items():
 d=params
 parts=name.split('/')
 for key in parts[:-1]:d=d.setdefault(key,{})
 d[parts[-1]]=val.astype(np.float64)
def sliced(t,k):return {name:sliced(x,k) if isinstance(x,dict) else x[k] for name,x in t.items()}
ps=[sliced(params['VmapSACCritic_0'],k) for k in range(2)]
# Independent float64 NumPy forward and analytic Jacobian, for the same float32 input/parameter values.
def dense(x,j,p):return x@p['kernel']+p['bias'],np.einsum('nha,hk->nka',j,p['kernel'])
def norm(x,j,p):
 mean=x.mean(-1,keepdims=True);z=x-mean
 var=(x*x).mean(-1,keepdims=True)-mean*mean
 inv=1/np.sqrt(np.maximum(var,0)+1e-6)
 dz=j-j.mean(1,keepdims=True)
 dvar=2*np.mean(z[:,:,None]*j,axis=1,keepdims=True)
 deriv=dz*inv[:,:,None]-0.5*z[:,:,None]*inv[:,:,None]**3*dvar
 return z*inv*p['scale']+p['bias'],deriv*p['scale'][None,:,None]
def one(p,obs,act):
 x=np.concatenate((obs,act),axis=1).astype(float);j=np.zeros((len(x),x.shape[1],act.shape[1]));j[:,obs.shape[1]:,:]=np.eye(act.shape[1])
 x,j=dense(x,j,p['encoder']['Dense_0'])
 for k in range(2):
  block=p['encoder'][f'ResidualBlock_{k}'];skip,dskip=x,j
  x,j=norm(x,j,block['LayerNorm_0']);x,j=dense(x,j,block['Dense_0']);j=j*(x>0)[:,:,None];x=np.maximum(x,0);x,j=dense(x,j,block['Dense_1']);x,j=x+skip,j+dskip
 x,j=norm(x,j,p['encoder']['LayerNorm_0']);x,j=dense(x,j,p['predictor']['Dense_0']);return x[:,0],j[:,0,:]
def pair(obs,act):
 out=[one(p,obs,act) for p in ps];q=np.stack([x[0] for x in out]);jac=np.stack([x[1] for x in out]);which=np.argmin(q,axis=0);return q,jac[which,np.arange(len(obs)),:]
q64,g64=pair(a['observation'],a['action']);ref=q64.reshape(-1)
summary={}
for key,expected in [('q_panel',ref),('q_eager',ref),('q_sliced',ref),('g_panel',g64),('g_eager',g64),('g_sliced',g64)]:
 d=np.abs(a[key].astype(float)-expected);summary[key]={'max_abs_vs_numpy64':float(d.max()),'max_abs_over_ref_max':float(d.max()/np.abs(expected).max())}
summary['min_Q_gap_numpy64']=float(np.min(np.abs(q64[0]-q64[1])))
summary['min_selection_differences_panel_vs_numpy']=int(np.sum(np.argmin(a['q_panel'].reshape(2,-1),axis=0)!=np.argmin(q64,axis=0)))
summary['min_selection_differences_eager_vs_numpy']=int(np.sum(np.argmin(a['q_eager'].reshape(2,-1),axis=0)!=np.argmin(q64,axis=0)))
# Independent finite differences validate the analytic derivative; no production dtype/precision changed.
for h in [1e-4,1e-5,1e-6]:
 fd=[]
 for j in range(3):
  ap=a['action'].astype(float).copy();am=ap.copy();ap[:,j]+=h;am[:,j]-=h
  qp,_=pair(a['observation'],ap);qm,_=pair(a['observation'],am);fd.append((qp.min(0)-qm.min(0))/(2*h))
 fd=np.stack(fd,axis=1);summary[f'finite_difference_h_{h}']={'max_abs_vs_analytic':float(np.max(np.abs(fd-g64)))}
np.savez(r/'numpy64_reference.npz',q=q64,g=g64)
(r/'numpy64_reference.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
