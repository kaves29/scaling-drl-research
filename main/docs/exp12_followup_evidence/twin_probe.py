import sys,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from tests.test_exp12_twin_critic import _twin_agent
from tests.test_exp12_diagnostics import OBS
from experiments.exp12 import fork
from scale_rl.agents.sac.sac_network import SACCritic
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
g=_twin_agent();rng=np.random.default_rng(0)
panel={'observation':rng.normal(size=(256,OBS)).astype(np.float32),'action':rng.uniform(-1,1,(256,3)).astype(np.float32)}
obs,act=map(jnp.asarray,(panel['observation'],panel['action']))
# The local twin adapter permits the earlier single-only fork implementation to be investigated without editing it.
q_fn=lambda a:g.critic.network_def.apply({'params':g.critic.params},obs,a)
with jax.default_matmul_precision('highest'):
 eager_q=q_fn(act);eager_g=jax.grad(lambda a:jnp.minimum(*q_fn(a)).sum())(act)
 compiled_q,compiled_g=jax.jit(lambda a:(q_fn(a),jax.grad(lambda b:jnp.minimum(*q_fn(b)).sum())(a)))(act)
 separate_q=jax.jit(q_fn)(act);separate_g=jax.jit(jax.grad(lambda a:jnp.minimum(*q_fn(a)).sum()))(act)
 # Independently sliced single networks, composed before JIT, check axis/min semantics.
 single=SACCritic('residual',2,8,jnp.float32)
 ps=[jax.tree.map(lambda x:x[k],g.critic.params['VmapSACCritic_0']) for k in range(2)]
 manual=lambda a:jnp.stack([single.apply({'params':p},obs,a) for p in ps])
 sliced_q,sliced_g=jax.jit(lambda a:(manual(a),jax.grad(lambda b:jnp.minimum(*manual(b)).sum())(a)))(act)
 if hasattr(fork,'is_twin'):
  panel_out=fork.panel_q_and_grad(g.critic,panel)
 else:panel_out={'q':np.asarray(compiled_q).reshape(-1),'dq_da':np.asarray(compiled_g)}
arrays={'q_eager':np.asarray(eager_q).reshape(-1),'q_panel':panel_out['q'],'g_eager':np.asarray(eager_g),'g_panel':panel_out['dq_da'],'q_separate_jit':np.asarray(separate_q).reshape(-1),'g_separate_jit':np.asarray(separate_g),'q_sliced':np.asarray(sliced_q).reshape(-1),'g_sliced':np.asarray(sliced_g),**panel}
rows=[];summaries={}
for field,prefix in [('q','q'),('dq_da','g')]:
 expected=arrays[prefix+'_eager'];observed=arrays[prefix+'_panel'];d=np.abs(observed.astype(float)-expected.astype(float));rel=np.divide(d,np.abs(expected),out=np.full_like(d,np.inf),where=expected!=0);rel[(expected==0)&(observed==0)]=0
 tol=np.zeros_like(d) if field=='q' else 1e-7+1e-5*np.abs(expected)
 fail=d>tol
 for index in np.ndindex(expected.shape):
  if expected[index]!=observed[index]:rows.append({'field':field,'index':index,'expected_eager':float(expected[index]),'observed_panel':float(observed[index]),'expected_hex':float(expected[index]).hex(),'observed_hex':float(observed[index]).hex(),'abs_diff':float(d[index]),'rel_diff':float(rel[index]),'existing_bound':float(tol[index]),'fails_existing_criterion':bool(fail[index])})
 def maxrow(a):
  idx=np.unravel_index(np.argmax(a),a.shape);return {'index':idx,'expected':float(expected[idx]),'observed':float(observed[idx]),'abs':float(d[idx]),'relative':float(rel[idx]),'existing_bound':float(tol[idx])}
 summaries[field]={'different':int(np.sum(d!=0)),'outside_existing_criterion':int(fail.sum()),'max_abs':maxrow(d),'max_relative':maxrow(rel),'panel_equals_separate_jit':np.array_equal(observed,arrays[prefix+'_separate_jit']),'panel_equals_sliced_single_reference':np.array_equal(observed,arrays[prefix+'_sliced'])}
np.savez(out/'twin_arrays.npz',**arrays)
# Save float32 parameters as binary evidence for an independent NumPy reference.
flat={}
def flatten(t,path=''):
 for k,v in t.items():
  name=f'{path}/{k}' if path else k
  if hasattr(v,'items'):flatten(v,name)
  else:flat[name]=np.asarray(v)
flatten(g.critic.params);np.savez(out/'twin_params.npz',**flat)
(out/'twin.json').write_text(json.dumps({'module':sys.modules['scale_rl.agents.sac.sac_network'].__file__,'jax':jax.__version__,'backend':jax.default_backend(),'summaries':summaries,'differences':rows},indent=2,default=lambda x:int(x)))
print(json.dumps(summaries,indent=2,default=lambda x:int(x)))
