import json,sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scale_rl.agents.sac.sac_network import SACCritic,SACClippedDoubleCritic
out=Path('/tmp/exp12-followup-evidence')
a=np.load(out/'twin_current_1/twin_arrays.npz')
flat=np.load(out/'twin_current_1/twin_params.npz')
p={}
for key in flat.files:
 node=p
 parts=key.split('/')
 for part in parts[:-1]:node=node.setdefault(part,{})
 node[parts[-1]]=jnp.asarray(flat[key])
obs,act=jnp.asarray(a['observation']),jnp.asarray(a['action'])
net=SACClippedDoubleCritic('residual',2,8,jnp.float32)
try:
 net.init(jax.random.PRNGKey(0),observations=obs,actions=act)
 init={'success':True}
except Exception as e:init={'success':False,'type':type(e).__name__,'message':str(e)}
single=SACCritic('residual',2,8,jnp.float32)
ps=[jax.tree.map(lambda x:x[k],p['VmapSACCritic_0']) for k in range(2)]
q=lambda x:jnp.stack([single.apply({'params':s},obs,x) for s in ps])
with jax.default_matmul_precision('highest'):
 eq=q(act);eg=jax.grad(lambda x:jnp.minimum(*q(x)).sum())(act)
 cq,cg=jax.jit(lambda x:(q(x),jax.grad(lambda y:jnp.minimum(*q(y)).sum())(x)))(act)
eq,eg,cq,cg=map(np.asarray,(eq,eg,cq,cg))
np.savez(out/'pre_exp12_single_pair.npz',eager_q=eq,eager_g=eg,jit_q=cq,jit_g=cg)
r={'source':sys.modules['scale_rl.agents.sac.sac_network'].__file__,'original_twin_initialization':init,'equivalent_single_pair':{'q_different':int(np.sum(eq!=cq)),'q_max_abs':float(np.max(np.abs(eq.astype(float)-cq))),'g_different':int(np.sum(eg!=cg)),'g_max_abs':float(np.max(np.abs(eg.astype(float)-cg))),'g_outside_original_criterion':int(np.sum(np.abs(eg.astype(float)-cg)>1e-7+1e-5*np.abs(eg)))},'matches_current_independent_sliced_jit_q':np.array_equal(cq.reshape(-1),a['q_sliced']),'matches_current_independent_sliced_jit_g':np.array_equal(cg,a['g_sliced'])}
(out/'pre_exp12_twin_probe.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
