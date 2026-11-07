"""Save the real warm-up replay (seed 990, min_length random transitions) exactly as probe_fresh_checks builds it."""
import sys, types
from pathlib import Path
MAIN = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(MAIN), str(MAIN / "scripts")]
import numpy as np
import probe_fresh_checks as pfc

env, group, out, blocks, width = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
pfc._stub_wandb()
args = types.SimpleNamespace(env=env, env_group=group, override=[])
t = pfc._fresh_trainer(args, blocks, width, 990)
n = t.buffer._num_in_buffer
np.savez(out, observations=t.buffer._observations[:n], actions=t.buffer._actions[:n],
         rms_mean=np.asarray(t.agent.obs_rms.mean), rms_var=np.asarray(t.agent.obs_rms.var),
         epsilon=np.asarray(t.agent.epsilon), num=n,
         obs_norm=np.asarray(t.agent._normalize(t.buffer._observations[:n])))
import flax.serialization, pickle
open(out.replace('.npz','_critic.msgpack'),'wb').write(flax.serialization.to_bytes(t._sac_agent.critic.params))
print(env, n, t.buffer._observations.shape, t.buffer._actions.shape)
t.close()
