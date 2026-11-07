"""Re-runs the production probe code (experiments.exp12.probe.run_probe) on a saved warm-up replay."""
import os, sys, time
from pathlib import Path
MAIN = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(MAIN), str(MAIN / "scripts")]
import numpy as np, jax, jax.numpy as jnp, flax.serialization
from experiments.exp12 import probe, trigger
from experiments.exp1 import compose_config
from experiments.exp12.run_probes import original_critic_def
from scale_rl.agents.sac.sac_network import SACCritic

DATA = Path(os.environ["EXP12_SCI_DATA"])  # outside the repository
ENVS = {"hopper": ("hopper-hop", "dmc_medium"), "dog": ("dog-run", "dmc_hard")}

class Buffer:
    def __init__(self, d):
        self._num_in_buffer = int(d["num"]); self._observations = d["observations"]; self._actions = d["actions"]

class Agent:
    def __init__(self, d):
        self.mean, self.var, self.eps = d["rms_mean"], d["rms_var"], d["epsilon"]
    def _normalize(self, o):  # scale_rl/agents/wrappers/normalization.py
        return (o - self.mean) / np.sqrt(self.var + self.eps)

def setup(env, blocks, width, pool=None, overrides=()):
    name, group = ENVS[env]
    cfg = compose_config(str(MAIN / "configs"), "base_exp12", [f"env_name={name}", f"env={group}", "seed=990", "run_role=dev",
                         f"critic_num_blocks={blocks}", f"critic_hidden_dim={width}", *overrides])
    d = np.load(DATA / f"{env}_D2W512.npz")
    pc = probe.probe_config(cfg)
    if pool is not None:
        import math
        pc = probe.ProbeConfig(pc.rounds, pc.steps, pc.checks, pool, pc.batch_size, pc.target_scale, math.gcd(pool, pc.eval_chunk))
    return cfg, Agent(d), Buffer(d), original_critic_def(cfg), probe.critic_optimizer(cfg.agent), pc, trigger.trigger_config(cfg)

def init_params(net, key, d_obs, d_act):  # as probe_fresh_checks.null_mode
    return net.init(key, jnp.zeros((1, d_obs)), jnp.zeros((1, d_act)))["params"]

def trainer_params(env, blocks, width, net, d_obs, d_act):
    f = DATA / f"{env}_D{blocks}W{width}_critic.msgpack"
    tmpl = init_params(net, jax.random.PRNGKey(0), d_obs, d_act)
    return flax.serialization.from_bytes(tmpl, f.read_bytes())
