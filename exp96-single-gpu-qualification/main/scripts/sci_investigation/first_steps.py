"""Approved fit, first K AdamW steps only (same _fit internals): pool loss and output spread after each step."""
import sys, json
import numpy as np, jax, jax.numpy as jnp, optax
from lab import *

env, blocks, width, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cfg, agent, buf, net, tx, pc, tc = setup(env, blocks, width)
d_obs, d_act = buf._observations.shape[1], buf._actions.shape[1]
f = DATA / f"{env}_D{blocks}W{width}_critic.msgpack"
params = (trainer_params(env, blocks, width, net, d_obs, d_act) if f.exists()
          else init_params(net, jax.random.PRNGKey(10_000), d_obs, d_act))
obs, act = probe.sample_pool(agent, buf, probe.probe_rng(990, 0, 0), pc.pool_size)
tk, ik = jax.random.split(probe.probe_key(990, 0, 0))
base = probe._base_targets(net, tk, obs, act, pc.eval_chunk, pc.target_scale)
idx = jax.random.randint(ik, (pc.steps, pc.batch_size), 0, pc.pool_size)
sub = slice(0, 2560)
pred = jax.jit(lambda p: net.apply({"params": p}, obs[sub], act[sub]).reshape(-1))
q0 = pred(params)
offset = jnp.mean(probe._chunked(lambda o, a: net.apply({"params": params}, o, a).reshape(-1), obs, act, pc.eval_chunk))
targets = offset + base


def loss_fn(p, bi):
    return jnp.mean((net.apply({"params": p}, obs[bi], act[bi]).reshape(-1) - targets[bi]) ** 2)


@jax.jit
def step(p, s, bi):
    l, g = jax.value_and_grad(loss_fn)(p, bi)
    u, s = tx.update(g, s, p)
    return optax.apply_updates(p, u), s, l, optax.global_norm(g), optax.global_norm(u)


s, p, rows = tx.init(params), params, []
for k in range(K):
    p, s, l, gn, un = step(p, s, idx[k])
    q = pred(p)
    rows.append({"step": k + 1, "minibatch_loss_before": round(float(l), 4), "grad_norm": round(float(gn), 4),
                 "update_norm": round(float(un), 4), "q_sd": round(float(jnp.std(q)), 4),
                 "q_mean_shift": round(float(jnp.mean(q - q0)), 4),
                 "pool_mse_2560": round(float(jnp.mean((q - targets[sub]) ** 2)), 4)})
print(json.dumps({"env": env, "arch": f"D{blocks}W{width}",
                  "params": int(sum(x.size for x in jax.tree_util.tree_leaves(params))),
                  "q0_sd": round(float(jnp.std(q0)), 4), "b": round(float(jnp.var(base)), 4), "steps": rows}))
