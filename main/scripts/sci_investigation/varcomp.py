"""Variance components of the probe score across fresh initialisations (null-mode inits), on shared rounds.

Every init is fitted on the same 5 rounds (pools, teacher targets, minibatch order of one check), with the
production probe_round/_fit. Score P[i, r] = b_r - final_loss[i, r]. For any two inits the null pair's
per-round L is P[i, :] - P[j, :], which is exactly what null_mode computes for a pair at that check index.
A persistent init effect (an init that fits worse on every round) is what a rounds-only bootstrap reads as a
real difference.
"""
import sys, json, time
import numpy as np, jax
from lab import *

env, blocks, width, n_pairs, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
CHECK = 1
cfg, agent, buf, net, tx, pc, tc = setup(env, blocks, width)
d_obs, d_act = buf._observations.shape[1], buf._actions.shape[1]
inits = []
for pair in range(n_pairs):  # probe_fresh_checks.null_mode keys: fresh = keys[0], current = keys[1]
    keys = jax.random.split(jax.random.PRNGKey(10_000 + pair), 2)
    inits += [init_params(net, k, d_obs, d_act) for k in keys]
n = len(inits)
final = np.full((n, pc.rounds), np.nan); b = np.full(pc.rounds, np.nan); curves = np.full((n, pc.rounds, pc.steps), np.nan)
for r in range(pc.rounds):
    obs, act = probe.sample_pool(agent, buf, probe.probe_rng(990, CHECK, r), pc.pool_size)
    for i, p in enumerate(inits):
        t = time.time()
        res = jax.device_get(probe.probe_round(net, {"x": (net, p)}, tx, obs, act, probe.probe_key(990, CHECK, r), pc))["x"]
        final[i, r], b[r], curves[i, r] = res["final_loss"], res["b"], res["losses"]
        np.savez(out, final=final, b=b, curves=curves)
        print(json.dumps({"round": r, "init": i, "final": float(res["final_loss"]), "s": round(time.time() - t, 1)}), flush=True)
print("VARCOMP_DONE", flush=True)
