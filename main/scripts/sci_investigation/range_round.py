"""One round of the range-mode probe (seed 990, check 0), identical to run_probe's loop body for round r."""
import sys, time, json
import numpy as np, jax
from lab import *
env, blocks, width, r, pool = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
cfg, agent, buf, net, tx, pc, tc = setup(env, blocks, width, pool)
params = trainer_params(env, blocks, width, net, buf._observations.shape[1], buf._actions.shape[1])
obs, act = probe.sample_pool(agent, buf, probe.probe_rng(990, 0, r), pc.pool_size)
t = time.time()
out = jax.device_get(probe.probe_round(net, {"fresh": (net, params)}, tx, obs, act, probe.probe_key(990, 0, r), pc))["fresh"]
dt = time.time() - t
tag = f"{env}_D{blocks}W{width}_pool{pool}_r{r}"
np.savez(os.path.join(os.environ["EXP12_SCI_OUT"], f"range_{tag}.npz"), **out)
print(json.dumps({"tag": tag, "seconds": round(dt, 1), "b": float(out["b"]), "final": float(out["final_loss"]),
                  "score": float(out["score"]), "P_over_b": float(out["score"] / out["b"]), "offset": float(out["offset"]),
                  "loss_first": float(out["losses"][:10].mean()), "loss_last": float(out["losses"][-50:].mean())}), flush=True)
