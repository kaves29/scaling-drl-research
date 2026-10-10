import numpy as np, jax
from lab import *
cfg, agent, buf, net, tx, pc, tc = setup("dog", 1, 32)
d_o, d_a = buf._observations.shape[1], buf._actions.shape[1]
a, b = (init_params(net, k, d_o, d_a) for k in jax.random.split(jax.random.PRNGKey(10_000), 2))
same = probe.summarize(probe.run_probe(agent, buf, net, {"current": (net, a), "fresh": (net, a)}, tx, 990, 1, pc))
ab = probe.summarize(probe.run_probe(agent, buf, net, {"current": (net, b), "fresh": (net, a)}, tx, 990, 1, pc))
ba = probe.summarize(probe.run_probe(agent, buf, net, {"current": (net, a), "fresh": (net, b)}, tx, 990, 1, pc))
print("identical inits: L == 0 exactly:", bool(np.all(same["loss_rounds"] == 0)), " fires:", trigger.bootstrap_interval(same["loss_rounds"], 990, 1, 10000, .95)[0] > 0)
print("swap negates L exactly:", bool(np.array_equal(ab["loss_rounds"], -ba["loss_rounds"])), ab["loss_rounds"])
