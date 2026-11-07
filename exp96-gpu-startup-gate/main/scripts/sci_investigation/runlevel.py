"""Illustrative run-level fork probability of the approved trigger (2 consecutive firing checks among 1..19) under
a normal model L_rk = mu + e_rk: mu persistent per run (SD tau), e iid N(0,1). Exact 5^5 bootstrap enumeration
(agrees with the production 10,000-resample function on 400/400 random inputs)."""
import itertools, json
import numpy as np
IDX = np.array(list(itertools.product(range(5), repeat=5)))
def fires(x):  # x: (n, 5) -> bool (n,)
    s = np.sort(x[:, IDX], axis=2)[:, :, 1:4].mean(2)
    return np.percentile(s, 2.5, axis=1) > 0
def run_level(tau, runs=600, checks=19, seed=0):
    g = np.random.default_rng(seed)
    mu = g.normal(size=(runs, 1, 1)) * tau
    f = fires((mu + g.normal(size=(runs, checks, 5))).reshape(-1, 5)).reshape(runs, checks)
    two = (f[:, 1:] & f[:, :-1]).any(1)
    return float(f.mean()), float(two.mean())
out = {}
for tau in (0.0, 0.3, 0.6, 1.0):
    per_check, run2 = run_level(tau)
    out[f"tau/sigma_w={tau}"] = {"per_check_fire": round(per_check, 4), "run_level_two_consecutive": round(run2, 4)}
print(json.dumps(out, indent=1))
