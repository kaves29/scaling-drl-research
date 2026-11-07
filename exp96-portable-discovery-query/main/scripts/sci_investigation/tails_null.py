"""Per-check fire rate of the approved rule (exact 5^5 bootstrap, IQM, 2.5% lower bound > 0) for IID symmetric,
zero-centred round losses with different tails. No pair effect anywhere."""
import itertools, json
import numpy as np
IDX = np.array(list(itertools.product(range(5), repeat=5)))
def rate(x):
    s = np.sort(x[:, IDX], axis=2)[:, :, 1:4].mean(2)
    return float((np.percentile(s, 2.5, axis=1) > 0).mean())
g = np.random.default_rng(7); n = 8000
dists = {
    "normal": g.normal(size=(n, 5)),
    "laplace": g.laplace(size=(n, 5)),
    "student_t3": g.standard_t(3, size=(n, 5)),
    "student_t1.5": g.standard_t(1.5, size=(n, 5)),
    "cauchy": g.standard_cauchy(size=(n, 5)),
    "contaminated_13pct_x10": g.normal(size=(n, 5)) * np.where(g.random((n, 5)) < 0.13, 10, 1),
    "difference_of_lognormal_sd1.5": np.exp(1.5 * g.normal(size=(n, 5))) - np.exp(1.5 * g.normal(size=(n, 5))),
}
print(json.dumps({k: round(rate(v), 4) for k, v in dists.items()}, indent=1))
