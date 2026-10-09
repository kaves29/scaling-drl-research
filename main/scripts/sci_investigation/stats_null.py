"""Exact operating characteristics of the approved per-check rule (5 rounds, IQM, percentile, lower bound > 0)."""
import itertools, json, math, sys
import numpy as np
from scipy import stats
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from experiments.exp12.trigger import bootstrap_interval

# Exact bootstrap distribution of the 5-sample trimmed mean: all 5^5 ordered resamples, equally likely.
IDX = np.array(list(itertools.product(range(5), repeat=5)))
def exact_low(x):
    s = np.sort(x[IDX], axis=1)[:, 1:4].mean(1)
    return np.percentile(s, 2.5)  # population quantile with the same linear interpolation as the code

rng = np.random.default_rng(1)
# 1. Agreement of production (10,000 resamples) with the exact enumeration on random inputs.
agree = sum((bootstrap_interval(x, 990, k, 10000, 0.95)[0] > 0) == (exact_low(x) > 0)
            for k, x in enumerate(rng.normal(size=(400, 5))))
# 2. IID symmetric null, production function.
def rate(tau, n=4000, seed=2):
    g = np.random.default_rng(seed)
    x = g.normal(size=(n, 1)) * tau + g.normal(size=(n, 5))
    return np.mean([bootstrap_interval(v, 990, k, 10000, 0.95)[0] > 0 for k, v in enumerate(x)]), x
r0, x0 = rate(0.0)
iqm_sd0 = np.std(np.sort(x0, 1)[:, 1:4].mean(1))
rows = []
for tau in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
    r, x = rate(tau, 3000, 3)
    rows.append({"tau_over_sigma_w": tau, "fire_rate": round(float(r), 4),
                 "sd_pair_iqm_over_sigma_w": round(float(np.std(np.sort(x, 1)[:, 1:4].mean(1))), 4)})
# 3. Binomial / homogeneity on the reported counts.
p0 = float(r0)
counts = {"D2W512": 3, "D4W1024": 13, "D4W1536": 5}
tab = np.array([[c, 100 - c] for c in counts.values()])
chi2, pchi, _, _ = stats.chi2_contingency(tab)
out = {
    "production_vs_exact_agreement": f"{agree}/400",
    "iid_null_fire_rate_production": p0,
    "iid_null_sd_of_pair_iqm_over_sigma_w": float(iqm_sd0),
    "P(X>5 | n=100, p=iid rate)": float(stats.binom.sf(5, 100, p0)),
    "P(at least one of 3 sizes > 5/100 | iid)": float(1 - stats.binom.cdf(5, 100, p0) ** 3),
    "P(X>=13 | n=100, p=iid rate)": float(stats.binom.sf(12, 100, p0)),
    "P(X>=13 | n=100, p=0.05)": float(stats.binom.sf(12, 100, 0.05)),
    "clopper_pearson_95_D4W1024": [float(stats.beta.ppf(0.025, 13, 88)), float(stats.beta.ppf(0.975, 14, 87))],
    "homogeneity_chi2_p_3_13_5": float(pchi),
    "fisher_13_vs_5": float(stats.fisher_exact([[13, 87], [5, 95]])[1]),
    "fisher_13_vs_3": float(stats.fisher_exact([[13, 87], [3, 97]])[1]),
    "pooled_rate_21_of_300": 21 / 300,
    "P(X>=13 | n=100, p=0.07)": float(stats.binom.sf(12, 100, 0.07)),
    "pair_effect_model": rows,
}
print(json.dumps(out, indent=1))
