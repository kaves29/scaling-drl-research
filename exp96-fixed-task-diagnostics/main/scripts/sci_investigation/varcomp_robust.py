"""Robust init-effect tests on the init x round final-loss matrix: Friedman (ranks within round) and log-scale ANOVA."""
import json, sys
import numpy as np
from scipy import stats
for path in sys.argv[1:]:
    f = np.load(path)["final"]
    n, r = f.shape
    fr = stats.friedmanchisquare(*f)  # each init's 5 rounds, blocked by round
    lf = np.log(f); g = lf.mean(); a = lf.mean(1) - g; rr = lf.mean(0) - g; e = lf - g - a[:, None] - rr[None, :]
    ms_a = r * np.sum(a ** 2) / (n - 1); ms_e = np.sum(e ** 2) / ((n - 1) * (r - 1))
    print(json.dumps({"file": path.rsplit("/", 1)[-1], "friedman_p": round(float(fr.pvalue), 4),
                      "log_anova_F": round(float(ms_a / ms_e), 3), "log_anova_p": round(float(stats.f.sf(ms_a / ms_e, n - 1, (n - 1) * (r - 1))), 4),
                      "median_final_by_init": np.round(np.median(f, 1), 6).tolist(),
                      "single_round_outliers_(>5x_round_median)": int((f > 5 * np.median(f, 0)[None, :]).sum())}))
