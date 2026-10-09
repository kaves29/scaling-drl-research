"""Init x round decomposition of the fresh-critic probe score; firing of every init pair under the approved rule."""
import itertools, json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from experiments.exp12.trigger import bootstrap_interval
from experiments.exp12.probe import iqm

for path in sys.argv[1:]:
    d = np.load(path)
    final, b = d["final"], d["b"]
    done = ~np.isnan(final).any(0)
    final, b = final[:, done], b[done]
    n, r = final.shape
    score = (b[None, :].astype(np.float32) - final.astype(np.float32)).astype(np.float64)
    grand = score.mean()
    a = score.mean(1) - grand
    rr = score.mean(0) - grand
    resid = score - grand - a[:, None] - rr[None, :]
    ms_init = r * np.sum(a ** 2) / (n - 1)
    ms_res = np.sum(resid ** 2) / ((n - 1) * (r - 1))
    var_init = max((ms_init - ms_res) / r, 0.0)
    f_stat = ms_init / ms_res
    from scipy import stats
    p_init = float(stats.f.sf(f_stat, n - 1, (n - 1) * (r - 1)))
    fires, mirror, pairs = 0, 0, 0
    rows = []
    for i, j in itertools.combinations(range(n), 2):
        loss = (score[i] - score[j]).astype(np.float32)  # fresh = i, current = j
        lo, hi = bootstrap_interval(loss, 990, 1, 10000, 0.95)
        pairs += 1; fires += lo > 0; mirror += hi < 0
        rows.append({"pair": [i, j], "L_iqm": round(iqm(loss), 6), "positive_rounds": int((loss > 0).sum()), "fires_low>0": bool(lo > 0), "fires_high<0": bool(hi < 0)})
    out = {"file": path.rsplit("/", 1)[-1], "inits": n, "rounds_complete": r,
           "final_loss_by_init_mean": np.round(final.mean(1), 6).tolist(),
           "final_loss_rank_by_round": [list(np.argsort(final[:, k])) for k in range(r)],
           "sd_init_effect": float(np.sqrt(var_init)), "sd_residual": float(np.sqrt(ms_res)),
           "tau_over_sigma_w": float(np.sqrt(var_init / ms_res)) if ms_res > 0 else None,
           "F_init": float(f_stat), "p_init_effect": p_init,
           "pairs": pairs, "pairs_firing_low_gt_0": int(fires), "pairs_firing_high_lt_0": int(mirror),
           "two_sided_pair_detection_rate": round((fires + mirror) / pairs, 3),
           "null_mode_pairs_(0,1)(2,3)(4,5)": [x for x in rows if x["pair"] in ([0, 1], [2, 3], [4, 5])]}
    print(json.dumps(out, indent=1, default=int))
