#!/usr/bin/env python3
"""POST-HOC, exploratory checks on validated null_pairs rows (not part of the pre-specified readings).

1. Shape of the pooled round losses (skew, excess kurtosis, overall sign balance).
2. Sign overdispersion: the number of positive rounds per pair is Binomial(5, 1/2) under no pair effect and
   symmetric rounds; its across-pair variance (dispersion statistic ~ chi2(k)) is robust to heavy tails.
3. Fire-count permutation test: shuffle the 5k pooled round losses into pairs (removing any pair structure while
   keeping the empirical distribution exactly), apply the production rule (bootstrap_interval, triggered) and
   compare the observed lower-bound fire count with the permutation distribution.

    python scripts/sci_investigation/null_exploratory.py --original DIR --out FILE [--permutations 1000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp12.trigger import bootstrap_interval, triggered  # noqa: E402
from scripts.sci_investigation.null_followup import load, validate  # noqa: E402

SEED = 20261010


def fires(L, seed, resamples, confidence):
    ci = np.array([bootstrap_interval(L[p], seed, p + 1, resamples, confidence) for p in range(len(L))])
    return int(sum(triggered(x, 0.0) for x in ci[:, 0])), int(np.sum(ci[:, 1] < 0))


def explore(root, arch, seed, n_pairs, perms):
    rows, errors = load(root / f"null_pairs_{arch}.jsonl")
    pairs, verrors = validate(rows, arch, "dog-run", seed, n_pairs, 10_000, 0.95, 0.0)
    if errors or verrors:
        return {"integrity_errors": errors + verrors}
    L = np.stack([pairs[p]["_loss"] for p in sorted(pairs)])
    flat = L.reshape(-1)
    k = len(L)
    pos = np.sum(L > 0, 1)
    dispersion = float(np.sum((pos - 2.5) ** 2) / 1.25)
    obs_low, obs_high = fires(L, seed, 10_000, 0.95)
    rng = np.random.default_rng(SEED)
    perm = np.array([fires(rng.permutation(flat).reshape(k, 5), seed, 10_000, 0.95) for _ in range(perms)])
    return {
        "pooled_round_losses": {"n": int(flat.size), "mean": float(flat.mean()), "median": float(np.median(flat)),
                                "skew": float(stats.skew(flat)), "excess_kurtosis": float(stats.kurtosis(flat)),
                                "fraction_positive": float(np.mean(flat > 0)),
                                "sign_test_p_two_sided": float(stats.binomtest(int(np.sum(flat > 0)), int(np.sum(flat != 0))).pvalue)},
        "positive_rounds_histogram_0_to_5": np.bincount(pos, minlength=6).tolist(),
        "expected_under_binomial": [round(k * stats.binom.pmf(j, 5, 0.5), 2) for j in range(6)],
        "sign_dispersion_chi2": dispersion, "sign_dispersion_p": float(stats.chi2.sf(dispersion, k)),
        "observed_fires_low_high": [obs_low, obs_high],
        "permutation_fire_count_low_mean": float(perm[:, 0].mean()),
        "permutation_fire_count_low_95pct": float(np.percentile(perm[:, 0], 95)),
        "permutation_p_low_fires": float((1 + np.sum(perm[:, 0] >= obs_low)) / (1 + perms)),
        "permutation_p_two_sided_fires": float((1 + np.sum(perm.sum(1) >= obs_low + obs_high)) / (1 + perms)),
        "fired_pairs": [{"pair": p, "loss_rounds": [round(float(x), 7) for x in L[p]], "ci_low": pairs[p]["ci_low"]}
                        for p in sorted(pairs) if pairs[p]["fired"]],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", required=True)
    p.add_argument("--archs", nargs="+", default=["D2W512", "D4W1024", "D4W1536"])
    p.add_argument("--seed", type=int, default=990)
    p.add_argument("--pairs", type=int, default=100)
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = {"POST_HOC": True, "permutation_seed": SEED, "permutations": a.permutations,
           "archs": {arch: explore(Path(a.original), arch, a.seed, a.pairs, a.permutations) for arch in a.archs}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for arch, e in out["archs"].items():
        if "integrity_errors" in e:
            print(arch, "integrity errors", e["integrity_errors"][:3])
            continue
        print(f"{arch}: fires low/high {e['observed_fires_low_high']}, permutation mean low "
              f"{e['permutation_fire_count_low_mean']:.2f} (95th pct {e['permutation_fire_count_low_95pct']:.0f}), "
              f"p_low {e['permutation_p_low_fires']:.4f}, p_two_sided {e['permutation_p_two_sided_fires']:.4f}; "
              f"sign dispersion p {e['sign_dispersion_p']:.4f}; kurtosis {e['pooled_round_losses']['excess_kurtosis']:.1f}")


if __name__ == "__main__":
    sys.exit(main())
