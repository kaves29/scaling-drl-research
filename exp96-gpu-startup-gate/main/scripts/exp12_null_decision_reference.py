"""Independent count-level inference only; does not adopt a trigger calibration."""

import argparse
import json
from pathlib import Path

from scipy.stats import beta, binom, fisher_exact


def count_summary(k, n):
    return {
        "fired": k,
        "valid_pairs": n,
        "rate": k / n,
        "clopper_pearson_95": [
            float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0,
            float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0,
        ],
        "binomial_one_sided_p_at_approved_0_05": float(binom.sf(k - 1, n, 0.05)),
        "approved_count_gate_pass": k <= 0.05 * n,
    }


def consecutive_probability(p, checks=19):
    # Independent Bernoulli illustration only: survival ends with a false/true check.
    zero, one = 1.0, 0.0
    for _ in range(checks):
        zero, one = (zero + one) * (1 - p), zero * p
    return 1 - zero - one


def report():
    return {
        "evidence_status": "user-reported counts, no raw replication artifacts inspected",
        "original": count_summary(13, 100),
        "seed991": count_summary(6, 100),
        "two_sided_fisher_difference_p_independence_illustration": float(
            fisher_exact([[13, 87], [6, 94]]).pvalue
        ),
        "independent_19_check_illustrations": [
            {
                "per_check_rate": p,
                "probability_any_consecutive_pair": consecutive_probability(p),
            }
            for p in (0.05, 0.06, 0.13)
        ],
        "warning": "Independent Bernoulli illustrations do not estimate actual longitudinal fork rates. Fisher assumes independent samples; matched initialization pairs require raw discordances/paired inference. No threshold, confidence or production rule changed.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    with args.out.open("x") as stream:
        json.dump(report(), stream, indent=2)
