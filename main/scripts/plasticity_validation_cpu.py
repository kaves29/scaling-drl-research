"""Independent CPU references and toy simulations; no calibration or GPU execution."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("EXP12_JAX_CACHE_DIR", "off")
os.environ.setdefault("MUJOCO_GL", "disable")

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import betaincinv

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))

BLOCK_B_SHA256 = "4f87ba9a4cada6be2eefb7a8b366200e5dce3a57edb167982136409238e258b2"
BOOT_STREAM = int.from_bytes(b"BOOT", "big")


def reference_iqm(values):
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    cut = ordered.size // 4
    return float(np.mean(ordered[cut : ordered.size - cut]))


def linear_quantile(values, probability):
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    rank = (len(ordered) - 1) * probability
    lo, hi = math.floor(rank), math.ceil(rank)
    return float(ordered[lo] + (rank - lo) * (ordered[hi] - ordered[lo]))


def reference_interval(values, seed, check):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return (math.nan, math.nan)
    rng = np.random.default_rng([seed, BOOT_STREAM, check])
    indices = rng.integers(0, len(values), (10_000, len(values)))
    ordered = np.sort(values[indices], axis=1)
    cut = len(values) // 4
    statistics = ordered[:, cut : len(values) - cut].mean(axis=1)
    tail = (1 - 0.95) / 2
    return tuple(linear_quantile(statistics, p) for p in (tail, 1 - tail))


def exact_five_round_intervals(samples):
    """Enumerate all 5**5 bootstrap draws, collapsed into 126 weighted multisets."""
    samples = np.sort(np.asarray(samples, dtype=np.float64), axis=1)
    if samples.shape[1] != 5:
        raise ValueError("exact enumeration requires five rounds")
    multisets = list(itertools.combinations_with_replacement(range(5), 5))
    weights, coefficients = [], []
    for indices in multisets:
        counts = np.bincount(indices, minlength=5)
        weights.append(
            math.factorial(5) // math.prod(math.factorial(int(c)) for c in counts)
        )
        coefficients.append(np.bincount(indices[1:4], minlength=5) / 3)
    assert sum(weights) == 5**5
    statistics = samples @ np.asarray(coefficients).T
    order = statistics.argsort(axis=1)
    ordered = np.take_along_axis(statistics, order, axis=1)
    cumulative = np.cumsum(np.asarray(weights)[order], axis=1)
    bounds = []
    for probability in (0.025, 0.975):
        rank = (5**5 - 1) * probability
        lo, hi = math.floor(rank), math.ceil(rank)
        lower = ordered[np.arange(len(samples)), (cumulative > lo).argmax(axis=1)]
        upper = ordered[np.arange(len(samples)), (cumulative > hi).argmax(axis=1)]
        bounds.append(lower + (rank - lo) * (upper - lower))
    return np.stack(bounds, axis=1)


def reference_trigger(records):
    eligible = {
        row["check_index"]: row for row in records if 1 <= row["check_index"] <= 19
    }
    for check in range(2, 20):
        pair = [eligible.get(check - 1), eligible.get(check)]
        if all(row is not None and row.get("triggered", False) for row in pair):
            return {
                "check_index": check,
                "interaction_step": eligible[check]["interaction_step"],
            }
    return None


def iid_consecutive_probability(probability, checks=19):
    no_previous, one_previous = 1.0, 0.0
    for _ in range(checks):
        no_previous, one_previous = (
            (no_previous + one_previous) * (1 - probability),
            no_previous * probability,
        )
    return 1 - no_previous - one_previous


def verify_implementation():
    import jax
    from experiments.exp12 import probe, trigger

    assert jax.default_backend() == "cpu"
    rng = np.random.default_rng(20261008)
    fixtures = [rng.normal(size=5) for _ in range(64)]
    fixtures += [np.zeros(5), np.ones(5), -np.ones(5), np.arange(-2.0, 3.0)]
    for check, values in enumerate(fixtures, 1):
        assert probe.iqm(values) == reference_iqm(values)
        np.testing.assert_allclose(
            trigger.bootstrap_interval(values, 990, check, 10_000, 0.95),
            reference_interval(values, 990, check),
            rtol=0,
            atol=2e-15,
        )
    for values in ([], [0, 0, math.nan, 0, 0], [0, 0, math.inf, 0, 0]):
        assert np.isnan(trigger.bootstrap_interval(values, 990, 1, 10_000, 0.95)).all()
    for low in (-1.0, 0.0, np.nextafter(0.0, 1.0), math.nan, math.inf):
        expected = bool(math.isfinite(low) and low > 0)
        assert trigger.triggered(low, 0) == expected
    for indices in ([1], [1, 2], [18, 19], [19, 20], [0, 1], [1, 3], [1, 2, 3]):
        records = [
            dict(check_index=k, interaction_step=k * 100, triggered=True)
            for k in indices
        ]
        assert trigger.f_star(records, 20, 2, 0.95) == reference_trigger(records)
    records = [
        dict(check_index=k, interaction_step=k * 100, triggered=k != 2)
        for k in (1, 2, 3)
    ]
    assert trigger.f_star(records, 20, 2, 0.95) == reference_trigger(records) is None
    for probability in (0.025, 0.05, 0.13):
        direct = sum(
            probability ** sum(sequence) * (1 - probability) ** (6 - sum(sequence))
            for sequence in itertools.product((0, 1), repeat=6)
            if any(a and b for a, b in zip(sequence, sequence[1:]))
        )
        np.testing.assert_allclose(
            iid_consecutive_probability(probability, checks=6),
            direct,
            rtol=0,
            atol=1e-15,
        )
    current = np.array([0, 1, 100, 2, 3], dtype=np.float32)
    fresh = np.array([0, 1, 2, 3, 100], dtype=np.float32)
    result = {"current": {"score": current}, "fresh": {"score": fresh}}
    independent_loss = np.array([float(f) - float(c) for f, c in zip(fresh, current)])
    np.testing.assert_array_equal(probe.paired_loss(result), independent_loss)
    assert probe.summarize(result)["loss_iqm"] == reference_iqm(independent_loss)
    assert reference_iqm(independent_loss) != reference_iqm(fresh) - reference_iqm(
        current
    )
    twin = {}
    for q in (1, 2):
        twin[f"fresh_q{q}"] = {"score": rng.uniform(-1, 1, 5).astype(np.float32)}
        twin[f"current_q{q}"] = {"score": rng.uniform(-1, 1, 5).astype(np.float32)}
    expected = np.array(
        [
            np.float32(
                (
                    np.float32(
                        twin["fresh_q1"]["score"][r] - twin["current_q1"]["score"][r]
                    )
                    + np.float32(
                        twin["fresh_q2"]["score"][r] - twin["current_q2"]["score"][r]
                    )
                )
                / np.float32(2)
            )
            for r in range(5)
        ]
    )
    np.testing.assert_array_equal(probe.paired_loss(twin), expected)
    enumerated = list(itertools.product(range(5), repeat=5))
    for values in fixtures[:4]:
        statistics = [reference_iqm(values[list(indices)]) for indices in enumerated]
        np.testing.assert_allclose(
            exact_five_round_intervals(values[None, :])[0],
            [linear_quantile(statistics, p) for p in (0.025, 0.975)],
            rtol=0,
            atol=2e-15,
        )
    return {
        "backend": "cpu",
        "iqm_and_10000_resample_ci_fixtures": len(fixtures),
        "pairing_example_iqm_of_differences": reference_iqm(independent_loss),
        "pairing_example_difference_of_iqms": reference_iqm(fresh)
        - reference_iqm(current),
        "invalid_interval_strict_trigger_twin_and_eligibility_checks": "PASS",
        "exact_enumeration_crosschecks": 4,
    }


def evidence_arithmetic(path):
    evidence = path.read_bytes()
    assert hashlib.sha256(evidence).hexdigest() == BLOCK_B_SHA256
    rows = [
        json.loads(line)
        for line in evidence.decode().splitlines()
        if line.startswith('{"arch":')
    ]
    assert len(rows) == 9 and len({(r["arch"], r["pool_size"]) for r in rows}) == 9
    output = []
    for row in rows:
        score, baseline, loss = (
            np.asarray(row[k])
            for k in ("score_rounds", "b_rounds", "final_loss_rounds")
        )
        np.testing.assert_array_equal(
            score.astype(np.float32),
            baseline.astype(np.float32) - loss.astype(np.float32),
        )
        ratio = reference_iqm(score) / reference_iqm(baseline)
        assert ratio == row["score_over_b"]
        assert reference_iqm(score) == row["score_iqm"]
        assert reference_iqm(baseline) == row["b_iqm"]
        output.append(
            {
                "arch": row["arch"],
                "pool": row["pool_size"],
                "ratio": ratio,
                "score_iqm": reference_iqm(score),
                "baseline_iqm": reference_iqm(baseline),
                "final_loss_iqm": reference_iqm(loss),
                "final_loss_rounds": loss.tolist(),
                "per_round_ratios": (score / baseline).tolist(),
                "alternative_not_approved_one_minus_loss_iqm_over_b_iqm": 1
                - reference_iqm(loss) / reference_iqm(baseline),
                "configured_pool_gate": (
                    ratio >= 0.9 if row["is_configured_pool"] else None
                ),
            }
        )
    return {"sha256": BLOCK_B_SHA256, "embedded_range_rows_verified": output}


def toy_statistics(checks, mc_checks):
    rng = np.random.default_rng(20261008)
    epsilon = rng.normal(size=(checks, 5))
    pair_effect = rng.normal(size=checks)
    models = []
    for sigma_pair in (0.0, 0.25, 0.5, 1.0):
        values = epsilon + sigma_pair * pair_effect[:, None]
        bounds = exact_five_round_intervals(values)
        conditional_mean = sigma_pair * pair_effect
        models.append(
            {
                "pair_effect_sd_over_round_noise_sd": sigma_pair,
                "marginal_round_correlation": sigma_pair**2 / (1 + sigma_pair**2),
                "checks": checks,
                "positive_zero_exclusion_rate": float(np.mean(bounds[:, 0] > 0)),
                "negative_zero_exclusion_rate": float(np.mean(bounds[:, 1] < 0)),
                "coverage_of_conditional_normal_iqm": float(
                    np.mean(
                        (bounds[:, 0] <= conditional_mean)
                        & (bounds[:, 1] >= conditional_mean)
                    )
                ),
            }
        )
    monte_carlo = np.asarray(
        [reference_interval(x, 990, k + 1) for k, x in enumerate(epsilon[:mc_checks])]
    )
    exact = exact_five_round_intervals(epsilon[:mc_checks])
    return {
        "label": "CPU normal-location toys, not critic fits, CUDA evidence, or new gates",
        "seed": 20261008,
        "iid_continuous_symmetric_null_unanimous_positive_probability": 0.5**5,
        "coverage_upper_bound_from_unanimous_sign_events": 1 - 2 * 0.5**5,
        "exact_empirical_bootstrap_models": models,
        "unchanged_10000_resample_iid_gaussian_control": {
            "checks": mc_checks,
            "fires": int(np.sum(monte_carlo[:, 0] > 0)),
            "fire_rate": float(np.mean(monte_carlo[:, 0] > 0)),
            "finite_resampling_vs_exact_decision_disagreements": int(
                np.sum((monte_carlo[:, 0] > 0) != (exact[:, 0] > 0))
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-b-report", type=Path, required=True)
    parser.add_argument("--checks", type=int, default=12_000)
    parser.add_argument("--mc-checks", type=int, default=1_000)
    args = parser.parse_args()
    if not 0 < args.mc_checks <= args.checks:
        parser.error("require 0 < mc-checks <= checks")
    out = {
        "reference_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=MAIN, text=True
        ).strip(),
        "implementation_verification": verify_implementation(),
        "block_b_arithmetic": evidence_arithmetic(args.block_b_report),
        "toy_statistics": toy_statistics(args.checks, args.mc_checks),
        "iid_binomial_illustrations_not_acceptance_tests": {
            "p_ge13_of100_when_p05": sum(
                math.comb(100, k) * 0.05**k * 0.95 ** (100 - k) for k in range(13, 101)
            ),
            "exact_95pct_interval_for_13_of100": [
                float(betaincinv(13, 88, 0.025)),
                float(betaincinv(14, 87, 0.975)),
            ],
        },
        "iid_19check_consecutive_fork_illustrations_not_production_rates": {
            str(p): iid_consecutive_probability(p) for p in (0.025, 0.05, 0.13)
        },
        "warmup_replay_occupancy_illustration": [
            {
                "pool": n,
                "replay_rows": 5000,
                "expected_distinct_rows": 5000 * -math.expm1(n * math.log1p(-1 / 5000)),
                "mean_minibatch_draws_per_pool_position": 1000 * 256 / n,
                "evaluation_chunk": math.gcd(n, 2560),
            }
            for n in (1600, 6400, 25600)
        ],
    }
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
