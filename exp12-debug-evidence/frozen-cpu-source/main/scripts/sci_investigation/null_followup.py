#!/usr/bin/env python3
"""Fresh-pair null follow-up: validate Block B's null_pairs_*.jsonl, test for persistent pair-level effects, compare
with an optional seed-991 replication, and evaluate (never adopt) the pre-specified p95 alternative threshold.

Read-only on its inputs; deterministic (fixed permutation seeds; the production bootstrap is seeded per row).

    python scripts/sci_investigation/null_followup.py --original /abs/blockB_22706349/gpu3/null_dog_run \\
        [--replication /abs/null_replicate_seed991] --out /abs/null_followup.json

Exit status: 0 = analysis complete and every supplied file passed the integrity checks; 2 = at least one integrity
error (the statistics are still reported but must not be interpreted); 1 = usage error.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp12.probe import iqm  # noqa: E402
from experiments.exp12.trigger import bootstrap_interval, triggered  # noqa: E402

ARCHS = ("D2W512", "D4W1024", "D4W1536")
ROUNDS = 5
PAIR_KEY_OFFSET = 10_000  # scripts/probe_fresh_checks.py NULL_PAIR_KEY_OFFSET (unchanged since b4a90cb)
APPROVED_RATE_LIMIT = 0.05  # pre-specified NULL rule (docs/exp12_decisions.md, 2026-10-04)
APPROVED_MIN_PAIRS = 100
NO_DIFFERENCE_FLOOR = 0.048  # rule's fire rate for IID zero-centred rounds (exact 5^5 enumeration; reference only)
ALPHA = 0.01  # pre-specified for the diagnostics below; not a qualification criterion
PERMUTATIONS = 20_000
PERM_SEED = 20261009
REQUIRED = ("arch", "env", "seed", "pair", "check_index", "score_fresh_rounds", "score_current_rounds",
            "b_rounds", "loss_rounds", "loss_iqm", "valid", "ci_low", "ci_high", "fired")


def _same(a, b):
    return bool(np.array_equal(np.asarray(a, np.float64), np.asarray(b, np.float64), equal_nan=True))


def load(path: Path):
    """Rows with their line numbers, plus parse errors."""
    rows, errors = [], []
    if not path.is_file():
        return rows, [f"{path.name}: missing"]
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            errors.append(f"{path.name}:{n}: blank line")
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{path.name}:{n}: malformed JSON ({e.msg})")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path.name}:{n}: not a JSON object")
            continue
        rows.append((n, row))
    return rows, errors


def validate(rows, arch, env, seed, n_pairs, resamples=None, confidence=None, null_threshold=None):
    """Integrity of one arch's rows; returns (pair -> row, errors). Recomputes every stored statistic."""
    errors, by_pair, seen = [], {}, {}
    for n, row in rows:
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            errors.append(f"line {n}: missing fields {missing}")
            continue
        pair = row["pair"]
        if type(pair) is not int or pair < 0:
            errors.append(f"line {n}: invalid pair {pair!r}")
            continue
        if pair in seen:
            identical = row == seen[pair][1]
            errors.append(f"line {n}: duplicate pair {pair} (first at line {seen[pair][0]}, "
                          f"{'identical' if identical else 'DIFFERENT'} content)")
            continue
        seen[pair] = (n, row)
        for field, expected in (("arch", arch), ("env", env), ("seed", seed), ("check_index", pair + 1)):
            if row[field] != expected:
                errors.append(f"pair {pair}: {field} {row[field]!r} != {expected!r}")
        if "init_keys" in row and row["init_keys"] != [PAIR_KEY_OFFSET + pair, 0, 1]:
            errors.append(f"pair {pair}: init_keys {row['init_keys']} differ from the null-mode keys")
        arrays = {}
        for k in ("score_fresh_rounds", "score_current_rounds", "b_rounds", "loss_rounds"):
            v = row[k]
            if not isinstance(v, list) or len(v) != ROUNDS or not all(isinstance(x, (int, float)) for x in v):
                errors.append(f"pair {pair}: {k} is not {ROUNDS} numbers")
                break
            arrays[k] = np.asarray(v, np.float32)
        else:
            loss = arrays["score_fresh_rounds"] - arrays["score_current_rounds"]
            if not np.array_equal(loss, arrays["loss_rounds"], equal_nan=True):
                errors.append(f"pair {pair}: loss_rounds != float32(score_fresh - score_current)")
            valid = bool(np.isfinite(loss).all())
            r_samples = row.get("resamples", resamples)
            conf = row.get("confidence", confidence)
            thr = row.get("null_threshold", null_threshold)
            for field, value, ref in (("resamples", r_samples, resamples), ("confidence", conf, confidence),
                                      ("null_threshold", thr, null_threshold)):
                if value is None:
                    errors.append(f"pair {pair}: {field} unknown")
                elif ref is not None and value != ref:
                    errors.append(f"pair {pair}: {field} {value} differs from the other rows' {ref}")
            if None in (r_samples, conf, thr):
                continue
            resamples, confidence, null_threshold = r_samples, conf, thr
            low, high = bootstrap_interval(loss, seed, pair + 1, int(r_samples), float(conf))
            expected = {"valid": valid, "loss_iqm": iqm(loss) if valid else float("nan"),
                        "ci_low": low, "ci_high": high, "fired": triggered(low, float(thr)) and valid}
            for field, value in expected.items():
                if not _same(row[field], value):
                    errors.append(f"pair {pair}: stored {field} {row[field]!r} != recomputed {value!r}")
            by_pair[pair] = {**row, "_loss": loss.astype(np.float64)}
    absent = sorted(set(range(n_pairs)) - set(seen))
    extra = sorted(p for p in seen if p >= n_pairs)
    if absent:
        errors.append(f"missing pairs {absent[:20]}{'...' if len(absent) > 20 else ''} ({len(absent)} total)")
    if extra:
        errors.append(f"pairs beyond {n_pairs - 1}: {extra[:20]}")
    if "provenance" in (rows[0][1] if rows else {}):
        prov = {json.dumps(r.get("provenance"), sort_keys=True) for _, r in rows}
        if len(prov) != 1:
            errors.append(f"{len(prov)} different provenance blocks in one file")
        code = rows[0][1]["provenance"].get("code", {})
        if code.get("dirty") is not False:
            errors.append("provenance: source tree dirty or unknown")
    return {p: by_pair[p] for p in sorted(by_pair) if p < n_pairs}, errors


def check_summary(path: Path, pairs, n_pairs):
    """Recompute null_summary_<arch>.json from the validated rows (the original code kept the LAST duplicate)."""
    if not path.is_file():
        return [f"{path.name}: missing"]
    s = json.loads(path.read_text())
    rows = [pairs[p] for p in sorted(pairs)]
    if not rows:
        return [f"{path.name}: no valid rows to compare"]
    losses = np.array([r["loss_iqm"] for r in rows])
    expected = {"null_pairs": len(rows), "per_check_fire_rate": float(np.mean([r["fired"] for r in rows])),
                "would_be_null_threshold_p95_of_L": float(np.percentile(losses, 95)),
                "iqm_of_L": iqm(losses), "std_of_L": float(np.std(losses, ddof=1))}
    errors = [f"{path.name}: {k} {s.get(k)!r} != recomputed {v!r}" for k, v in expected.items() if not _same(s.get(k, np.nan), v)]
    if s.get("null_pairs") != n_pairs:
        errors.append(f"{path.name}: null_pairs {s.get('null_pairs')} != expected {n_pairs}")
    return errors


def _perm_pvalue(observed, null_draws):
    return float((1 + np.sum(null_draws >= observed)) / (1 + len(null_draws)))


def persistence(pairs):
    """Within one dataset: is there a component shared by a pair's five rounds?"""
    rows = [pairs[p] for p in sorted(pairs) if pairs[p]["valid"]]
    L = np.stack([r["_loss"] for r in rows])  # (pairs, 5)
    k = len(L)
    trim = lambda x: np.sort(x, axis=-1)[..., 1:4].mean(-1)  # IQM of 5 = mean of the middle 3
    observed = float(np.var(trim(L), ddof=1))
    rng = np.random.default_rng(PERM_SEED)
    flat = L.reshape(-1)
    draws = np.array([np.var(trim(rng.permutation(flat).reshape(k, ROUNDS)), ddof=1) for _ in range(PERMUTATIONS)])
    same_sign = int(np.sum(np.all(L > 0, 1) | np.all(L < 0, 1)))
    grand, means = L.mean(), L.mean(1)
    ms_between = ROUNDS * np.sum((means - grand) ** 2) / (k - 1)
    ms_within = np.sum((L - means[:, None]) ** 2) / (k * (ROUNDS - 1))
    f = ms_between / ms_within
    return {
        "valid_pairs": k,
        "fires_lower_bound_gt_0": int(sum(r["fired"] for r in rows)),
        "fires_upper_bound_lt_0": int(sum(r["ci_high"] < 0 for r in rows)),
        "binomial_p_lower_fires_vs_floor": float(stats.binom.sf(sum(r["fired"] for r in rows) - 1, k, NO_DIFFERENCE_FLOOR)),
        "pairs_all_rounds_same_sign": same_sign,
        "same_sign_expected_under_iid_symmetric": round(k * 2 / 32, 2),
        "binomial_p_same_sign_excess": float(stats.binom.sf(same_sign - 1, k, 2 / 32)),
        "between_pair_var_of_iqm": observed,
        "permutation_p_between_pair_var": _perm_pvalue(observed, draws),
        "anova_F": float(f), "anova_p": float(stats.f.sf(f, k - 1, k * (ROUNDS - 1))),
        "tau_over_sigma_w_estimate": float(math.sqrt(max((ms_between - ms_within) / ROUNDS, 0.0) / ms_within)),
        "kruskal_wallis_p": float(stats.kruskal(*L).pvalue),
        "within_pair_round_sd_pooled": float(math.sqrt(ms_within)),
        "sd_of_pair_iqm": float(np.std(trim(L), ddof=1)),
    }


def replication(orig, rep):
    """Same initialisation pairs (keys do not depend on --seed), new replay and probe streams."""
    common = sorted(p for p in set(orig) & set(rep) if orig[p]["valid"] and rep[p]["valid"])
    a = np.array([orig[p]["loss_iqm"] for p in common])
    b = np.array([rep[p]["loss_iqm"] for p in common])
    rho = float(stats.spearmanr(a, b).statistic)
    rng = np.random.default_rng(PERM_SEED + 1)
    rb = stats.rankdata(b)
    ra = stats.rankdata(a)
    draws = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(PERMUTATIONS)])
    fo = np.array([orig[p]["fired"] for p in common])
    fr = np.array([rep[p]["fired"] for p in common])
    table = [[int(np.sum(fo & fr)), int(np.sum(fo & ~fr))], [int(np.sum(~fo & fr)), int(np.sum(~fo & ~fr))]]
    return {
        "matched_valid_pairs": len(common),
        "spearman_rho_loss_iqm": rho, "permutation_p_one_sided": _perm_pvalue(rho, draws),
        "pearson_r_loss_iqm": float(np.corrcoef(a, b)[0, 1]),
        "sign_agreement": int(np.sum(np.sign(a) == np.sign(b))),
        "fired_both_origOnly_repOnly_neither": [table[0][0], table[0][1], table[1][0], table[1][1]],
        "fisher_p_refire_one_sided": float(stats.fisher_exact(table, alternative="greater")[1]),
        "replication_fire_rate": float(np.mean([rep[p]["fired"] for p in sorted(rep)])),
    }


def p95_alternative(orig, rep=None):
    """REPORT ONLY: the pre-specified option (threshold = 95th percentile of pair-level L). Not adopted."""
    def thr(pairs, keep=lambda p: True):
        return float(np.percentile([pairs[p]["loss_iqm"] for p in pairs if pairs[p]["valid"] and keep(p)], 95))

    def rate(pairs, t, keep=lambda p: True):
        sel = [p for p in pairs if keep(p)]
        return float(np.mean([triggered(pairs[p]["ci_low"], t) and pairs[p]["valid"] for p in sel]))

    even, odd = (lambda p: p % 2 == 0), (lambda p: p % 2 == 1)
    t = thr(orig)
    out = {"threshold_p95_of_L": t, "in_sample_fire_rate": rate(orig, t),
           "note_in_sample": "calibrated on the same pairs, so <= ~5% almost by construction",
           "cross_validated_fire_rate_even_to_odd": rate(orig, thr(orig, even), odd),
           "cross_validated_fire_rate_odd_to_even": rate(orig, thr(orig, odd), even),
           "threshold_in_units_of_sd_of_L": t / float(np.std([orig[p]["loss_iqm"] for p in orig if orig[p]["valid"]], ddof=1))}
    if rep:
        out["out_of_sample_fire_rate_on_replication"] = rate(rep, t)
        out["replication_threshold_p95_of_L"] = thr(rep)
    return out


def classify(integrity_ok, pers, rep):
    if not integrity_ok:
        return "EVIDENCE INVALID: integrity errors; do not interpret (re-run or retrieve the complete original rows)"
    within = pers["permutation_p_between_pair_var"] < ALPHA
    if rep is None:
        return ("pair-level effect within this replay (permutation p < 0.01); cross-seed persistence untested"
                if within else "no pair-level effect detected within this replay; cross-seed persistence untested")
    across = rep["spearman_rho_loss_iqm"] > 0 and rep["permutation_p_one_sided"] < ALPHA
    if across:
        return "PERSISTENT INITIALISATION-PAIR EFFECT: pair differences reproduce on a new replay and new probe streams"
    if within:
        return "PAIR EFFECT SPECIFIC TO THE ORIGINAL REPLAY/STREAMS: present within the original, not reproduced"
    if rep["replication_fire_rate"] <= APPROVED_RATE_LIMIT:
        return "NO PAIR EFFECT DETECTED; replication within 5%: the original excess is consistent with chance or an artefact"
    return "RATE EXCESS REPRODUCED WITHOUT A PAIR EFFECT: look at replay, teacher and numerics"


def analyse(args):
    report, integrity_ok = {"archs": {}}, True
    for arch in args.archs:
        entry = {}
        datasets = [("original", Path(args.original), args.original_seed)]
        if args.replication and arch in args.replication_archs:
            datasets.append(("replication", Path(args.replication), args.replication_seed))
        validated = {}
        for name, root, seed in datasets:
            rows, errors = load(root / f"null_pairs_{arch}.jsonl")
            pairs, verrors = validate(rows, arch, args.env, seed, args.pairs, args.resamples, args.confidence,
                                      args.null_threshold)
            errors += verrors + check_summary(root / f"null_summary_{arch}.json", pairs, args.pairs)
            validated[name] = pairs
            n = len(pairs)
            rate = float(np.mean([pairs[p]["fired"] for p in pairs])) if pairs else float("nan")
            entry[name] = {
                "integrity_errors": errors,
                "pairs_validated": n,
                "invalid_pairs": int(sum(not pairs[p]["valid"] for p in pairs)),
                "approved_null_rule": {"fire_rate": rate, "pairs": n,
                                       "pass": bool(n >= APPROVED_MIN_PAIRS and rate <= APPROVED_RATE_LIMIT),
                                       "rule": "per-check fire rate <= 5% with >= 100 pairs (pre-specified; unchanged)"},
                "persistence": persistence(pairs) if n > 2 else None,
            }
            integrity_ok &= not errors
        rep = (replication(validated["original"], validated["replication"])
               if "replication" in validated and validated["original"] and validated["replication"] else None)
        entry["replication_comparison"] = rep
        if validated["original"]:
            entry["p95_alternative_REPORT_ONLY"] = p95_alternative(validated["original"], validated.get("replication"))
        ok = all(not entry[n]["integrity_errors"] for n, _, _ in datasets)
        entry["diagnostic_reading"] = classify(ok, entry["original"]["persistence"], rep) if entry["original"]["persistence"] else "no rows"
        report["archs"][arch] = entry
    import scipy

    report["settings"] = {"alpha": ALPHA, "permutations": PERMUTATIONS, "perm_seed": PERM_SEED,
                          "no_difference_floor_reference": NO_DIFFERENCE_FLOOR,
                          "numpy": np.__version__, "scipy": scipy.__version__}
    return report, integrity_ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", required=True, help="directory holding Block B's null_pairs_<arch>.jsonl")
    p.add_argument("--replication", default=None, help="directory of the seed-991 replication (optional)")
    p.add_argument("--archs", nargs="+", default=list(ARCHS))
    p.add_argument("--replication_archs", nargs="+", default=["D4W1024"])
    p.add_argument("--env", default="dog-run")
    p.add_argument("--pairs", type=int, default=100)
    p.add_argument("--original_seed", type=int, default=990)
    p.add_argument("--replication_seed", type=int, default=991)
    p.add_argument("--resamples", type=int, default=10_000, help="expected trigger.resamples (configs/base_exp12.yaml)")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--null_threshold", type=float, default=0.0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    report, ok = analyse(args)
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    for arch, e in report["archs"].items():
        o = e["original"]
        integrity = "OK" if not o["integrity_errors"] else f"{len(o['integrity_errors'])} ERRORS"
        rule = o["approved_null_rule"]
        print(f"{arch}: integrity {integrity}; approved rule fire rate {rule['fire_rate']:.3f} "
              f"({'PASS' if rule['pass'] else 'FAIL'}); reading: {e['diagnostic_reading']}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
