"""CPU-only, fail-closed combination of original five and D's five new targets.

Descriptive ten-target L uses the pinned probe's IQM. The binary verdict is
explicitly OUR existing bootstrap rule, not a claim about Lyle et al.'s rule.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import numpy as np

from experiments.exp12.probe import iqm, probe_key
from experiments.exp12.trigger import bootstrap_interval
from scripts.sci_investigation.compare_probe_decisions import exact_binomial_interval


def combine(first_root, extra_root, first_commit, extra_commit, pairs=100):
    first_prov = json.loads((first_root / "provenance.json").read_text())
    extra_prov = json.loads((extra_root / "provenance.json").read_text())
    for prov, commit in ((first_prov, first_commit), (extra_prov, extra_commit)):
        if prov["code"] != {"commit": commit, "dirty": False}:
            raise ValueError("source provenance differs from expected clean commit")
        if (prov["mode"], prov["arch"], prov["env"], prov["seed"]) != (
            "null", "D4W1024", "dog-run", 991
        ):
            raise ValueError("incompatible null provenance")
    if first_prov.get("round_start", 0) != 0 or extra_prov.get("round_start") != 5:
        raise ValueError("incorrect target-round partition")
    if extra_prov.get("rounds") != 5:
        raise ValueError("D requires exactly five additional target functions")
    first_paths = sorted(first_root.glob("pair_*.json"))
    extra_paths = sorted(extra_root.glob("pair_*.json"))
    names = [f"pair_{i:03d}.json" for i in range(pairs)]
    if [p.name for p in first_paths] != names or [p.name for p in extra_paths] != names:
        raise ValueError("missing, extra, or duplicate pair files")
    output = {"first_commit": first_commit, "extra_commit": extra_commit,
              "first_five_root": str(first_root), "extra_five_root": str(extra_root),
              "pairs": pairs, "ten_target_loss_iqm": [], "ten_target_loss_mean": [], "per_pair": [],
              "our_existing_trigger": {}}
    fired = 0
    digest = hashlib.sha256()
    for pair, (a_path, d_path) in enumerate(zip(first_paths, extra_paths)):
        a_bytes, d_bytes = a_path.read_bytes(), d_path.read_bytes()
        digest.update(hashlib.sha256(a_bytes).digest())
        digest.update(hashlib.sha256(d_bytes).digest())
        a, d = json.loads(a_bytes), json.loads(d_bytes)
        if (a["pair"], d["pair"], a["check_index"], d["check_index"], a["seed"], d["seed"]) != (
            pair, pair, pair + 1, pair + 1, 991, 991
        ):
            raise ValueError(f"pair {pair}: mismatched identity")
        if len(a["null_2000"]["loss_rounds"]) != 5 or len(d["null_2000"]["loss_rounds"]) != 5:
            raise ValueError(f"pair {pair}: five losses required in each half")
        if d.get("round_indices") != [5, 6, 7, 8, 9]:
            raise ValueError(f"pair {pair}: target indices differ")
        expected_keys = [np.asarray(jax.device_get(jax.random.split(probe_key(991, pair + 1, r))[0]),
                                    dtype=np.uint32).tolist() for r in range(5, 10)]
        if d.get("target_keys") != expected_keys or len({tuple(k) for k in expected_keys}) != 5:
            raise ValueError(f"pair {pair}: target keys differ or repeat")
        values = np.asarray(a["null_2000"]["loss_rounds"] + d["null_2000"]["loss_rounds"],
                            dtype=np.float32)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError(f"pair {pair}: invalid ten-target losses")
        low, high = bootstrap_interval(values, 991, pair + 1, 10000, .95)
        verdict = bool(np.isfinite(low) and low > 0)
        fired += verdict
        output["ten_target_loss_iqm"].append(iqm(values))
        output["ten_target_loss_mean"].append(float(np.mean(values, dtype=np.float64)))
        output["per_pair"].append({"pair": pair, "loss_rounds": values.tolist(),
                                    "loss_iqm": iqm(values),
                                    "loss_mean": float(np.mean(values, dtype=np.float64)), "ci_low": low,
                                    "ci_high": high, "our_trigger_fired": verdict})
    output["pair_files_sha256_digest"] = digest.hexdigest()
    output["descriptive"] = {"mean_ten_target_loss_iqm": float(np.mean(output["ten_target_loss_iqm"])),
                             "median_ten_target_loss_iqm": float(np.median(output["ten_target_loss_iqm"])),
                             "mean_ten_target_loss_mean": float(np.mean(output["ten_target_loss_mean"]))}
    output["our_existing_trigger"] = {"fired": fired, "pairs": pairs,
                                       "exact_binomial_95": exact_binomial_interval(fired, pairs),
                                       "method": "project's 10000-resample 95% percentile-bootstrap lower bound > 0; not attributed to Lyle et al."}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-five-root", required=True, type=Path)
    parser.add_argument("--extra-five-root", required=True, type=Path)
    parser.add_argument("--first-commit", required=True)
    parser.add_argument("--extra-commit", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = combine(args.first_five_root, args.extra_five_root,
                     args.first_commit, args.extra_commit)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
