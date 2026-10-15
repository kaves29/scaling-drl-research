"""Read-only CPU summary of the seven-run pilot's archived Myo probe curves.

This archive contains curves, not complete replay/checkpoint states. Initial
minibatch loss is not initial full-pool error, and no gradient can be recovered
from these arrays. The program never changes archived inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PILOT_SHA = "f8619dd5f494399d9febc79fc9092010234c714d"
ARCHITECTURES = ("D2W512", "D4W1024", "D4W1536")


def inspect(root: Path) -> dict:
    manifest = root / "manifest.json"
    metadata = json.loads(manifest.read_text())
    if metadata.get("commit") != PILOT_SHA:
        raise ValueError("pilot manifest source differs")
    out = {"pilot_commit": PILOT_SHA, "archive_root": str(root), "runs": {}}
    for arch in ARCHITECTURES:
        path = root / "results/exp12/exp1/runs" / f"exp1_{arch}_myo-key-turn_seed1" / "probe_curves.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checks = {}
        with np.load(path, allow_pickle=False) as data:
            for check in (0, 2, 20):
                prefix = f"check_{check:02d}/"
                arms = ("fresh",) if check == 0 else ("current", "fresh")
                record = {}
                for arm in arms:
                    b = data[prefix + arm + "_b"]
                    final = data[prefix + arm + "_final_loss"]
                    losses = data[prefix + arm + "_losses"]
                    offset = data[prefix + arm + "_offset"]
                    if (b.shape != (5,) or final.shape != (5,) or offset.shape != (5,)
                            or losses.shape != (5, 2000) or not all(np.isfinite(x).all()
                                                                     for x in (b, final, losses, offset))):
                        raise ValueError(f"{path}: invalid five-function check {check} {arm}")
                    record[arm] = {
                        "final_mse_over_b_rounds": (final / b).astype(float).tolist(),
                        "mean_final_mse_over_b": float(np.mean(final / b)),
                        "mean_first_100_minibatch_losses": float(np.mean(losses[:, :100])),
                        "mean_last_100_minibatch_losses": float(np.mean(losses[:, -100:])),
                        "mean_first_1000_minibatch_losses": float(np.mean(losses[:, :1000])),
                        "mean_last_1000_minibatch_losses": float(np.mean(losses[:, 1000:])),
                        "offsets": offset.astype(float).tolist(),
                    }
                if check:
                    if not np.array_equal(data[prefix + "current_b"], data[prefix + "fresh_b"]):
                        raise ValueError(f"{path}: target variance differs within paired round")
                    offset_gap = data[prefix + "fresh_offset"] - data[prefix + "current_offset"]
                    record["fresh_minus_current_target_offset_rounds"] = offset_gap.astype(float).tolist()
                    record["all_full_targets_identical"] = bool(np.all(offset_gap == 0))
                checks[str(check)] = record
        out["runs"][arch] = {"probe_curves_sha256": digest, "checks": checks}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.archive_root.resolve(strict=True))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        if not args.output.is_absolute() or args.output.exists():
            raise ValueError("output must be a new absolute path")
        args.output.write_text(rendered)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
