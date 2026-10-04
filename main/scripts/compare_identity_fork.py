#!/usr/bin/env python3
"""Identity-fork gate (amendment (b), decision D2): the control and an identity arm must agree bit for bit.

Both snapshots are taken fork.identity_snapshot_steps interaction steps after
the fork: the control's in <run_dir>/fork/identity_control, the identity arm's
in <arm_dir>/identity_identity. Compared: every agent leaf (params, optimizer
state, keys), obs normalisation, the replay buffer, all metadata (env RNG and
replay record, global RNGs, counters, meters, logs) except the WandB run id,
the probe records, and the post-fork evaluations; plus the Check 1 panel
values (Q and dQ/da) of the control after its restore and of the identity arm.

    python scripts/compare_identity_fork.py --run_dir /abs/parent_run --arm_dir /abs/identity_arm \\
        --out /abs/identity_<arch>_<env>.json

Exit 0 = PASS (bit-identical), 1 = FAIL. The JSON lists every differing component.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from experiments.exp12 import fork  # noqa: E402
from experiments.exp12.state import latest_state_dir, load_meta, state_differences  # noqa: E402

SNAPSHOT_IGNORE_META = ("wandb_run_id", "extra_state")


def _without_arm(rows):
    return [{k: v for k, v in r.items() if k != "arm"} for r in rows]


def compare(run_dir: Path, arm_dir: Path) -> dict:
    control = latest_state_dir(fork.fork_dir(run_dir) / "identity_control")
    identity = latest_state_dir(arm_dir / "identity_identity")
    if control is None or identity is None:
        missing = [str(p) for p, d in ((fork.fork_dir(run_dir) / "identity_control", control),
                                       (arm_dir / "identity_identity", identity)) if d is None]
        return {"pass": False, "error": f"snapshot missing: {missing}"}
    diffs = state_differences(control, identity, ignore_meta=SNAPSHOT_IGNORE_META)
    meta_c, meta_i = load_meta(control), load_meta(identity)
    a, b = meta_c["extra_state"], meta_i["extra_state"]
    if a.get("probe_records") != b.get("probe_records"):
        diffs.append("extra_state:probe_records")
    if _without_arm(a.get("post_fork_evals", [])) != _without_arm(b.get("post_fork_evals", [])):
        diffs.append("extra_state:post_fork_evals")
    if a.get("fork") != b.get("fork"):
        diffs.append("extra_state:fork")
    if "injection" in b:
        diffs.append("extra_state:injection (the identity arm was injected)")
    panel_c = fork.load_npz(fork.fork_dir(run_dir) / "check1_control.npz")
    panel_i = fork.load_npz(arm_dir / "check1_after.npz")
    for k in panel_c:
        if not np.array_equal(panel_c[k], panel_i[k]):
            diffs.append(f"check1_panel:{k}")
    return {"pass": not diffs, "differences": diffs, "control_snapshot": str(control),
            "identity_snapshot": str(identity), "interaction_step": meta_c["interaction_step"],
            "identity_interaction_step": meta_i["interaction_step"],
            "fork_step": a.get("fork", {}).get("fork_step")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--arm_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    result = {"run_dir": args.run_dir, "arm_dir": args.arm_dir,
              **compare(Path(args.run_dir), Path(args.arm_dir))}
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
