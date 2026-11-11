"""Full-size, full-branch-horizon GPU engineering qualification, one checkpoint.

Reference training is shortened to 10k raw steps; plasticity and all four branch
budgets are unchanged. This is not a million-step scientific result.
"""

import argparse
import json
from pathlib import Path
import pickle
from unittest.mock import patch

from experiments.exp1 import compose_config
from experiments.online_exp2 import OnlineExp2
from experiments.exp12.precision import set_matmul_precision


def qualify(regime, seed, output):
    import jax

    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU qualification requires actual GPU execution")
    set_matmul_precision()
    cfg = compose_config(
        str(Path(__file__).parents[2] / "configs"),
        "online_exp2_pilot_" + regime,
        [
            "seed=" + str(seed),
            "online_exp2.reference_raw_steps=10000",
            "online_exp2.checkpoint_raw_steps=[10000]",
            "online_exp2.probe_checkpoints=[10000]",
        ],
        allow_pilot=True,
    )
    interrupted = output / "interruption_injected.json"
    original_check = OnlineExp2._check_resource_cap

    def interrupt(runner):
        original_check(runner)
        state_path = output / "diagnostics/raw_00010000/small_learning.pkl"
        if state_path.exists() and not interrupted.exists():
            state = pickle.loads(state_path.read_bytes())
            if state["raw_env_steps"] >= 5000 and not state["complete"]:
                interrupted.write_text(
                    json.dumps({"raw_env_steps": state["raw_env_steps"]})
                )
                raise RuntimeError("expected qualification restart")

    # Only reference duration/checkpoint schedule differs from the approved pilot.
    # Full-size networks, source batch/optimizer settings and full assays remain.
    with patch(
        "experiments.online_exp2.validate", return_value=(5000, [10000], [10000])
    ):
        if not interrupted.exists():
            try:
                with patch.object(OnlineExp2, "_check_resource_cap", interrupt):
                    OnlineExp2(cfg, str(output.resolve())).run()
            except RuntimeError as exc:
                if str(exc) != "expected qualification restart":
                    raise
            else:
                raise AssertionError(
                    "qualification did not reach learning-branch interruption"
                )
        runner = OnlineExp2(cfg, str(output.resolve()))
        runner.run()
        records = [row for row in runner.rows if row["type"] == "pilot_diagnostic"]
        if len(records) != 1 or any(
            counter["raw_env_steps"] != 10000
            for counter in records[0]["branch_counters"].values()
        ):
            raise AssertionError("full four-condition raw-step evidence missing")
        probe_rows = [
            row
            for row in runner.rows
            if row["type"] == "probe" and row["check"] == 10000
        ]
        if {row["architecture"] for row in probe_rows} != {"small", "large"}:
            raise AssertionError("full plasticity evidence missing")
        receipt = {
            "source_commit": runner.source_commit,
            "source_sha256": runner.source_hash,
            "regime": regime,
            "seed": seed,
            "full_branch_horizon": True,
            "plasticity_probe": True,
            "restart": interrupted.exists(),
            "branch_counters": records[0]["branch_counters"],
            "reference_raw_steps": runner.trainer.raw_env_steps,
        }
        (output / "qualification_result.json").write_text(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--regime", choices=("reference_small", "reference_large"), required=True
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    qualify(args.regime, args.seed, args.output)


if __name__ == "__main__":
    main()
