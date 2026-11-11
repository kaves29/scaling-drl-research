"""Run one member of the owner-approved revised online Exp2 pilot population.

Usage:
  python scripts/run_revised_exp2_pilot.py --run-index 0 --root /kaggle/working/exp2_pilot
"""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp1 import compose_config
from experiments.online_exp2 import OnlineExp2
from experiments.exp12.precision import configure_compilation_cache, set_matmul_precision
from experiments.exp12.revised_exp2_population import population

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def run_name(member):
    return (f"{member['environment']}_seed{member['seed']}_"
            f"{member['reference_regime']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--time-budget-seconds", type=int, default=0,
                        help="0 = no limit; alarm after a complete periodic save window")
    args = parser.parse_args()
    members = population()
    if not 0 <= args.run_index < len(members):
        raise ValueError(f"run index outside population: {args.run_index}")
    member = members[args.run_index]
    root = Path(args.root).resolve() / run_name(member)
    if (root / "analysis.json").exists():
        print(f"SKIP complete: {run_name(member)}", flush=True)
        return
    cfg = compose_config(str(CONFIG_DIR), "online_exp2_revised_pilot", [
        f"env_name={member['environment']}", f"seed={member['seed']}",
        f"online_exp2.environments=[{member['environment']}]",
        f"online_exp2.training_seeds=[{member['seed']}]",
        f"online_exp2.reference_regime={member['reference_regime']}",
    ], allow_pilot=True)
    set_matmul_precision()
    configure_compilation_cache()

    if args.time_budget_seconds > 0:
        def timeout(signum, frame):
            raise TimeoutError(f"session time budget reached for {run_name(member)}")
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.time_budget_seconds)

    print(f"START {run_name(member)} -> {root}", flush=True)
    runner = OnlineExp2(cfg, str(root))
    runner.run()
    print(f"DONE {run_name(member)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
