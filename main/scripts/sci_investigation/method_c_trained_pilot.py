"""Development-only Method C check on two naturally trained Hopper-Hop states.

The configured 250,000-interaction Exp1 budget determines the unchanged 20-check
grid. Training stops after check 2 solely to bound this validation workload.
This entry point does not mark a parent complete or launch an Exp2 arm.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.hardware import configure_hardware_env

configure_hardware_env()

import jax
import numpy as np

from experiments.exp1 import compose_config, record_metadata
from experiments.exp12.precision import configure_compilation_cache, runtime_info, set_matmul_precision
from experiments.exp12.run_probes import RunProbes
from experiments.exp12.trainer import Exp12Trainer
from utils.run_metadata import code_version


def validation_steps(cfg):
    """Use the full-budget Exp1 grid; stop only after its second scheduled check."""
    if (cfg.env_name, int(cfg.seed), int(cfg.critic_num_blocks), int(cfg.critic_hidden_dim)) != (
        "hopper-hop", 990, 4, 1536
    ):
        raise ValueError("this pilot is pinned to Hopper-Hop D4W1536 seed 990")
    if (int(cfg.num_interaction_steps), int(cfg.probe.checks), int(cfg.probe.rounds),
            int(cfg.probe.steps), int(cfg.probe.pool_size), int(cfg.probe.batch_size)) != (
        250_000, 20, 5, 2000, 25_600, 256
    ):
        raise ValueError("unexpected Method C probe or full-budget check grid")
    if (int(cfg.trigger.resamples), float(cfg.trigger.confidence),
            int(cfg.trigger.consecutive_checks), float(cfg.trigger.null_threshold)) != (10000, .95, 2, 0.0):
        raise ValueError("unexpected bootstrap or trigger settings")
    if cfg.run_role != "dev" or cfg.testing.force_trigger_check is not None:
        raise ValueError("development role and natural checks are required")
    first = int(cfg.num_interaction_steps) // int(cfg.probe.checks)
    if first <= int(cfg.buffer.min_length):
        raise ValueError("fresh critic must be captured before the first check")
    return first, 2 * first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    code = code_version()
    if code["commit"] != args.expected_commit or code["dirty"] is not False:
        parser.error("exact clean source revision required")
    if not args.out_dir.is_absolute() or args.out_dir.exists():
        parser.error("--out-dir must be an unused absolute path")
    set_matmul_precision()
    runtime = runtime_info()
    if runtime["platform"] != "gpu" or "A100" not in runtime["device_kind"]:
        parser.error("an NVIDIA A100 GPU backend is required")

    configure_compilation_cache()
    cfg = compose_config(str(Path(__file__).resolve().parents[2] / "configs"), "base_exp12", [
        "env_name=hopper-hop", "env=dmc_medium", "seed=990", "run_role=dev",
        "critic_num_blocks=4", "critic_hidden_dim=1536", "probe.steps=2000",
    ])
    first, second = validation_steps(cfg)
    args.out_dir.mkdir(parents=True)
    run_dir = args.out_dir / "run"
    run_dir.mkdir()
    record_metadata(cfg, run_dir, resumed=False)
    np.random.seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    trainer = Exp12Trainer(cfg, str(run_dir))
    try:
        trainer.start()
        probes = RunProbes(trainer, str(run_dir))
        checkpoints = []

        def after_step(current):
            probes.maybe_check(current)
            if current.interaction_step in (first, second):
                probes.write_csv()
                state = current.save(run_dir / "state", keep_previous=True)
                checkpoints.append({"check_index": current.interaction_step // first,
                                    "interaction_step": current.interaction_step,
                                    "update_step": current.update_step,
                                    "state_dir": str(state.relative_to(args.out_dir))})

        trainer.train(second, after_step=after_step, before_first_update=probes.capture_fresh)
        if len(checkpoints) != 2 or len(probes.records) != 3:
            raise RuntimeError("fresh check and both trained checks were not captured")
        receipt = {"code": code, "runtime": runtime, "role": "exploratory Method C validation",
                   "configured_interaction_budget": int(cfg.num_interaction_steps),
                   "check_grid": [first, second], "checkpoints": checkpoints,
                   "f_star": probes.f_star, "probe_records": probes.records,
                   "note": "stopped at check 2; no completed parent or Exp2 arm"}
        (args.out_dir / "receipt.json").write_text(json.dumps(receipt, allow_nan=False, indent=2))
        print(json.dumps({"checkpoints": checkpoints, "f_star": probes.f_star}), flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
