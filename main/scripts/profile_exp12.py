#!/usr/bin/env python3
"""Measures Exp 1/2 compute cost per critic size (section 6 of the brief).

For each architecture: steady-state training it/s with probes off, wall time of
one full probe check (rounds x {current, fresh} x steps, after a compile
warm-up), the projected probe overhead over a whole run (checks per run x check
time / run training time), and peak device memory (GPU only). With --angle1 it
also times the existing angle_1 loop on the same config, for the
"probes-off training unchanged" comparison (angle_1 only accepts its own core
environments, e.g. dog-run, dog-trot, humanoid-run). WandB is stubbed, nothing is logged
remotely, and outputs go to a temporary directory.

Example (CUDA):
    python scripts/profile_exp12.py --env dog-run --env_group dmc_hard \\
        --archs D2W512 D4W1024 D6W1536 --train_steps 2000 --angle1 --out profile_dog_run.json
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.hardware import configure_hardware_env  # noqa: E402

configure_hardware_env()

import jax  # noqa: E402
import numpy as np  # noqa: E402

ARCHS = {"D2W512": (2, 512), "D4W1024": (4, 1024), "D6W1536": (6, 1536)}


class _Run:
    name, id, summary = "profile", "profile", {}

    def log(self, *a, **k):
        pass

    def finish(self):
        pass


def _stub_wandb():
    import wandb

    def init(*a, **k):
        wandb.run = _Run()
        return wandb.run

    mock.patch("wandb.init", side_effect=init).start()
    mock.patch("wandb.log").start()


def _peak_memory_bytes():
    stats = jax.devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use")


def _overrides(args, blocks, width, extra=()):
    return [f"env_name={args.env}", f"env={args.env_group}", f"seed={args.seed}",
            f"critic_num_blocks={blocks}", f"critic_hidden_dim={width}", *args.override, *extra]


def profile_arch(args, name, blocks, width):
    from experiments.exp1 import compose_config
    from experiments.exp12.probe import run_probe
    from experiments.exp12.run_probes import RunProbes
    from experiments.exp12.trainer import Exp12Trainer

    config_path = str(Path(__file__).resolve().parents[1] / "configs")
    cfg = compose_config(config_path, "base_exp12", _overrides(args, blocks, width))
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    trainer = Exp12Trainer(cfg, tempfile.mkdtemp(prefix="profile_exp12_"))
    trainer.start()
    warm = int(cfg.buffer.min_length) + args.warmup_steps
    trainer.train(warm)
    jax.block_until_ready(trainer._sac_agent.critic.params)
    t0 = time.perf_counter()
    trainer.train(warm + args.train_steps)
    jax.block_until_ready(trainer._sac_agent.critic.params)
    train_s = time.perf_counter() - t0
    it_per_s = args.train_steps / train_s

    probes = RunProbes(trainer, tempfile.mkdtemp(prefix="profile_probe_"))
    fresh = trainer._sac_agent.critic.params
    critics = {"current": (probes.critic_def, trainer._sac_agent.critic.params), "fresh": (probes.critic_def, fresh)}
    run_probe(trainer.agent, trainer.buffer, probes.critic_def, critics, probes.tx, cfg.seed, 1, probes.cfg)
    times = []
    for rep in range(args.probe_repeats):
        t0 = time.perf_counter()
        run_probe(trainer.agent, trainer.buffer, probes.critic_def, critics, probes.tx, cfg.seed, 2 + rep, probes.cfg)
        times.append(time.perf_counter() - t0)
    check_s = float(np.median(times))
    n = int(cfg.num_interaction_steps)
    run_train_s = n / it_per_s
    probe_total_s = (probes.cfg.checks + 1) * check_s
    result = {
        "arch": name, "env": args.env, "device": str(jax.devices()[0]),
        "train_it_per_s_probes_off": it_per_s,
        "probe_check_s": check_s,
        "probe_check_s_all": times,
        "num_interaction_steps": n,
        "projected_run_train_h": run_train_s / 3600,
        "projected_run_probe_h": probe_total_s / 3600,
        "probe_overhead_pct_of_wallclock": 100 * probe_total_s / (run_train_s + probe_total_s),
        "peak_device_bytes": _peak_memory_bytes(),
    }
    trainer.close()
    return result


def profile_angle1(args, blocks, width):
    """Wall time of a short angle_1 run vs the same exp1 run (probes off), end to end."""
    from hydra.core.global_hydra import GlobalHydra

    from experiments import angle_1, exp1

    config_path = str(Path(__file__).resolve().parents[1] / "configs")
    steps = int(args.compare_steps)
    common = _overrides(args, blocks, width, [f"env.num_env_steps={2 * steps}"])
    out = {}
    for label, fn, config, extra in [
        ("angle_1", angle_1.run, "base_sac", ["updates_per_interaction_step=2"]),
        ("exp1_probes_off", exp1.run, "base_exp12", ["probe.enabled=false"]),
    ]:
        overrides = common + extra
        GlobalHydra.instance().clear()
        t0 = time.perf_counter()
        fn({"experiment": label, "config_path": config_path, "config_name": config, "overrides": overrides,
            "checkpoint_dir": tempfile.mkdtemp(prefix=f"cmp_{label}_"), "checkpoint_interval": 10**9,
            "checkpoint_start_frac": 0.99})
        out[f"{label}_wall_s"] = time.perf_counter() - t0
    out["ratio_exp1_over_angle1"] = out["exp1_probes_off_wall_s"] / out["angle_1_wall_s"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="dog-run")
    parser.add_argument("--env_group", default="dmc_hard")
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--seed", type=int, default=990)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--probe_repeats", type=int, default=2)
    parser.add_argument("--angle1", action="store_true", help="also time angle_1 vs exp1 (probes off)")
    parser.add_argument("--compare_steps", type=int, default=6000)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    _stub_wandb()
    results = []
    for name in args.archs:
        blocks, width = ARCHS[name]
        r = profile_arch(args, name, blocks, width)
        if args.angle1:
            r.update(profile_angle1(args, blocks, width))
        print(json.dumps(r), flush=True)
        results.append(r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
