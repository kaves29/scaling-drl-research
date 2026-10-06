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

Also (each optional):
  --diagnostics_off  training it/s with the actor diagnostics disabled, for their overhead;
  --fork_timing      wall time to save and to restore the complete training state with the
                     replay buffer filled to the size it has at a fork at 95% of the budget
                     (synthetic transitions; capped at buffer.max_length), and its size on disk;
  --eval_cost        one post-fork evaluation (fork.eval_episodes episodes, fresh env), and the
                     projected F1 cost: 26 evaluations per arm against the arm's 25%-of-budget
                     training time at the measured it/s.

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

from experiments.exp12.precision import configure_compilation_cache, set_matmul_precision  # noqa: E402

set_matmul_precision()
configure_compilation_cache()

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


def _jobs_per_gpu(peak):
    """floor(0.9 * device memory / this process's peak); None off GPU. A starting point only:
    it ignores XLA's preallocation (set XLA_PYTHON_CLIENT_PREALLOCATE=false for concurrent jobs)."""
    limit = (jax.devices()[0].memory_stats() or {}).get("bytes_limit")
    return int(0.9 * limit // peak) if limit and peak else None


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
    extra = {}
    if args.diagnostics_off:
        extra.update(_train_rate(args, cfg, blocks, width, ["diagnostics.enabled=false"]))
        extra["diagnostics_overhead_pct"] = 100 * (extra["train_it_per_s_diagnostics_off"] / it_per_s - 1)
    if args.fork_timing:
        extra.update(_fork_timing(trainer, cfg))
    if args.eval_cost:
        extra.update(_eval_cost(trainer, cfg, it_per_s))
    result = {
        "arch": name, "env": args.env, "device": str(jax.devices()[0]), **extra,
        "train_it_per_s_probes_off": it_per_s,
        "probe_check_s": check_s,
        "probe_check_s_all": times,
        "num_interaction_steps": n,
        "projected_run_train_h": run_train_s / 3600,
        "projected_run_probe_h": probe_total_s / 3600,
        "probe_overhead_pct_of_wallclock": 100 * probe_total_s / (run_train_s + probe_total_s),
        "peak_device_bytes": _peak_memory_bytes(),
        "recommended_jobs_per_gpu_upper_bound": _jobs_per_gpu(_peak_memory_bytes()),
    }
    trainer.close()
    return result


def _train_rate(args, cfg, blocks, width, extra):
    from experiments.exp1 import compose_config
    from experiments.exp12.trainer import Exp12Trainer

    config_path = str(Path(__file__).resolve().parents[1] / "configs")
    c = compose_config(config_path, "base_exp12", _overrides(args, blocks, width, extra))
    np.random.seed(c.seed)
    random.seed(c.seed)
    t = Exp12Trainer(c, tempfile.mkdtemp(prefix="profile_rate_"))
    t.start()
    warm = int(c.buffer.min_length) + args.warmup_steps
    t.train(warm)
    jax.block_until_ready(t._sac_agent.critic.params)
    t0 = time.perf_counter()
    t.train(warm + args.train_steps)
    jax.block_until_ready(t._sac_agent.critic.params)
    t.close()
    return {"train_it_per_s_diagnostics_off": args.train_steps / (time.perf_counter() - t0)}


def _fork_timing(trainer, cfg):
    """Save and restore of the complete state with the buffer at its size for a fork at 95% of B."""
    from experiments.exp12.state import BUFFER_ARRAYS, latest_state_dir
    from experiments.exp12.trainer import Exp12Trainer

    buf = trainer.buffer
    n = min(int(0.95 * int(cfg.num_interaction_steps)), int(cfg.buffer.max_length))
    rng = np.random.default_rng(0)
    for k in BUFFER_ARRAYS:
        arr = getattr(buf, k)
        arr[:n] = rng.standard_normal((n,) + arr.shape[1:]).astype(arr.dtype)
    buf._num_in_buffer, buf._current_idx = n, n % int(cfg.buffer.max_length)
    root = Path(tempfile.mkdtemp(prefix="profile_fork_"))
    t0 = time.perf_counter()
    trainer.save(root / "state")
    save_s = time.perf_counter() - t0
    state = latest_state_dir(root / "state")
    size = sum(f.stat().st_size for f in state.rglob("*") if f.is_file())
    other = Exp12Trainer(cfg, str(root / "restored"))
    t0 = time.perf_counter()
    other.restore(state, new_wandb_run=True)
    restore_s = time.perf_counter() - t0
    other.close()
    return {"fork_buffer_transitions": n, "fork_save_s": save_s, "fork_restore_s": restore_s,
            "fork_state_bytes": size}


def _eval_cost(trainer, cfg, it_per_s):
    from experiments.exp12 import fork

    n = int(cfg.num_interaction_steps)
    plan = fork.fork_plan(cfg, n // 2, cfg.probe.checks // 2)
    fork.post_fork_eval(trainer, plan, 0, "control")  # compile warm-up
    t0 = time.perf_counter()
    fork.post_fork_eval(trainer, plan, 1, "control")
    eval_s = time.perf_counter() - t0
    evals = plan["horizon_steps"] // plan["eval_every_steps"] + 1
    arm_train_s = plan["horizon_steps"] / it_per_s
    return {"post_fork_eval_s": eval_s, "post_fork_evals_per_arm": evals,
            "post_fork_eval_episodes": plan["eval_episodes"],
            "post_fork_eval_total_h_per_arm": evals * eval_s / 3600,
            "post_fork_arm_train_h": arm_train_s / 3600,
            "post_fork_eval_overhead_pct": 100 * evals * eval_s / (arm_train_s + evals * eval_s)}


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
    parser.add_argument("--diagnostics_off", action="store_true", help="also time training with diagnostics off")
    parser.add_argument("--fork_timing", action="store_true", help="time complete-state save/restore at fork size")
    parser.add_argument("--eval_cost", action="store_true", help="time one post-fork evaluation (F1 cost)")
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
