#!/usr/bin/env python3
"""Probe checks on FRESH (untrained) critics only, for the CUDA plan.

--mode range: the probe's dynamic range at the real fresh check. The buffer is
  filled to buffer.min_length transitions exactly as in a run (SimBa random
  warm-up, obs_rms updated), then the untrained critic is probed with the real
  settings at each --pools size: per-round scores, IQM, std and range across the
  5 rounds. With check index 0 and pool 25,600 this is the run's own check-0 probe.

--mode null: false-trigger rate of the Exp 1 trigger when neither critic has
  lost plasticity. Each pair is two independently initialised fresh critics of
  the same architecture, probed on the same pool, target and minibatches as a
  real check. L = P(fresh A) - P(fresh B) has zero mean by construction, so the
  share of pairs whose 95% lower bound of L is > 0 is the false-trigger rate.

Training never starts in either mode, so nothing about degradation is
measured. WandB is stubbed.

Examples (CUDA, from main/):
    python scripts/probe_fresh_checks.py --mode range --env dog-run --env_group dmc_hard \\
        --archs D2W512 D4W1024 D6W1536 --out fresh_range_dog_run.json
    python scripts/probe_fresh_checks.py --mode null --env dog-run --env_group dmc_hard \\
        --archs D2W512 D4W1024 D6W1536 --null_pairs 20 --out fresh_null_dog_run.json
"""

import argparse
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.hardware import configure_hardware_env  # noqa: E402

configure_hardware_env()

import jax  # noqa: E402
import numpy as np  # noqa: E402

ARCHS = {"D2W512": (2, 512), "D4W1024": (4, 1024), "D6W1536": (6, 1536)}


class _Run:
    name, id, summary = "fresh-checks", "fresh-checks", {}

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


def _fresh_trainer(args, blocks, width, seed):
    """A trainer whose buffer holds min_length warm-up transitions and whose critic is untrained."""
    from experiments.exp1 import compose_config
    from experiments.exp12.trainer import Exp12Trainer

    config_path = str(Path(__file__).resolve().parents[1] / "configs")
    cfg = compose_config(config_path, "base_exp12", [
        f"env_name={args.env}", f"env={args.env_group}", f"seed={seed}", "run_role=dev",
        f"critic_num_blocks={blocks}", f"critic_hidden_dim={width}", *args.override,
    ])
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    trainer = Exp12Trainer(cfg, tempfile.mkdtemp(prefix="fresh_checks_"))
    trainer.start()

    class Ready(Exception):
        pass

    def stop(_):
        raise Ready

    try:
        trainer.train(int(cfg.num_interaction_steps), before_first_update=stop)
    except Ready:
        pass
    assert trainer.update_step == 0 and len(trainer.buffer) == int(cfg.buffer.min_length)
    return trainer


def _probe_cfg(trainer, pool_size):
    from experiments.exp12.probe import ProbeConfig, probe_config

    base = probe_config(trainer.cfg)
    # gcd keeps the evaluation chunking identical to a real run at the configured pool size.
    return ProbeConfig(base.rounds, base.steps, base.checks, pool_size, base.batch_size, base.target_scale,
                       math.gcd(pool_size, base.eval_chunk))


def range_mode(args, name, blocks, width):
    from experiments.exp12.probe import critic_optimizer, iqm, run_probe

    trainer = _fresh_trainer(args, blocks, width, args.seed)
    critic = trainer._sac_agent.critic
    tx = critic_optimizer(trainer.cfg.agent)
    rows = []
    for pool in args.pools:
        cfg = _probe_cfg(trainer, pool)
        out = run_probe(trainer.agent, trainer.buffer, critic.network_def,
                        {"fresh": (critic.network_def, critic.params)}, tx, args.seed, 0, cfg)["fresh"]
        score = out["score"]
        rows.append({
            "arch": name, "env": args.env, "seed": args.seed, "pool_size": pool,
            "score_rounds": score.tolist(), "score_iqm": iqm(score), "score_std": float(np.std(score, ddof=1)),
            "score_range": float(score.max() - score.min()), "b_rounds": out["b"].tolist(),
            "final_loss_rounds": out["final_loss"].tolist(),
            "first_minus_last_minibatch_loss": float(np.mean(out["losses"][:, :10]) - np.mean(out["losses"][:, -10:])),
        })
        print(json.dumps(rows[-1]), flush=True)
    trainer.close()
    return rows


def null_mode(args, name, blocks, width):
    from experiments.exp12.probe import critic_optimizer, iqm, probe_config, run_probe, summarize
    from experiments.exp12.trigger import bootstrap_interval

    trainer = _fresh_trainer(args, blocks, width, args.seed)
    critic = trainer._sac_agent.critic
    tx = critic_optimizer(trainer.cfg.agent)
    cfg = probe_config(trainer.cfg)
    obs = jax.numpy.zeros((1, trainer.buffer._observations.shape[1]))
    act = jax.numpy.zeros((1, trainer.buffer._actions.shape[1]))
    rows = []
    for pair in range(args.null_pairs):
        keys = jax.random.split(jax.random.PRNGKey(10_000 + pair), 2)
        a, b = (critic.network_def.init(k, obs, act)["params"] for k in keys)
        result = run_probe(trainer.agent, trainer.buffer, critic.network_def,
                           {"current": (critic.network_def, b), "fresh": (critic.network_def, a)},
                           tx, args.seed, 1 + pair, cfg)
        s = summarize(result)
        low, high = bootstrap_interval(s["loss_rounds"], seed=args.seed, check_index=1 + pair)
        rows.append({"arch": name, "pair": pair, "loss_rounds": s["loss_rounds"].tolist(),
                     "loss_iqm": s["loss_iqm"], "ci_low": low, "ci_high": high, "triggered": bool(low > 0)})
        print(json.dumps(rows[-1]), flush=True)
    rate = float(np.mean([r["triggered"] for r in rows]))
    summary = {"arch": name, "env": args.env, "null_pairs": args.null_pairs, "false_trigger_rate": rate,
               "spread_of_loss_iqm": float(np.std([r["loss_iqm"] for r in rows], ddof=1)),
               "median_abs_loss_iqm": float(np.median(np.abs([r["loss_iqm"] for r in rows]))),
               "iqm_of_loss_iqm": iqm([r["loss_iqm"] for r in rows])}
    print(json.dumps(summary), flush=True)
    trainer.close()
    return rows + [summary]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("range", "null"), required=True)
    parser.add_argument("--env", default="dog-run")
    parser.add_argument("--env_group", default="dmc_hard")
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--seed", type=int, default=990)
    parser.add_argument("--pools", nargs="+", type=int, default=[1_600, 6_400, 25_600])
    parser.add_argument("--null_pairs", type=int, default=20)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    _stub_wandb()
    rows = []
    for name in args.archs:
        blocks, width = ARCHS[name]
        rows += (range_mode if args.mode == "range" else null_mode)(args, name, blocks, width)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
