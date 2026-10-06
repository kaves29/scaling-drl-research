#!/usr/bin/env python3
"""Probe calibration checks on FRESH (untrained) critics only, for the CUDA plan.

Training never starts in either mode, so nothing about degradation is measured.
The buffer is filled to buffer.min_length transitions exactly as at a run's
fresh check (SimBa random warm-up, obs_rms updated). WandB is stubbed.

--mode range: the probe's dynamic range. The untrained critic is probed with the
  real settings at each --pools size, giving per-round P and b, the IQM, and the
  std and range of P across rounds. At the configured pool size (25,600) this is
  the run's own check-0 probe. The verdict applies the pre-specified rule
  (docs/exp12_decisions.md, 2026-10-04): PASS when 0.1*b <= IQM(P) <= 0.9*b at
  the configured pool, at every size.

--mode null: the trigger's behaviour when neither critic has lost plasticity.
  Each pair is two independently initialised fresh critics of the same
  architecture, compared exactly like current vs fresh (same pool, target and
  minibatches; trigger settings from the config). Every pair's per-round P, b
  and L, its IQM and its bootstrap interval are appended to
  <out_dir>/null_pairs_<arch>.jsonl as soon as the pair finishes, so a restarted
  run resumes and alternative rules can be evaluated offline. The summary gives
  the per-check fire rate and the 95th percentile of the pairs' L (the would-be
  null-calibrated threshold). Neither is adopted automatically.

Examples (CUDA, from main/):
    python scripts/probe_fresh_checks.py --mode range --env dog-run --env_group dmc_hard \\
        --archs D2W512 D4W1024 D4W1536 --out_dir /abs/path/fresh_range_dog_run
    python scripts/probe_fresh_checks.py --mode null --env dog-run --env_group dmc_hard \\
        --archs D2W512 D4W1024 D4W1536 --null_pairs 100 --out_dir /abs/path/fresh_null_dog_run
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

from experiments.exp12.precision import configure_compilation_cache, set_matmul_precision  # noqa: E402

set_matmul_precision()
configure_compilation_cache()

import jax  # noqa: E402
import numpy as np  # noqa: E402

ARCHS = {"D2W512": (2, 512), "D4W1024": (4, 1024), "D4W1536": (4, 1536)}
RANGE_LOW, RANGE_HIGH = 0.1, 0.9
NULL_FIRE_RATE_LIMIT = 0.05
NULL_PAIR_KEY_OFFSET = 10_000


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


def range_mode(args, name, blocks, width, out_dir):
    from experiments.exp12.probe import critic_optimizer, iqm, probe_config, run_probe

    trainer = _fresh_trainer(args, blocks, width, args.seed)
    critic = trainer._sac_agent.critic
    tx = critic_optimizer(trainer.cfg.agent)
    configured_pool = probe_config(trainer.cfg).pool_size
    rows = []
    for pool in args.pools:
        out = run_probe(trainer.agent, trainer.buffer, critic.network_def,
                        {"fresh": (critic.network_def, critic.params)}, tx, args.seed, 0,
                        _probe_cfg(trainer, pool))["fresh"]
        score, b = out["score"], out["b"]
        ratio = iqm(score) / iqm(b)
        rows.append({
            "arch": name, "env": args.env, "seed": args.seed, "pool_size": pool,
            "is_configured_pool": pool == configured_pool,
            "score_rounds": score.tolist(), "b_rounds": b.tolist(), "final_loss_rounds": out["final_loss"].tolist(),
            "score_iqm": iqm(score), "b_iqm": iqm(b), "score_over_b": ratio,
            "score_std": float(np.std(score, ddof=1)), "score_range": float(score.max() - score.min()),
            "within_10_90_pct_of_b": bool(RANGE_LOW <= ratio <= RANGE_HIGH),
        })
        print(json.dumps(rows[-1]), flush=True)
    trainer.close()
    with open(out_dir / f"range_{name}.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def _done_pairs(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["pair"]: r for r in map(json.loads, f) if r.get("pair") is not None}


def null_mode(args, name, blocks, width, out_dir):
    from experiments.exp12.probe import critic_optimizer, iqm, probe_config, run_probe, summarize
    from experiments.exp12.trigger import bootstrap_interval, trigger_config, triggered

    trainer = _fresh_trainer(args, blocks, width, args.seed)
    critic = trainer._sac_agent.critic
    tx = critic_optimizer(trainer.cfg.agent)
    cfg = probe_config(trainer.cfg)
    tc = trigger_config(trainer.cfg)
    obs = jax.numpy.zeros((1, trainer.buffer._observations.shape[1]))
    act = jax.numpy.zeros((1, trainer.buffer._actions.shape[1]))
    path = out_dir / f"null_pairs_{name}.jsonl"
    done = _done_pairs(path)
    for pair in range(args.null_pairs):
        if pair in done:
            continue
        keys = jax.random.split(jax.random.PRNGKey(NULL_PAIR_KEY_OFFSET + pair), 2)
        a, b = (critic.network_def.init(k, obs, act)["params"] for k in keys)
        result = run_probe(trainer.agent, trainer.buffer, critic.network_def,
                           {"current": (critic.network_def, b), "fresh": (critic.network_def, a)},
                           tx, args.seed, 1 + pair, cfg)
        s = summarize(result)
        low, high = bootstrap_interval(s["loss_rounds"], args.seed, 1 + pair, tc.resamples, tc.confidence)
        row = {
            "arch": name, "env": args.env, "seed": args.seed, "pair": pair, "check_index": 1 + pair,
            "init_keys": [NULL_PAIR_KEY_OFFSET + pair, 0, 1],
            "score_fresh_rounds": s["score_fresh_rounds"].tolist(),
            "score_current_rounds": s["score_current_rounds"].tolist(),
            "b_rounds": result["fresh"]["b"].tolist(),
            "loss_rounds": s["loss_rounds"].tolist(), "loss_iqm": s["loss_iqm"], "valid": s["valid"],
            "ci_low": low, "ci_high": high, "resamples": tc.resamples, "confidence": tc.confidence,
            "null_threshold": tc.null_threshold, "fired": triggered(low, tc.null_threshold) and s["valid"],
        }
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        done[pair] = row
        print(json.dumps(row), flush=True)
    rows = [done[p] for p in sorted(done) if p < args.null_pairs]
    losses = np.array([r["loss_iqm"] for r in rows])
    rate = float(np.mean([r["fired"] for r in rows]))
    summary = {
        "arch": name, "env": args.env, "null_pairs": len(rows), "per_check_fire_rate": rate,
        "fire_rate_exceeds_5pct": rate > NULL_FIRE_RATE_LIMIT,
        "would_be_null_threshold_p95_of_L": float(np.percentile(losses, 95)),
        "iqm_of_L": iqm(losses), "std_of_L": float(np.std(losses, ddof=1)),
        "note": "Reported only. Adopting a null-calibrated threshold is the lead's decision.",
    }
    with open(out_dir / f"null_summary_{name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)
    trainer.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("range", "null"), required=True)
    parser.add_argument("--env", default="dog-run")
    parser.add_argument("--env_group", default="dmc_hard")
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--seed", type=int, default=990)
    parser.add_argument("--pools", nargs="+", type=int, default=[1_600, 6_400, 25_600])
    parser.add_argument("--null_pairs", type=int, default=100)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        raise ValueError("--out_dir must be an absolute path")
    out_dir.mkdir(parents=True, exist_ok=True)
    _stub_wandb()
    results = []
    for name in args.archs:
        blocks, width = ARCHS[name]
        results.append((range_mode if args.mode == "range" else null_mode)(args, name, blocks, width, out_dir))
    if args.mode == "range":
        configured = [r for rows in results for r in rows if r["is_configured_pool"]]
        verdict = {
            "rule": f"{RANGE_LOW} * b <= IQM(P) <= {RANGE_HIGH} * b at the configured pool, at every size",
            "per_size": {r["arch"]: r["within_10_90_pct_of_b"] for r in configured},
            "PASS": bool(configured) and all(r["within_10_90_pct_of_b"] for r in configured)
            and len(configured) == len(args.archs),
        }
        with open(out_dir / "range_verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)
        print(json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
