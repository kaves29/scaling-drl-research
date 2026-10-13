#!/usr/bin/env python3
"""CPU study: does a non-bit-identical resume shift Exp 1 outcomes? (evidence, not a gate)

A cross-process resume on CUDA can differ from an uninterrupted run only through floating-point
rounding in the compiled programs (the restored state, RNG keys, replay, environments and
counters are exact; tests cover that). This study emulates that rounding directly: a run is
killed mid-budget, and on resume every float32 leaf of the actor, critic, target critic and
temperature states (parameters and AdamW moments) is moved by one ULP in a random direction
drawn from an RNG independent of training. Everything else is restored exactly.

For each seed it records, for the uninterrupted run, an exact resume and a perturbed resume:
post-resume evaluation returns, post-resume probe plasticity loss (IQM per check) and the
critic's distance from the uninterrupted run. The perturbation's effect on an outcome is the
paired difference (perturbed - uninterrupted) across seeds; it is compared with the
between-seed spread of the same outcome. Tiny networks and budgets: this measures the
mechanism, not the production effect size.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "tests"))

STEPS, INTERVAL, CRASH_STEP = 600, 30, 300
ENV = ("myo-reach", "myosuite_simba")


def overrides(seed, results_root):
    return [
        f"env_name={ENV[0]}", f"env={ENV[1]}", f"seed={seed}", "run_role=dev",
        "actor_num_blocks=1", "actor_hidden_dim=16", "critic_num_blocks=1", "critic_hidden_dim=16",
        f"env.num_env_steps={2 * STEPS}", "buffer.min_length=10", "buffer.max_length=2000",
        "buffer.sample_batch_size=16", "evaluation_per_interaction_step=60",
        "logging_per_interaction_step=30", "num_eval_episodes=3",
        "probe.steps=40", "probe.pool_size=64", "probe.batch_size=16", "probe.eval_chunk=32",
        f"results_root={results_root}",
    ]


def _perturb(tree, rng, scale=None):
    """One ULP in a random direction (scale None), or a relative change of +-scale."""
    import jax
    import numpy as np

    def leaf(x):
        a = np.asarray(x)
        if a.dtype != np.float32 or a.size == 0:
            return x
        sign = np.where(rng.random(a.shape) < 0.5, np.float32(-1), np.float32(1))
        if scale is None:
            moved = np.nextafter(a, sign * np.float32(np.inf))
        else:
            moved = a * (np.float32(1) + sign * np.float32(scale))
        return jax.numpy.asarray(moved.astype(np.float32))

    return jax.tree_util.tree_map(leaf, tree)


def run_child(spec_path):
    """One exp1 process: optionally killed at CRASH_STEP, optionally perturbed on restore."""
    import numpy as np

    from exp12_helpers import CONFIG_PATH, patch_wandb

    spec = json.loads(Path(spec_path).read_text())
    patch_wandb()
    from experiments import exp1
    from experiments.exp12.trainer import Exp12Trainer

    if spec.get("crash_step") is not None:
        original_train = Exp12Trainer.train

        def train(self, last_step, after_step=None, **kwargs):
            def hook(t):
                if after_step is not None:
                    after_step(t)
                if t.interaction_step == spec["crash_step"]:
                    os._exit(3)

            return original_train(self, last_step, after_step=hook, **kwargs)

        Exp12Trainer.train = train
    if spec.get("perturb_seed") is not None:
        original_restore = Exp12Trainer.restore

        def restore(self, state_dir, **kwargs):
            original_restore(self, state_dir, **kwargs)
            rng = np.random.default_rng([20261013, spec["perturb_seed"]])  # independent of training
            agent = self._sac_agent
            for name in ("_actor", "_critic", "_target_critic", "_temperature"):
                setattr(agent, name, _perturb(getattr(agent, name), rng, spec.get("perturb_scale")))

        Exp12Trainer.restore = restore
    exp1.run({"experiment": "exp1", "config_path": CONFIG_PATH, "config_name": "base_exp12",
              "overrides": spec["overrides"], "checkpoint_dir": spec["checkpoint_dir"],
              "checkpoint_interval": INTERVAL, "checkpoint_start_frac": 0.0})


def _launch(work, name, seed, **spec):
    run_dir = work / name
    spec = {"overrides": overrides(seed, str(work / (name + "_results"))),
            "checkpoint_dir": str(run_dir), **spec}
    path = work / (name + ".json")
    path.write_text(json.dumps(spec))
    env = dict(os.environ, JAX_PLATFORMS="cpu", MUJOCO_GL=os.environ.get("MUJOCO_GL", "disable"),
               EXP12_JAX_CACHE_DIR="off", WANDB_MODE="disabled")
    return subprocess.run([sys.executable, __file__, "child", str(path)], env=env,
                          capture_output=True, text=True)


def _outcomes(run_dir, reference=None):
    import numpy as np

    from exp12_helpers import load_agent_tree
    from experiments.exp12.state import latest_state_dir, load_meta

    state = latest_state_dir(Path(run_dir) / "state")
    meta = load_meta(state)
    evals = [r["return"] for r in meta["eval_rows"] if r["interaction_step"] > CRASH_STEP]
    checks = [r["loss_iqm"] for r in meta["extra_state"]["probe_records"]
              if r["check_index"] > 0 and r["interaction_step"] > CRASH_STEP]
    out = {"final_step": int(meta["interaction_step"]),
           "post_resume_eval_return": float(np.mean(evals)),
           "post_resume_probe_loss_iqm": float(np.mean(checks)),
           "post_resume_triggered_checks": sum(bool(r["triggered"]) for r in meta["extra_state"]["probe_records"]
                                               if r["check_index"] > 0 and r["interaction_step"] > CRASH_STEP)}
    params = load_agent_tree(state)["critic"]["params"]
    flat = np.concatenate([np.ravel(np.asarray(x, np.float64)) for x in _leaves(params)])
    out["_critic"] = flat
    if reference is not None:
        out["critic_relative_distance"] = float(np.linalg.norm(flat - reference) / np.linalg.norm(reference))
    return out


def _leaves(tree):
    import jax

    return jax.tree_util.tree_leaves(tree)


def study(seeds, out, exact_seeds, scale=None):
    rows = []
    work = Path(tempfile.mkdtemp(prefix="resume_perturbation_"))
    try:
        for seed in seeds:
            ref = _launch(work, f"s{seed}_reference", seed)
            if ref.returncode:
                raise RuntimeError(ref.stderr[-3000:])
            reference = _outcomes(work / f"s{seed}_reference")
            row = {"seed": seed, "uninterrupted": {k: v for k, v in reference.items() if k != "_critic"}}
            kinds = (("perturbed", seed),) + ((("exact", None),) if seed in exact_seeds else ())
            for kind, perturb in kinds:
                name = f"s{seed}_{kind}"
                killed = _launch(work, name, seed, crash_step=CRASH_STEP)
                if killed.returncode != 3:
                    raise RuntimeError(f"{name}: expected the kill at step {CRASH_STEP}: {killed.stderr[-2000:]}")
                resumed = _launch(work, name, seed, perturb_seed=perturb, perturb_scale=scale)
                if resumed.returncode:
                    raise RuntimeError(resumed.stderr[-3000:])
                result = _outcomes(work / name, reference["_critic"])
                row[kind] = {k: v for k, v in result.items() if k != "_critic"}
            rows.append(row)
            print(json.dumps(row), flush=True)
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    summary = summarize(rows)
    Path(out).write_text(json.dumps({"design": __doc__, "steps": STEPS, "crash_step": CRASH_STEP,
                                     "env": ENV, "perturbation": "1 ULP" if scale is None else f"relative {scale}", "rows": rows, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


def summarize(rows, resamples=10_000):
    import numpy as np

    rng = np.random.default_rng(0)
    summary = {}
    for metric in ("post_resume_eval_return", "post_resume_probe_loss_iqm"):
        base = np.array([r["uninterrupted"][metric] for r in rows])
        diff = np.array([r["perturbed"][metric] - r["uninterrupted"][metric] for r in rows])
        boot = rng.choice(diff, (resamples, diff.size)).mean(axis=1)
        summary[metric] = {
            "between_seed_sd": float(base.std(ddof=1)),
            "paired_difference_mean": float(diff.mean()),
            "paired_difference_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "paired_difference_sd": float(diff.std(ddof=1)),
            "seeds_changed": int(np.sum(diff != 0)),
        }
    summary["critic_relative_distance_perturbed"] = [r["perturbed"]["critic_relative_distance"] for r in rows]
    exact = [r["exact"] for r in rows if "exact" in r]
    summary["exact_resumes_identical"] = all(
        e["critic_relative_distance"] == 0.0
        and e["post_resume_eval_return"] == r["uninterrupted"]["post_resume_eval_return"]
        for e, r in zip(exact, [r for r in rows if "exact" in r]))
    summary["exact_resume_count"] = len(exact)
    summary["seeds"] = len(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    child = sub.add_parser("child")
    child.add_argument("spec")
    run = sub.add_parser("study")
    run.add_argument("--seeds", type=int, default=20)
    run.add_argument("--exact-seeds", type=int, default=3, help="seeds that also get an exact resume")
    run.add_argument("--scale", type=float, default=None,
                     help="relative perturbation instead of one ULP (e.g. 1e-3, about TF32 rounding)")
    run.add_argument("--first-seed", type=int, default=1001)
    run.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "child":
        run_child(args.spec)
    else:
        seeds = list(range(args.first_seed, args.first_seed + args.seeds))
        study(seeds, args.out, set(seeds[:args.exact_seeds]), args.scale)


if __name__ == "__main__":
    main()
