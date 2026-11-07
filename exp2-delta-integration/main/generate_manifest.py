"""Current five-target Exp1/Exp2 grid and shared path/config helpers.

The older Angle 1 manifest is retired; its exact source remains in Git history.
"""

import argparse
import json
import os
from pathlib import Path

from utils.paths import RESULTS_ROOT, require_absolute

DONE_MARKER = "DONE"


def _has_any_files(path):
    """Distinguish an empty interrupted directory from a begun run."""
    return any(p.is_file() for p in path.rglob("*"))


EXP12_ARCHS = [("D2W512", 2, 512), ("D4W1024", 4, 1024), ("D4W1536", 4, 1536)]
EXP12_ENVS = [
    ("dog-run", "dmc_hard"), ("dog-trot", "dmc_hard"), ("humanoid-run", "dmc_hard"),
    ("humanoid-walk", "dmc_hard"), ("humanoid-stand", "dmc_hard"),
    ("swimmer-swimmer15", "dmc_medium"), ("hopper-hop", "dmc_medium"),
    ("myo-key-turn", "myosuite_simba"), ("myo-pen-twirl", "myosuite_simba"),
    ("myo-pose-hard", "myosuite_simba"), ("myo-reach", "myosuite_simba"),
    ("h1-reach-v0", "humanoid_bench"), ("h1-run-v0", "humanoid_bench"),
]
EXP12_BUDGETS = {"dmc_hard": 1_000_000, "dmc_medium": 500_000, "myosuite_simba": 1_000_000,
                 "humanoid_bench": 2_000_000}
EXP12_FORKING = ("D4W1024", "D4W1536")
EXP12_SEEDS = [1, 2, 3, 4, 5]
EXP12_CHECKS = 20
EXP12_ACTION_REPEAT = 2


def exp12_paths(ckpt_root, arch_name, env_name, seed):
    run_dir = os.path.join(ckpt_root, "exp1", arch_name, env_name, f"seed_{seed}")
    arm_dir = os.path.join(ckpt_root, "exp2_arm", arch_name, env_name, f"seed_{seed}", "injected")
    return run_dir, arm_dir


def exp12_overrides(arch, env_name, env_group, seed, results_root):
    _, blocks, hidden = arch
    return [f"critic_num_blocks={blocks}", f"critic_hidden_dim={hidden}", f"env_name={env_name}",
            f"env={env_group}", f"seed={seed}", f"results_root={results_root}"]


def exp12_command(experiment, overrides, ckpt_dir, interval, log_path):
    cmd = f"python -u run.py --experiment {experiment} --config_name base_exp12 "
    cmd += "".join(f"--overrides {o} " for o in overrides)
    return (cmd + f"--checkpoint_dir {ckpt_dir} --checkpoint_interval {interval} "
            f"--checkpoint_start_frac 0.0 > {log_path} 2>&1")


def classify_exp12(ckpt_dir):
    """done (DONE marker), fresh (nothing on disk) or resume (exp1/exp2_arm resume from
    state/LATEST, from the saved fork state, or start over before their first save)."""
    ckpt = Path(ckpt_dir)
    if (ckpt / DONE_MARKER).exists():
        return "done"
    if not ckpt.exists() or not _has_any_files(ckpt):
        return "fresh"
    return "resume"


def _device_slug(device_kind):
    return "".join(c if c.isalnum() else "_" for c in device_kind).strip("_") or "unknown"


def add_exp12_grid(manifests, status, ckpt_root, results_root, injection_m=None, seeds=EXP12_SEEDS):
    """Exp 1 jobs into manifests["exp12_exp1_jobs.txt"]; injected-arm jobs (forks only) into
    manifests["exp2_arms_<device>.txt"]. status: state -> list of checkpoint dirs."""
    if injection_m is not None and injection_m not in ("last", "half", "all"):
        raise ValueError("--injection-m must be last, half or all")
    logs = os.path.join(ckpt_root, "logs")
    exp1_jobs = manifests.setdefault("exp12_exp1_jobs.txt", [])
    for env_name, env_group in EXP12_ENVS:
        interval = EXP12_BUDGETS[env_group] // EXP12_ACTION_REPEAT // EXP12_CHECKS  # one save per check
        for arch in EXP12_ARCHS:
            for seed in seeds:
                run_dir, arm_dir = exp12_paths(ckpt_root, arch[0], env_name, seed)
                overrides = exp12_overrides(arch, env_name, env_group, seed, results_root)
                state = classify_exp12(run_dir)
                status.setdefault(f"exp1 {state}", []).append(run_dir)
                if state != "done":
                    log = os.path.join(logs, f"exp1_{arch[0]}_{env_name}_seed{seed}.log")
                    exp1_jobs.append(exp12_command("exp1", overrides, run_dir, interval, log))
                fork_json = Path(run_dir) / "fork" / "fork.json"
                if arch[0] not in EXP12_FORKING or not (Path(run_dir) / "fork" / "FORK_READY").exists():
                    continue
                arm_state = classify_exp12(arm_dir)
                status.setdefault(f"exp2 arm {arm_state}", []).append(arm_dir)
                if arm_state == "done":
                    continue
                if injection_m is None:
                    status.setdefault("exp2 arm waiting for --injection-m", []).append(arm_dir)
                    continue
                device = json.loads(fork_json.read_text()).get("device", {}).get("device_kind", "unknown")
                arm_overrides = overrides + [f"fork.source={run_dir}", "fork.arm=injected", f"injection.m={injection_m}"]
                log = os.path.join(logs, f"exp2_injected_{arch[0]}_{env_name}_seed{seed}.log")
                manifests.setdefault(f"exp2_arms_{_device_slug(device)}.txt", []).append(
                    exp12_command("exp2_arm", arm_overrides, arm_dir, interval, log))
    os.makedirs(logs, exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--grid", choices=("exp12",), default="exp12")
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--injection-m", default=None)
    args = parser.parse_args(argv)
    results_root = require_absolute(args.results_root, "--results-root")
    ckpt_root = require_absolute(args.ckpt_root, "--ckpt-root")
    status, manifests = {}, {}
    add_exp12_grid(manifests, status, ckpt_root, results_root, args.injection_m)
    existing = sorted({*Path.cwd().glob("exp12_*_jobs.txt"), *Path.cwd().glob("exp2_arms_*.txt")})
    if existing:
        raise ValueError(f"existing Exp12 manifests must be preserved; generate in a fresh directory: {existing}")
    for path, jobs in manifests.items():
        with open(path, "x") as stream:
            stream.write("\n".join(jobs) + ("\n" if jobs else ""))
        print(f"{path}: queued {len(jobs)}")
    for state, dirs in sorted(status.items()):
        print(f"{state}: {len(dirs)}")


if __name__ == "__main__":
    main()
