#!/usr/bin/env python3
"""Versioned 96-parent Method C campaign; never submits a Slurm job.

The historical Exp1/Exp2 launcher remains unchanged. This module builds a
source-bound plan and uses its durable ledger and resource admission primitives.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))
from generate_manifest import EXP12_ARCHS, exp12_overrides
from scripts import exp12_dynamic_scheduler as durable
from scripts.exp12_launch_plan import sha, require_pinned_stack

VERSION = "method-c-ten-96-v1"
ENVS = (("dog-run", "dmc_hard"), ("humanoid-run", "dmc_hard"),
        ("myo-key-turn", "myosuite_simba"), ("myo-pen-twirl", "myosuite_simba"))
SEEDS = tuple(range(1, 9))
INTERACTIONS = 500_000
CHECKS = 50
GPU = durable.GPU_KIND


def _config_hash(overrides):
    from experiments.exp1 import compose_config
    from omegaconf import OmegaConf
    from utils.run_metadata import _hash, _strip_locations
    cfg = compose_config(str(MAIN / "configs"), "base_exp12_ten", overrides)
    return _hash(_strip_locations(OmegaConf.to_container(cfg, resolve=True)))


def _source(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full source SHA required")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MAIN, text=True).strip()
    if actual != commit or subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=MAIN
    ).strip():
        raise ValueError("clean source checkout at the exact approved commit required")


def rows(root):
    from scale_rl.envs.myosuite import MYOSUITE_TASKS_DICT
    if MYOSUITE_TASKS_DICT.get("myo-pen-twirl") != "myoHandPenTwirlFixed-v0":
        raise ValueError("MyoSuite pen-twirl environment ID changed")
    result = []
    for env, group in ENVS:
        for arch in EXP12_ARCHS:
            for seed in SEEDS:
                name = arch[0]
                checkpoint = root / "checkpoints" / "exp1" / name / env / f"seed_{seed}"
                overrides = exp12_overrides(arch, env, group, seed, str(root / "results"))
                # The environment configs use raw steps and action_repeat=2.
                overrides.append("env.num_env_steps=1000000")
                cfg = __import__("experiments.exp1", fromlist=["compose_config"]).compose_config(
                    str(MAIN / "configs"), "base_exp12_ten", overrides)
                if (int(cfg.num_interaction_steps), int(cfg.probe.rounds),
                    int(cfg.probe.steps), int(cfg.probe.checks), int(cfg.action_repeat)) != (
                    INTERACTIONS, 10, 2000, CHECKS, 2):
                    raise ValueError("Method C ten study config differs from approved protocol")
                # The worker replaces this placeholder with its verified pinned
                # interpreter; the immutable plan must not depend on the host
                # used to create it.
                command = ["python", "-u", "run.py", "--experiment", "exp1",
                           "--config_name", "base_exp12_ten"]
                for override in overrides:
                    command += ["--overrides", override]
                command += ["--checkpoint_dir", str(checkpoint), "--checkpoint_interval", "10000",
                            "--checkpoint_start_frac", "0.0"]
                result.append({"index": len(result), "run_key": f"exp1_{name}_{env}_seed{seed}",
                               "study": VERSION,
                               "architecture": name, "environment": env, "seed": seed,
                               "group": group, "checkpoint_dir": str(checkpoint),
                               "budget": 1_000_000, "config_hash": _config_hash(overrides),
                               "overrides": overrides, "command": command})
    if len(result) != 96 or len({r["run_key"] for r in result}) != 96:
        raise ValueError("96-parent population is incomplete or duplicated")
    return result


def _plan(root, commit, approval, *, staged=False):
    if not Path(root).is_absolute():
        raise ValueError("campaign root must be absolute")
    root = Path(root).resolve()
    if root.is_relative_to(MAIN.parent):
        raise ValueError("campaign root must be outside checkout")
    return {"schema_version": VERSION, "commit": commit, "root": str(root),
            "approval": str(Path(approval).resolve()),
            "approval_sha256": None if staged else sha(approval),
            "entries": rows(root)}


def _approval(plan):
    approval = durable._read(plan["approval"])
    if plan["approval_sha256"] is not None and sha(plan["approval"]) != plan["approval_sha256"]:
        raise ValueError("campaign approval changed")
    if (approval.get("approved") is not True or approval.get("expected_commit") != plan["commit"]
        or approval.get("study_version") != VERSION or not approval.get("decision_reference")
        or approval.get("parent_count") != 96):
        raise ValueError("explicit 96-parent source-bound study approval required")
    for name in ("ten_function_method", "scientific_calibration", "gpu_smoke",
                 "allocation_and_storage", "cross_seed_analysis"):
        decision = approval.get("gates", {}).get(name, {})
        if decision.get("status") != "APPROVED" or not decision.get("decision_reference"):
            raise ValueError(f"unapproved study gate: {name}")
    smoke = approval["gates"]["gpu_smoke"]
    evidence = smoke.get("evidence", {})
    if (not evidence.get("path") or not evidence.get("sha256")
        or sha(evidence["path"]) != evidence["sha256"]):
        raise ValueError("intact GPU smoke evidence required for campaign approval")
    report = durable._read(evidence["path"])
    if (report.get("verdict") != "PASS" or report.get("gpu") != GPU
        or report.get("expected_commit") != plan["commit"]
        or report.get("study_version") != VERSION):
        raise ValueError("A100 GPU smoke does not qualify this exact study source")
    return approval


def load(queue_root, *, require_approval=True):
    root = Path(queue_root).resolve()
    queue = durable._read(root / "queue.json")
    plan_path = Path(queue["plan"])
    plan = durable._read(plan_path)
    if (plan.get("schema_version") != VERSION or queue.get("schema_version") != VERSION
        or sha(plan_path) != queue.get("plan_sha256") or plan["commit"] != queue["commit"]
        or plan["entries"] != rows(Path(plan["root"]))):
        raise ValueError("96-parent queue/plan/configuration mismatch")
    if require_approval:
        _approval(plan)
    expected = [{"id": durable.identity("exp1", row, plan["commit"]), "kind": "exp1", "row": row}
                for row in plan["entries"]]
    if queue["parents"] != expected or len({x["id"] for x in expected}) != 96:
        raise ValueError("96-parent durable queue differs from plan")
    if durable._read(Path(plan["root"]) / "EXP96_DYNAMIC_QUEUE.json") != {
        "queue": str(root), "plan_sha256": sha(plan_path), "commit": plan["commit"]}:
        raise ValueError("campaign queue binding mismatch")
    if durable._read(Path(plan["root"]) / "source_freeze.json") != {
        "study_version": VERSION, "source_commit": plan["commit"],
        "plan_sha256": sha(plan_path), "approval_sha256": plan["approval_sha256"],
        "parent_count": 96, "environment_ids": {
            "myo-key-turn": "myoHandKeyTurnFixed-v0",
            "myo-pen-twirl": "myoHandPenTwirlFixed-v0"}}:
        raise ValueError("source freeze manifest differs from plan")
    return queue, plan


def prepare(root, commit, approval, queue_root, *, staged=False):
    _source(commit)
    if not Path(queue_root).is_absolute():
        raise ValueError("queue root must be absolute")
    plan = _plan(root, commit, approval, staged=staged)
    if not staged:
        _approval(plan)
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    if root.exists() or queue_root.exists() or queue_root.is_relative_to(MAIN.parent):
        raise ValueError("fresh campaign and queue paths outside checkout required")
    queue_root.mkdir(parents=True)
    root.mkdir(parents=True)
    plan_path = root / "exp96_plan.json"
    durable._write_new(plan_path, plan)
    durable._write_new(root / "EXP96_DYNAMIC_QUEUE.json", {
        "queue": str(queue_root), "plan_sha256": sha(plan_path), "commit": commit})
    durable._write_new(root / "source_freeze.json", {
        "study_version": VERSION, "source_commit": commit,
        "plan_sha256": sha(plan_path), "approval_sha256": plan["approval_sha256"],
        "parent_count": 96, "environment_ids": {
            "myo-key-turn": "myoHandKeyTurnFixed-v0",
            "myo-pen-twirl": "myoHandPenTwirlFixed-v0"}})
    entries = [{"id": durable.identity("exp1", row, commit), "kind": "exp1", "row": row}
               for row in plan["entries"]]
    durable._write_new(queue_root / "queue.json", {
        "schema_version": VERSION, "plan": str(plan_path), "plan_sha256": sha(plan_path),
        "commit": commit, "parents": entries})
    (queue_root / "slurm").mkdir()
    (queue_root / "submissions").mkdir()
    return plan


def validate_parent(item, commit, queue_root):
    """Validate complete evidence before the shared ledger can mark an item complete."""
    import numpy as np
    from experiments.exp12.state import latest_state_dir, load_meta, load_agent_tree
    run = Path(item["row"]["checkpoint_dir"])
    if not (run / "DONE").is_file():
        return None
    metadata = durable._metadata_matches(run, item, commit)
    cfg = metadata["resolved_config"]
    if (metadata["identity"].get("run_role") != "confirmatory"
        or cfg.get("project_name") != "echocritic_method_c_ten_96"
        or (cfg["probe"]["rounds"], cfg["probe"]["steps"], cfg["probe"]["checks"]) != (10, 2000, 50)
        or cfg["trigger"]["null_threshold"] != 0
        or cfg["trigger"]["resamples"] != 10_000
        or cfg["trigger"]["consecutive_checks"] != 2
        or int(cfg["num_interaction_steps"]) != INTERACTIONS):
        raise ValueError("completion has incompatible Method C ten protocol")
    from utils.run_metadata import _hash, _strip_locations
    if metadata["config_hash"] != _hash(_strip_locations(cfg)):
        raise ValueError("resolved configuration hash mismatch")
    done = durable._read(run / "DONE")
    state = latest_state_dir(run / "state")
    meta = load_meta(state)
    end = INTERACTIONS
    if (run / "fork" / "fork.json").exists():
        fork = durable._read(run / "fork" / "fork.json")
        end = fork["control_end_step"]
        if not (run / "fork" / "FORK_READY").exists():
            raise ValueError("incomplete natural fork evidence")
        verified_fork = durable._natural_fork(item, commit)
        if verified_fork != fork:
            raise ValueError("natural fork validation disagrees with saved plan")
    if meta["interaction_step"] != end or done["interaction_step"] != end:
        raise ValueError("run did not reach its endpoint")
    if meta["update_step"] != 2 * (end - 4999) or meta["update_counter"] != 0:
        raise ValueError("SAC update counters inconsistent")
    required = {"numpy_rng_state", "python_rng_state", "train_env", "eval_env",
                "observations", "timestep", "agent_window_buffers", "actor_diagnostics"}
    if not required <= meta.keys():
        raise ValueError("checkpoint lacks complete state")
    tree = load_agent_tree(state)
    if not {"actor", "critic", "target_critic", "temperature", "rng"} <= tree.keys():
        raise ValueError("agent state incomplete")
    import jax
    if any(not np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(tree)):
        raise ValueError("nonfinite model/optimizer state")
    from scripts.exp12_launch_plan import validate_replay_normalization
    validate_replay_normalization(state)
    count = validate_probe_artifacts(run, cfg, end)
    if not meta.get("wandb_run_id") or not meta.get("metrics_rows") or not meta.get("eval_rows"):
        raise ValueError("W&B identity, training metrics or evaluation returns missing")
    logs = list((run / "logs").glob("*.csv"))
    if not any("eval_episodes" in p.name for p in logs) or not any("eval_episodes" not in p.name for p in logs):
        raise ValueError("local training/evaluation CSV artifacts missing")
    receipt = durable._receipt_path(queue_root, item)
    durable._write_new(receipt, {"engineering_verdict": "COMPLETE", "study": VERSION,
                                 "expected_commit": commit, "id": item["id"], "endpoint": end,
                                 "checks": count, "config_hash": item["row"]["config_hash"]})
    return receipt


def validate_probe_artifacts(run, cfg, end):
    """Require every scheduled ten-round curve and its ordered CSV record."""
    import numpy as np
    import pandas as pd
    run = Path(run)
    checks = pd.read_csv(run / "probes" / "probe_checks.csv")
    count = end // 10_000
    expected_steps = [int(cfg["buffer"]["min_length"])] + [i * 10_000 for i in range(1, count + 1)]
    if (checks["check_index"].tolist() != list(range(count + 1))
        or checks["interaction_step"].tolist() != expected_steps):
        raise ValueError("missing, duplicate or mistimed plasticity checks")
    required_columns = {"loss_iqm", "ci_low", "ci_high", "triggered", "valid"}
    required_columns |= {f"{stem}_r{r}" for stem in ("score_current", "score_fresh", "loss")
                         for r in range(10)}
    if not required_columns <= set(checks.columns):
        raise ValueError("plasticity CSV lacks ten-round metrics/confidence bounds")
    if not checks.loc[checks.check_index > 0, list(required_columns - {"triggered", "valid"})].notna().all().all():
        raise ValueError("plasticity CSV contains missing measurements")
    for k in range(count + 1):
        with np.load(run / "probes" / f"check_{k:02d}.npz") as payload:
            if not payload.files or any(not np.isfinite(payload[x]).all() for x in payload.files):
                raise ValueError(f"probe check {k} missing/nonfinite")
            for name in (("fresh_score",) if k == 0 else ("fresh_score", "current_score")):
                if name not in payload or payload[name].shape != (10,):
                    raise ValueError(f"probe check {k} lacks ten {name} rounds")
            for name in (("fresh_q1_losses", "fresh_q2_losses") if k == 0 else
                         ("fresh_q1_losses", "fresh_q2_losses", "current_q1_losses", "current_q2_losses")):
                if name not in payload or payload[name].shape != (10, 2000):
                    raise ValueError(f"probe check {k} lacks continuous 2,000-step {name} curves")
    return count


def allocation_command(queue_root, approval_path):
    queue, plan = load(queue_root)
    approval = durable._read(approval_path)
    _study_bound(plan, approval)
    r, _ = _allocation(approval, queue_root, queue["commit"])
    logs = Path(queue_root) / "slurm"
    cmd = ["env", f"EXPECTED_COMMIT={queue['commit']}", f"EXP96_QUEUE={Path(queue_root).resolve()}",
           f"EXP96_ALLOCATION_APPROVAL={Path(approval_path).resolve()}",
           f"EXP96_APPROVAL_SHA256={sha(approval_path)}", "sbatch", "--parsable", "--export=ALL",
           "--nodes=1", "--ntasks=1",
           f"--cpus-per-task={r['cpus']}", f"--mem={r['memory_gb']}G", f"--time={r['wall_minutes']}",
           "--gres=gpu:1", f"--output={logs}/%x_%j.out", f"--error={logs}/%x_%j.err",
           "scripts/sbatch_exp96_allocation.sh"]
    return shlex.join(cmd)


def pending_command(queue_root, approval_path, *, cpus, memory_gb, wall_minutes):
    """Print a normal queued allocation; approval is enforced at job startup."""
    queue, _ = load(queue_root, require_approval=False)
    _source(queue["commit"])
    if (type(cpus) is not int or type(memory_gb) is not int
        or type(wall_minutes) is not int or cpus <= 0 or memory_gb <= 0
        or not 1 <= wall_minutes <= 2880):
        raise ValueError("invalid pending one-A100 resource request")
    logs = Path(queue_root).resolve() / "slurm"
    cmd = ["env", "-u", "SBATCH_HOLD", "-u", "SBATCH_BEGIN", "-u", "SBATCH_DEPENDENCY",
           f"EXPECTED_COMMIT={queue['commit']}", f"EXP96_QUEUE={Path(queue_root).resolve()}",
           f"EXP96_ALLOCATION_APPROVAL={Path(approval_path).resolve()}",
           "sbatch", "--parsable", "--export=ALL", "--job-name=exp96-pending-v1",
           "--nodes=1", "--ntasks=1", f"--cpus-per-task={cpus}", f"--mem={memory_gb}G",
           f"--time={wall_minutes}", "--gres=gpu:1",
           f"--output={logs}/%x_%j.out", f"--error={logs}/%x_%j.err",
           "scripts/sbatch_exp96_allocation.sh"]
    return shlex.join(cmd)


def startup_gate(queue_root, approval_path):
    """Return the approval digest only after every training prerequisite passes."""
    queue, plan = load(queue_root)
    _source(queue["commit"])
    require_pinned_stack()
    digest = sha(approval_path)
    approval = durable._read(approval_path)
    _study_bound(plan, approval)
    resources, _ = _allocation(approval, queue_root, queue["commit"])
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise ValueError("real Slurm allocation required")
    description = subprocess.check_output(
        ["scontrol", "show", "job", "--oneliner", job_id], text=True)
    from scripts.exp12_launch_plan import allocation_matches
    allocation_matches({"cpus": resources["cpus"], "memory_gb": resources["memory_gb"],
                        "wall_minutes": resources["wall_minutes"]}, os.environ, description)
    if int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) != 1:
        raise ValueError("one allocated GPU required")
    return digest


def _study_bound(plan, allocation):
    if (plan["approval_sha256"] is None
        and allocation.get("study_approval_sha256") != sha(plan["approval"])):
        raise ValueError("staged study approval is not hash-bound to allocation approval")


def startup_gate_receipt(queue_root, approval_path):
    """Record the startup decision without touching claims or parent artifacts."""
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise ValueError("real Slurm allocation required")
    path = Path(queue_root).resolve() / "slurm" / f"startup_gate_{job_id}.json"
    receipt = {"job_id": job_id, "queue": str(Path(queue_root).resolve()),
               "allocation_approval": str(Path(approval_path).resolve()),
               "expected_commit": os.environ.get("EXPECTED_COMMIT")}
    try:
        digest = startup_gate(queue_root, approval_path)
    except Exception as exc:
        receipt.update(verdict="BLOCKED", reason=f"{type(exc).__name__}: {exc}")
        durable._write_new(path, receipt)
        raise
    receipt.update(verdict="PASS", approval_sha256=digest)
    durable._write_new(path, receipt)
    return digest


def _allocation(approval, queue_root, commit):
    if (approval.get("approved") is not True or approval.get("expected_commit") != commit
        or approval.get("queue_sha256") != sha(Path(queue_root) / "queue.json")
        or not approval.get("decision_reference")):
        raise ValueError("allocation approval/source/queue mismatch")
    resources = approval["resources"]
    if (resources.get("gpus") != 1 or type(resources.get("wall_minutes")) is not int
        or type(resources.get("cpus")) is not int
        or type(resources.get("memory_gb")) is not int
        or resources["cpus"] <= 0 or resources["memory_gb"] <= 0
        or not 1 <= resources["wall_minutes"] <= 2880):
        raise ValueError("invalid one-A100 allocation")
    admission = approval["admission"]
    cap = admission.get("max_processes_per_gpu")
    if admission.get("packing") != "ten_target" or type(cap) is not int or not 1 <= cap <= 6:
        raise ValueError("ten-target packing policy required")
    if admission.get("validated_concurrency") is not True:
        raise ValueError("high-density packing needs a source-bound passing smoke")
    if set(admission.get("architectures", {})) != {x[0] for x in EXP12_ARCHS}:
        raise ValueError("all architecture resource profiles required")
    for profile in admission["architectures"].values():
        if set(profile.get("estimated_minutes_by_group", {})) != {"dmc_hard", "myosuite_simba"}:
            raise ValueError("runtime profiles must cover both environment groups")
        for field in ("cpus", "host_ram_gb", "gpu_memory_gb", "min_interactions_per_second"):
            value = profile.get(field)
            if type(value) not in (int, float) or value <= 0:
                raise ValueError(f"missing positive resource admission field: {field}")
        if any(type(v) is not int or v <= 0 for v in profile["estimated_minutes_by_group"].values()):
            raise ValueError("positive runtime estimates required")
    for field in ("gpu_headroom_gb", "host_headroom_gb", "stop_margin_minutes"):
        value = admission.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"missing admission margin: {field}")
    if admission["stop_margin_minutes"] >= resources["wall_minutes"]:
        raise ValueError("stop margin consumes allocation")
    for profile in admission["architectures"].values():
        if any(minutes + admission["stop_margin_minutes"] >= resources["wall_minutes"]
               for minutes in profile["estimated_minutes_by_group"].values()):
            raise ValueError("a parent runtime estimate cannot fit this allocation")
        if (profile["cpus"] > resources["cpus"]
            or profile["host_ram_gb"] + admission["host_headroom_gb"] > resources["memory_gb"]
            or profile["gpu_memory_gb"] + admission["gpu_headroom_gb"] > 40):
            raise ValueError("resource profile cannot fit the approved allocation")
    return resources, admission


def worker(queue_root, approval_path):
    """One GPU worker. Other allocations coordinate through the shared Ledger."""
    import jax
    import jaxlib
    from datetime import datetime
    queue, plan = load(queue_root)
    _source(queue["commit"])
    require_pinned_stack()
    if sys.version_info[:3] != (3, 12, 13) or jax.__version__ != "0.4.34" or jaxlib.__version__ != "0.4.34":
        raise ValueError("pinned Python/JAX environment required")
    if jax.default_backend() != "gpu" or not jax.local_devices(backend="cpu"):
        raise ValueError("CPU and CUDA backends required")
    if sha(approval_path) != os.getenv("EXP96_APPROVAL_SHA256"):
        raise ValueError("allocation approval changed since submission")
    approval = durable._read(approval_path)
    _study_bound(plan, approval)
    resources, admission = _allocation(approval, queue_root, queue["commit"])
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise ValueError("real Slurm allocation required")
    description = subprocess.check_output(["scontrol", "show", "job", "--oneliner", job_id], text=True)
    from scripts.exp12_launch_plan import allocation_matches
    allocation_matches({"cpus": resources["cpus"], "memory_gb": resources["memory_gb"],
                        "wall_minutes": resources["wall_minutes"]}, os.environ, description)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0] or int(os.getenv("SLURM_GPUS_ON_NODE", "0")) != 1:
        raise ValueError("one visible allocated A100 required")
    token = visible[0]
    durable.gpu_snapshot(token)
    match = re.search(r"\bEndTime=(\S+)", description)
    if not match:
        raise ValueError("Slurm end time unavailable")
    deadline = datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S").timestamp()
    stop_at = deadline - 60 * admission["stop_margin_minutes"]
    ledger = durable.Ledger(queue_root)
    active, seen = [], set()
    validating, unrecorded = {}, []
    with ThreadPoolExecutor(max_workers=2) as validator:
      while time.time() < stop_at or active or validating or unrecorded:
        if (sha(approval_path) != os.environ["EXP96_APPROVAL_SHA256"]
            or (plan["approval_sha256"] is None
                and sha(plan["approval"]) != approval["study_approval_sha256"])
            or sha(queue["plan"]) != queue["plan_sha256"]
            or sha(Path(queue_root) / "queue.json") != approval["queue_sha256"]):
            raise ValueError("allocation/plan/queue changed during execution")
        for job in list(active):
            if job["process"].poll() is not None:
                active.remove(job)
                validating[validator.submit(durable._finish, job, ledger, queue["commit"], queue_root)] = job
        for job in [j for j in unrecorded if j["retry_finish_at"] <= time.time()]:
            unrecorded.remove(job)
            validating[validator.submit(durable._finish, job, ledger, queue["commit"], queue_root)] = job
        durable._collect_finished(validating, unrecorded, time.time())
        if time.time() >= deadline - 30 and (active or validating or unrecorded):
            if not validating:
                durable._abandon_unrecorded(unrecorded)
            # Live training processes retain their last committed checkpoint;
            # Slurm ends them at the approved wall limit for later recovery.
            return 4
        if time.time() >= stop_at:
            if not active and not validating:
                durable._abandon_unrecorded(unrecorded)
                break
            time.sleep(1)
            continue
        durable.update_rates(active)
        pending = [x for x in queue["parents"] if x["id"] not in seen and ledger.state(x) == "pending"]
        started = False
        for item in pending:
            adapter = {"cpus_per_node": resources["cpus"], "memory_gb_per_node": resources["memory_gb"]}
            if not durable.can_admit(item, token, active, admission, adapter, deadline,
                                     durable.gpu_snapshot(token), durable.available_host_gb(),
                                     validating.values()):
                continue
            job = durable._start(item, token, approval, queue_root, ledger, job_id)
            if job is not None:
                seen.add(item["id"])
                active.append(job)
                started = True
                break
        if not active and not pending and not validating and not unrecorded:
            return 0
        time.sleep(1 if started else durable.POLL_SECONDS)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    for opt in ("root", "commit", "approval", "queue"):
        p.add_argument("--" + opt, required=True)
    p.add_argument("--staged", action="store_true",
                   help="freeze identities before approval; startup still requires every gate")
    p = sub.add_parser("allocation-command")
    p.add_argument("--queue", required=True)
    p.add_argument("--approval", required=True)
    p = sub.add_parser("pending-command")
    p.add_argument("--queue", required=True)
    p.add_argument("--approval", required=True)
    p.add_argument("--cpus", type=int, required=True)
    p.add_argument("--memory-gb", type=int, required=True)
    p.add_argument("--wall-minutes", type=int, required=True)
    p = sub.add_parser("startup-gate")
    p.add_argument("--queue", required=True)
    p.add_argument("--approval", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--queue", required=True)
    p.add_argument("--approval", required=True)
    p = sub.add_parser("review-retry")
    p.add_argument("--queue", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--decision", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args.root, args.commit, args.approval, args.queue, staged=args.staged)
    elif args.action == "allocation-command":
        print(allocation_command(args.queue, args.approval))
    elif args.action == "pending-command":
        print(pending_command(args.queue, args.approval, cpus=args.cpus,
                              memory_gb=args.memory_gb, wall_minutes=args.wall_minutes))
    elif args.action == "startup-gate":
        print(startup_gate_receipt(args.queue, args.approval))
    elif args.action == "review-retry":
        queue, _ = load(args.queue)
        matches = [x for x in queue["parents"] if x["id"] == args.id]
        if len(matches) != 1:
            raise ValueError("retry identity missing or ambiguous")
        durable.Ledger(args.queue).review_retry(matches[0], queue["commit"], args.decision)
    else:
        return worker(args.queue, args.approval)


if __name__ == "__main__":
    sys.exit(main())
