#!/usr/bin/env python3
"""Discover and execute only approved natural Exp2 arms from an Exp96 queue.

This scheduler may be newer than the parent source. The trainer always runs in
the clean, detached checkout at the parent commit. One process per A100 is the
conservative starting policy; higher density needs a separate measured approval.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from scripts import exp12_dynamic_scheduler as durable
from scripts import exp96_campaign
from scripts.exp12_launch_plan import sha

ARCHS = {"D4W1024", "D4W1536"}
GROUPS = {"dmc_hard", "myosuite_simba"}


def clean_checkout(path, commit):
    root = Path(path).resolve()
    if (
        not (root / "run.py").is_file()
        or not (root / "configs" / "base_exp12_ten.yaml").is_file()
    ):
        raise ValueError("frozen trainer checkout lacks current experiment files")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        != commit
    ):
        raise ValueError("trainer checkout differs from parent source commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root
    ).strip():
        raise ValueError("frozen trainer checkout must be clean")
    return root


def qualified_parent(queue_root):
    try:
        return exp96_campaign.load(queue_root)
    except ValueError as legacy_error:
        queue, plan = exp96_campaign.load(queue_root, require_approval=False)
        decision = durable._read(plan["approval"])
        evidence = decision.get("gates", {}).get("gpu_smoke", {}).get("evidence", {})
        if (
            not evidence.get("path")
            or durable._read(evidence["path"]).get("schema") != "exp96-multigpu-754-v1"
        ):
            raise legacy_error
        from scripts.exp96_multigpu import study

        study(plan)  # Real multi-GPU source-bound evidence; no receipt relabeling.
        return queue, plan


def approved(queue_root, path):
    queue, plan = qualified_parent(queue_root)
    approval = durable._read(path)
    if (
        approval.get("approved") is not True
        or approval.get("study_version") != exp96_campaign.VERSION
        or approval.get("expected_commit") != queue["commit"]
        or approval.get("queue_sha256") != sha(Path(queue_root) / "queue.json")
        or not approval.get("decision_reference")
        or approval.get("injection_m") not in ("last", "half", "all")
    ):
        raise ValueError("separate, source-bound Exp2 approval is missing")
    scheduler = Path(__file__).resolve().parents[2]
    if (
        approval.get("scheduler_commit")
        != subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=scheduler, text=True
        ).strip()
        or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=scheduler
        ).strip()
    ):
        raise ValueError("Exp2 scheduler differs from clean approved source")
    evidence = approval.get("positive_control", {})
    if (
        not evidence.get("path")
        or not evidence.get("sha256")
        or sha(evidence["path"]) != evidence["sha256"]
    ):
        raise ValueError("intact, source-bound positive control required")
    control = durable._read(evidence["path"])
    if (
        control.get("status") != "m_chosen"
        or control.get("chosen_m") != approval["injection_m"]
        or control.get("source_code") != {"commit": queue["commit"], "dirty": False}
        or control.get("analysis_code")
        != {"commit": approval["scheduler_commit"], "dirty": False}
        or control.get("allow_any_setting") is not False
        or control.get("trigger_check", {}).get("forced") is not False
    ):
        raise ValueError("positive control does not qualify parent source and chosen m")
    ident = control.get("run_identity") or {}
    if (
        ident.get("run_role") != "dev"
        or ident.get("architecture") != "D4W1536"
        or ident.get("environment") != "dog-run"
        or type(ident.get("seed")) is not int
        or ident["seed"] <= 8
    ):
        raise ValueError(
            "positive control is not independent of the 96 confirmatory parents"
        )
    probes = control.get("probe") or {}
    if any(
        len(probes.get(name, {}).get("score_rounds", [])) != 10
        for name in (
            "fresh",
            "degraded",
            "injected_last",
            "injected_half",
            "injected_all",
        )
    ):
        raise ValueError("ten-target positive control evidence is required for Exp96")
    if control.get("probe_configuration") != {
        "rounds": 10,
        "steps": 2000,
        "checks": 50,
        "pool_size": 25600,
        "batch_size": 256,
    }:
        raise ValueError("ten-target positive control uses another probe configuration")
    if control.get("qualification_criteria") != {
        "natural_unforced_trigger": True,
        "healthy_reference": "fresh critic at check 0",
        "noise_threshold": 0.10,
        "similar_within": 0.10,
        "rounds": 10,
    }:
        raise ValueError("ten-target positive control uses another selection criterion")
    import numpy as np
    from experiments.exp12.m_selection import evaluate

    loss = control.get("loss_rounds") or {}
    expected = ("degraded", "injected_last", "injected_half", "injected_all")
    if set(loss) != set(expected):
        raise ValueError("positive control lacks four paired loss series")
    try:
        losses = {name: np.asarray(loss[name], np.float32) for name in expected}
        fresh = np.asarray(probes["fresh"]["score_rounds"], np.float32)
        for name in expected:
            measured = np.asarray(probes[name]["score_rounds"], np.float32)
            if not np.array_equal(losses[name], fresh - measured):
                raise ValueError("positive control score/loss pairing differs")
        selected = evaluate(losses, 0.10, 0.10, rounds=10)
    except (TypeError, ValueError) as exc:
        raise ValueError("positive control loss evidence is malformed") from exc
    if selected["stop"] is not None or selected["chosen_m"] != control["chosen_m"]:
        raise ValueError("positive control m is inconsistent with paired losses")
    clean_checkout(approval["trainer_checkout"], queue["commit"])
    return queue, plan, approval


def discover(queue_root, approval_path):
    queue, _, approval = approved(queue_root, approval_path)
    ledger_root = arm_queue(queue_root, approval)
    arms = durable.discover_arms(
        ledger_root,
        queue,
        approval["injection_m"],
        arm_root=approval.get("arm_output_root"),
    )
    if any(arm["row"]["architecture"] not in ARCHS for arm in arms):
        raise ValueError("non-forking architecture has an injected arm")
    return arms


def arm_queue(parent_queue, approval):
    """Keep new arm claims and outputs outside the frozen parent campaign."""
    root = Path(approval.get("arm_queue_root", parent_queue)).resolve()
    if "arm_queue_root" in approval:
        parent = Path(parent_queue).resolve()
        output = Path(approval["arm_output_root"])
        if (
            not Path(approval["arm_queue_root"]).is_absolute()
            or not output.is_absolute()
            or root == parent
            or root.is_relative_to(parent)
            or not root.is_dir()
        ):
            raise ValueError(
                "separate absolute prepared Exp2 queue/output paths required"
            )
        binding = durable._read(root / "parent_binding.json")
        if binding != {
            "parent_queue": str(parent),
            "queue_sha256": approval["queue_sha256"],
            "parent_commit": approval["expected_commit"],
            "scheduler_commit": approval["scheduler_commit"],
            "arm_output_root": str(output.resolve()),
        }:
            raise ValueError("Exp2 queue parent/source binding changed")
    return root


def resources(approval, visible, affinity):
    requested = approval.get("resources", {})
    if (
        type(requested.get("nodes")) is not int
        or not 1 <= requested["nodes"] <= 2
        or requested.get("gpus_per_node") != 4
        or requested.get("account") != "biqc-delta-gpu"
        or requested.get("partition") != "gpuA100x4"
        or len(visible) != requested["gpus_per_node"]
        or len(set(visible)) != len(visible)
        or type(requested.get("cpus_per_node")) is not int
        or requested["cpus_per_node"] < 4 * len(visible)
        or len(affinity) < requested["cpus_per_node"]
        or type(requested.get("memory_gb_per_node")) is not int
        or requested["memory_gb_per_node"] < 8 * len(visible)
        or type(requested.get("wall_minutes")) is not int
        or not 1 <= requested["wall_minutes"] <= 2880
    ):
        raise ValueError("actual allocation differs from approved Exp2 resources")
    profiles = approval.get("admission", {}).get("architectures", {})
    admission = approval.get("admission", {})
    for field in ("gpu_headroom_gb", "host_headroom_gb", "stop_margin_minutes"):
        if type(admission.get(field)) is not int or admission[field] <= 0:
            raise ValueError(f"positive Exp2 admission {field} is required")
    if admission["stop_margin_minutes"] >= requested["wall_minutes"]:
        raise ValueError("Exp2 stop margin consumes the allocation")
    if set(profiles) != ARCHS:
        raise ValueError(
            "measured Exp2 arm profiles required for both fork architectures"
        )
    for arch, profile in profiles.items():
        if set(profile.get("estimated_arm_minutes_by_group", {})) != GROUPS:
            raise ValueError(f"{arch} arm runtimes must cover both environment groups")
        if any(
            type(value) is not int
            or value <= 0
            or value + admission["stop_margin_minutes"] >= requested["wall_minutes"]
            for value in profile["estimated_arm_minutes_by_group"].values()
        ):
            raise ValueError(f"{arch} arm runtimes do not fit the allocation")
        for key in (
            "cpus",
            "host_ram_gb",
            "gpu_memory_gb",
            "min_interactions_per_second",
        ):
            if type(profile.get(key)) not in (int, float) or profile[key] <= 0:
                raise ValueError(f"{arch} measured {key} is missing")
        if (
            profile["gpu_memory_gb"] + admission["gpu_headroom_gb"] > 40
            or profile["host_ram_gb"] + admission["host_headroom_gb"]
            > requested["memory_gb_per_node"]
            or profile["cpus"] > requested["cpus_per_node"] // len(visible)
        ):
            raise ValueError(f"{arch} profile does not fit one approved A100 lane")
    if approval.get("admission", {}).get("max_processes_per_gpu") != 1:
        raise ValueError("Exp2 packing beyond one arm per GPU is unqualified")
    return requested, profiles


def worker(queue_root, approval_path):
    import jax
    import jaxlib
    from scripts.exp12_launch_plan import require_pinned_stack

    queue, _, approval = approved(queue_root, approval_path)
    ledger_root = arm_queue(queue_root, approval)
    from scripts.exp96_exp2_delta import funding_gate

    funding_gate(approval, os.environ.get("SLURM_JOB_ID"))
    if sha(approval_path) != os.getenv("EXP2_APPROVAL_SHA256"):
        raise ValueError("Exp2 approval changed after submission")
    require_pinned_stack()
    if (
        sys.version_info[:3] != (3, 12, 13)
        or jax.__version__ != "0.4.34"
        or jaxlib.__version__ != "0.4.34"
        or jax.default_backend() != "gpu"
        or not jax.local_devices(backend="cpu")
    ):
        raise ValueError("pinned CPU and CUDA environment required")
    job_id = os.getenv("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("real Slurm allocation required")
    visible = [
        value for value in os.getenv("CUDA_VISIBLE_DEVICES", "").split(",") if value
    ]
    requested, profiles = resources(approval, visible, os.sched_getaffinity(0))
    if (
        int(os.getenv("SLURM_GPUS_ON_NODE", "0")) != requested["gpus_per_node"]
        or int(os.getenv("SLURM_JOB_NUM_NODES", "0")) != requested["nodes"]
    ):
        raise ValueError("Slurm GPU/node allocation differs from Exp2 approval")
    description = subprocess.check_output(
        ["scontrol", "show", "job", "--oneliner", job_id], text=True
    )
    from scripts.exp12_launch_plan import allocation_matches

    allocation_matches(
        {
            "cpus": requested["cpus_per_node"],
            "memory_gb": requested["memory_gb_per_node"],
            "wall_minutes": requested["wall_minutes"],
        },
        os.environ,
        description,
    )
    from scripts.exp96_exp2_delta import verify_billing

    verify_billing(requested, description)
    match = re.search(r"\bEndTime=(\S+)", description)
    if not match or match[1] in ("Unknown", "N/A"):
        raise ValueError("Slurm end time unavailable")
    deadline = datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S").timestamp()
    lanes = [
        sorted(os.sched_getaffinity(0))[i :: len(visible)] for i in range(len(visible))
    ]
    for token in visible:
        durable.gpu_snapshot(token)
    ledger = durable.Ledger(ledger_root)
    active, validating, unrecorded, seen = [], {}, [], set()
    trainer = clean_checkout(approval["trainer_checkout"], queue["commit"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            if (
                sha(approval_path) != os.environ["EXP2_APPROVAL_SHA256"]
                or sha(Path(queue_root) / "queue.json") != approval["queue_sha256"]
            ):
                raise ValueError("immutable Exp2 launch inputs changed")
            arms = discover(queue_root, approval_path)
            for job in list(active):
                if job["process"].poll() is not None:
                    active.remove(job)
                    validating[
                        pool.submit(
                            durable._finish, job, ledger, queue["commit"], ledger_root
                        )
                    ] = job
            durable._collect_finished(validating, unrecorded, time.time())
            for job in list(unrecorded):
                if job["retry_finish_at"] <= time.time():
                    unrecorded.remove(job)
                    validating[
                        pool.submit(
                            durable._finish, job, ledger, queue["commit"], ledger_root
                        )
                    ] = job
            pending = [
                arm
                for arm in arms
                if arm["id"] not in seen and ledger.state(arm) == "pending"
            ]
            if not active and not validating and not unrecorded and not pending:
                return 0
            if (
                time.time()
                >= deadline - 60 * approval["admission"]["stop_margin_minutes"]
            ):
                if not active and not validating:
                    durable._abandon_unrecorded(unrecorded)
                    return 0
            else:
                started = False
                for token, lane in zip(visible, lanes):
                    if any(job["gpu"] == token for job in active):
                        continue
                    snapshot = durable.gpu_snapshot(token)
                    for arm in pending:
                        profile = profiles[arm["row"]["architecture"]]
                        estimate = profile["estimated_arm_minutes_by_group"][
                            arm["row"]["group"]
                        ]
                        if (
                            time.time()
                            + 60
                            * (estimate + approval["admission"]["stop_margin_minutes"])
                            >= deadline
                            or sum(job["profile"]["host_ram_gb"] for job in active)
                            + profile["host_ram_gb"]
                            + approval["admission"]["host_headroom_gb"]
                            > requested["memory_gb_per_node"]
                            or profile["host_ram_gb"]
                            + approval["admission"]["host_headroom_gb"]
                            > durable.available_host_gb()
                            or snapshot[1]
                            + 1024
                            * (
                                profile["gpu_memory_gb"]
                                + approval["admission"]["gpu_headroom_gb"]
                            )
                            > snapshot[0]
                        ):
                            continue
                        child_approval = {
                            "expected_commit": queue["commit"],
                            "admission": {"architectures": profiles},
                        }
                        job = durable._start(
                            arm,
                            token,
                            child_approval,
                            ledger_root,
                            ledger,
                            job_id,
                            training_checkout=trainer,
                            cpu_ids=set(lane),
                        )
                        if job:
                            active.append(job)
                            seen.add(arm["id"])
                            started = True
                            break
                if (
                    pending
                    and not active
                    and not validating
                    and not unrecorded
                    and not started
                ):
                    print(
                        "Exp2 allocation has no admissible arms; release GPUs for review",
                        flush=True,
                    )
                    return 3
            if time.time() >= deadline - 30 and (active or validating or unrecorded):
                durable._abandon_unrecorded(unrecorded)
                return 4  # Slurm ends live children; their checkpoints survive.
            time.sleep(1 if active else 3)


def submission_command(queue_root, approval_path):
    _, _, approval = approved(queue_root, approval_path)
    from scripts.exp96_exp2_delta import funding_gate

    funding_gate(approval)
    r = approval["resources"]
    if (
        r.get("nodes") not in (1, 2)
        or r.get("gpus_per_node") != 4
        or type(r.get("wall_minutes")) is not int
        or not 1 <= r["wall_minutes"] <= 2880
        or type(r.get("cpus_per_node")) is not int
    ):
        raise ValueError("unsupported Exp2 allocation")
    resources(approval, ["0", "1", "2", "3"], set(range(r["cpus_per_node"])))
    if not discover(queue_root, approval_path):
        raise ValueError("no naturally eligible injected arms; do not reserve GPUs")
    logs = arm_queue(queue_root, approval) / "slurm"
    logs.mkdir(exist_ok=True)
    command = [
        "env",
        f"EXP2_QUEUE={Path(queue_root).resolve()}",
        f"EXP2_APPROVAL={Path(approval_path).resolve()}",
        f"EXP2_APPROVAL_SHA256={sha(approval_path)}",
        f"EXP2_SCHEDULER_COMMIT={approval['scheduler_commit']}",
        f"EXP2_SCHEDULER_MAIN={Path(__file__).resolve().parents[1]}",
        "sbatch",
        "--parsable",
        "--export=ALL",
        f"--nodes={r['nodes']}",
        f"--account={r['account']}",
        f"--partition={r['partition']}",
        "--ntasks-per-node=1",
        "--gpus-per-node=4",
        f"--cpus-per-task={r['cpus_per_node']}",
        f"--mem={r['memory_gb_per_node']}G",
        f"--time={r['wall_minutes']}",
        f"--output={logs}/exp2_%j.out",
        f"--error={logs}/exp2_%j.err",
        "scripts/sbatch_exp96_exp2_arms.sh",
    ]
    return shlex.join(command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("discover", "worker", "submission-command", "review-retry"):
        command = sub.add_parser(action)
        command.add_argument("--queue", required=True)
        command.add_argument("--approval", required=True)
        if action == "review-retry":
            command.add_argument("--id", required=True)
            command.add_argument("--decision", required=True)
    args = parser.parse_args(argv)
    if args.action == "discover":
        print(len(discover(args.queue, args.approval)))
    elif args.action == "submission-command":
        print(submission_command(args.queue, args.approval))
    elif args.action == "review-retry":
        queue, _, _ = approved(args.queue, args.approval)
        matches = [
            arm for arm in discover(args.queue, args.approval) if arm["id"] == args.id
        ]
        if len(matches) != 1:
            raise ValueError("arm retry identity is missing or ambiguous")
        _, _, approval = approved(args.queue, args.approval)
        durable.Ledger(arm_queue(args.queue, approval)).review_retry(
            matches[0], queue["commit"], args.decision
        )
    else:
        return worker(args.queue, args.approval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
