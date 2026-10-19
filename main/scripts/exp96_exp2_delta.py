"""CPU-only Delta staging and separate Exp2 spending admission. Never submits jobs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))
SCHEMA = "exp96-exp2-delta-v1"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def su_cost(resources):
    """Observed MAX_TRES gpuA100x4 billing, normalized by1000."""
    if resources["partition"] != "gpuA100x4":
        raise ValueError("unreviewed Exp2 partition")
    numbers = [
        resources[k]
        for k in (
            "nodes",
            "gpus_per_node",
            "cpus_per_node",
            "memory_gb_per_node",
            "wall_minutes",
        )
    ]
    if any(type(x) is not int or x <= 0 for x in numbers):
        raise ValueError("positive integer allocation resources required")
    if (
        resources["nodes"] > 2
        or resources["gpus_per_node"] > 4
        or resources["cpus_per_node"] > 64
        or resources["memory_gb_per_node"] > 240
        or resources["wall_minutes"] > 2880
    ):
        raise ValueError("request exceeds reviewed Delta per-node topology/limits")
    return (
        math.ceil(
            max(
                resources["cpus_per_node"] * 62.5,
                resources["memory_gb_per_node"] * 16,
                resources["gpus_per_node"] * 1000,
            )
        )
        / 1000
        * resources["nodes"]
        * resources["wall_minutes"]
        / 60
    )


def funding_gate(approval, job_id=None, *, resources=None):
    """No borrowing from Exp1's600SU authorization or unsigned Exp2 budgets."""
    budget = read(approval["funding_path"])
    limit = budget.get("limit_su")
    if (
        budget.get("schema") != SCHEMA
        or budget.get("approved") is not True
        or not budget.get("decision_reference")
        or type(limit) not in (int, float)
        or not math.isfinite(limit)
        or limit <= 0
    ):
        raise ValueError("separate explicit Exp2 GPU spending authorization required")
    for key in ("expected_commit", "scheduler_commit", "queue_sha256"):
        if budget.get(key) != approval[key]:
            raise ValueError("Exp2 funding source/queue mismatch")
    entries = budget.get("entries", {})
    total = 0.0
    for entry in entries.values():
        reserved = su_cost(entry["resources"])
        if entry.get("reserved_su") != reserved:
            raise ValueError("invalid/unreconciled Exp2 spending entry")
        # Count every allocation at its full maximum. Unverified actual_su
        # annotations can never create additional spending headroom.
        total += reserved
    if total > limit + 1e-9:
        raise ValueError("cumulative Exp2 spending ceiling exceeded")
    requested = resources or approval["resources"]
    su_cost(requested)
    if job_id is not None:
        matching = [e for e in entries.values() if e.get("job_id") == str(job_id)]
        if (
            len(matching) != 1
            or matching[0]["resources"] != requested
            or matching[0].get("status") != "SUBMITTED"
        ):
            raise ValueError("actual Exp2 job lacks its unique full-cost reservation")
    return budget


def verify_billing(resources, description):
    match = re.search(r"\bAllocTRES=(\S+)", description)
    billing = re.search(r"(?:^|,)billing=(\d+)(?:,|$)", match[1]) if match else None
    reserved_rate = su_cost(resources) * 60 / resources["wall_minutes"] * 1000
    if (
        "Account=biqc-delta-gpu" not in description
        or "Partition=gpuA100x4" not in description
        or not billing
        or int(billing[1]) > reserved_rate + 1e-6
    ):
        raise ValueError(
            "actual allocation billing/account exceeds approved Exp2 reservation"
        )


def prepare(root, parent_queue, trainer, scheduler_commit, *, seed=991):
    from scripts import exp96_campaign
    from scripts.exp96_exp2_arms import clean_checkout

    exp96_campaign._source(scheduler_commit)
    queue, plan = exp96_campaign.load(parent_queue, require_approval=False)
    trainer = clean_checkout(trainer, queue["commit"])
    root = Path(root)
    parent_queue = Path(parent_queue).resolve()
    if (
        not root.is_absolute()
        or root.exists()
        or root.is_relative_to(MAIN.parent)
        or root.is_relative_to(parent_queue.parent)
        or type(seed) is not int
        or seed <= 8
    ):
        raise ValueError(
            "fresh isolated preparation root and independent dev seed>8 required"
        )
    root.mkdir(parents=True)
    arm_queue = root / "arm_queue"
    output = root / "arms_output"
    arm_queue.mkdir()
    output.mkdir()
    (arm_queue / "slurm").mkdir()
    binding = {
        "parent_queue": str(parent_queue),
        "queue_sha256": digest(parent_queue / "queue.json"),
        "parent_commit": queue["commit"],
        "scheduler_commit": scheduler_commit,
        "arm_output_root": str(output),
    }
    write(arm_queue / "parent_binding.json", binding)
    candidates = [
        p
        for p in queue["parents"]
        if p["row"]["architecture"] in ("D4W1024", "D4W1536")
    ]
    assert len(candidates) == len({p["id"] for p in candidates}) == 64
    write(
        root / "parent_candidates.json",
        {
            **binding,
            "parent_count": 96,
            "potential_fork_parents": candidates,
            "actual_arms": [],
            "injection_m": None,
            "reason": "Only real natural fork plus owner-approved m can create an arm identity",
            "control_results_root": str(Path(plan["root"]) / "results"),
            "injected_results_root": str(output / "results"),
        },
    )
    approval = read(MAIN / "configs/exp96_exp2_approval.template.json")
    approval.update(
        approved=False,
        expected_commit=queue["commit"],
        scheduler_commit=scheduler_commit,
        queue_sha256=binding["queue_sha256"],
        trainer_checkout=str(trainer),
        arm_queue_root=str(arm_queue),
        arm_output_root=str(output),
        funding_path=str(root / "funding.draft.json"),
        injection_m=None,
    )
    write(root / "approval.draft.json", approval)
    write(
        root / "funding.draft.json",
        {
            "schema": SCHEMA,
            "approved": False,
            "limit_su": 0,
            "decision_reference": "",
            "expected_commit": queue["commit"],
            "scheduler_commit": scheduler_commit,
            "queue_sha256": binding["queue_sha256"],
            "entries": {},
        },
    )
    development = root / "development" / f"D4W1536_dog_seed{seed}"
    qualifiers = root / "positive_control"
    common = dict(
        partition="gpuA100x4",
        nodes=1,
        gpus_per_node=1,
        cpus_per_node=16,
        memory_gb_per_node=60,
    )
    dev_resources = dict(common, wall_minutes=1440)
    pc_resources = dict(common, wall_minutes=120)
    dev_command = [
        "python",
        "-u",
        "run.py",
        "--experiment",
        "exp1",
        "--config_name",
        "base_exp12_ten",
    ]
    for value in (
        "env_name=dog-run",
        "env=dmc_hard",
        "critic_num_blocks=4",
        "critic_hidden_dim=1536",
        "run_role=dev",
        f"seed={seed}",
        "env.num_env_steps=1000000",
        f"results_root={root/'development_results'}",
    ):
        dev_command += ["--overrides", value]
    dev_command += [
        "--checkpoint_dir",
        str(development),
        "--checkpoint_interval",
        "10000",
        "--checkpoint_start_frac",
        "0.0",
    ]
    manifest = {
        "schema": SCHEMA,
        **binding,
        "expected_commit": queue["commit"],
        "funding_path": str(root / "funding.draft.json"),
        "exp2_gpu_spending_authorized": False,
        "development": {
            "command": dev_command,
            "cwd": str(trainer),
            "resources": dev_resources,
        },
        "positive-control": {
            "command": [
                "python",
                "-u",
                "scripts/positive_control.py",
                "--run_dir",
                str(development),
                "--out_dir",
                str(qualifiers),
            ],
            "cwd": str(MAIN),
            "resources": pc_resources,
        },
        "planning_cost_su": {
            "development_cap": su_cost(dev_resources),
            "positive_control_cap": su_cost(pc_resources),
        },
        "arm_qualification": "After owner m approval, reuse natural forks for both architectures and both suites; measured profiles and recovery still required",
    }
    write(root / "qualification_plan.json", manifest)
    import shlex

    lines = [
        "# Commands only. NOT submitted. Explicit separate funding and job reservation required."
    ]
    for phase in ("development", "positive-control"):
        r = manifest[phase]["resources"]
        command = [
            "env",
            f"EXP2_SCHEDULER_MAIN={MAIN}",
            f"EXP2_QUALIFICATION_PLAN={root/'qualification_plan.json'}",
            f"EXP2_PHASE={phase}",
            "sbatch",
            "--parsable",
            "--account=biqc-delta-gpu",
            "--partition=gpuA100x4",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            "--cpus-per-task=16",
            "--mem=60G",
            f"--time={r['wall_minutes']}",
            f"--output={root}/{phase}_%j.out",
            f"--error={root}/{phase}_%j.err",
            str(MAIN / "scripts/sbatch_exp96_exp2_qualification.sh"),
        ]
        lines.append(shlex.join(command))
    (root / "qualification_commands.txt").write_text("\n".join(lines) + "\n")
    return manifest


def qualification(plan_path, phase):
    plan = read(plan_path)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or phase not in ("development", "positive-control"):
        raise ValueError("real approved qualification allocation required")
    funding_gate(plan, job_id, resources=plan[phase]["resources"])
    from scripts.exp96_exp2_arms import clean_checkout

    clean_checkout(MAIN, plan["scheduler_commit"])
    clean_checkout(plan["development"]["cwd"], plan["expected_commit"])
    r = plan[phase]["resources"]
    if (
        int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) != 1
        or int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != 1
    ):
        raise ValueError("qualification requires one allocated A100")
    from scripts.exp12_launch_plan import allocation_matches

    view = subprocess.check_output(
        ["scontrol", "show", "job", "-o", job_id], text=True, timeout=20
    )
    if "Account=biqc-delta-gpu" not in view or "Partition=gpuA100x4" not in view:
        raise ValueError(
            "qualification account/partition differs from reviewed resources"
        )
    allocation_matches(
        {
            "cpus": r["cpus_per_node"],
            "memory_gb": r["memory_gb_per_node"],
            "wall_minutes": r["wall_minutes"],
        },
        os.environ,
        view,
    )
    verify_billing(r, view)
    command = [sys.executable, *plan[phase]["command"][1:]]
    return subprocess.call(command, cwd=plan[phase]["cwd"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "qualification"))
    parser.add_argument("--root")
    parser.add_argument("--parent-queue")
    parser.add_argument("--trainer-main")
    parser.add_argument("--scheduler-commit")
    parser.add_argument("--plan")
    parser.add_argument("--phase", choices=("development", "positive-control"))
    args = parser.parse_args()
    if args.action == "prepare":
        print(
            json.dumps(
                prepare(
                    args.root,
                    args.parent_queue,
                    args.trainer_main,
                    args.scheduler_commit,
                ),
                indent=2,
            )
        )
        return 0
    return qualification(args.plan, args.phase)


if __name__ == "__main__":
    sys.exit(main())
