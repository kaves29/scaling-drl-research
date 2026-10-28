"""Review-only six-cell plan and literal request builder; never submits jobs."""

import argparse
import json
from pathlib import Path
import re

BASE = "844e3c0cc24998b25c3d1e667bcbfe056b1b336c"
SCHEDULE = "even_milestones_v2"
TEACHER = dict(block_type="residual", num_blocks=2, hidden_dim=512, dtype="float32")
CELLS = [
    ("dog-small", "dog-run", "D2W512", 0, True),
    ("myo-medium", "myo-pen-twirl", "D4W1024", 0, False),
    ("dog-large", "dog-run", "D4W1536", 0, False),
    ("myo-small", "myo-pen-twirl", "D2W512", 1, True),
    ("dog-medium", "dog-run", "D4W1024", 1, False),
    ("myo-large", "myo-pen-twirl", "D4W1536", 1, False),
]
ALLOCATION = dict(
    nodes=1,
    GPUs=2,
    run_processes=6,
    processes_per_GPU=[3, 3],
    waves=1,
    walltime_seconds=21600,
    max_aggregate_reserved_GPU_hours=12,
)


def validate_design(plan):
    if plan["base_sha"] != BASE or any(
        plan.get(k) is not False for k in ("approved", "executable", "submitted")
    ):
        raise ValueError("review flags and approved base required")
    actual = [
        (r["id"], r["environment"], r["architecture"], r["gpu"], r["bank_producer"])
        for r in plan["runs"]
    ]
    if actual != CELLS or any(r["seed"] != 0 for r in plan["runs"]):
        raise ValueError("exact six-cell grid, seed and GPU assignment required")
    expected = dict(
        schedule=SCHEDULE,
        teacher=TEACHER,
        teacher_architecture="D2W512",
        target_scale=100000.0,
        pool_size=25600,
        batch_size=256,
        chunk=2560,
        warmup=5000,
        endpoint_interactions=500000,
        cadence_interactions=50000,
        reported_K=[2, 4, 6, 8],
        AB_fitted_K=list(range(1, 9)),
        C_fitted_K=[2, 4, 6, 8],
        steps_per_target=2000,
        observations_per_run=11,
        updates_per_observation=40000,
        updates_per_run=440000,
        updates_total=2640000,
        fixed_fresh_architecture_offset=True,
    )
    if plan["diagnostic"] != expected or plan["requested_allocation"] != ALLOCATION:
        raise ValueError("scientific schedule/teacher or allocation cap changed")
    return {
        "status": "DESIGN_VERIFIED_NOT_CLUSTER_QUALIFIED",
        "submission_allowed": False,
    }


def request_for(
    plan,
    run_id,
    *,
    main,
    root,
    project,
    source_sha,
    checkpoint_interval,
    bank_timeout_seconds,
    allocation_deadline_unix,
):
    """Caller must separately verify cluster/approvals; build no resource guesses.

    This only returns data. Delta infrastructure writes unique request/config
    hashes, sets the recorded GPU UUID/W&B identity and runs the reviewed runner.
    """
    validate_design(plan)
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("exact reviewed source SHA required")
    if not Path(root).is_absolute() or not Path(main).is_absolute():
        raise ValueError("absolute source/output roots required")
    if (
        isinstance(checkpoint_interval, bool)
        or not isinstance(checkpoint_interval, int)
        or checkpoint_interval <= 0
    ):
        raise ValueError("explicit positive checkpoint interval required")
    row = next(r for r in plan["runs"] if r["id"] == run_id)
    depth, width = map(int, row["architecture"][1:].split("W"))
    environment = row["environment"]
    suite = "dmc_hard" if environment == "dog-run" else "myosuite_simba"
    output = Path(root) / "runs" / run_id
    producer = "dog-small" if environment == "dog-run" else "myo-small"
    alternative = dict(
        enabled=True,
        bank=str(Path(root) / "banks" / environment),
        reference=str(output / "reference"),
        output=str(output / "alternative"),
        teacher=dict(TEACHER),
        teacher_architecture="D2W512",
        architecture=row["architecture"],
        environment=environment,
        seed=0,
        source=source_sha,
        bank_producer=row["bank_producer"],
        schedule=SCHEDULE,
    )
    if not row["bank_producer"]:
        alternative["bank_wait"] = dict(
            timeout_seconds=bank_timeout_seconds,
            allocation_deadline_unix=allocation_deadline_unix,
            producer_status=str(Path(root) / "status" / f"{producer}.json"),
        )
    args = dict(
        config_path=str(Path(main) / "configs"),
        config_name="base_exp12",
        checkpoint_dir=str(output),
        checkpoint_interval=checkpoint_interval,
        checkpoint_start_frac=0.0,
        overrides=[
            f"env={suite}",
            f"env_name={environment}",
            "seed=0",
            "run_role=dev",
            f"critic_num_blocks={depth}",
            f"critic_hidden_dim={width}",
            "fork.enabled=false",
            "probe.steps=2000",
            "probe.pool_size=25600",
            "probe.batch_size=256",
            "probe.eval_chunk=2560",
            "probe.target_scale=100000",
            "buffer.min_length=5000",
            "evaluation_per_interaction_step=50000",
            f"project_name={project}",
        ],
    )
    return {"args": args, "alternative": alternative}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    print(
        json.dumps(validate_design(json.loads(Path(args.plan).read_text())), indent=2)
    )


if __name__ == "__main__":
    main()
