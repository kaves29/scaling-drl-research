"""Kaggle process scheduler: one run per GPU unless concurrency is qualified.

Each child sees exactly one CUDA device. The two-job ceiling never enables
multi-GPU training of a single reference run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.exp12.revised_exp2_population import population


def devices():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        dict(
            zip(
                ("index", "uuid", "name", "total_mib", "free_mib"),
                [part.strip() for part in line.split(",")],
            )
        )
        for line in output.splitlines()
    ]


def concurrency(gpu, requested, receipt, source_commit):
    if requested not in (1, 2):
        raise ValueError("at most two independent runs per GPU")
    if requested == 1:
        return 1
    if not receipt or receipt.get("source_commit") != source_commit:
        return 1
    evidence = receipt.get("devices", {}).get(gpu["uuid"], {})
    # Evidence must cover the four real full-horizon branches, probe, and restart.
    if not all(
        receipt.get(key) is True
        for key in (
            "full_branch_horizon",
            "plasticity_probe",
            "restart",
            "concurrent_pass",
        )
    ):
        return 1
    peak = evidence.get("concurrent_peak_mib")
    if (
        evidence.get("jobs") != 2
        or not isinstance(peak, (int, float))
        or peak <= 0
        or peak > float(gpu["free_mib"])
    ):
        return 1
    return 2


def pair_is_qualified(gpu, active, candidate, receipt):
    """A measured mixed-regime dog pair cannot admit two large runs or Myo."""
    if len(active) != 1 or not receipt:
        return False
    identity = lambda row: (row["environment"], row["reference_regime"])
    requested = sorted([identity(active[0]), identity(candidate)])
    evidence = receipt.get("devices", {}).get(gpu["uuid"], {})
    return any(
        sorted(identity(row) for row in workload) == requested
        for workload in evidence.get("qualified_workloads", [])
        if len(workload) == 2
    )


def run_one(row, output):
    from experiments.exp1 import compose_config
    from experiments.online_exp2 import OnlineExp2
    from experiments.exp12.precision import set_matmul_precision

    set_matmul_precision()
    cfg = compose_config(
        str(Path(__file__).parents[2] / "configs"),
        "online_exp2_pilot_" + row["reference_regime"],
        [
            "env="
            + (
                "myosuite_hard" if row["environment"].startswith("myo-") else "dmc_hard"
            ),
            "env_name=" + row["environment"],
            "seed=" + str(row["seed"]),
        ],
        allow_pilot=True,
    )
    OnlineExp2(cfg, str(output.resolve())).run()


def schedule(output, jobs_per_gpu=1, qualification=None):
    output.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    receipt = json.loads(qualification.read_text()) if qualification else None
    inventory = devices()
    if not inventory:
        raise RuntimeError("no NVIDIA devices detected")
    slots = [
        (gpu, concurrency(gpu, jobs_per_gpu, receipt, source)) for gpu in inventory
    ]
    (output / "schedule.json").write_text(
        json.dumps(
            {
                "source_commit": source,
                "population": population(),
                "devices": inventory,
                "concurrency": {g["uuid"]: count for g, count in slots},
            },
            indent=2,
        )
    )
    pending, active, failed = list(population()), [], []
    try:
        while pending or active:
            for gpu, count in slots:
                while (
                    pending and sum(job["gpu"] == gpu["uuid"] for job in active) < count
                ):
                    current = [
                        job["row"] for job in active if job["gpu"] == gpu["uuid"]
                    ]
                    if current and not pair_is_qualified(
                        gpu, current, pending[0], receipt
                    ):
                        break
                    row = pending.pop(0)
                    name = f"{row['environment']}_seed{row['seed']}_{row['reference_regime']}"
                    directory = output / name
                    directory.mkdir(exist_ok=True)
                    if (directory / "analysis.json").exists():
                        analysis = json.loads((directory / "analysis.json").read_text())
                        if (
                            analysis.get("source_commit") != source
                            or analysis.get("environment") != row["environment"]
                            or analysis.get("training_seed") != row["seed"]
                            or analysis.get("reference_regime")
                            != row["reference_regime"]
                        ):
                            raise ValueError(
                                "completed result identity mismatch; select a new output root"
                            )
                        continue
                    # A completed analysis is the only completion marker. Existing states resume.
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=gpu["index"],
                        MUJOCO_EGL_DEVICE_ID="0",
                        XLA_PYTHON_CLIENT_PREALLOCATE="false",
                        JAX_PLATFORMS="cuda",
                        WANDB_MODE="online",
                        PYTHONPATH=str(Path(__file__).parents[2])
                        + os.pathsep
                        + os.environ.get("PYTHONPATH", ""),
                        OPENBLAS_NUM_THREADS="1",
                        OMP_NUM_THREADS="1",
                    )
                    log = (directory / "process.log").open("a")
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "experiments.exp12.revised_exp2_scheduler",
                            "--child",
                            json.dumps(row),
                            "--output",
                            str(directory),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    active.append(
                        dict(
                            process=process,
                            log=log,
                            name=name,
                            gpu=gpu["uuid"],
                            row=row,
                        )
                    )
            for job in list(active):
                code = job["process"].poll()
                if code is not None:
                    job["log"].close()
                    active.remove(job)
                    if code:
                        failed.append({"run": job["name"], "exit_code": code})
                    print(f"{job['name']}: exit {code}", flush=True)
            if active:
                time.sleep(5)
    finally:
        for job in active:
            job["process"].terminate()
            job["process"].wait()
            job["log"].close()
    (output / "scheduler_result.json").write_text(
        json.dumps({"failures": failed}, indent=2)
    )
    if failed:
        raise RuntimeError(
            f"{len(failed)} reference runs failed; inspect process.log and rerun to resume"
        )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs-per-gpu", type=int, choices=(1, 2), default=1)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.child:
        run_one(json.loads(args.child), args.output)
    else:
        schedule(args.output, args.jobs_per_gpu, args.qualification)


if __name__ == "__main__":
    main()
