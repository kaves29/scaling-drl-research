"""Fail-closed admission and reporting for one diagnostic continuation."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import os
import zipfile

import numpy as np

FROZEN = "f597ee9898e4d401912092c336c3e5330cf2fda1"
FIRST_SOURCE = "bcf6cf3c90726c77f998e5795fedf0d62c5f11e2"
FIRST_JOB = "23120792"
CEILING_SECONDS = 14400
CHILD_SECONDS = 3600
CORE_FILES = (
    "experiments/exp12/probe.py",
    "experiments/exp12/fixed_task.py",
    "experiments/exp12/diagnostic_artifacts.py",
    "experiments/exp12/precision.py",
    "scale_rl/agents/sac/sac_network.py",
    "scale_rl/networks/layers.py",
    "scale_rl/networks/utils.py",
    "configs/agent/sac.yaml",
)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def command(args, cwd=None):
    result = subprocess.run(
        args, cwd=cwd, env={**os.environ, "TZ": "UTC"},
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr[:1500]}")
    return result.stdout


def accounting(job_ids):
    raw = command([
        "sacct", "-D", "-nP", "-j", ",".join(job_ids),
        "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,Partition",
    ])
    result = {}
    for line in raw.splitlines():
        fields = line.split("|")
        if len(fields) < 6 or fields[0] not in job_ids:
            continue
        job, state, exit_code, seconds, tres, partition = fields[:6]
        if job in result:
            raise ValueError("duplicate/requeued accounting requires human review")
        resources = dict(x.split("=", 1) for x in tres.split(",") if "=" in x)
        elapsed = int(seconds)
        gpu = int(resources.get("gres/gpu", "0"))
        terminal = state.split()[0] in {
            "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
        }
        if not terminal or (elapsed and gpu != 1):
            raise ValueError("unsettled or unexpected GPU accounting")
        result[job] = {
            "state": state, "exit_code": exit_code, "elapsed_seconds": elapsed,
            "gpu_count": gpu, "physical_gpu_seconds": elapsed * gpu,
            "alloc_tres": tres, "partition": partition,
            "resource_weighted_SU_estimate": elapsed / 3600 * float(resources.get("billing", 0)) / 1000,
        }
    if set(result) != set(job_ids):
        raise ValueError("accounting missing; no GPU admission")
    return result


def budget_admission(usage, reservations):
    if set(usage) != {"23120674", FIRST_JOB}:
        raise ValueError("unexpected investigation job set")
    if any(str(x["job_id"]) not in usage for x in reservations):
        raise ValueError("another reservation already exists")
    parent = usage[FIRST_JOB]
    if parent["state"] != "COMPLETED" or parent["exit_code"] != "0:0" or parent["gpu_count"] != 1:
        raise ValueError("first GPU stage did not complete cleanly")
    seconds = sum(x["physical_gpu_seconds"] for x in usage.values())
    if seconds + CHILD_SECONDS > CEILING_SECONDS:
        raise ValueError("insufficient physical A100 budget")
    return seconds


def validate_first(root, first_main, current_main):
    root = Path(root)
    status, runtime, results = [read_json(root / n) for n in ["status.json", "runtime.json", "results.json"]]
    if status["status"] != "COMPLETED_BUDGETED_STAGE":
        raise ValueError("first stage failed or unfinished")
    if runtime["job_id"] != FIRST_JOB or runtime["expected_commit"] != FIRST_SOURCE:
        raise ValueError("first job/source binding mismatch")
    env = runtime["runtime"]
    if env["platform"] != "gpu" or "A100" not in env["device_kind"] or env["matmul_precision"] != "tensorfloat32":
        raise ValueError("wrong hardware/precision")
    if results["source"] != FIRST_SOURCE or results["frozen_source"] != FROZEN:
        raise ValueError("result source binding mismatch")
    hashes = {}
    for name in CORE_FILES:
        old = (Path(first_main) / name).read_bytes()
        new = (Path(current_main) / name).read_bytes()
        if old != new:
            raise ValueError("numerical core changed: " + name)
        hashes[name] = hashlib.sha256(old).hexdigest()
    reproduced = [x["original_reproduction"] for x in results["rows"] if "original_reproduction" in x]
    required = {("myo-pen-twirl", "D2W512", 5), ("humanoid-run", "D4W1536", 1)}
    summaries = []
    for env_name, arch, seed in sorted(required):
        directory = root / f"original_exp1_{arch}_{env_name}_seed{seed}_r0"
        with np.load(directory / "task.npz", allow_pickle=False) as task:
            if task["obs"].shape[0] != 25600 or task["indices"].shape != (8000, 256):
                raise ValueError("incomplete reconstruction task")
            if any(not np.isfinite(task[n]).all() for n in ("obs", "act", "base")):
                raise ValueError("nonfinite original numerical task")
        for kind in ("fresh", "latest"):
            rows = [x for x in reproduced if (x["environment"], x["architecture"], x["seed"], x["kind"], x["round"]) == (env_name, arch, seed, kind, 0)]
            if len(rows) != 1:
                raise ValueError("required original fit absent/duplicated")
            row = rows[0]
            if not all(np.isfinite(row[k]) for k in ("final_mse", "reference_mse", "offset_error", "variance_error", "curve_max_abs_error")):
                raise ValueError("nonfinite original fit receipt")
            if row["variance_error"] > .002 or row["offset_error"] > .05 or abs(row["final_mse"] - row["reference_mse"]) > max(.02, .1 * abs(row["reference_mse"])):
                raise ValueError("historical reconstruction mismatch")
            with np.load(directory / f"{kind}_original_fit.npz", allow_pickle=False) as data:
                losses = data["losses"]
                if losses.shape != (2000,) or not np.isfinite(losses).all():
                    raise ValueError("incomplete original curve")
                summaries.append({**row, "initial_minibatch_mse": float(losses[0]), "early_peak_mse": float(losses[:100].max())})
        equivalent = read_json(directory / "segmented_equivalence.json")
        fresh = next(x for x in summaries if (x["environment"], x["kind"]) == (env_name, "fresh"))
        if not np.isfinite(equivalent["final_abs_error"]) or equivalent["final_abs_error"] > max(1e-4, .005 * abs(fresh["final_mse"])):
            raise ValueError("GPU segmented equivalence failed")
    # The frozen runner reaches COMPLETED only after its exact repeat checks.
    return {"original_fits": summaries, "numerical_core_sha256": hashes, "runtime": runtime}


def write_report(auto_root, first_root, usage=None, error=None):
    root = Path(auto_root)
    rows = []
    for stage, location in [("first", Path(first_root)), ("continuation", root / "gpu")]:
        p = location / "results.json"
        if not p.exists():
            continue
        data = read_json(p)
        for row in data["rows"]:
            if "measurements" not in row:
                continue
            for point in row["measurements"]:
                rows.append({"stage": stage, "name": row["name"], "environment": row["environment"], "architecture": row["architecture"], "seed": row["seed"], "kind": row["kind"], "task_id": row.get("task_id", "legacy-unbound"), **point})
    columns = sorted({k for r in rows for k in r})
    with (root / "fitting_measurements.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Bounded EchoCritic continuation", "", "This is diagnostic evidence, not an approved detector change or a causal actor claim.", ""]
    if error:
        lines.extend(["BLOCKED: " + error, "No further GPU job is admitted by this report.", ""])
    if usage is not None:
        atomic_json(root / "accounting.json", usage)
        seconds = sum(x["physical_gpu_seconds"] for x in usage.values())
        lines.extend([f"Allocated physical A100 hours: {seconds / 3600:.8f} / 4.", "Weighted charges in accounting.json are Slurm resource-time estimates, not posted ACCESS transactions.", ""])
    for row in rows:
        if row["step"] not in (0, 2000, 4000, 8000):
            continue
        lines.append(f"- {row['stage']} {row['name']}, update {row['step']}: MSE={row['mse']:.7g}, mean residual={row['mean_error']:.7g}, centered residual variance={row['centered_error_variance']:.7g}.")
    lines.extend(["", "Interpretation limits: zero-offset fixed tasks are supplementary contrasts; different starting functions remain a confound. Three-target screens are not IQM10 qualification. Missing/partial conditions are not passes. No scientific correction is applied automatically. Owner review is required before a scientific metric change."])
    (root / "findings.md").write_text("\n".join(lines) + "\n")
    with zipfile.ZipFile(root / "review_package.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(root.glob("*.json")):
            archive.write(p, p.name)
        for name in ("findings.md", "fitting_measurements.csv"):
            archive.write(root / name, name)
