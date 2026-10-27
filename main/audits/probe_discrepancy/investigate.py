"""Reconstruct paired signs; prepare retrieval inventory and an unapproved pilot."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audits.echocritic_forensic.evidence import (
    CAMPAIGN,
    prepare_output,
    read_records,
    reconstruct_probes,
    write_csv,
)
from audits.probe_difficulty.accounting import (
    ENVIRONMENTS,
    ARCHITECTURES,
    seed_interval,
)
from audits.probe_discrepancy.source_checks import (
    controlled_routing,
    function,
    original_source,
    structural_checks,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_representatives(rows):
    """Exploratory outcome-selected retrieval targets, not an unbiased new sample."""
    values = sorted(rows, key=lambda x: x["seed"])
    if len(values) != 8 or len({x["seed"] for x in values}) != 8:
        raise ValueError("eight unique independent seed labels required")
    median = np.median([x["paired_loss_iqm"] for x in values])
    typical = min(values, key=lambda x: (abs(x["paired_loss_iqm"] - median), x["seed"]))
    exceptional = max(
        (x for x in values if x["id"] != typical["id"]),
        key=lambda x: (abs(x["paired_loss_iqm"] - median), -x["seed"]),
    )
    return typical, exceptional


def paired_tables(records, probes):
    identity = {r["id"]: r for r in records if r["cohort"] == "exp96"}
    rows = []
    for p in probes:
        if p["id"] not in identity or p["check"] == 0:
            continue
        run = identity[p["id"]]
        rows.append(
            {
                "id": p["id"],
                "environment": run["environment"],
                "architecture": run["architecture"],
                "seed": run["seed"],
                "check": p["check"],
                "interaction_step": p["interaction_step"],
                "paired_loss_iqm": p["paired_deficit"],
                "trained_better": p["paired_deficit"] < 0,
                "source": run["source"],
                "raw_row_index": next(
                    i
                    for i, h in enumerate(run["history"])
                    if h.get("probe/check_index") == p["check"]
                ),
            }
        )
    groups = defaultdict(list)
    for row in rows:
        if row["check"] in (5, 35):
            groups[row["environment"], row["architecture"], row["check"]].append(row)
    summaries = []
    for index, ((env, arch, check), values) in enumerate(sorted(groups.items())):
        if len(values) != 8:
            raise ValueError("shared-check coverage failure")
        estimate = seed_interval([v["paired_loss_iqm"] for v in values], 80000 + index)
        summaries.append(
            {
                "environment": env,
                "architecture": arch,
                "check": check,
                "trained_better_seeds": sum(v["trained_better"] for v in values),
                "trained_worse_seeds": sum(v["paired_loss_iqm"] > 0 for v in values),
                **estimate,
            }
        )
    return rows, groups, summaries


def retrieval_inventory(records, groups, package):
    by_id = {r["id"]: r for r in records}
    rows, selected = [], []
    for env in ENVIRONMENTS:
        for arch in ARCHITECTURES:
            for label, item in zip(
                ("typical", "exceptional"),
                choose_representatives(groups[env, arch, 35]),
            ):
                run = by_id[item["id"]]
                root = (
                    Path(run["config"]["results_root"]).parent
                    / "checkpoints/exp1"
                    / arch
                    / env
                    / f"seed_{run['seed']}"
                )
                selected.append({**item, "selection": label})
                for check in (5, 35):
                    rows.append(
                        {
                            "id": run["id"],
                            "environment": env,
                            "architecture": arch,
                            "seed": run["seed"],
                            "selection": label,
                            "check": check,
                            "expected_probe_path": str(
                                root / f"probes/check_{check:02d}.npz"
                            ),
                            "source": CAMPAIGN,
                            "sha256": None,
                            "completeness": "not recovered; remote existence unverified",
                            "failure_reason": "Delta root not mounted; TCP proxy refused; no configured VPN/credentials",
                            "additional_requested_paths": "fresh_critic/; run_metadata.json; probes/probe_checks.csv; state/LATEST; fork/state/LATEST",
                            "snapshot_warning": "LATEST need not match requested check; older routine state may have been pruned",
                        }
                    )
    archive = Path(package) / "raw_archived/D2W512_myo-pen-twirl_seed5_check50.npz"
    rows.append(
        {
            "id": "63b389qb",
            "environment": "myo-pen-twirl",
            "architecture": "D2W512",
            "seed": 5,
            "selection": "available original",
            "check": 50,
            "expected_probe_path": str(archive),
            "source": CAMPAIGN,
            "sha256": sha(archive),
            "completeness": "b/final_loss/score/offset/losses for ten paired rounds; no pool/parameters/RMS/teacher",
            "failure_reason": None,
            "additional_requested_paths": None,
            "snapshot_warning": None,
        }
    )
    return rows, selected


# Each scientific condition is replicated unchanged on both allocated GPUs.
PILOT_CELLS = (
    ("dog-run", "D4W1536"),
    ("humanoid-run", "D4W1024"),
    ("myo-key-turn", "D4W1536"),
    ("myo-pen-twirl", "D4W1024"),
    ("myo-pen-twirl", "D2W512"),
)


def validate_pilot(manifest):
    from audits.probe_discrepancy.plan import validate_design

    validate_design(manifest)
    if manifest["launch_approved"] or manifest["executable"]:
        raise ValueError("generator produces an unapproved draft only")


def pilot_design(selected, records):
    lookup = {r["id"]: r for r in records}
    chosen = [
        next(
            r
            for r in selected
            if (r["environment"], r["architecture"]) == cell
            and r["selection"] == "typical"
        )
        for cell in PILOT_CELLS
    ]
    # Retain the one directly auditable original NPZ case as the small anchor.
    chosen[-1] = next(
        r
        for r in records
        if r["cohort"] == "exp96"
        and r["environment"] == "myo-pen-twirl"
        and r["architecture"] == "D2W512"
        and r["seed"] == 5
    )
    runs = []
    for i, row in enumerate(chosen, 1):
        run = lookup[row["id"]]
        cfg = run["config"]
        scientific = {
            k: v
            for k, v in cfg.items()
            if k not in ("results_root", "project_name", "server")
        }
        encoded = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
        for gpu in (0, 1):
            runs.append(
                {
                    "pilot_id": f"C{i:02d}G{gpu}",
                    "pair_id": f"C{i:02d}",
                    "gpu": gpu,
                    "environment": row["environment"],
                    "architecture": row["architecture"],
                    "seed": row["seed"],
                    "selection": "same historical seed replicated on both GPUs; small anchor has original archived NPZ",
                    "original_run_id": row["id"],
                    "training_source": CAMPAIGN,
                    "source_resolved_config_sha256": hashlib.sha256(
                        encoded
                    ).hexdigest(),
                    "source_scientific_config_sha256": hashlib.sha256(
                        json.dumps(
                            scientific, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                    "source_scientific_config": scientific,
                    "arm": "unchanged original protocol with proposed observational retention",
                    "historical_runtime_seconds_PROXY_NOT_GPU_USAGE": run["meta"][
                        "summary"
                    ].get("_runtime"),
                    "nominal_endpoint_interactions": 500000,
                    "control_extension": "preserve original natural-fork untreated extension",
                    "retained_checks": [0, 5, 35, 50],
                    "output_relative": f"runs/C{i:02d}G{gpu}",
                    "wandb_id": f"echocritic-c{i:02d}g{gpu}",
                    "wandb_group": "discrepancy-10-simultaneous-v2",
                }
            )
    hours = (
        sum(r["historical_runtime_seconds_PROXY_NOT_GPU_USAGE"] or 0 for r in runs)
        / 3600
    )
    manifest = {
        "schema": "echocritic-discrepancy-pilot-design-v2",
        "launch_approved": False,
        "executable": False,
        "new_campaign_root": None,
        "count": 10,
        "training_source": CAMPAIGN,
        "diagnostic_implementation_source": None,
        "diagnostic_status": "adapter not implemented/reviewed/qualified",
        "selection_warning": "ten executions, five outcome-selected scientific seed/configurations; matched GPU replicas are not independent seed evidence",
        "schedule": {
            "mode": "simultaneous",
            "processes_per_gpu": [5, 5],
            "waves": 1,
            "replacement_runs": False,
            "automatic_retries": False,
            "barrier": "all ten children ready before any adapter is released",
        },
        "allocation_proposal": {
            "account": "biqc-delta-gpu",
            "partition": "gpuA100x4",
            "nodes": 1,
            "gpus": 2,
            "ntasks": 1,
            "cpus": 32,
            "host_memory_GiB": 128,
            "wall_minutes": 2880,
            "gpu_model": "NVIDIA A100-SXM4-40GB",
        },
        "process_environment": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_PLATFORMS": "cuda,cpu",
            "JAX_ENABLE_X64": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "matmul_precision": "tensorfloat32; unchanged source rule",
        "retention_contract": {
            "all_checks": [
                "original round b/MSE/scores/offsets",
                "2000 pre-update minibatch losses",
                "source/config/checkpoint/run identity hashes",
            ],
            "selected_checks": [
                "raw replay indices/pool/actions and hashes",
                "RMS state and normalized inputs",
                "teacher tree/keys/phase/base and literal own-offset targets",
                "fresh/current initial trees, Q/centered residual/covariance",
                "full-pool Q and MSE at 0/100/500/1000/2000, separate from minibatches",
                "optimizer/training RNG/probe RNG/evaluation RNG provenance",
            ],
            "telemetry": [
                "per-process RSS/CPU",
                "GPU UUID and VRAM/utilization",
                "compile and steady-state timing",
                "pool/fit/checkpoint durations and bytes",
                "shared filesystem I/O/free space",
                "barrier/exec timing",
            ],
            "pairing": "same raw/normalized inputs, teacher/base and minibatch stream; original own offsets retained",
            "failure_policy": "retain partial/failed outputs, no deletion, replacement or automatic retry",
        },
        "resource_estimate": {
            "packing_requested": True,
            "packing_qualified": False,
            "sum_historical_runtime_hours_NOT_ALLOCATION_HOURS": hours,
            "reserved_gpu_hours_at_proposed_wall_cap_NOT_AUTHORIZED": 96,
            "cpus_per_child_proposed": 3,
            "controller_cpus_proposed": 2,
            "storage_estimate_GiB_NOT_AUTHORIZED": [200, 500],
            "throughput_VRAM_storage_actual_measurements": None,
            "feasibility": "unknown; do not divide historical runtimes by five or assume 48h completion",
        },
        "runs": runs,
    }
    validate_pilot(manifest)
    return manifest


def plots(rows, output):
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), sharex=True)
    for i, env in enumerate(ENVIRONMENTS):
        for j, arch in enumerate(ARCHITECTURES):
            ax = axes[i, j]
            for seed in range(1, 9):
                values = [
                    r
                    for r in rows
                    if (r["environment"], r["architecture"], r["seed"])
                    == (env, arch, seed)
                ]
                ax.plot(
                    [r["interaction_step"] / 1000 for r in values],
                    [r["paired_loss_iqm"] for r in values],
                    label=str(seed),
                    alpha=0.7,
                    lw=0.8,
                )
            ax.axhline(0, color="black", lw=0.6)
            ax.axvline(350, color="grey", ls=":", lw=0.7)
            ax.set_title(f"{env} / {arch}", fontsize=9)
            if j == 0:
                ax.set_ylabel("MSE trained − MSE fresh\n(source paired-score IQM)")
            if i == 3:
                ax.set_xlabel("Interactions (thousands)")
    axes[0, 0].legend(title="Training seed", ncol=4, fontsize=6)
    fig.suptitle(
        "Original paired contrast: negative = trained better; observed checks only"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output / "paired_all12.pdf")
    fig.savefig(output / "paired_all12.png", dpi=130)
    plt.close(fig)


def run(package, out):
    output = prepare_output(out, package)
    records = read_records(package)
    probes, errors = reconstruct_probes(records)
    if not all(row["passed"] for row in errors):
        raise ValueError("original score reconstruction failed")
    rows, groups, summaries = paired_tables(records, probes)
    inventory, selected = retrieval_inventory(records, groups, package)
    manifest = pilot_design(selected, records)
    source = original_source("main/experiments/exp12/probe.py")
    checks = structural_checks(source)
    if not all(r["passed"] for r in checks):
        raise ValueError("source structural guard failed")
    routing = controlled_routing(source)
    with np.load(
        Path(package) / "raw_archived/D2W512_myo-pen-twirl_seed5_check50.npz",
        allow_pickle=False,
    ) as a:
        result = {kind: {"score": a[kind + "_score"]} for kind in ("fresh", "current")}
        paired = function(source, "paired_loss", {"np": np})(result)
        real_difference = a["current_final_loss"].astype(float) - a[
            "fresh_final_loss"
        ].astype(float)
        max_error = float(np.max(np.abs(paired.astype(float) - real_difference)))
        np.testing.assert_array_equal(paired, a["fresh_score"] - a["current_score"])
    for name, table in [
        ("paired_seed_checks", rows),
        ("paired_common_summaries", summaries),
        ("recovery_inventory", inventory),
        ("recovery_selected_seeds", selected),
        ("original_source_guards", checks),
    ]:
        write_csv(output / f"{name}.csv", table)
    (output / "pilot_manifest.DRAFT.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    plots(rows, output)
    receipt = {
        "source": CAMPAIGN,
        "canonical_parents": 96,
        "excluded_pilots": 7,
        "paired_checks": len(rows),
        "targeted_missing_check_files": 48,
        "available_original_check_files": 1,
        "source_guards": len(checks),
        "controlled_routing": routing,
        "archive_max_paired_float32_vs_real_mse_delta": max_error,
        "pilot_runs": len(manifest["runs"]),
        "new_fits": 0,
        "model_forwards": 0,
        "gpu_jobs": 0,
        "remote_storage_inventory_completed": False,
        "pilot_approved": False,
        "pilot_manifest_sha256": sha(output / "pilot_manifest.DRAFT.json"),
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args.package, args.out)


if __name__ == "__main__":
    main()
