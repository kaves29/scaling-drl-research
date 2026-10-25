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


def validate_pilot(manifest):
    runs = manifest["runs"]
    if manifest["launch_approved"] or manifest["executable"] or len(runs) != 14:
        raise ValueError(
            "manifest must remain an unapproved non-executable 14-run design"
        )
    if len({r["original_run_id"] for r in runs}) != 14:
        raise ValueError("duplicate original run identity")
    if manifest["new_campaign_root"] is not None:
        raise ValueError("owner has not reviewed a new output root")
    counts = defaultdict(int)
    for row in runs:
        cfg = row["source_scientific_config"]
        if (
            cfg["probe"]["steps"],
            cfg["probe"]["rounds"],
            cfg["probe"]["checks"],
            cfg["probe"]["pool_size"],
        ) != (2000, 10, 50, 25600):
            raise ValueError("historical probe settings changed")
        if cfg["num_interaction_steps"] != 500000 or row["training_source"] != CAMPAIGN:
            raise ValueError("historical source/budget changed")
        if (
            cfg["agent"]["critic_learning_rate"] != 1e-4
            or cfg["agent"]["critic_weight_decay"] != 0.01
        ):
            raise ValueError("optimizer changed")
        counts[row["architecture"]] += 1
    if dict(counts) != {"D4W1536": 8, "D4W1024": 4, "D2W512": 2}:
        raise ValueError("wrong pilot coverage")


def pilot_design(selected, records):
    lookup = {r["id"]: r for r in records}
    chosen = [
        r
        for r in selected
        if r["architecture"] == "D4W1536"
        or (r["architecture"] == "D4W1024" and r["selection"] == "typical")
        or (
            r["architecture"] == "D2W512"
            and r["selection"] == "typical"
            and r["environment"] in ("dog-run", "myo-pen-twirl")
        )
    ]
    runs = []
    for i, row in enumerate(chosen, 1):
        run = lookup[row["id"]]
        # Full source config remains separately hash-bound. These fields preserve
        # scientific settings without embedding the original writable output root.
        cfg = run["config"]
        scientific = {
            k: v
            for k, v in cfg.items()
            if k not in ("results_root", "project_name", "server")
        }
        encoded = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
        runs.append(
            {
                "pilot_id": f"P{i:02d}",
                "environment": row["environment"],
                "architecture": row["architecture"],
                "seed": row["seed"],
                "selection": row["selection"],
                "original_run_id": row["id"],
                "training_source": CAMPAIGN,
                "source_resolved_config_sha256": hashlib.sha256(encoded).hexdigest(),
                "source_scientific_config": scientific,
                "arm": "original protocol with proposed observation-only retention",
                "historical_runtime_seconds_PROXY_NOT_GPU_USAGE": run["meta"][
                    "summary"
                ].get("_runtime"),
                "nominal_endpoint_interactions": 500000,
                "control_extension": "preserve source natural-fork untreated continuation; max nominal/fork+125000, not artificially truncated",
                "retained_checks": [0, 5, 35, 50],
            }
        )
    runtimes = [r["historical_runtime_seconds_PROXY_NOT_GPU_USAGE"] for r in runs]
    hours = sum(v for v in runtimes if v is not None) / 3600
    manifest = {
        "schema": "echocritic-discrepancy-pilot-design-v1",
        "launch_approved": False,
        "executable": False,
        "new_campaign_root": None,
        "count": 14,
        "training_source": CAMPAIGN,
        "diagnostic_implementation_source": None,
        "diagnostic_status": "retention specification only; adapter not implemented or qualified by this task",
        "selection_warning": "outcome-selected replication of known typical/exceptional seeds, not independent confirmatory evidence or unbiased environment sample",
        "gpu_model_proposal": "NVIDIA A100-SXM4-40GB; owner must verify original hardware and approve one common model",
        "matmul_precision": "tensorfloat32; no x64/bfloat16/fp16 changes",
        "resource_estimate": {
            "one_process_per_gpu": True,
            "packing": False,
            "cpu_per_process_proposed": 16,
            "host_ram_GiB_per_process_proposed": 64,
            "max_gpus_proposed": 1,
            "sum_historical_runtime_hours_NOT_ALLOCATION_HOURS": hours,
            "planning_range_gpu_hours_NOT_AUTHORIZED": [hours, 2 * hours],
            "basis": "sum selected W&B runtime values as serial one-GPU planning proxy; factor2 contingency is unmeasured, not allocation accounting",
            "storage_estimate_GiB_NOT_AUTHORIZED": [200, 500],
            "stop_before_launch_until_profiles_quota_storage_reviewed": True,
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
