"""Read-only retained-data accounting. Imports no model/training/JAX modules."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from audits.echocritic_forensic.evidence import (
    CAMPAIGN,
    check_derived,
    verify_source_snapshots,
    PILOT,
    finite,
    prepare_output,
    read_records,
    reconstruct_probes,
    source_iqm,
    write_csv,
)

PLAN_PATH = Path(__file__).with_name("analysis_plan.json")
PLAN = json.loads(PLAN_PATH.read_text())
VARIANCE_FLOOR = 1e-12
ARCHITECTURES = ("D2W512", "D4W1024", "D4W1536")
ENVIRONMENTS = ("dog-run", "humanoid-run", "myo-key-turn", "myo-pen-twirl")
METRICS = (
    "train/q1_mean",
    "train/critic_loss",
    "train/critic_gnorm",
    "train/critic_srank",
    "train/critic_DR0.2",
    "eval/avg_return",
)


def ratio(mse, variance):
    if variance is None or mse is None or variance <= VARIANCE_FLOOR:
        return None
    return float(mse / variance)


def component_bounds(fresh, current=None):
    """Structural bounds, not component estimates; ideal score arithmetic."""
    lower_b = max(0.0, fresh, current if current is not None else 0.0)
    if lower_b > 1:
        raise ValueError("scores outside structural sine-variance support")
    out = {"b_lower_bound": lower_b, "b_upper_bound": 1.0}
    for name, score in (("fresh", fresh), ("current", current)):
        if score is not None:
            out[name + "_mse_lower_bound"] = lower_b - score
            out[name + "_mse_upper_bound"] = 1 - score
    return out


def constant_fresh_witness(rows):
    """Feasibility of an algebraic constant-MSE history, not a measured fit."""
    lower = max(
        max(
            0.0,
            -r["fresh_score"],
            (
                (r["current_score"] - r["fresh_score"])
                if r["current_score"] is not None
                else 0.0
            ),
        )
        for r in rows
    )
    upper = min(1 - r["fresh_score"] for r in rows)
    return {
        "feasible": lower <= upper,
        "fresh_mse_lower": lower,
        "fresh_mse_upper": upper,
        "witness_fresh_mse": (lower + upper) / 2 if lower <= upper else None,
    }


def seed_interval(values, seed):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"n_seeds": 0, "mean": None, "ci_low": None, "ci_high": None}
    if len(values) == 1:
        return {"n_seeds": 1, "mean": float(values[0]), "ci_low": None, "ci_high": None}
    rng = np.random.default_rng([PLAN["bootstrap_seed"], seed])
    sample = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    low, high = np.quantile(sample, [0.025, 0.975])
    return {
        "n_seeds": len(values),
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def archived_arrays(package, records):
    origin = json.loads((package / "archived_probe_origin.json").read_text())
    path = package / origin["package_copy"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != origin["sha256"]:
        raise ValueError("original archive checksum mismatch")
    run = next(
        r
        for r in records
        if (r["environment"], r["architecture"], r["seed"])
        == ("myo-pen-twirl", "D2W512", 5)
        and r["cohort"] == "exp96"
    )
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    return run["id"], path, origin, arrays


def rounds_and_points(records, archive_run, archive, source_points):
    raw_rows, points = [], []
    source_by = {(p["id"], p["check"]): p for p in source_points}
    for run in records:
        if run["cohort"] != "exp96":
            continue
        initial = None
        for ordinal, raw in enumerate(run["history"]):
            if not finite(raw.get("probe/check_index")):
                continue
            check = int(raw["probe/check_index"])
            fresh = [raw[f"probe/score_fresh_r{i}"] for i in range(10)]
            current = (
                [raw[f"probe/score_current_r{i}"] for i in range(10)] if check else None
            )
            base = {k: run[k] for k in ("id", "environment", "architecture", "seed")}
            if check == 0:
                initial = fresh
            if initial is None:
                raise ValueError("missing initial fresh anchor")
            for round_index in range(10):
                f = float(fresh[round_index])
                c = float(current[round_index]) if current is not None else None
                row = {
                    **base,
                    "check": check,
                    "round": round_index,
                    "interaction_step": int(raw["probe/interaction_step"]),
                    "history_row_index": ordinal,
                    "history_logging_step": raw["_step"],
                    "fresh_score": f,
                    "current_score": c,
                    "source_paired_deficit": (
                        float(raw[f"probe/loss_r{round_index}"]) if check else None
                    ),
                    "b": None,
                    "fresh_final_pool_mse": None,
                    "current_final_pool_mse": None,
                    "fresh_mse_over_b": None,
                    "current_mse_over_b": None,
                    "fresh_offset": None,
                    "current_offset": None,
                    "ratio_status": "variance_unobserved",
                    "component_evidence": "scores_only; b/MSE not logged",
                }
                row.update(component_bounds(f, c))
                if run["id"] == archive_run and check == 50:
                    j = round_index
                    row["b"] = float(archive["fresh_b"][j])
                    row["ratio_status"] = (
                        "defined"
                        if row["b"] > VARIANCE_FLOOR
                        else "undefined_b_at_or_below_floor"
                    )
                    if archive["fresh_b"][j] != archive["current_b"][j]:
                        raise ValueError("paired variance mismatch")
                    for kind in ("fresh", "current"):
                        b = archive[kind + "_b"][j]
                        mse = archive[kind + "_final_loss"][j]
                        score = archive[kind + "_score"][j]
                        if (
                            np.float32(b - mse) != score
                            or float(score) != row[kind + "_score"]
                        ):
                            raise ValueError("stored-array/score binding mismatch")
                        row[kind + "_final_pool_mse"] = float(mse)
                        row[kind + "_mse_over_b"] = ratio(float(mse), float(b))
                        row[kind + "_offset"] = float(archive[kind + "_offset"][j])
                    row["component_evidence"] = (
                        "original archived NPZ and exact W&B score binding"
                    )
                    row["algebraic_mse_difference"] = (
                        row["current_final_pool_mse"] - row["fresh_final_pool_mse"]
                    )
                    row["paired_fp32_minus_mse_difference"] = (
                        row["source_paired_deficit"] - row["algebraic_mse_difference"]
                    )
                raw_rows.append(row)
            point = {
                **base,
                "check": check,
                "interaction_step": int(raw["probe/interaction_step"]),
                "fresh_iqm": source_iqm(fresh, 10),
                "fresh_round_mean": float(np.mean(fresh)),
                "initial_fresh_iqm": source_iqm(initial, 10),
                "initial_fresh_round_mean": float(np.mean(initial)),
                "fresh_delta_iqm": source_iqm(fresh, 10) - source_iqm(initial, 10),
                "current_iqm": source_iqm(current, 10) if current is not None else None,
                "current_round_mean": (
                    float(np.mean(current)) if current is not None else None
                ),
                "paired_iqm": source_by[run["id"], check].get("paired_deficit"),
                "paired_round_mean": (
                    float(
                        np.mean(
                            np.asarray(fresh, np.float32)
                            - np.asarray(current, np.float32)
                        )
                    )
                    if check
                    else None
                ),
                "anchored_gap": (
                    source_iqm(initial, 10) - source_iqm(current, 10) if check else None
                ),
                "b_iqm": None,
                "fresh_final_pool_mse_iqm": None,
                "current_final_pool_mse_iqm": None,
                "fresh_mse_over_b_iqm": None,
                "current_mse_over_b_iqm": None,
            }
            # Exact additive score operand mean uses float64 differences, not
            # separately rounded FP32 paired values; source paired IQM retained.
            if check:
                point["paired_operand_mean"] = (
                    point["fresh_round_mean"] - point["current_round_mean"]
                )
            if run["id"] == archive_run and check == 50:
                point["b_iqm"] = source_iqm(archive["fresh_b"], 10)
                for kind in ("fresh", "current"):
                    point[kind + "_final_pool_mse_iqm"] = source_iqm(
                        archive[kind + "_final_loss"], 10
                    )
                    ratios = [
                        ratio(float(m), float(b))
                        for m, b in zip(
                            archive[kind + "_final_loss"], archive[kind + "_b"]
                        )
                    ]
                    point[kind + "_mse_over_b_iqm"] = (
                        source_iqm(ratios, 10)
                        if all(v is not None for v in ratios)
                        else None
                    )
            these_rounds = raw_rows[-10:]
            for kind in ("fresh", "current"):
                for bound in ("lower", "upper"):
                    point[kind + "_mse_" + bound + "_iqm_bound"] = (
                        source_iqm(
                            [
                                r[kind + "_mse_" + bound + "_bound"]
                                for r in these_rounds
                            ],
                            10,
                        )
                        if kind == "fresh" or check
                        else None
                    )
            points.append(point)
    return raw_rows, points


def trends(points):
    grouped = defaultdict(list)
    for p in points:
        if p["check"] <= 35:
            grouped[p["id"]].append(p)
    rows = []
    for rid, values in sorted(grouped.items()):
        values.sort(key=lambda p: p["check"])
        f = np.array([p["fresh_iqm"] for p in values])
        t = np.array([p["interaction_step"] for p in values]) / 500000
        delta = np.diff(f)
        first, last = values[1], values[-1]
        rows.append(
            {
                **{k: last[k] for k in ("id", "environment", "architecture", "seed")},
                "fresh_delta_iqm_0_to_35": float(f[-1] - f[0]),
                "fresh_slope_per_budget": float(np.polyfit(t, f, 1)[0]),
                "adjacent_increases": int(np.sum(delta > 0)),
                "adjacent_decreases": int(np.sum(delta < 0)),
                "monotone_nonincreasing": bool(np.all(delta <= 0)),
                "fraction_postchecks_below_initial": float(np.mean(f[1:] < f[0])),
                "fresh_round_mean_change_1_to_35": last["fresh_round_mean"]
                - first["fresh_round_mean"],
                "trained_round_mean_change_1_to_35": last["current_round_mean"]
                - first["current_round_mean"],
                "paired_operand_mean_change_1_to_35": last["paired_operand_mean"]
                - first["paired_operand_mean"],
                "paired_iqm_35": last["paired_iqm"],
                "anchored_gap_35": last["anchored_gap"],
                "fresh_component_cause": "unidentified: missing b time series",
            }
        )
    return rows


def grouped_intervals(points):
    fields = (
        "fresh_iqm",
        "current_iqm",
        "paired_iqm",
        "anchored_gap",
        "fresh_delta_iqm",
        "b_iqm",
        "fresh_final_pool_mse_iqm",
        "current_final_pool_mse_iqm",
        "fresh_mse_over_b_iqm",
        "current_mse_over_b_iqm",
        "fresh_mse_lower_iqm_bound",
        "fresh_mse_upper_iqm_bound",
        "current_mse_lower_iqm_bound",
        "current_mse_upper_iqm_bound",
    )
    groups = defaultdict(list)
    for p in points:
        groups[p["environment"], p["architecture"], p["check"]].append(p)
    rows = []
    for group_id, ((env, arch, check), values) in enumerate(sorted(groups.items())):
        for field in fields:
            data = [
                p[field]
                for p in sorted(values, key=lambda p: p["seed"])
                if p[field] is not None
            ]
            rows.append(
                {
                    "environment": env,
                    "architecture": arch,
                    "check": check,
                    "interaction_step": values[0]["interaction_step"],
                    "metric": field,
                    **seed_interval(data, group_id),
                }
            )
    return rows


def domain_contrasts(points, common_max):
    lookup = {
        (p["environment"], p["architecture"], p["seed"], p["check"]): p for p in points
    }
    seed_rows, intervals = [], []
    for arch_id, arch in enumerate(ARCHITECTURES):
        for check in range(1, common_max + 1):
            for field in (
                "fresh_iqm",
                "current_iqm",
                "paired_iqm",
                "anchored_gap",
                "fresh_delta_iqm",
            ):
                differences = []
                for seed in range(1, 9):
                    dmc = np.mean(
                        [lookup[e, arch, seed, check][field] for e in ENVIRONMENTS[:2]]
                    )
                    myo = np.mean(
                        [lookup[e, arch, seed, check][field] for e in ENVIRONMENTS[2:]]
                    )
                    row = {
                        "architecture": arch,
                        "seed": seed,
                        "check": check,
                        "interaction_step": check * 10000,
                        "metric": field,
                        "dmc_equal_task_mean": float(dmc),
                        "myo_equal_task_mean": float(myo),
                        "myo_minus_dmc": float(myo - dmc),
                    }
                    seed_rows.append(row)
                    differences.append(myo - dmc)
                intervals.append(
                    {
                        "architecture": arch,
                        "check": check,
                        "metric": field,
                        "interaction_step": check * 10000,
                        **seed_interval(differences, 100000 + arch_id * 1000 + check),
                    }
                )
    return seed_rows, intervals


def witnesses(round_rows):
    grouped = defaultdict(list)
    for row in round_rows:
        if row["check"] <= 35:
            grouped[row["id"], row["round"]].append(row)
    summaries, detail = [], []
    for (rid, rep), values in sorted(grouped.items()):
        info = constant_fresh_witness(values)
        summaries.append({"id": rid, "round": rep, **info})
        if not info["feasible"]:
            continue
        mf = info["witness_fresh_mse"]
        for row in values:
            pf, pc = row["fresh_score"], row["current_score"]
            b = pf + mf
            error_mf = 0.75 - pf
            if b < 0 or b > 1 or error_mf < 0:
                raise ValueError("invalid witness")
            detail.append(
                {
                    "id": rid,
                    "round": rep,
                    "check": row["check"],
                    "evidence_type": "ALGEBRAIC WITNESS; not observed components/teacher history",
                    "constant_fresh_mse": mf,
                    "variance_driven_b": b,
                    "variance_driven_current_mse": b - pc if pc is not None else None,
                    "constant_variance_b": 0.75,
                    "error_driven_fresh_mse": error_mf,
                    "error_driven_current_mse": 0.75 - pc if pc is not None else None,
                }
            )
    return summaries, detail


def precision_and_curves(archive):
    precision, curve_points, curve_stats = [], [], []
    for kind in ("fresh", "current"):
        for rep in range(10):
            a = float(archive[kind + "_offset"][rep])
            b = float(archive[kind + "_b"][rep])
            m = float(archive[kind + "_final_loss"][rep])
            losses = archive[kind + "_losses"][rep]
            delta = (
                max(
                    abs(float(np.spacing(np.float32(a - 1)))),
                    abs(float(np.spacing(np.float32(a + 1)))),
                )
                / 2
            )
            # Conservative positive-sum roundoff envelope for25600 float32
            # residual squares; a fixed-prediction bound, not trajectory bound.
            u = np.finfo(np.float32).eps / 2
            gamma = 25600 * u / (1 - 25600 * u)
            residual_mse_upper = m / ((1 - gamma) * (1 - u) ** 3)
            initial_mb = float(losses[0])
            gamma_batch = 256 * u / (1 - 256 * u)
            initial_residual_rms_lower = math.sqrt(
                initial_mb / ((1 + gamma_batch) * (1 + u) ** 6)
            )
            centered_mb_abs_lower = max(0.0, initial_residual_rms_lower - 1 - delta)
            precision.append(
                {
                    "kind": kind,
                    "round": rep,
                    "actual_offset": a,
                    "initial_minibatch_mse": initial_mb,
                    "max_abs_initial_minibatch_Q_minus_stored_offset_lower_bound": centered_mb_abs_lower,
                    "centered_bound_scope": "minibatch forward only, not full-pool variance; sine range and normal float32 arithmetic",
                    "base_variance": b,
                    "final_pool_mse": m,
                    "max_offset_plus_base_rounding_abs": delta,
                    "target_variance_perturbation_bound": 2 * delta * math.sqrt(b)
                    + delta * delta,
                    "fixed_prediction_mse_perturbation_bound": 2
                    * delta
                    * math.sqrt(residual_mse_upper)
                    + delta * delta,
                    "does_not_bound": "teacher sine phase error or accumulated optimization-path change",
                }
            )
            for step in (0, 100, 500, 1000):
                curve_points.append(
                    {
                        "kind": kind,
                        "round": rep,
                        "fit_updates_completed": step,
                        "evaluation_domain": "changing minibatch; pre-update",
                        "mse": float(losses[step]),
                        "initial_or_intermediate_pool_mse": None,
                    }
                )
            curve_points.append(
                {
                    "kind": kind,
                    "round": rep,
                    "fit_updates_completed": 2000,
                    "evaluation_domain": "full pool; post-fit",
                    "mse": m,
                    "initial_or_intermediate_pool_mse": m,
                }
            )
            curve_stats.append(
                {
                    "kind": kind,
                    "round": rep,
                    "first_minibatch_mse": float(losses[0]),
                    "first100_minibatch_median": float(np.median(losses[:100])),
                    "last100_minibatch_median": float(np.median(losses[-100:])),
                    "max_minibatch_mse": float(np.max(losses)),
                    "max_over_first_minibatch": float(np.max(losses) / losses[0]),
                    "final_full_pool_mse": m,
                    "initial_full_pool_mse": None,
                    "initial_to_final_pool_improvement": None,
                }
            )
    return precision, curve_points, curve_stats


def associations(records, points):
    by_run = defaultdict(list)
    for p in points:
        if 0 < p["check"] <= 35:
            by_run[p["id"]].append(p)
    rows = []
    for run in records:
        if run["cohort"] != "exp96":
            continue
        for field in METRICS:
            history = sorted(
                [
                    (h["_step"] / run["config"]["action_repeat"], float(h[field]))
                    for h in run["history"]
                    if finite(h.get(field))
                ]
            )
            paired = []
            for p in by_run[run["id"]]:
                matches = [
                    (t, v)
                    for t, v in history
                    if t <= p["interaction_step"]
                    and p["interaction_step"] - t
                    <= (0 if field.startswith("eval/") else 2000)
                ]
                if matches:
                    paired.append(
                        (p["interaction_step"], p["fresh_iqm"], matches[-1][1])
                    )
            if len(paired) < 5:
                continue
            data = np.asarray(paired)
            t = data[:, 0] / 500000
            x, y = data[:, 1], data[:, 2]
            design = np.column_stack([np.ones(len(t)), t, t * t])
            residual = [
                v - design @ np.linalg.lstsq(design, v, rcond=None)[0] for v in (x, y)
            ]

            def rho(a, b):
                return (
                    float(spearmanr(a, b).statistic)
                    if np.ptp(a) > 0 and np.ptp(b) > 0
                    else None
                )

            rows.append(
                {
                    **{
                        k: run[k] for k in ("id", "environment", "architecture", "seed")
                    },
                    "paired_with": field,
                    "n_repeated_points_not_seeds": len(paired),
                    "fresh_score_spearman": rho(x, y),
                    "quadratic_time_residual_spearman": (
                        rho(*residual) if np.ptp(x) > 0 and np.ptp(y) > 0 else None
                    ),
                    "association_not_causal": True,
                    "fresh_mse_or_variance_association": None,
                }
            )
    return rows


def source_predicted_sampling(points):
    rows = []
    for p in points:
        n = p["interaction_step"]
        expected_unique = -n * math.expm1(25600 * math.log1p(-1 / n))
        rows.append(
            {
                "id": p["id"],
                "check": p["check"],
                "interaction_step": n,
                "evidence_type": "source-predicted empty-start n_step1 single-env occupancy, NOT saved replay measurement",
                "conditional_occupancy": n,
                "uniform_draws": 25600,
                "expected_unique_indices": expected_unique,
                "minimum_duplicate_index_fraction": max(0, 25600 - n) / 25600,
                "expected_duplicate_index_fraction": 1 - expected_unique / 25600,
                "physical_state_coverage": None,
                "normalizer_statistics": None,
            }
        )
    return rows


def artifact_inventory(package, records, search_roots):
    files = []
    for root in search_roots:
        root = Path(root)
        for p in root.rglob("*"):
            if p.is_file() and (
                p.suffix.lower() in (".npz", ".pkl", ".ckpt", ".zip", ".gz")
                or p.name in ("_CHECKPOINT_METADATA", "_METADATA", "LATEST")
            ):
                original = p.is_relative_to(package / "raw_archived")
                files.append(
                    {
                        "path": str(p),
                        "bytes": p.stat().st_size,
                        "classification": (
                            "original supplied campaign probe"
                            if original
                            else (
                                "uploaded evidence container"
                                if "attachments" in str(p)
                                else "generated/synthetic or unbound workspace artifact; not campaign evidence"
                            )
                        ),
                    }
                )
    file_names = Counter()
    field_names = Counter()
    unresolved = []
    for run in records:
        directory = package / "raw_api" / run["id"]
        listed = json.loads((directory / "files.json").read_text())
        file_names.update(f["name"] for f in listed)
        for item in listed:
            if item["name"].startswith("artifact/"):
                unresolved.append(
                    {
                        "id": run["id"],
                        "cohort": run["cohort"],
                        "listed_reference": item["name"],
                        "listed_bytes": item["size"],
                        "listed_md5": item["md5"],
                        "body_available": (
                            directory / "files" / item["name"]
                        ).is_file(),
                        "interpretation": "unresolved listed placeholder; not proof of available checkpoint contents",
                    }
                )
        field_names.update({key for h in run["history"] for key in h})
    return files, {
        "wandb_file_inventory": dict(file_names),
        "unresolved_manifest_references": unresolved,
        "probe_history_fields": {
            k: v for k, v in field_names.items() if k.startswith("probe/")
        },
        "normalizer_input_teacher_state_fields": [
            k
            for k in field_names
            if any(
                word in k.lower()
                for word in (
                    "normalizer",
                    "obs_rms",
                    "teacher_output",
                    "base_target",
                    "pool_observation",
                )
            )
        ],
    }


def archive_anchor_bounds(round_rows, archive_run):
    initial = {
        r["round"]: r for r in round_rows if r["id"] == archive_run and r["check"] == 0
    }
    result = []
    for row in round_rows:
        if row["id"] != archive_run or row["check"] != 50:
            continue
        p0 = initial[row["round"]]["fresh_score"]
        b, mf = row["b"], row["fresh_final_pool_mse"]
        result.append(
            {
                "id": archive_run,
                "round": row["round"],
                "initial_fresh_score": p0,
                "final_fresh_score": row["fresh_score"],
                "observed_final_b": b,
                "observed_final_fresh_mse": mf,
                "b_change_lower": b - 1,
                "b_change_upper": b - p0,
                "variance_decrease_required_by_nonnegative_initial_mse": b < p0,
                "fresh_mse_change_lower": mf - (1 - p0),
                "fresh_mse_change_upper": mf,
                "constant_fresh_mse_to_endpoint_feasible": p0 + mf <= 1,
                "unobserved_initial_b_under_constant_mse_witness": p0 + mf,
            }
        )
    return result


def candidate_accounting(trend_rows, associations_rows, precision, archive_run):
    association_map = defaultdict(list)
    for row in associations_rows:
        association_map[row["id"]].append(row)
    rows = []
    for parent in trend_rows:
        base = {k: parent[k] for k in ("id", "environment", "architecture", "seed")}
        measured = (
            (
                "fresh_score_drift",
                parent["fresh_delta_iqm_0_to_35"],
                "score change, not identified fresh-MSE change",
            ),
            (
                "contemporary_paired_gap",
                parent["paired_iqm_35"],
                "relative final MSE difference under source pairing; variance cancels",
            ),
            (
                "initial_reference_gap",
                parent["anchored_gap_35"],
                "descriptive cross-task anchor",
            ),
        )
        for factor, value, status in measured:
            rows.append(
                {
                    **base,
                    "candidate_factor": factor,
                    "direct_evidence": "recorded round scores",
                    "observed_value_or_association": value,
                    "causal_contribution": None,
                    "causal_status": status,
                    "missing_artifact": None,
                }
            )
        for factor, missing in (
            ("target_variance_drift", "per-round b at successive checks"),
            (
                "fresh_final_MSE_drift",
                "per-round final pool MSE or b at successive checks",
            ),
            (
                "replay_and_normalization",
                "raw/normalized replay pools and obs_rms snapshots",
            ),
            (
                "teacher_sine_precision",
                "teacher outputs/base targets and actual input/parameter arrays",
            ),
            ("initial_centered_Q", "initial pool predictions and base targets"),
            (
                "full_pool_catchup",
                "initial/intermediate full-pool errors; DMC curves absent",
            ),
        ):
            rows.append(
                {
                    **base,
                    "candidate_factor": factor,
                    "direct_evidence": "unavailable",
                    "observed_value_or_association": None,
                    "causal_contribution": None,
                    "causal_status": "unidentified; no missing values imputed",
                    "missing_artifact": missing,
                }
            )
        unique_first = -5000 * math.expm1(25600 * math.log1p(-1 / 5000))
        unique_late = -350000 * math.expm1(25600 * math.log1p(-1 / 350000))
        rows.append(
            {
                **base,
                "candidate_factor": "source_predicted_distinct_index_increase",
                "direct_evidence": "conditional source/occupancy expectation, NOT observed sample counts",
                "observed_value_or_association": unique_late - unique_first,
                "causal_contribution": None,
                "causal_status": "same source support schedule in every condition; performance effect unidentified",
                "missing_artifact": "actual pool index arrays and input/target difficulty components",
            }
        )
        for association in association_map[parent["id"]]:
            rows.append(
                {
                    **base,
                    "candidate_factor": "fresh_score_association_with_"
                    + association["paired_with"],
                    "direct_evidence": "observed common-window time series",
                    "observed_value_or_association": association[
                        "quadratic_time_residual_spearman"
                    ],
                    "causal_contribution": None,
                    "causal_status": "quadratic-stage-adjusted repeated-measure association, not mechanism",
                    "missing_artifact": "b/MSE and original input/prediction decomposition",
                }
            )
        if parent["id"] == archive_run:
            bound = max(
                p["fixed_prediction_mse_perturbation_bound"]
                for p in precision
                if p["kind"] == "current"
            )
            rows.append(
                {
                    **base,
                    "candidate_factor": "actual_offset_target_rounding_at_check50",
                    "direct_evidence": "original offset and final pool MSE arrays",
                    "observed_value_or_association": bound,
                    "causal_contribution": None,
                    "causal_status": "fixed-prediction direct-error upper bound only; optimization-path effect unbounded",
                    "missing_artifact": "teacher, targets, final predictions and GPU computation trace",
                }
            )
    return rows


def trend_groups(trend_rows):
    grouped = defaultdict(list)
    for row in trend_rows:
        grouped[row["environment"], row["architecture"]].append(row)
    fields = (
        "fresh_delta_iqm_0_to_35",
        "fresh_slope_per_budget",
        "fraction_postchecks_below_initial",
        "fresh_round_mean_change_1_to_35",
        "trained_round_mean_change_1_to_35",
        "paired_operand_mean_change_1_to_35",
    )
    rows = []
    for index, ((env, arch), values) in enumerate(sorted(grouped.items())):
        for field in fields:
            rows.append(
                {
                    "environment": env,
                    "architecture": arch,
                    "metric": field,
                    **seed_interval([r[field] for r in values], 800000 + index),
                }
            )
    return rows


def run(package, metric_package, output, search_roots):
    package, metric_package = Path(package).resolve(), Path(metric_package).resolve()
    output = prepare_output(output, package, metric_package)
    records = read_records(package)
    for record in records:
        if record["cohort"] != "exp96":
            continue
        cfg = record["config"]
        if not (
            cfg["num_interaction_steps"] == 500000
            and cfg["num_train_envs"] == 1
            and cfg["n_step"] == 1
            and cfg["buffer"]["max_length"] >= 500000
            and cfg["probe"]["pool_size"] == 25600
            and cfg["probe"]["checks"] == 50
            and cfg["probe"]["rounds"] == 10
        ):
            raise ValueError(
                "conditional occupancy/schedule assumptions do not match config"
            )
    source_points, source_checks = reconstruct_probes(records)
    derived_checks = check_derived(package, metric_package, source_points, records)
    source_checksums = verify_source_snapshots(
        package, Path(__file__).resolve().parents[3]
    )
    if any(not row["passed"] for row in derived_checks + source_checksums):
        raise ValueError("supplied analysis or frozen source mismatch")
    archive_run, archive_path, origin, archive = archived_arrays(package, records)
    round_rows, points = rounds_and_points(records, archive_run, archive, source_points)
    counts = Counter(p["check"] for p in points)
    common_max = max(k for k, n in counts.items() if n == 96)
    trend_rows = trends(points)
    groups = grouped_intervals(points)
    domain_seeds, domain_groups = domain_contrasts(points, common_max)
    witness_summary, witness_rows = witnesses(round_rows)
    precision, curve_points, curve_stats = precision_and_curves(archive)
    association_rows = associations(records, points)
    inventory, availability = artifact_inventory(package, records, search_roots)
    datasets = {
        "round_components": round_rows,
        "probe_points": points,
        "seed_trends": trend_rows,
        "condition_intervals": groups,
        "domain_seed_contrasts": domain_seeds,
        "domain_intervals": domain_groups,
        "identification_witness_summary": witness_summary,
        "ALGEBRAIC_WITNESSES_NOT_OBSERVED": witness_rows,
        "archived_precision_bounds": precision,
        "archived_curve_checkpoints": curve_points,
        "archived_curve_summaries": curve_stats,
        "archived_endpoint_anchor_bounds": archive_anchor_bounds(
            round_rows, archive_run
        ),
        "stage_adjusted_associations": association_rows,
        "source_predicted_sampling_NOT_OBSERVED": source_predicted_sampling(points),
        "workspace_artifact_inventory": inventory,
        "unresolved_wandb_manifest_references": availability[
            "unresolved_manifest_references"
        ],
        "candidate_factor_accounting": candidate_accounting(
            trend_rows, association_rows, precision, archive_run
        ),
        "condition_trend_intervals": trend_groups(trend_rows),
    }
    for name, rows in datasets.items():
        write_csv(output / (name + ".csv"), rows)
    receipt = {
        "plan_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "input_source": CAMPAIGN,
        "pilot_source": PILOT,
        "run_records": len(records),
        "raw_manifest_files_verified": len(
            json.loads((package / "raw_api_manifest.json").read_text())
        ),
        "campaign_parent_runs": 96,
        "pilot_records_excluded": 7,
        "probe_points": len(points),
        "round_records": len(round_rows),
        "components_observed_rounds": sum(r["b"] is not None for r in round_rows),
        "components_missing_rounds": sum(r["b"] is None for r in round_rows),
        "max_all96_common_check": common_max,
        "supplied_analysis_comparisons": len(derived_checks),
        "supplied_source_blobs_verified": len(source_checksums),
        "source_iqm_checks": len(source_checks),
        "source_iqm_failures": [r for r in source_checks if not r["passed"]],
        "fresh_declines_common35": sum(
            r["fresh_delta_iqm_0_to_35"] < 0 for r in trend_rows
        ),
        "monotone_fresh_declines_common35": sum(
            r["monotone_nonincreasing"] for r in trend_rows
        ),
        "negative_fresh_slopes_common35": sum(
            r["fresh_slope_per_budget"] < 0 for r in trend_rows
        ),
        "constant_fresh_mse_algebraic_witnesses_feasible": sum(
            r["feasible"] for r in witness_summary
        ),
        "witness_trajectories_checked": len(witness_summary),
        "original_archive": {
            "run_id": archive_run,
            "path": str(archive_path),
            "sha256": origin["sha256"],
            "arrays": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in archive.items()
            },
        },
        "availability": availability,
        "rows_written": {k: len(v) for k, v in datasets.items()},
        "causal_status": "b/MSE drift not identified from scores alone; one original endpoint has components; no original DMC curves/parameters/inputs available",
        "new_fits": 0,
        "rl_training": 0,
        "gpu_jobs": 0,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if receipt["source_iqm_failures"]:
        raise ValueError("original IQM mismatch")
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in ("original_archive", "availability", "source_iqm_failures")
            },
            indent=2,
        )
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--metric-package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--search-root", action="append", default=[])
    args = parser.parse_args()
    run(args.package, args.metric_package, args.out, args.search_root)


if __name__ == "__main__":
    main()
