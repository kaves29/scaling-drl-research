"""Numerical sanity inspection of Angle 2A/2B/2C smoke-test outputs (values, not exit codes)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

MATCHUPS = ("matchup_1", "matchup_2")


def _stats(values):
    a = np.asarray(values, dtype=np.float64).ravel()
    finite = a[np.isfinite(a)]
    out = {"n": int(a.size), "nan": int(np.isnan(a).sum()), "inf": int(np.isinf(a).sum())}
    if finite.size:
        out.update(
            min=float(finite.min()), max=float(finite.max()), mean=float(finite.mean()),
            std=float(finite.std()), n_unique=int(np.unique(np.round(finite, 8)).size),
            frac_zero=float(np.mean(finite == 0.0)),
        )
    return out


def _suspicious(name, st, flags, expect_variance=True):
    if st["n"] == 0:
        flags.append(f"{name}: empty")
        return
    if st["nan"] or st["inf"]:
        flags.append(f"{name}: {st['nan']} NaN / {st['inf']} inf of {st['n']}")
    if expect_variance and st["n"] > 1 and st.get("std", 0.0) == 0.0:
        flags.append(f"{name}: zero variance (all values = {st.get('min')})")
    if st.get("frac_zero", 0.0) == 1.0:
        flags.append(f"{name}: all exactly zero")


def _npz_stats(path, flags, prefix):
    out = {}
    with np.load(path, allow_pickle=False) as npz:
        for key in npz.files:
            arr = npz[key]
            if arr.dtype.kind in "fiu":
                st = _stats(arr)
                st["shape"] = list(arr.shape)
                out[key] = st
                _suspicious(f"{prefix}:{key}", st, flags, expect_variance=arr.size > 1)
            else:
                out[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return out


def inspect_angle2a(results, env, seed, flags):
    report = {}
    for m in MATCHUPS:
        d = results / "angle_2a" / env / f"seed{seed}" / m
        if not (d / "probes.csv").exists():
            flags.append(f"2A {m}: probes.csv missing")
            continue
        meta = json.loads((d / "run_metadata.json").read_text())
        df = pd.read_csv(d / "probes.csv")
        r = {"metadata": meta, "num_probes": len(df), "by_source": df["source"].value_counts().to_dict()}
        for col in ("q_d", "q_r", "mc_return", "mc_return_se", "mc_return_std", "diagonal_error"):
            st = _stats(df[col])
            r[col] = st
            _suspicious(f"2A {m}.{col}", st, flags)
        if np.allclose(df["q_d"], df["q_r"]):
            flags.append(f"2A {m}: Q_D == Q_R on every probe (same critic?)")
        zero_std = df[df["mc_return_std"] == 0.0]["probe_id"].tolist()
        if zero_std:
            flags.append(f"2A {m}: {len(zero_std)} probes with zero MC rollout spread (deterministic policy?): {zero_std}")
        if df["num_rollouts"].nunique() != 1:
            flags.append(f"2A {m}: unequal rollouts per probe {df['num_rollouts'].tolist()}")
        e_d = df.loc[df["source"] == "D", "diagonal_error"]
        e_r = df.loc[df["source"] == "R", "diagonal_error"]
        r["E_D"] = _stats(e_d)
        r["E_R"] = _stats(e_r)
        r["q_minus_mc_by_source"] = {
            s: _stats(np.where(g["source"] == "D", g["q_d"], g["q_r"]) - g["mc_return"])
            for s, g in df.groupby("source")
        }
        r["probes"] = df.round(4).to_dict(orient="records")
        r["arrays"] = _npz_stats(d / "probes_arrays.npz", flags, f"2A {m} npz")
        report[m] = r

    cache = results / "angle_2a_pool_null" / env / "pool_null_distribution.json"
    if cache.exists():
        pool = json.loads(cache.read_text())
        report["pool_null"] = pool
        vals = [v for p in pool.get("pairs", []) for v in (p.get("e_a"), p.get("e_b"))]
        st = _stats(vals)
        report["pool_null_values"] = st
        _suspicious("2A pool_null e_a/e_b", st, flags)
    else:
        flags.append("2A pool null cache missing")
    return report


def inspect_angle2b(results, env, seed, flags):
    report = {}
    for m in MATCHUPS:
        d = results / "angle_2b" / env / f"seed{seed}" / m
        if not (d / "run_metadata.json").exists():
            flags.append(f"2B {m}: outputs missing")
            continue
        meta = json.loads((d / "run_metadata.json").read_text())
        null = pd.read_csv(d / "null_distribution.csv")
        r = {"metadata": meta, "null_pairs": null.round(6).to_dict(orient="records")}
        for metric in ("d_dir", "d_mag", "d_grad"):
            st = _stats(null[metric])
            r[f"null_{metric}"] = st
            _suspicious(f"2B {m} null.{metric}", st, flags)
            for side in ("primary", "secondary"):
                v = meta.get(side, {}).get(metric)
                if v is None or not np.isfinite(v):
                    flags.append(f"2B {m} {side}.{metric} = {v}")
                elif v == 0.0:
                    flags.append(f"2B {m} {side}.{metric} is exactly 0")
        if meta.get("primary") == meta.get("secondary"):
            flags.append(f"2B {m}: primary == secondary exactly")
        r["arrays"] = _npz_stats(d / "gradients.npz", flags, f"2B {m} npz")
        report[m] = r
    return report


def inspect_angle2c(results, env, seed, flags):
    report = {}
    for m in MATCHUPS:
        d = results / "angle_2c" / env / f"seed{seed}" / m
        if not (d / "run_metadata.json").exists():
            flags.append(f"2C {m}: outputs missing")
            continue
        meta = json.loads((d / "run_metadata.json").read_text())
        null = pd.read_csv(d / "null_distribution.csv")
        r = {"metadata": meta, "null_pairs": null.round(6).to_dict(orient="records")}
        for col in ("direction_for_null", "magnitude_for_null", "raw_offset", "instability_ratio"):
            st = _stats(null[col])
            r[f"null_{col}"] = st
            _suspicious(f"2C {m} null.{col}", st, flags)
        for side in ("primary", "secondary"):
            for k, v in (meta.get(side) or {}).items():
                if isinstance(v, (int, float)) and not np.isfinite(v):
                    flags.append(f"2C {m} {side}.{k} = {v}")
        r["arrays"] = _npz_stats(d / "properties.npz", flags, f"2C {m} npz")
        report[m] = r
    return report


def inspect_all(results, env, seed):
    results = Path(results)
    flags = []
    report = {
        "angle2a": inspect_angle2a(results, env, seed, flags),
        "angle2b": inspect_angle2b(results, env, seed, flags),
        "angle2c": inspect_angle2c(results, env, seed, flags),
    }
    report["flags"] = flags
    return report
