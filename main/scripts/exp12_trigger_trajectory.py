#!/usr/bin/env python3
"""READ-ONLY plasticity-check trajectory of a development run (why it did / did not trigger).

Lightweight: Python >= 3.10 and numpy only (no JAX, pandas, matplotlib, experiments package).
Inputs: the inspector report (scripts/exp12_pilot_inspect.py --out) and the run's saved
probe artifacts (probes/probe_checks.csv, probes/check_XX.npz). Writes, outside the run:
  <out>/trajectory.csv     one row per check
  <out>/summary.json       pattern classification and quality flags
  <out>/trajectory.html    self-contained charts (inline SVG) + table

Firing is evaluated only with the registered rule recorded in the run's own config:
a check fires when its percentile-bootstrap lower bound of IQM(L) exceeds
trigger.null_threshold; f*_run needs trigger.consecutive_checks consecutive firing checks
among the eligible ones. Nothing here changes, loosens or re-tunes that rule.

The "near boundary" label and the pattern classes are DESCRIPTIVE, fixed in this file
before any pilot data were seen, and are not trigger criteria:
  near boundary     : lower bound <= threshold and within 25% of the interval width of it
  NEGATIVE          : >= 75% of eligible IQM(L) < 0 and >= 50% have the whole interval < 0
  POSITIVE_UNDERPOWERED : >= 50% of eligible IQM(L) > 0, none fires, >= 1 near-boundary check
  NEAR_ZERO_UNCERTAIN   : >= 75% of eligible intervals contain the threshold
  UNDETERMINED      : none of the above, or any missing/invalid data
Precedence when several match: NEGATIVE, then POSITIVE_UNDERPOWERED (a positive-leaning
special case of an interval containing the boundary), then NEAR_ZERO_UNCERTAIN. NEGATIVE
cannot co-occur with either other class under these thresholds. All matches are reported.
"""

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np

NEAR = 0.25


def _f(text):
    return float("nan") if text in ("", None) else float(text)


def _iqm(values):
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = v.size
    if n == 0 or not np.isfinite(v).all():
        return float("nan")
    # scipy.stats.trim_mean(proportiontocut=0.25), as experiments/exp12/probe.py: iqm
    cut = int(0.25 * n)
    return float(v[cut:n - cut].mean())


def _tail_drop(curves):
    """Mean relative loss decrease over the final 10% of fit steps (> 0: still improving)."""
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2 or curves.shape[1] < 10:
        return float("nan")
    w = max(1, curves.shape[1] // 10)
    start, end = curves[:, -w - 1], curves[:, -1]
    return float(np.mean((start - end) / np.maximum(np.abs(start), 1e-12)))


def load(report_path):
    report = json.loads(Path(report_path).read_text())
    run = Path(report["exp1_run_dirs"][0])
    rule = report["trigger"]["rule"]
    recomputed = {r["check_index"]: r for r in report["trigger"]["checks"]}
    with open(run / "probes" / "probe_checks.csv", newline="") as stream:
        csv_rows = {int(float(r["check_index"])): r for r in csv.DictReader(stream)}
    n_checks = int(report["identity"].get("num_checks") or max(csv_rows))
    return report, run, rule, recomputed, csv_rows, n_checks


def trajectory(report_path):
    report, run, rule, recomputed, csv_rows, n_checks = load(report_path)
    threshold = float(rule["null_threshold"])
    last_eligible = int(rule["eligible_checks"].split("..")[1])
    total_steps = report["horizon"]["num_interaction_steps"]
    rows, problems = [], []
    for k in range(0, n_checks + 1):
        npz_path = run / "probes" / f"check_{k:02d}.npz"
        row = {"check_index": k}
        if k not in csv_rows:
            problems.append(f"check {k}: missing from probe_checks.csv")
            rows.append(row)
            continue
        c = csv_rows[k]
        row["interaction_step"] = int(float(c["interaction_step"]))
        row["budget_fraction"] = row["interaction_step"] / total_steps
        arrays = {}
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
        else:
            problems.append(f"check {k}: {npz_path.name} missing")
        networks = sorted({n.rsplit("_", 1)[0] for n in arrays if n.endswith("_b")})
        def first(prefix, key):
            names = [n for n in networks if n.startswith(prefix)]
            return [arrays[f"{n}_{key}"] for n in names]
        fresh_b, fresh_s = first("fresh", "b"), first("fresh", "score")
        row["b_iqm"] = _iqm(np.mean(fresh_b, axis=0)) if fresh_b else float("nan")
        row["fresh_P_iqm"] = _f(c.get("score_fresh_iqm"))
        row["fresh_P_over_b"] = (_iqm(np.mean(fresh_s, axis=0)) / row["b_iqm"]
                                 if fresh_s and row["b_iqm"] else float("nan"))
        row["fresh_fit_still_improving"] = _tail_drop(np.concatenate(first("fresh", "losses"))) if fresh_s else float("nan")
        if k == 0:
            rows.append(row)
            continue
        r = recomputed[k]
        cur_s = first("current", "score")
        row.update({
            "loss_iqm": r["loss_iqm"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "ci_width": r["ci_high"] - r["ci_low"],
            "loss_rounds": r["loss_rounds"],
            "rounds_positive": int(sum(x > 0 for x in r["loss_rounds"])),
            "round_sd": float(np.std(r["loss_rounds"], ddof=1)),
            "current_P_iqm": _f(c.get("score_current_iqm")),
            "current_P_over_b": (_iqm(np.mean(cur_s, axis=0)) / row["b_iqm"]
                                 if cur_s and row["b_iqm"] else float("nan")),
            "current_fit_still_improving": _tail_drop(np.concatenate(first("current", "losses"))) if cur_s else float("nan"),
            "eligible": k <= last_eligible,
            "fires_registered_rule": bool(math.isfinite(r["ci_low"]) and r["ci_low"] > threshold),
            "boundary_gap": r["ci_low"] - threshold,  # > 0 fires; < 0 = shortfall
        })
        width = row["ci_width"]
        row["near_boundary"] = bool(row["boundary_gap"] <= 0 and width > 0
                                    and -row["boundary_gap"] <= NEAR * width)
        flags = []
        if not all(math.isfinite(x) for x in r["loss_rounds"]):
            flags.append("nonfinite round")
        if row["round_sd"] > abs(row["loss_iqm"]) and abs(row["loss_iqm"]) > 0:
            flags.append("round spread exceeds |IQM|")
        if math.isfinite(row["fresh_P_over_b"]) and row["fresh_P_over_b"] < 0.9:
            flags.append("fresh P/b < 0.9 (range rule (w), descriptive)")
        for who in ("fresh", "current"):
            drop = row[f"{who}_fit_still_improving"]
            if math.isfinite(drop) and drop > 0.01:
                flags.append(f"{who} fit still improving at budget end ({drop:.1%} over last 10%)")
        row["quality_flags"] = flags
        rows.append(row)
    return report, rows, problems, threshold, last_eligible


def classify(rows, problems, threshold):
    e = [r for r in rows if r.get("eligible") and math.isfinite(r.get("loss_iqm", float("nan")))]
    n = len(e)
    if not n:
        return "UNDETERMINED", {"reason": "no eligible checks"}
    stats = {
        "eligible_checks": n,
        "iqm_negative": sum(r["loss_iqm"] < 0 for r in e),
        "iqm_positive": sum(r["loss_iqm"] > 0 for r in e),
        "interval_entirely_negative": sum(r["ci_high"] < 0 for r in e),
        "interval_contains_threshold": sum(r["ci_low"] <= threshold <= r["ci_high"] for r in e),
        "fires": sum(r["fires_registered_rule"] for r in e),
        "near_boundary": [r["check_index"] for r in e if r["near_boundary"]],
        "max_lower_bound": max(r["ci_low"] for r in e),
        "max_lower_bound_check": max(e, key=lambda r: r["ci_low"])["check_index"],
        "median_iqm": float(np.median([r["loss_iqm"] for r in e])),
        "median_half_width": float(np.median([r["ci_width"] / 2 for r in e])),
        "ols_slope_iqm_per_check": float(np.polyfit([r["check_index"] for r in e], [r["loss_iqm"] for r in e], 1)[0]),
    }
    classes = []
    if stats["iqm_negative"] >= 0.75 * n and stats["interval_entirely_negative"] >= 0.5 * n:
        classes.append("NEGATIVE")
    if stats["iqm_positive"] >= 0.5 * n and stats["fires"] == 0 and stats["near_boundary"]:
        classes.append("POSITIVE_UNDERPOWERED")
    if stats["interval_contains_threshold"] >= 0.75 * n:
        classes.append("NEAR_ZERO_UNCERTAIN")
    if problems or not classes:
        return "UNDETERMINED", {**stats, "matched": classes, "problems": problems}
    return classes[0], {**stats, "matched": classes, "problems": problems}


def _svg_chart(points, width=720, height=300, title="", y_label="", threshold=None, bands=None,
               series=(), eligible_until=None):
    """Minimal inline SVG: x = check index; series = [(label, css var, [(x, y, tooltip)])]."""
    left, right, top, bottom = 76, 16, 28, 40
    xs = [p[0] for s in series for p in s[2]] + ([b[0] for b in bands] if bands else [])
    ys = [p[1] for s in series for p in s[2] if math.isfinite(p[1])]
    if bands:
        ys += [v for b in bands for v in b[1:3] if math.isfinite(v)]
    span = (max(ys) - min(ys)) or 1.0
    off_scale = threshold is not None and not (min(ys) - 3 * span <= threshold <= max(ys) + 3 * span)
    if threshold is not None and not off_scale:
        ys.append(threshold)
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.08 or 1e-6
    lo, hi = lo - pad, hi + pad
    x0, x1 = min(xs), max(xs)
    sx = lambda x: left + (x - x0) / max(x1 - x0, 1) * (width - left - right)
    sy = lambda y: top + (hi - y) / (hi - lo) * (height - top - bottom)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    for i in range(5):
        y = lo + (hi - lo) * i / 4
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" class="grid"/>'
                   f'<text x="{left - 6}" y="{sy(y) + 4:.1f}" class="tick" text-anchor="end">{y:.3g}</text>')
    for x in range(int(x0), int(x1) + 1):
        out.append(f'<text x="{sx(x):.1f}" y="{height - bottom + 16}" class="tick" text-anchor="middle">{x}</text>')
    out.append(f'<text x="{(left + width - right) / 2}" y="{height - 6}" class="axis" text-anchor="middle">check index (k/20 of training)</text>')
    out.append(f'<text x="12" y="{top + (height - top - bottom) / 2}" class="axis" text-anchor="middle" '
               f'transform="rotate(-90 12 {top + (height - top - bottom) / 2})">{html.escape(y_label)}</text>')
    if eligible_until is not None and eligible_until < x1:
        out.append(f'<rect x="{sx(eligible_until + 0.5):.1f}" y="{top}" width="{sx(x1) - sx(eligible_until + 0.5) + 8:.1f}" '
                   f'height="{height - top - bottom}" class="ineligible"><title>check {x1}: not eligible to fork</title></rect>')
    if threshold is not None and off_scale:
        where = "above" if threshold > hi else "below"
        out.append(f'<text x="{width - right}" y="{top - 8}" class="thlabel" text-anchor="end">'
                   f'firing boundary {threshold:g} is far {where} this range (off-scale)</text>')
    elif threshold is not None:
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(threshold):.1f}" y2="{sy(threshold):.1f}" class="threshold"/>'
                   f'<text x="{width - right}" y="{sy(threshold) - 6:.1f}" class="thlabel" text-anchor="end">'
                   f'fires only if lower bound &gt; {threshold:g}</text>')
    for x, low, high, tip in bands or ():
        out.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(low):.1f}" y2="{sy(high):.1f}" class="band">'
                   f'<title>{html.escape(tip)}</title></line>')
    for label, var, pts in series:
        finite = [(x, y) for x, y, _ in pts if math.isfinite(y)]
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(finite))
        out.append(f'<path d="{path}" fill="none" stroke="var({var})" stroke-width="2"/>')
        for x, y, tip in pts:
            if math.isfinite(y):
                out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="var({var})" class="mark">'
                           f'<title>{html.escape(tip)}</title></circle>')
        if len(series) > 1 and finite:
            x, y = finite[-1]
            out.append(f'<text x="{sx(x) - 4:.1f}" y="{sy(y) - 8:.1f}" class="dlabel" text-anchor="end">{html.escape(label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def write_html(path, report, rows, verdict, stats, threshold, last_eligible):
    checks = [r for r in rows if r.get("check_index", 0) >= 1 and "loss_iqm" in r]
    tip = lambda r: (f"check {r['check_index']} (step {r['interaction_step']}): IQM L {r['loss_iqm']:.4g}, "
                     f"95% CI [{r['ci_low']:.4g}, {r['ci_high']:.4g}], gap to boundary {r['boundary_gap']:.4g}"
                     + (" — near boundary" if r["near_boundary"] else ""))
    loss_chart = _svg_chart(None, title="IQM plasticity loss with 95% bootstrap interval", y_label="L = P(fresh) − P(current)",
                            threshold=threshold, bands=[(r["check_index"], r["ci_low"], r["ci_high"], tip(r)) for r in checks],
                            series=[("IQM L", "--series-1", [(r["check_index"], r["loss_iqm"], tip(r)) for r in checks])],
                            eligible_until=last_eligible)
    fit = [r for r in rows if "fresh_P_over_b" in r]
    fit_chart = _svg_chart(None, title="Probe fit relative to target variance", y_label="P / b",
                           series=[("fresh", "--series-1", [(r["check_index"], r["fresh_P_over_b"],
                                                            f"check {r['check_index']}: fresh P/b {r['fresh_P_over_b']:.4g}") for r in fit]),
                                   ("current", "--series-2", [(r["check_index"], r.get("current_P_over_b", float("nan")),
                                                              f"check {r['check_index']}: current P/b {r.get('current_P_over_b', float('nan')):.4g}") for r in fit])])
    head = ["k", "step", "IQM L", "CI low", "CI high", "gap", "fires", "near", "fresh P", "current P", "flags"]
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
            r["check_index"], r["interaction_step"], f"{r['loss_iqm']:.4g}", f"{r['ci_low']:.4g}", f"{r['ci_high']:.4g}",
            f"{r['boundary_gap']:.4g}", "yes" if r["fires_registered_rule"] else "no", "yes" if r["near_boundary"] else "",
            f"{r['fresh_P_iqm']:.4g}", f"{r['current_P_iqm']:.4g}", "; ".join(r["quality_flags"]))) + "</tr>"
        for r in checks)
    ident = report["identity"]
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pilot trigger trajectory</title>
<style>
:root {{ color-scheme: light; --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --grid:#e6e5e0; --series-1:#2a78d6; --series-2:#eb6834; --band:#2a78d6; --boundary:#52514e; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --surface-1:#1a1a19;
  --text-primary:#f0efec; --text-secondary:#c3c2b7; --grid:#33322f; --series-1:#3987e5; --series-2:#d95926;
  --band:#3987e5; --boundary:#c3c2b7; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --surface-1:#1a1a19; --text-primary:#f0efec; --text-secondary:#c3c2b7;
  --grid:#33322f; --series-1:#3987e5; --series-2:#d95926; --band:#3987e5; --boundary:#c3c2b7; }}
body {{ background:var(--surface-1); color:var(--text-primary); font:14px/1.5 system-ui, sans-serif; margin:0; padding:16px; }}
main {{ max-width:760px; margin:auto; }} svg {{ width:100%; height:auto; }}
.grid {{ stroke:var(--grid); }} .tick,.axis,.dlabel {{ fill:var(--text-secondary); font-size:11px; }}
.threshold {{ stroke:var(--boundary); stroke-dasharray:4 3; }} .thlabel {{ fill:var(--text-secondary); font-size:11px; }}
.band {{ stroke:var(--band); stroke-opacity:.45; stroke-width:6; stroke-linecap:round; }}
.ineligible {{ fill:var(--grid); opacity:.6; }} .mark {{ stroke:var(--surface-1); stroke-width:2; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }} td,th {{ border-bottom:1px solid var(--grid); padding:3px 6px; text-align:right; }}
.wrap {{ overflow-x:auto; }} h1 {{ font-size:18px; }} h2 {{ font-size:15px; margin-top:24px; }} p {{ color:var(--text-secondary); }}
</style></head><body><main>
<h1>Why the pilot did not trigger: {html.escape(ident['architecture'])} {html.escape(ident['environment'])}, seed {ident['seed']}</h1>
<p>Inspector verdict {html.escape(report['verdict'])}. Descriptive pattern: <strong>{html.escape(verdict)}</strong>.
Registered rule: a check fires only if its 95% lower bound exceeds {threshold:g}; f*_run needs
{report['trigger']['rule']['consecutive_checks']} consecutive firing checks among checks {report['trigger']['rule']['eligible_checks']}.</p>
<h2>IQM plasticity loss with 95% bootstrap interval</h2>{loss_chart}
<h2>Probe fit relative to target variance (P/b)</h2>{fit_chart}
<h2>Table</h2><div class="wrap"><table><thead><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr></thead><tbody>{body}</tbody></table></div>
<h2>Summary statistics</h2><pre>{html.escape(json.dumps(stats, indent=2, default=str))}</pre>
</main></body></html>"""
    Path(path).write_text(doc)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True, help="pilot_inspection.json from exp12_pilot_inspect.py")
    parser.add_argument("--out", required=True, help="new output directory outside the run")
    args = parser.parse_args(argv)
    report, rows, problems, threshold, last_eligible = trajectory(args.report)
    run_root = Path(report["root"]).resolve()
    out = Path(args.out).resolve()
    if out.is_relative_to(run_root):
        raise SystemExit("output must be outside the pilot root (read-only analysis)")
    out.mkdir(parents=True, exist_ok=False)
    verdict, stats = classify(rows, problems, threshold)
    columns = ["check_index", "interaction_step", "budget_fraction", "eligible", "loss_iqm", "ci_low", "ci_high",
               "ci_width", "boundary_gap", "fires_registered_rule", "near_boundary", "rounds_positive", "round_sd",
               "fresh_P_iqm", "current_P_iqm", "b_iqm", "fresh_P_over_b", "current_P_over_b",
               "fresh_fit_still_improving", "current_fit_still_improving", "quality_flags"]
    with (out / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("; ".join(map(str, v)) if isinstance(v, list) else v) for k, v in row.items()})
    summary = {"inspector_verdict": report["verdict"], "pattern": verdict, "statistics": stats,
               "rule": report["trigger"]["rule"], "missing_or_quality": problems,
               "checks_with_quality_flags": {r["check_index"]: r["quality_flags"] for r in rows if r.get("quality_flags")},
               "note": "Descriptive only; the registered trigger is unchanged and no check is re-evaluated under any other rule."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_html(out / "trajectory.html", report, rows, verdict, stats, threshold, last_eligible)
    print(json.dumps({"pattern": verdict, **{k: stats.get(k) for k in (
        "eligible_checks", "iqm_negative", "iqm_positive", "interval_entirely_negative",
        "interval_contains_threshold", "near_boundary", "max_lower_bound", "max_lower_bound_check",
        "median_iqm", "median_half_width", "ols_slope_iqm_per_check")}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
