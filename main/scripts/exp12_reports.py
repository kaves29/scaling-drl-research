#!/usr/bin/env python3
"""Summaries for the unattended Exp 1/2 GPU jobs (docs/exp12_cuda_commands.md). Never imports jax.

    python scripts/exp12_reports.py blockA <out_dir>                 # writes <out_dir>/summary.txt
    python scripts/exp12_reports.py range-check <range_dir> ARCH...  # exit 0 iff P/b >= 0.9 at the configured pool
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

# Amendment (w): the fresh critic must fit the probe target at the configured pool.
RANGE_MIN_P_OVER_B = 0.9
OLD_RANGE_RULE = (0.1, 0.9)  # superseded by (w); reported as information only

_TEST = re.compile(r"^(\w+) \(([\w.]+)\)")
_SUBTEST = re.compile(r"^\s+(\w+) \(([\w.]+)\) \((.*)\) \.\.\. (FAIL|ERROR)\s*$")
_INLINE = re.compile(r" \.\.\. (ok|FAIL|ERROR|skipped.*|expected failure|unexpected success)\s*$")
_BARE = re.compile(r"^(ok|FAIL|ERROR|skipped .*|expected failure|unexpected success)\s*$")
_HEADER = re.compile(r"^(FAIL|ERROR): (\w+) \(([\w.]+)\)(?: \((.*)\))?\s*$")
_RAN = re.compile(r"^Ran (\d+) tests? in")
_FINAL = re.compile(r"^(OK|FAILED)(?: \((.*)\))?\s*$")


def _test_id(name, paren):
    return paren if paren.endswith("." + name) else f"{paren}.{name}"  # 3.11+ vs older unittest


def _status(token):
    if token == "ok":
        return "PASS"
    if token.startswith("skipped"):
        return "SKIP"
    return token.upper()


def parse_unittest_log(text):
    """Every test of a `unittest -v` log with its status, robust to output interleaved with the status.

    A status written after other output (warnings, prints) lands on a later line and is still attributed
    to the pending test. Failing subtests are listed one by one from the FAIL/ERROR headers, which
    unittest always prints. A test with no status anywhere counts as passed only when the totals
    (Ran N, failures, errors, skipped) leave no other possibility; otherwise it is UNKNOWN and a
    warning says so."""
    tests, order, pending = {}, [], None
    failures, ran, final, counts = {}, None, None, {}
    for line in text.splitlines():
        m = _SUBTEST.match(line)
        if m:
            continue  # subtests are taken from the FAIL/ERROR headers below
        m = _HEADER.match(line)
        if m:
            failures.setdefault(_test_id(m.group(2), m.group(3)), []).append((m.group(1), m.group(4)))
            pending = None
            continue
        m = _RAN.match(line)
        if m:
            ran, pending = int(m.group(1)), None
            continue
        m = _FINAL.match(line)
        if m and ran is not None:
            final = line.strip()
            for part in (m.group(2) or "").split(","):
                if "=" in part:
                    k, v = part.strip().split("=")
                    counts[k] = int(v)
            continue
        m = _TEST.match(line)
        if m and ran is None:
            pending = _test_id(m.group(1), m.group(2))
            if pending not in tests:
                order.append(pending)
                tests[pending] = None
            m = _INLINE.search(line)
            if m:
                tests[pending], pending = _status(m.group(1)), None
            continue
        if pending is not None:
            m = _INLINE.search(line) or _BARE.match(line.strip())
            if m:
                tests[pending], pending = _status(m.group(1)), None
    for tid, entries in failures.items():
        if tid not in tests:
            order.append(tid)
        tests[tid] = "ERROR" if any(kind == "ERROR" for kind, _ in entries) else "FAIL"
    warnings = []
    n_fail = sum(len(v) for v in failures.values())
    expected_fail = counts.get("failures", 0) + counts.get("errors", 0)
    if ran is not None and ran != len(order):
        warnings.append(f"the log lists {len(order)} tests but unittest ran {ran}")
    if final is not None and n_fail != expected_fail:
        warnings.append(f"{n_fail} FAIL/ERROR entries found but unittest reports {expected_fail}")
    unknown = [t for t in order if tests[t] is None]
    skips_left = counts.get("skipped", 0) - sum(1 for t in order if tests[t] == "SKIP")
    if unknown:
        if skips_left == 0 and not warnings and final is not None:
            for t in unknown:
                tests[t] = "PASS"  # no failure, error or skip left for it in the totals
        else:
            for t in unknown:
                tests[t] = "UNKNOWN"
            warnings.append(f"{len(unknown)} tests have no status in the log")
    if final is None:
        warnings.append("no final unittest line: the run did not finish")
    return {"tests": [(t, tests[t]) for t in order], "failures": failures, "ran": ran, "final": final,
            "warnings": warnings}


def format_unittest(parsed):
    lines = []
    for tid, status in parsed["tests"]:
        lines.append(f"{status:<8} {tid}")
        for kind, params in parsed["failures"].get(tid, []):
            if params:
                lines.append(f"  {kind:<6} subtest ({params})")
    n = {s: sum(1 for _, x in parsed["tests"] if x == s) for s in ("PASS", "FAIL", "ERROR", "SKIP", "UNKNOWN")}
    n_sub = sum(1 for v in parsed["failures"].values() for _, p in v if p)
    lines.append(f"tests: {len(parsed['tests'])} listed ({', '.join(f'{k} {v}' for k, v in n.items() if v)}); "
                 f"failing subtests: {n_sub}; unittest: Ran {parsed['ran']} | {parsed['final']}")
    lines.extend(f"WARNING: {w}" for w in parsed["warnings"])
    return lines


def _iqm(values):
    v = np.sort(np.asarray(values, np.float64))
    k = int(np.floor(0.25 * len(v)))
    return float(v[k:len(v) - k].mean())  # = scipy trim_mean(0.25), as experiments.exp12.probe.iqm


def range_rows(range_dir):
    rows = []
    for f in sorted(Path(range_dir).glob("range_D*.json")):
        rows.extend(json.loads(f.read_text()))
    return rows


def range_criterion(range_dir, archs):
    """Amendment (w): PASS iff P/b >= 0.9 at the configured pool for every arch. The superseded
    10-90% rule is evaluated for information."""
    configured = {r["arch"]: r for r in range_rows(range_dir) if r["is_configured_pool"]}
    per_size = {a: (a in configured and configured[a]["score_over_b"] >= RANGE_MIN_P_OVER_B) for a in archs}
    old = {a: (a in configured and OLD_RANGE_RULE[0] <= configured[a]["score_over_b"] <= OLD_RANGE_RULE[1])
           for a in archs}
    return {"pass": all(per_size.values()), "per_size": per_size, "old_rule_per_size": old,
            "missing": [a for a in archs if a not in configured]}


def format_range(range_dir, archs):
    lines = ["arch      pool    P_IQM      b_IQM      P/b     P_SD       P_range    final_loss_IQM  "
             "final_loss_SD  final_loss_range  old_10-90%"]
    for r in range_rows(range_dir):
        fl = r["final_loss_rounds"]
        mark = "  (configured pool)" if r["is_configured_pool"] else ""
        lines.append(f"{r['arch']:<9} {r['pool_size']:<7} {r['score_iqm']:<10.4g} {r['b_iqm']:<10.4g} "
                     f"{r['score_over_b']:<7.3f} {r['score_std']:<10.3g} {r['score_range']:<10.3g} "
                     f"{_iqm(fl):<15.3g} {np.std(fl, ddof=1):<14.3g} {max(fl) - min(fl):<17.3g} "
                     f"{r['within_10_90_pct_of_b']}{mark}")
    c = range_criterion(range_dir, archs)
    lines.append(f"criterion (w): P/b >= {RANGE_MIN_P_OVER_B} at the configured pool, every size: "
                 f"{'PASS' if c['pass'] else 'FAIL'} {c['per_size']}" + (f" missing {c['missing']}" if c["missing"] else ""))
    lines.append(f"superseded 10-90% rule (information only): {c['old_rule_per_size']}")
    lines.append("A fresh-critic range check cannot show sensitivity; that comes from the positive control and "
                 "the dev run. The fresh-pair null measures noise.")
    return lines


def _section(lines, title, fn):
    lines.extend(["", f"== {title} =="])
    try:
        fn()
    except Exception as e:  # a missing or partial output is reported, not fatal
        lines.append(f"(not available: {type(e).__name__}: {e})")


def _status_rows(path):
    return list(csv.DictReader(open(path), delimiter="\t")) if Path(path).exists() else []


def blockA_summary(out):
    out = Path(out)
    lines = [f"Exp 1/2 Block A summary ({out.name})", "", "step  result     exit  wall_s"]
    for r in _status_rows(out / "status.tsv"):
        lines.append(f"{r['step']:<5} {r['result']:<10} {r['exit_code']:<5} {r['wall_s']}")

    def a0():
        d = json.loads((out / "A0_matmul_precision.json").read_text())
        r, h = d["run_setting"], d["check1_highest"]
        lines.append(f"device {r['device_kind']} ({r['platform']}); jax {r['jax']} jaxlib {r['jaxlib']}; "
                     f"cuda {r['backend_platform_version']}")
        lines.append(f"run setting ({r['matmul_precision']}): float32 matmul max rel error "
                     f"{r['float32_matmul_max_rel_error']:.3g} (TF32 expected ~1e-4 to 1e-3)")
        lines.append(f"highest (Check 1): max rel error {h['float32_matmul_max_rel_error']:.3g} "
                     "(FP32 expected ~1e-7 to 1e-6)")
        if h["float32_matmul_max_rel_error"] >= 1e-5:
            lines.append("WARNING: 'highest' does not look like full FP32; Check 1 relies on it. Send me this file.")

    def a3():
        for r in json.loads((out / "A3_profile_dog_run_D6.json").read_text()):
            peak = r.get("peak_device_bytes")
            peak = f"{peak / 2**30:.2f} GiB" if peak else "n/a"
            lines.append(f"{r['arch']} {r['env']}: training {r['train_it_per_s_probes_off']:.1f} it/s (probes off); "
                         f"one probe check {r['probe_check_s']:.1f} s; probe overhead "
                         f"{r['probe_overhead_pct_of_wallclock']:.2f}% of a run; peak GPU memory {peak}")

    def a4():
        log = out / "A4.log"
        if not log.exists():
            lines.append("not run (RUN_A4 unset, or skipped after an A0 failure)")
            return
        hits = [l for l in log.read_text().splitlines() if "it_per_s_traced" in l]
        lines.append(hits[-1].strip() if hits else "no it/s line in A4.log")
        lines.extend(str(p) for p in sorted((out / "A4_trace_D6W1536_dog_run").rglob("*.trace.json.gz")))

    _section(lines, "A0 matmul precision", a0)
    _section(lines, "A2 fresh-critic range, dog-run",
             lambda: lines.extend(format_range(out / "A2_range_dog_run", ["D2W512", "D4W1024", "D6W1536"])))
    _section(lines, "A3 short profile, D6W1536 dog-run (TF32)", a3)
    _section(lines, "A1 smoke tests",
             lambda: lines.extend(format_unittest(parse_unittest_log((out / "A1.log").read_text()))))
    _section(lines, "A4 GPU trace", a4)
    return "\n".join(lines) + "\n"


def main(argv):
    if len(argv) >= 2 and argv[0] == "range-check":
        c = range_criterion(argv[1], argv[2:])
        print(f"range criterion (w): P/b >= {RANGE_MIN_P_OVER_B} at the configured pool: {c['per_size']} -> "
              f"{'PASS' if c['pass'] else 'FAIL'}; superseded 10-90% rule (information): {c['old_rule_per_size']}")
        return 0 if c["pass"] else 1
    if len(argv) == 2 and argv[0] == "blockA":
        text = blockA_summary(argv[1])
        (Path(argv[1]) / "summary.txt").write_text(text)
        print(text, end="")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
