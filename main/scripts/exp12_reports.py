#!/usr/bin/env python3
"""Summaries for the unattended Exp 1/2 GPU jobs (docs/exp12_cuda_commands.md). Never imports jax.

    python scripts/exp12_reports.py blockA <out_dir>                 # writes <out_dir>/summary.txt
    python scripts/exp12_reports.py blockB <out_dir>                 # writes <out_dir>/report.txt
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


STATUS_COLUMNS = ("lane", "step", "result", "exit_code", "wall_s", "start")


def blockB_status(out):
    rows = []
    for f in sorted(Path(out, "status").glob("*.tsv")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(dict(zip(STATUS_COLUMNS, line.split("\t"))))
    return rows


def _glob1(base, pattern):
    hits = sorted(Path(base).glob(pattern))
    return hits[0] if hits else None


def packing_table(out):
    """Per critic size and job count: it/s per job, slowdown vs 1 job, total throughput, memory, overlap."""
    rows = []
    for d in sorted(Path(out, "gpu2", "packing").glob("*_x*")):
        jobs = [json.loads(f.read_text()) for f in sorted(d.glob("job_*.json"))]
        arch, n = d.name.rsplit("_x", 1)
        mem = [int(x) for x in (d / "gpu_mem_mib.txt").read_text().split()] if (d / "gpu_mem_mib.txt").exists() else []
        common = (min(j["end"] for j in jobs) - max(j["start"] for j in jobs)) if jobs else 0.0
        rows.append({"arch": arch, "jobs": int(n), "completed": len(jobs), "it_per_s": [j["it_per_s"] for j in jobs],
                     "peak_gib": [(j["peak_bytes"] or 0) / 2**30 for j in jobs],
                     "gpu_total_gib": max(mem) / 1024 if mem else None,
                     "overlap": min(max(common, 0.0) / (j["end"] - j["start"]) for j in jobs) if jobs else 0.0,
                     "cores": [len(j["cores"]) for j in jobs]})
    lines = ["dog-run, probes off: probe overhead and post-fork evaluation are NOT included.",
             "arch      jobs  per-job it/s                 mean it/s  slowdown vs 1  total it/s  per-job peak GiB        "
             "GPU total GiB  overlap  cores/job"]
    base = {r["arch"]: float(np.mean(r["it_per_s"])) for r in rows if r["jobs"] == 1 and r["it_per_s"]}
    for r in rows:
        if not r["it_per_s"]:
            lines.append(f"{r['arch']:<9} {r['jobs']:<5} (no job finished; see gpu2/packing_{r['arch']}_x{r['jobs']}.log)")
            continue
        mean = float(np.mean(r["it_per_s"]))
        slow = f"{base[r['arch']] / mean:.2f}x" if r["arch"] in base else "n/a"
        total = f"{r['gpu_total_gib']:.2f}" if r["gpu_total_gib"] is not None else "n/a"
        flag = "yes" if r["overlap"] >= 0.9 else "NO"
        lines.append(f"{r['arch']:<9} {r['jobs']:<5} {', '.join(f'{x:.1f}' for x in r['it_per_s']):<28} {mean:<10.1f} "
                     f"{slow:<14} {sum(r['it_per_s']):<11.1f} {', '.join(f'{x:.2f}' for x in r['peak_gib']):<23} "
                     f"{total:<14} {r['overlap']:.2f} {flag:<3} {','.join(map(str, r['cores']))}"
                     + ("" if r["completed"] == r["jobs"] else f"  ONLY {r['completed']} of {r['jobs']} jobs finished"))
    lines.append("slowdown = 1-job it/s / mean it/s; overlap = share of each job's timed window during which all "
                 "jobs were timing (flag 'yes' at >= 0.9); GPU total = nvidia-smi maximum during the configuration.")
    return lines


def _a1_followup(out, mode):
    d = json.loads(Path(out, "gpu2", f"a1_followup_{mode}.json").read_text())
    lines = [f"[{mode}] device {d['device']}, XLA_FLAGS={d['XLA_FLAGS']}, run setting {d['run_setting']}",
             "  injection (blocks, m, pair): Q bit-identical TF32/highest; dQ/da deviation in eps units TF32/highest"]
    for r in d["injection"]:
        t, h = r["tensorfloat32"], r["highest"]
        lines.append(f"    {r['blocks']}, {r['m']:<4}, {r['pair']:<6}: {t['q_bit_identical']}/{h['q_bit_identical']}; "
                     f"{t['dq_da_dev_eps_units']:.1f}/{h['dq_da_dev_eps_units']:.1f}")
    for prec, v in d["diagnostics_on_off"].items():
        for k, c in v.items():
            lines.append(f"  diagnostics {k} [{prec}]: {c['leaves_differing']}/{c['leaves']} leaves differ, max abs "
                         f"{c['max_abs_dev']:.3g}, max dev/leaf max {c['max_dev_over_leaf_max']:.3g}")
    for prec, v in d["policy_kl"].items():
        lines.append(f"  policy KL [{prec}]: diagnostic {v['diagnostic']:.6g} vs closed form {v['closed_form']:.6g} "
                     f"(rel diff {v['rel_diff']:.3g}; min sigma {v['min_sigma']:.3g})")
    return lines


def blockB_report(out):
    import pandas as pd

    out = Path(out)
    status = blockB_status(out)
    by_step = {r["step"]: r for r in status}
    lines = [f"Exp 1/2 Block B report ({out.name})"]
    gates = []

    def gate(name, ok, detail):
        gates.append((name, "PASS" if ok is True else ("FAIL" if ok is False else ok), detail))

    def info():
        for f in ("commit.txt", "host.tsv", "gpus.csv"):
            if (out / f).exists():
                lines.extend(l for l in (out / f).read_text().splitlines() if l.strip())
    _section(lines, "Node", info)

    def steps():
        lines.append("lane  step                          result               exit  wall_s")
        for r in status:
            lines.append(f"{r['lane']:<5} {r['step']:<29} {r['result']:<20} {r['exit_code']:<5} {r['wall_s']}")
    _section(lines, "Steps", steps)

    # GPU 2: tests
    for mode in ("default", "deterministic"):
        log = out / "gpu2" / f"tests_{mode}.log"
        if log.exists():
            p = parse_unittest_log(log.read_text())
            bad = [(t, s) for t, s in p["tests"] if s not in ("PASS", "SKIP")]
            gate(f"GPU test suite, {mode} ops", (p["final"] or "").startswith("OK") and not p["warnings"],
                 f"{len(p['tests'])} tests; {p['final']}; not passed: {len(bad)}")

            def tests(p=p, bad=bad, mode=mode):
                lines.append(f"[{mode}] " + format_unittest(p)[-1])
                lines.extend(f"  {s:<8} {t}" + "".join(f"\n      {k} subtest ({q})" for k, q in p["failures"].get(t, []) if q)
                             for t, s in bad)
                lines.extend(f"  WARNING: {w}" for w in p["warnings"])
            _section(lines, f"GPU test suite ({mode} ops)", tests)
        else:
            gate(f"GPU test suite, {mode} ops", "NOT RUN", by_step.get(f"tests_{mode}", {}).get("result", "no log"))
    hb = out / "gpu2" / "tests_humanoid_bench.log"
    if hb.exists():
        p = parse_unittest_log(hb.read_text())
        gate("HumanoidBench tests (HB_ENV)", (p["final"] or "").startswith("OK") and not p["warnings"],
             f"{len(p['tests'])} tests; {p['final']}")
    else:
        gate("HumanoidBench tests (HB_ENV)", "UNAVAILABLE", by_step.get("tests_humanoid_bench", {}).get("result", "no log"))
    blog = out / "gpu2" / "break_checks.log"
    if blog.exists():
        text = blog.read_text().splitlines()
        ok = sum(1 for l in text if l.startswith("[OK] "))
        total = ok + sum(1 for l in text if l.startswith("[PROBLEM] "))  # the script's two result labels
        gate("break checks (default ops)", total > 0 and ok == total, f"{ok} of {total} OK")
    else:
        gate("break checks (default ops)", "NOT RUN", by_step.get("break_checks", {}).get("result", "no log"))

    for mode in ("default", "deterministic"):
        _section(lines, f"A1 follow-up measurements ({mode} ops)", lambda m=mode: lines.extend(_a1_followup(out, m)))

    _section(lines, "Packing test (GPU 2)", lambda: lines.extend(packing_table(out)))

    def speed():
        lines.append("probes off, 1 job, 4 cores, 600 timed steps; dog-run: the packing test's 1-job rows above")
        for r in status:
            if r["step"].startswith("speed_"):
                j = out / "gpu3" / "speed" / r["step"][len("speed_"):] / "job_0.json"
                if j.exists():
                    d = json.loads(j.read_text())
                    peak = f"{d['peak_bytes'] / 2**30:.2f} GiB" if d.get("peak_bytes") else "n/a"
                    lines.append(f"{d['env']:<14} {d['arch']:<9} {d['it_per_s']:.1f} it/s, peak GPU memory {peak}")
                else:
                    lines.append(f"{r['step']}: {r['result']}")
    _section(lines, "Per-suite training speed (GPU 3)", speed)

    def numerics():
        lines.append("name                                              mode / context                      deviation   tolerance")
        for f in sorted((out / "gpu2").glob("numerics_*.jsonl")):
            for line in f.read_text().splitlines():
                d = json.loads(line)
                dev = d.get("deviation", d.get("rel_diff"))
                ctx = {k: v for k, v in d.items() if k not in ("name", "deviation", "tolerance", "rel_diff")}
                lines.append(f"{d['name']:<49} [{f.stem[len('numerics_'):]}] {str(ctx)[:60]:<36} {dev:<11.3g} "
                             f"{d.get('tolerance', 'information')}")
        lines.append("A tolerance of None means not measured yet: the test skips and records the deviation; set each "
                     "tolerance from these numbers (10x the largest, one significant digit).")
    _section(lines, "GPU numerics measured by the test suite", numerics)

    def timings():
        for log, who in ((out / "gpu0" / "dev_run.log", "dev run (control)"), (out / "gpu1" / "arm_injected.log", "injected arm")):
            if not log.exists():
                continue
            text = log.read_text()
            for m in re.findall(r"\[exp1\] fork state saved at interaction_step \d+ in ([\d.]+) s", text):
                lines.append(f"{who}: fork state save {m} s")
            for m in re.findall(r"restored interaction_step \d+ from \S+ in ([\d.]+) s", text):
                lines.append(f"{who}: state restore {m} s")
            for m in re.findall(r"\[exp2_arm\] fork state restored in ([\d.]+) s", text):
                lines.append(f"{who}: fork state restore {m} s")
            evals = [float(x) for x in re.findall(r"post-fork evaluation \d+: ([\d.]+) s", text)]
            if evals:
                lines.append(f"{who}: {len(evals)} post-fork evaluations, wall per evaluation mean {np.mean(evals):.1f} s, "
                             f"max {max(evals):.1f} s, total {sum(evals) / 60:.1f} min")
    _section(lines, "Fork save/restore and post-fork evaluation wall times", timings)

    rng = out / "gpu2" / "range_hopper_hop"
    archs = ["D2W512", "D4W1024", "D6W1536"]
    if list(rng.glob("range_D*.json")):
        c = range_criterion(rng, sorted({r["arch"] for r in range_rows(rng)}) or archs)
        gate("hopper-hop range, amendment (w)", c["pass"], str(c["per_size"]))
        _section(lines, "hopper-hop fresh-critic range (B2)",
                 lambda: lines.extend(format_range(rng, sorted({r["arch"] for r in range_rows(rng)}))))
    else:
        gate("hopper-hop range, amendment (w)", "NOT RUN", by_step.get("range_hopper", {}).get("result", "no output"))

    def identity():
        lines.append("cell (cache mode)                      pass   parent wall s  arm wall s  arm cache hits/misses  differences/error")
        for r in status:
            if not r["step"].startswith("identity_"):
                continue
            mode, cell = r["step"][len("identity_"):].split("_", 1)
            if r["result"] == "SKIPPED_UNAVAILABLE":
                lines.append(f"{cell} ({mode}): not available on this install (see gpu2/suite_*.log)")
                continue
            j = out / "gpu2" / "identity" / mode / f"{cell}.json"
            d = json.loads(j.read_text()) if j.exists() else {}
            gate(f"identity fork {cell} ({mode} cache)", bool(d.get("pass")), r["result"])
            log = out / "gpu2" / f"{r['step']}.log"
            text = log.read_text() if log.exists() else ""
            walls = re.findall(r"\[identity\] (parent|identity-arm) process wall (\d+) s", text)
            wall = dict(walls)
            caches = re.findall(r"persistent compilation cache \S+: hits (\d+), misses (\d+)", text)
            arm_cache = f"{caches[1][0]}/{caches[1][1]}" if len(caches) > 1 else "n/a"
            what = (d.get("differences", [])[:8] or d.get("error", "")) if d else "no compare output"
            lines.append(f"{cell + ' (' + mode + ')':<38} {str(d.get('pass')):<6} {wall.get('parent', 'n/a'):<14} "
                         f"{wall.get('identity-arm', 'n/a'):<11} {arm_cache:<22} {what}")
        lines.append("cold: parent and arm compile separately (separate empty caches); warm: the arm reuses the parent's "
                     "executables from one shared cache. The arm's wall time difference is the cache's effect on start-up.")
    _section(lines, "Identity forks (reduced budget, Block C)", identity)

    def null():
        for f in sorted((out / "gpu3" / "null_dog_run").glob("null_summary_*.json")):
            d = json.loads(f.read_text())
            gate(f"fresh-pair null {d['arch']} (fire rate <= 5%)", not d["fire_rate_exceeds_5pct"],
                 f"{d['per_check_fire_rate']:.3f}")
            lines.append(f"{d['arch']}: pairs {d['null_pairs']}, per-check fire rate {d['per_check_fire_rate']:.3f}, "
                         f"p95 of L {d['would_be_null_threshold_p95_of_L']:.4g}, IQM of L {d['iqm_of_L']:.4g}, "
                         f"SD of L {d['std_of_L']:.4g}")
        if not list((out / "gpu3" / "null_dog_run").glob("null_summary_*.json")):
            gate("fresh-pair null", "NOT RUN", by_step.get("null_dog_run", {}).get("result", "no output"))
    _section(lines, "Fresh-pair null (GPU 3)", null)

    results = out / "gpu0" / "results"

    def dev():
        run = _glob1(results, "exp12/exp1/runs/*/run.csv")
        r = pd.read_csv(run).iloc[0] if run else None
        triggered = r is not None and pd.notna(r["f_star_check"])
        gate("dev run triggered (f*_run exists)", True if triggered else "NO TRIGGER",
             f"check {r['f_star_check']}, step {r['f_star_interaction_step']}" if triggered else
             f"no f*_run (dev run step: {by_step.get('dev_run', {}).get('result', 'not run')})")
        if r is not None:
            lines.append(f"run {r['run_key']}: status {r['status']}, f*_run check {r['f_star_check']} "
                         f"(step {r['f_star_interaction_step']}, fraction {r['f_star_fraction']}), fork step "
                         f"{r['fork_interaction_step']}, initial fresh score IQM {r['initial_fresh_score_iqm']:.4g}")
            chk = pd.read_csv(run.parent / "checks.csv")
            lines.append("check  step      L_IQM       CI_low      CI_high     triggered")
            for _, c in chk.iterrows():
                lines.append(f"{int(c['check_index']):<6} {int(c['interaction_step']):<9} {c['loss_iqm']:<11.4g} "
                             f"{c['ci_low']:<11.4g} {c['ci_high']:<11.4g} {c['triggered']}")
        fj = out / "gpu0" / "dev_run" / "fork" / "fork.json"
        if fj.exists():
            f = json.loads(fj.read_text())
            lines.append(f"fork: step {f['fork_step']} (check {f['fork_check_index']}), arm end {f['arm_end_step']}, "
                         f"control end {f['control_end_step']}, device {f['device'].get('device_kind')}")
    _section(lines, "Development run (GPU 0)", dev)

    def pc():
        p = json.loads((out / "gpu0" / "positive_control" / "positive_control.json").read_text())
        s = p.get("selection") or {}
        gate("positive control: m chosen", True if p["status"] == "m_chosen" else "STOP (exit 3)",
             f"m={p['chosen_m']}" if p["status"] == "m_chosen" else str(p.get("stop")))
        lines.append(f"status {p['status']}; stop: {p.get('stop')}; chosen m: {p['chosen_m']}")
        for k in ("l_trigger", "l_injected", "recovery", "noise", "noise_sd"):
            if k in s:
                lines.append(f"{k}: {s[k]}")
        for k in ("trigger_reproduction_max_abs_diff", "loss_iqm", "shared_offset_sensitivity"):
            if k in p:
                lines.append(f"{k}: {p[k]}")
    if by_step.get("positive_control", {}).get("result", "").startswith("SKIPPED"):
        gate("positive control: m chosen", "SKIPPED", by_step["positive_control"]["result"])
    else:
        _section(lines, "Positive control (GPU 0)", pc)

    def arm():
        m = (out / "gpu1" / "arm_m.txt").read_text().strip() if (out / "gpu1" / "arm_m.txt").exists() else None
        lines.append(f"injected arm m: {m}; step result {by_step.get('arm_injected', {}).get('result')}")
        c1 = _glob1(results, "exp12/exp2/*/check1_injected.json")
        c2 = _glob1(results, "exp12/exp2/*/check2.json")
        if c1:
            d = json.loads(c1.read_text())
            gate("injected arm Check 1", d["pass"], f"max {d['max_eps_units']:.3g} eps units ({d['matmul_precision']})")
            lines.append(f"Check 1: pass {d['pass']}, max {d['max_eps_units']:.4g} eps units, precision {d['matmul_precision']}")
        if c2:
            d = json.loads(c2.read_text())
            gate("injected arm Check 2", d["pass"], f"paired diff IQM {d['paired_difference_iqm']:.4g}, "
                 f"CI [{d['paired_difference_ci_low']:.4g}, {d['paired_difference_ci_high']:.4g}]")
            lines.append(f"Check 2: pass {d['pass']}, paired difference IQM {d['paired_difference_iqm']:.4g}, 95% CI "
                         f"[{d['paired_difference_ci_low']:.4g}, {d['paired_difference_ci_high']:.4g}]")
        for arm_name in ("control", "injected"):
            ev = _glob1(results, f"exp12/exp2/*/arm_{arm_name}/eval_episodes.csv")
            if ev:
                e = pd.read_csv(ev)
                last = e[e["eval_index"] == e["eval_index"].max()]
                lines.append(f"post-fork evaluations, {arm_name}: {e['eval_index'].nunique()} evaluations; last "
                             f"(index {int(last['eval_index'].iloc[0])}) mean return {last['return'].mean():.2f}")
    arm_status = by_step.get("arm_injected", {}).get("result", "NOT RUN")
    if arm_status.startswith("SKIPPED"):
        gate("injected arm", "SKIPPED", arm_status)
    _section(lines, "Injected arm (GPU 1)", arm)

    def preflight():
        for r in status:
            if r["step"].startswith("preflight_"):
                gate(r["step"], r["result"] == "PASS", r["result"])
                log = out / "gpu0" / f"{r['step']}.log"
                c1 = [l for l in log.read_text().splitlines() if "Check 1:" in l] if log.exists() else []
                lines.append(f"{r['step']}: {r['result']}" + (f"; {c1[-1].strip()}" if c1 else ""))
    _section(lines, "Preflight (GPU 0)", preflight)

    gate_lines = ["", "== Gates =="] + [f"{res:<14} {name}: {detail}" for name, res, detail in gates]
    overall = gates and all(res in ("PASS", "UNAVAILABLE") for _, res, _ in gates)  # unavailable suites do not fail it
    gate_lines.append(f"OVERALL: {'PASS' if overall else 'NOT PASS (see the gates above)'}")
    return "\n".join(lines[:1] + gate_lines + lines[1:]) + "\n"


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
    if len(argv) == 2 and argv[0] == "blockB":
        text = blockB_report(argv[1])
        (Path(argv[1]) / "report.txt").write_text(text)
        print(text, end="")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
