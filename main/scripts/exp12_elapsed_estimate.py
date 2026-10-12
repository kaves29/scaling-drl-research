#!/usr/bin/env python3
"""Planning estimate of Exp1+Exp2 GPU-hours and elapsed hours at fixed concurrency.

Inputs are the repository's measured single-job A100 training rates (Block B,
dog-run and myo-key-turn; the newer 42.44/s D4W1536 dog-run window) and the
UNMEASURED twin-critic HumanoidBench estimates. Every assumption is a parameter
and is printed. This is not a forecast, an approval or a resource request.
Elapsed time = greedy longest-processing-time list scheduling, one job per GPU,
arms released at their parent's fork; queue waits and allocation gaps excluded.
"""

import argparse
import heapq
import json

SEEDS = 5
INTERACTIONS = {"dmc_hard": 500_000, "dmc_medium": 250_000, "myosuite_simba": 500_000,
                "humanoid_bench": 1_000_000}
TASKS = {"dmc_hard": 5, "dmc_medium": 2, "myosuite_simba": 4, "humanoid_bench": 2}
# Measured Block B A100 single-job training rates (interactions/s), scaled by the
# newer D4W1536 dog-run measurement 42.44/46.3 for conservatism.
SCALE = 42.4411 / 46.3
MEASURED = {
    "dmc": {"D2W512": 112.5, "D4W1024": 65.8, "D4W1536": 46.3},
    "myosuite_simba": {"D2W512": 117.5, "D4W1024": 66.1, "D4W1536": 45.1},
}
HB_ESTIMATE = {"D2W512": 88.0, "D4W1024": 56.0, "D4W1536": 32.0}  # UNMEASURED twin estimate


def rate(arch, group, hb_factor):
    if group == "humanoid_bench":
        return HB_ESTIMATE[arch] * hb_factor
    key = "myosuite_simba" if group == "myosuite_simba" else "dmc"
    return MEASURED[key][arch] * SCALE


def jobs(overhead, hb_factor, fork_fraction, fork_at, control_ext):
    out = []
    for group, tasks in TASKS.items():
        for arch in ("D2W512", "D4W1024", "D4W1536"):
            hours = INTERACTIONS[group] / rate(arch, group, hb_factor) / 3600 * overhead
            for n in range(tasks * SEEDS):
                forks = arch != "D2W512" and n < round(tasks * SEEDS * fork_fraction)
                parent = hours * (1 + (control_ext if forks else 0))
                arm = hours * 0.25 if forks else 0.0
                out.append({"arch": arch, "group": group, "parent_h": parent,
                            "arm_h": arm, "fork_h": hours * fork_at})
    return out


def simulate(all_jobs, gpus):
    free = [0.0] * gpus
    heapq.heapify(free)
    released = []  # (release_time, duration)
    end = 0.0
    for job in sorted(all_jobs, key=lambda j: -j["parent_h"]):
        start = heapq.heappop(free)
        finish = start + job["parent_h"]
        heapq.heappush(free, finish)
        end = max(end, finish)
        if job["arm_h"]:
            released.append((start + job["fork_h"], job["arm_h"]))
    # Arms run after parents in this simple model, earliest-release first, but never
    # before their fork; with arms prioritised in the real scheduler this is an upper bound.
    for release, duration in sorted(released):
        start = max(heapq.heappop(free), release)
        finish = start + duration
        heapq.heappush(free, finish)
        end = max(end, finish)
    return end


def simulate_windows(all_jobs, windows, stop_margin_h=0.25):
    """Allocations as (start_h, gpus, hours). A job starts on a free GPU only if its
    whole duration fits before that allocation's end minus the stop margin (the
    scheduler's admission rule); arms become available at their parent's fork.
    Returns completed/remaining GPU-hours and the finish time of completed work."""
    slots = [[start, start + hours] for start, gpus, hours in windows for _ in range(gpus)]
    queue = sorted(({"kind": "parent", "h": j["parent_h"], "ready": 0.0, "job": j} for j in all_jobs),
                   key=lambda t: -t["h"])
    done_h, finish, pending, arms = 0.0, 0.0, list(queue), []
    while True:
        best = None
        # Earliest feasible start overall; ties go to arms, then the longest job.
        for rank, task in enumerate(sorted(arms, key=lambda t: t["ready"]) + pending):
            for slot in slots:
                start = max(slot[0], task["ready"])
                if start + task["h"] <= slot[1] - stop_margin_h:
                    key = (start, rank)
                    if best is None or key < best[0]:
                        best = (key, slot, task)
        if best is None:
            break
        (start, _), slot, task = best
        slot[0] = start + task["h"]
        (arms if task["kind"] == "arm" else pending).remove(task)
        done_h += task["h"]
        finish = max(finish, slot[0])
        job = task["job"]
        if task["kind"] == "parent" and job["arm_h"]:
            arms.append({"kind": "arm", "h": job["arm_h"], "ready": start + job["fork_h"], "job": job})
    remaining = sum(t["h"] for t in pending) + sum(t["h"] for t in arms)
    remaining += sum(t["job"]["arm_h"] for t in pending if t["job"]["arm_h"])
    return {"completed_gpu_h": round(done_h, 1), "remaining_gpu_h": round(remaining, 1),
            "parents_left": len(pending), "arms_left": len(arms) + sum(bool(t["job"]["arm_h"]) for t in pending),
            "last_finish_h": round(finish, 1),
            "capacity_gpu_h": sum(g * h for _, g, h in windows)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overhead", type=float, default=1.20,
                   help="multiplier for probes, evaluations, checkpoints, compile and start-up")
    p.add_argument("--hb-factor", type=float, default=1.0, help="scale of the unmeasured HB estimate")
    p.add_argument("--fork-fraction", type=float, default=1.0,
                   help="fraction of scaled parents that fork naturally (unknown; 1.0 = upper bound)")
    p.add_argument("--fork-at", type=float, default=0.6, help="fork time as a fraction of the parent")
    p.add_argument("--control-ext", type=float, default=0.05,
                   help="mean control extension beyond B for forked parents (cap 0.20)")
    p.add_argument("--gpus", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--windows", nargs="*", default=[],
                   help="allocations START_H:GPUS:HOURS, e.g. 0:4:35 6:8:48")
    args = p.parse_args(argv)
    work = jobs(args.overhead, args.hb_factor, args.fork_fraction, args.fork_at, args.control_ext)
    parent_h = sum(j["parent_h"] for j in work)
    arm_h = sum(j["arm_h"] for j in work)
    longest = max(j["parent_h"] for j in work)
    report = {
        "assumptions": vars(args) | {"rate_scale_vs_block_b": SCALE,
                                     "humanoid_bench": "UNMEASURED twin estimate"},
        "gpu_hours": {"exp1_parents": round(parent_h, 1), "exp2_arms": round(arm_h, 1),
                      "total": round(parent_h + arm_h, 1)},
        "longest_single_job_h": round(longest, 2),
        "elapsed_h": {g: round(simulate(work, g), 1) for g in args.gpus},
    }
    if args.windows:
        windows = [tuple(float(x) for x in w.split(":")) for w in args.windows]
        report["windows"] = simulate_windows(work, [(a, int(b), c) for a, b, c in windows])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
