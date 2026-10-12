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
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
