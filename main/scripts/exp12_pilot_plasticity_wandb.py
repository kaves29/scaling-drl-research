#!/usr/bin/env python3
"""Publish existing completed Exp1 probe rows to a separate W&B project once.

The local CSV remains authoritative. A durable publication claim prevents an
automatic replay of possibly sent history after a network/process failure.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import exp12_dynamic_scheduler as scheduler
from scripts.exp12_seven_parent_pilot import FROZEN, select

PROJECT = "Scaling-DRL-Experiment-1"


def records(path, action_repeat):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or int(rows[0]["check_index"]) != 0:
        raise ValueError("probe check zero is missing")
    baseline = float(rows[0]["score_fresh_iqm"])
    scores = [abs(float(row[name])) for row in rows for name in ("score_current_iqm", "score_fresh_iqm")
              if row.get(name) not in (None, "") and math.isfinite(float(row[name]))]
    scale = max([1.0, *scores])
    undefined = not math.isfinite(baseline) or abs(baseline) <= 1e-12 * scale
    output = []
    seen = set()
    for row in rows[1:]:
        check = int(row["check_index"])
        step = int(row["interaction_step"]) * action_repeat
        if check in seen or step <= 0:
            raise ValueError("duplicate or malformed probe check")
        seen.add(check)
        current, fresh = float(row["score_current_iqm"]), float(row["score_fresh_iqm"])
        if not math.isfinite(current) or not math.isfinite(fresh):
            raise ValueError("nonfinite plasticity score; preserve local evidence for review")
        values = {"plasticity/score_current_iqm": current,
                  "plasticity/score_fresh_iqm": fresh,
                  "training_env_steps": step}
        if not undefined and math.isfinite(current):
            values["plasticity/degradation_pct"] = 100 * (baseline - current) / abs(baseline)
        output.append(values)
    if len(output) < 20 or sorted(x["training_env_steps"] for x in output) != [
            x["training_env_steps"] for x in output]:
        raise ValueError("completed parent does not contain 20 ordered probe checks")
    return output, baseline, undefined


def publish(queue_root, item_id):
    queue = scheduler.load_queue(queue_root)
    chosen = {item["id"]: item for _, item in select(queue)}
    item = chosen[item_id]
    ledger = scheduler.Ledger(queue_root)
    if ledger.state(item) != "complete":
        raise ValueError("plasticity export requires validated completed parent")
    run = Path(item["row"]["checkpoint_dir"])
    metadata = scheduler._metadata_matches(run, item, FROZEN)
    output, baseline, undefined = records(
        run / "probes" / "probe_checks.csv", int(metadata["resolved_config"]["action_repeat"]))
    publication = Path(queue_root) / "pilot_wandb" / item_id
    publication.parent.mkdir(parents=True, exist_ok=True)
    publication.mkdir()  # atomic once-only claim; interrupted publication requires review
    manifest = {"source_commit": FROZEN, "item_id": item_id, "project": PROJECT,
                "run_key": item["row"]["run_key"], "baseline": baseline,
                "degradation_undefined": undefined, "records": output,
                "training_run_id": metadata.get("wandb_run_id")}
    scheduler._write_new(publication / "source_records.json", manifest)
    import wandb
    run_id = item_id[:16]
    try:
        with wandb.init(project=PROJECT, id=run_id, resume="allow",
                        group="Exp1-production-pilot", job_type="plasticity-sidecar",
                        name=item["row"]["run_key"], config={k: v for k, v in manifest.items() if k != "records"},
                        dir=str(publication)) as run_wandb:
            for row in output:
                run_wandb.log(row, step=row["training_env_steps"])
    except Exception:
        # The claim remains. Inspect W&B and the local CSV before manually retrying.
        raise
    scheduler._write_new(publication / "published.json", {"project": PROJECT,
                         "run_id": run_id, "item_id": item_id})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--id", help="one source-bound parent id; omit to publish all completed pilot parents")
    args = parser.parse_args()
    if args.id:
        publish(args.queue, args.id)
        return
    queue = scheduler.load_queue(args.queue)
    ledger = scheduler.Ledger(args.queue)
    for _, item in select(queue):
        if ledger.state(item) != "complete":
            continue
        publication = Path(args.queue) / "pilot_wandb" / item["id"]
        if (publication / "published.json").exists():
            continue
        if publication.exists():
            raise ValueError(f"interrupted publication {publication} requires W&B review")
        publish(args.queue, item["id"])


if __name__ == "__main__":
    main()
