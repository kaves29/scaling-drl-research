#!/usr/bin/env python3
"""Read every JSONL record; distinguish recorded stage completion from trace silence."""

import argparse
import collections
import json
from pathlib import Path


def analyze(path):
    events = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not events:
        raise ValueError("empty trace")
    pending, mismatches = [], []
    stages = collections.Counter()
    for index, event in enumerate(events, 1):
        if event["event"] == "begin":
            pending.append((event["stage"], index))
        elif event["event"] == "end":
            stages[event["stage"]] += 1
            if event.get("begin_recorded") is False:
                continue  # An error on a deliberately sampled-out fine call.
            if not pending or pending[-1][0] != event["stage"]:
                mismatches.append(dict(line=index, stage=event["stage"], open_stages=list(pending)))
            else:
                pending.pop()
    progress = [e for e in events if e["event"] == "progress"]
    return dict(records=len(events), event_counts=dict(collections.Counter(e["event"] for e in events)),
                span_seconds=events[-1]["monotonic_time"] - events[0]["monotonic_time"],
                unmatched_begins=pending, mismatched_ends=mismatches,
                completed_stages=dict(stages), terminal_event=events[-1],
                progress=[dict(interaction_step=e["interaction_step"], update_step=e["update_step"],
                               elapsed=e["monotonic_time"] - events[0]["monotonic_time"],
                               totals=e["totals"]) for e in progress],
                stage_timeline=[dict(stage=e["stage"], seconds=e.get("seconds"),
                                     elapsed=e["monotonic_time"] - events[0]["monotonic_time"],
                                     error=e.get("error"), module=e.get("module"))
                                for e in events if e["event"] == "end" and e["stage"] in
                                {"trainer_init", "start", "_evaluate", "structural_metrics", "update_many", "backend_compile"}])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace), indent=2))
