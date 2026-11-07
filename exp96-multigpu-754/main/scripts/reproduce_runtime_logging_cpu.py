#!/usr/bin/env python3
"""Isolated CPU fixture: real SAC/DMControl loop through logging boundary 6000.

This shrinks networks and replay capacity only in a local test fixture. It is
not a scientific configuration, calibration, or estimate of A100 performance.
The production sampling onset, UTD, batch size and logging boundary are retained.
"""

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root = Path(args.out_dir).resolve()
    root.mkdir()  # Refuse reused evidence directories.
    import jax
    import numpy as np
    from tests.exp12_helpers import compose, patch_wandb, tiny_overrides
    from experiments.exp12.trainer import Exp12Trainer
    from experiments.exp12.runtime_trace import RuntimeTrace
    from experiments.exp12.state import latest_state_dir, state_differences

    if any(d.platform != "cpu" for d in jax.devices()):
        raise RuntimeError("this fixture requires CPU-only JAX")
    patchers = patch_wandb()
    results = []
    try:
        for mode in ("plain", "trace_async", "trace_sync"):
            cfg = compose(tiny_overrides(env_name="dog-run", env_group="dmc_hard", seed=990, steps=7000,
                          extra=["buffer.min_length=5000", "buffer.max_length=7000", "buffer.sample_batch_size=256",
                                 "logging_per_interaction_step=2000", "evaluation_per_interaction_step=50000",
                                 "num_eval_episodes=5"]))
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)
            path = root / (mode + ".jsonl")
            context = contextlib.nullcontext() if mode == "plain" else RuntimeTrace(
                path, synchronize=mode == "trace_sync", boundary_detail=True)
            start = time.perf_counter()
            with context as trace:
                if trace:
                    trace.install()
                trainer = Exp12Trainer(cfg, str(root / mode))
                trainer.start()
                trainer.train(6001)
                jax.block_until_ready(trainer._sac_agent._critic)
                elapsed = time.perf_counter() - start
                rows = len(trainer.metrics_rows)
                updates = trainer.update_step
                trainer.save(root / mode / "state")
                trainer.close()
            results.append(dict(mode=mode, seconds=elapsed, interaction_step=6001,
                                update_step=updates, logging_rows=rows,
                                trace_bytes=path.stat().st_size if path.exists() else 0))
        plain = latest_state_dir(root / "plain/state")
        for result in results[1:]:
            result["complete_state_differences"] = state_differences(
                plain, latest_state_dir(root / result["mode"] / "state"))
        # Measure the writer separately; this is not an estimate of GPU overhead.
        write_start = time.perf_counter()
        with RuntimeTrace(root / "writer.jsonl") as trace:
            for index in range(10000):
                trace.emit("writer_fixture", index=index)
        write_seconds = time.perf_counter() - write_start
        report = dict(cpu_only=True, python=sys.version, jax=jax.__version__, devices=list(map(str, jax.devices())),
                      fixture_changes="D1W8 actor/critic, capacity 7000, 14000 raw-step fixture budget; no production configurations edited",
                      runs=results, writer=dict(records=10000, seconds=write_seconds),
                      order_warning="plain first, then async then synchronized; compilation-cache warming confounds elapsed comparisons")
        (root / "results.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if any(r.get("complete_state_differences") for r in results):
            raise RuntimeError("traced fixture changed complete state")
    finally:
        for patcher in patchers:
            patcher.stop()


if __name__ == "__main__":
    main()
