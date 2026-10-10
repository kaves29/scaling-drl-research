"""CPU inventory of all candidate task shapes and raw capture sizes; no source selection."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    from experiments.exp1 import compose_config
    from experiments.exp12.envs import create_envs
    from experiments.exp3.streams import StreamWriter
    from scale_rl.agents.wrappers.utils import RunningMeanStd
    from generate_manifest import EXP12_ENVS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    reports = []
    for name, group in EXP12_ENVS:
        if name not in ("dog-run", "humanoid-walk") and group != "myosuite_simba":
            continue
        cfg = compose_config(
            str(Path(__file__).resolve().parents[1] / "configs"),
            "base_exp12",
            [f"env_name={name}", f"env={group}", "seed=1"],
        )
        train, evaluation = create_envs(**cfg.env)
        try:
            obs, _ = train.reset()
            action = np.zeros(train.action_space.shape, dtype=np.float32)
            next_obs, reward, terminated, truncated, _ = train.step(action)
            transition = dict(
                observation=obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                next_observation=next_obs,
            )
            rms = RunningMeanStd(
                shape=train.observation_space.shape, dtype=train.observation_space.dtype
            )
            rms.update(obs)
            normalization = dict(mean=rms.mean, var=rms.var, count=rms.count)
            payload = sum(np.asarray(v).nbytes for v in transition.values())
            payload += (
                sum(np.asarray(v).nbytes for v in normalization.values()) + 16 + 8
            )
            with tempfile.TemporaryDirectory() as d:
                writer = StreamWriter(
                    Path(d) / "stream",
                    {"start_step": 0, "captures": ["agent_key"]},
                    1000,
                    autoflush=False,
                )
                start = time.perf_counter()
                for step in range(1, 1001):
                    writer.append(
                        step,
                        transition,
                        2,
                        normalization,
                        {"agent_key": np.array([0, 1], np.uint32)},
                    )
                append_seconds = time.perf_counter() - start

                # Python-container overhead is additional to dense array payloads.
                def size(x):
                    if isinstance(x, dict):
                        return sys.getsizeof(x) + sum(
                            sys.getsizeof(k) + size(v) for k, v in x.items()
                        )
                    return sys.getsizeof(x)

                pending_bytes = sys.getsizeof(writer.pending) + sum(
                    size(x) for x in writer.pending
                )
                start = time.perf_counter()
                writer.flush()
                flush_seconds = time.perf_counter() - start
                reports.append(
                    dict(
                        environment=name,
                        observation_dim=obs.shape[-1],
                        action_dim=action.shape[-1],
                        observation_dtype=str(obs.dtype),
                        rms_dtype=str(rms.mean.dtype),
                        dense_bytes_per_arrival=payload,
                        pending_python_bytes_per_arrival=pending_bytes / 1000,
                        npz_bytes_per_arrival=writer.manifest["chunks"][0]["bytes"]
                        / 1000,
                        cpu_append_seconds_1000=append_seconds,
                        cpu_flush_seconds_1000=flush_seconds,
                    )
                )
        finally:
            train.close()
            evaluation.close()
    report = dict(
        scope="CPU shape/storage reference with repeated engineering transitions, not recording overhead during training or source-cell selection",
        tasks=reports,
    )
    with open(args.out, "x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
