"""CPU simulator construction and exact next-transition restore for all 13 tasks."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("EXP12_JAX_CACHE_DIR", "off")

import copy
import importlib.util
import json
import random
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))

import jax
import numpy as np

from experiments.exp1 import compose_config
from experiments.exp12.envs import create_envs, env_restore_state, restore_env

TASKS = (
    *(
        (name, "dmc_hard")
        for name in (
            "dog-run",
            "dog-trot",
            "humanoid-run",
            "humanoid-walk",
            "humanoid-stand",
        )
    ),
    ("swimmer-swimmer15", "dmc_medium"),
    ("hopper-hop", "dmc_medium"),
    *(
        (name, "myosuite_simba")
        for name in ("myo-key-turn", "myo-pen-twirl", "myo-pose-hard", "myo-reach")
    ),
    ("h1-reach-v0", "humanoid_bench"),
    ("h1-run-v0", "humanoid_bench"),
)


def verify_task(name, group, seed):
    """Use canonical wrappers and counters without constructing or training an agent."""
    cfg = compose_config(
        str(MAIN / "configs"),
        "base_exp12",
        [f"env_name={name}", f"env={group}", f"seed={seed}"],
    )
    np_state, py_state = np.random.get_state(), random.getstate()
    train = evaluation = None
    try:
        np.random.seed(seed)
        random.seed(seed)
        train, evaluation = create_envs(**cfg.env)
        observation, _ = train.reset()
        eval_observation, _ = evaluation.reset()
        assert np.isfinite(observation).all() and np.isfinite(eval_observation).all()
        for _ in range(2):
            train.step(train.action_space.sample())
        snapshot = copy.deepcopy(env_restore_state(train))
        np_saved, py_saved = np.random.get_state(), random.getstate()
        action = train.action_space.sample()
        expected = copy.deepcopy(train.step(action))
        restore_env(train, snapshot)
        np.random.set_state(np_saved)
        random.setstate(py_saved)
        restored_action = train.action_space.sample()
        np.testing.assert_array_equal(action, restored_action)
        actual = train.step(restored_action)
        for field, (a, b) in enumerate(zip(actual[:4], expected[:4])):
            np.testing.assert_array_equal(
                a, b, err_msg=f"{name}: next-transition field {field}"
            )
            assert np.isfinite(a).all()
        return {
            "task": name,
            "seed": seed,
            "status": "PASS",
            "observation_shape": list(observation.shape),
            "action_shape": list(action.shape),
            "raw_budget": int(cfg.num_env_steps),
            "action_repeat": int(cfg.action_repeat),
            "next_transition_restore": "exact",
            "action_rng_restore": "exact",
        }
    finally:
        for env in (train, evaluation):
            if env is not None:
                env.close()
        np.random.set_state(np_state)
        random.setstate(py_state)


def main():
    assert jax.default_backend() == "cpu"
    rows = []
    for name, group, seed in (
        (name, group, seed) for name, group in TASKS for seed in range(1, 6)
    ):
        if (
            group == "humanoid_bench"
            and importlib.util.find_spec("humanoid_bench") is None
        ):
            rows.append(
                {
                    "task": name,
                    "seed": seed,
                    "status": "UNAVAILABLE",
                    "reason": "humanoid_bench absent",
                }
            )
            continue
        try:
            row = verify_task(name, group, seed)
        except Exception as error:
            row = {"task": name, "seed": seed, "status": "FAIL", "error": repr(error)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(
        json.dumps(
            {
                "label": "CPU simulator checks only; no training, CUDA or scientific qualification",
                "passed": sum(row["status"] == "PASS" for row in rows),
                "tasks": rows,
            },
            indent=2,
        )
    )
    return int(any(row["status"] == "FAIL" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
