#!/usr/bin/env python3
"""CPU source/configuration provenance and default-training compatibility audit."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import types
from unittest import mock

REVISIONS = {
    "pilot_source": "f8619dd5f494399d9febc79fc9092010234c714d",
    "exp96_source": "f597ee9898e4d401912092c336c3e5330cf2fda1",
    "pr8_source": "868cab4346f2f3124d1a459e34a2d66fed28c794",
    "batch1_base": "c05d77cc9899e50e857c170faf9937e5635c3663",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(
        JAX_PLATFORMS="cpu",
        JAX_PLATFORM_NAME="cpu",
        JAX_ENABLE_X64="false",
        EXP12_JAX_CACHE_DIR="off",
        WANDB_MODE="disabled",
    )
    sys.path.insert(0, str(args.repository / "main"))
    import jax
    import numpy as np
    import gymnasium as gym
    import optax
    from omegaconf import OmegaConf
    from experiments.exp1 import compose_config
    from experiments.exp12.fixed_task import tree_sha256
    from experiments.exp12.precision import runtime_info
    from scale_rl.agents.sac.sac_agent import SACAgent
    from scale_rl.buffers.numpy_buffer import NpyUniformBuffer

    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise RuntimeError("CPU FP32 audit only")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "PARTIAL",
        "source": subprocess.check_output(
            ["git", "-C", str(args.repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime": runtime_info(),
        "reconstructed_configs_not_runtime_records": True,
        "sources": {},
        "default_update_comparisons": [],
    }

    def save():
        (args.out / "receipt.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    save()

    def blob(revision, name):
        return subprocess.check_output(
            ["git", "-C", str(args.repository), "show", revision + ":" + name]
        )

    observation_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    old_module = None
    for label, revision in REVISIONS.items():
        directory = args.out / label
        directory.mkdir()
        archive = subprocess.check_output(
            ["git", "-C", str(args.repository), "archive", revision, "main/configs"]
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(directory, filter="data")
        cfg = compose_config(
            str((directory / "main/configs").resolve()),
            "base_exp12" if label == "pilot_source" else "base_exp12_ten",
            [],
        )
        report["sources"][label] = {
            "revision": revision,
            "agent": OmegaConf.to_container(cfg.agent, resolve=True),
            "training_source_sha256": hashlib.sha256(
                blob(revision, "main/scale_rl/agents/sac/sac_agent.py")
            ).hexdigest(),
            "generic_sac_yaml_sha256": hashlib.sha256(
                blob(revision, "main/configs/agent/sac.yaml")
            ).hexdigest(),
        }
        # Execute each actual historical constructor with resized fixture geometry.
        module = types.ModuleType("batch1_verified_" + label)
        module.__file__ = str(directory / "sac_agent.py")
        sys.modules[module.__name__] = module
        code = blob(revision, "main/scale_rl/agents/sac/sac_agent.py")
        Path(module.__file__).write_bytes(code)
        exec(compile(code, module.__file__, "exec"), module.__dict__)
        small = OmegaConf.to_container(cfg.agent, resolve=True)
        small.pop("agent_type")
        small.update(
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            actor_num_blocks=1,
            critic_num_blocks=1,
            normalize_observation=False,
            seed=7,
            critic_use_cdq=False,
        )
        with mock.patch.object(optax, "adamw", wraps=optax.adamw) as factory:
            historical = module.SACAgent(observation_space, action_space, dict(small))
        report["sources"][label]["resized_actual_adamw_calls"] = [
            call.kwargs for call in factory.call_args_list
        ]
        if label == "batch1_base":
            old_module = module
        del historical
        save()
    # The update dependencies are byte-identical to the completed previous task.
    dependencies = (
        "scale_rl/agents/sac/sac_update.py",
        "scale_rl/agents/sac/sac_network.py",
        "scale_rl/networks/trainer.py",
        "scale_rl/networks/policies.py",
        "scale_rl/networks/layers.py",
        "scale_rl/agents/wrappers/normalization.py",
    )
    report["unchanged_update_dependency_sha256"] = {}
    for name in dependencies:
        previous = blob(REVISIONS["batch1_base"], "main/" + name)
        if previous != (args.repository / "main" / name).read_bytes() or any(
            blob(revision, "main/" + name) != previous
            for revision in REVISIONS.values()
        ):
            raise ValueError("default comparison dependency changed: " + name)
        report["unchanged_update_dependency_sha256"][name] = hashlib.sha256(
            previous
        ).hexdigest()
    for twin in (False, True):
        small.update(critic_use_cdq=twin)
        old = old_module.SACAgent(observation_space, action_space, dict(small))
        new = SACAgent(observation_space, action_space, dict(small))
        fields = (
            "_rng",
            "_actor",
            "_critic",
            "_target_critic",
            "_temperature",
            "churn_ref_batch",
        )
        before = {field: tree_sha256(getattr(old, field)) for field in fields}
        if before != {field: tree_sha256(getattr(new, field)) for field in fields}:
            raise RuntimeError("default initialization changed")
        rng = np.random.default_rng(918)
        batches = {
            "observation": rng.normal(size=(4, 8, 4)).astype(np.float32),
            "next_observation": rng.normal(size=(4, 8, 4)).astype(np.float32),
            "action": rng.uniform(-1, 1, size=(4, 8, 2)).astype(np.float32),
            "reward": rng.normal(size=(4, 8)).astype(np.float32),
            "terminated": np.zeros((4, 8), np.float32),
            "truncated": np.zeros((4, 8), np.float32),
        }
        batches["terminated"][0, 0] = 1
        batches["truncated"][0, 1] = 1
        old_info = old.update_many(11, {k: v.copy() for k, v in batches.items()}, 3)
        new_info = new.update_many(11, {k: v.copy() for k, v in batches.items()}, 3)
        for field in fields:
            if tree_sha256(getattr(old, field)) != tree_sha256(getattr(new, field)):
                raise RuntimeError("default SAC update changed: " + field)
        for field in old_info:
            np.testing.assert_array_equal(old_info[field], new_info[field])
        report["default_update_comparisons"].append(
            {
                "twin": twin,
                "updates": 4,
                "first_update_step": 11,
                "initial_state_bitwise_equal": True,
                "final_state_bitwise_equal": True,
                "all_metrics_equal_nan_aware": True,
                "initial_hashes": before,
                "final_hashes": {
                    field: tree_sha256(getattr(new, field)) for field in fields
                },
            }
        )
        save()
    # Demonstrate the old unsupported multi-step truncation problem explicitly.
    legacy = types.ModuleType("batch1_legacy_replay")
    legacy.__file__ = "<immutable-base-replay>"
    exec(
        compile(
            blob(REVISIONS["batch1_base"], "main/scale_rl/buffers/numpy_buffer.py"),
            legacy.__file__,
            "exec",
        ),
        legacy.__dict__,
    )
    space = gym.spaces.Box(-100.0, 100.0, (1, 1), np.float32)
    replay = legacy.NpyUniformBuffer(space, space, 3, 0.5, 8, 1, 1, 1)
    replay.reset()
    for i in range(3):
        replay.add(
            dict(
                observation=np.array([[i]], np.float32),
                action=np.zeros((1, 1), np.float32),
                reward=np.array([1.0 if i == 0 else 7.0], np.float32),
                terminated=np.array([False]),
                truncated=np.array([i == 0]),
                next_observation=np.array([[10.0 if i == 0 else 1.0]], np.float32),
            )
        )
    old_target = float(replay._rewards[0] + 0.5**3 * replay._next_observations[0, 0])
    expected = 1.0 + 0.5 * 10.0
    assert old_target == 2.25 and expected == 6.0
    try:
        NpyUniformBuffer(space, space, 3, 0.5, 8, 1, 1, 1)
    except ValueError:
        report["unsupported_multistep"] = {
            "immutable_base_target": old_target,
            "one_step_truncated_prefix_target": expected,
            "current_constructor_rejected": True,
        }
    else:
        raise RuntimeError("unsupported multi-step configuration was accepted")
    report["status"] = "COMPLETE"
    report["qualification"] = (
        "resized CPU fixtures only; no GPU qualification or pilot-result interpretation"
    )
    save()
    print(
        json.dumps(
            {
                "status": report["status"],
                "source": report["source"],
                "out": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
