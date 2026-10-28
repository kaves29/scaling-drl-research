"""Dedicated alternative-only runner; the default delegates to exp1 unchanged.

No registration, class monkeypatching, automatic launch, or approved config edit.
Configuration and teacher choices are explicit caller inputs, never pilot defaults.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

import jax
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

from . import sequential_probe as probe
from .sequential_hooks import Observer


def run(args, alternative=None):
    # Keep the existing default entrypoint and its RunProbes lifecycle intact.
    from experiments import exp1

    if not alternative or not alternative.get("enabled", False):
        return exp1.run(args)

    from experiments.exp12.precision import (
        configure_compilation_cache,
        set_matmul_precision,
    )
    from experiments.exp12.state import latest_state_dir
    from experiments.exp12.trainer import Exp12Trainer
    from scale_rl.agents.sac.sac_network import SACCritic

    cfg = exp1.compose_config(
        args["config_path"], args["config_name"], args["overrides"]
    )
    # An alternative diagnostic does not supply the approved fork's trigger.
    # Reject instead of silently disabling forks or changing scientific settings.
    if cfg.fork.enabled:
        raise ValueError("alternative-only run requires explicit fork.enabled=false")
    if (
        cfg.agent.critic_use_cdq
        or cfg.agent.critic_sparsity
        or cfg.agent.mixed_precision
    ):
        raise ValueError(
            "alternative-only runner supports a single dense float32 critic"
        )
    if cfg.num_train_envs != 1 or cfg.updates_per_interaction_step < 1:
        raise ValueError(
            "fresh-capture lifecycle requires one train env and at least one update per step"
        )
    if cfg.evaluation_per_interaction_step <= cfg.buffer.min_length:
        raise ValueError("first evaluation checkpoint must follow replay warmup")
    for field in (
        "bank",
        "reference",
        "output",
        "teacher",
        "teacher_architecture",
        "architecture",
        "environment",
        "seed",
        "source",
        "bank_producer",
    ):
        if field not in alternative:
            raise ValueError(f"explicit alternative option required: {field}")
    if not isinstance(alternative["bank_producer"], bool):
        raise ValueError("bank_producer must be an explicit boolean")
    if (
        int(cfg.seed) != alternative["seed"]
        or cfg.env_name != alternative["environment"]
    ):
        raise ValueError("alternative config environment/seed mismatch")
    from scale_rl.common.logger import get_architecture_id

    if get_architecture_id(cfg) != alternative["architecture"]:
        raise ValueError("alternative architecture label differs from actual config")
    probe.Settings(
        steps=int(cfg.probe.steps),
        batch_size=int(cfg.probe.batch_size),
        chunk=int(cfg.probe.eval_chunk),
    ).validate(int(cfg.probe.pool_size))
    if cfg.evaluation_per_interaction_step <= 0 or args["checkpoint_interval"] <= 0:
        raise ValueError("positive explicit evaluation/checkpoint cadence required")
    teacher_config = alternative["teacher"]
    if (
        set(teacher_config) != {"block_type", "num_blocks", "hidden_dim", "dtype"}
        or teacher_config["dtype"] != "float32"
    ):
        raise ValueError(
            "explicit fixed teacher architecture and float32 dtype required"
        )
    teacher = SACCritic(**{**teacher_config, "dtype": np.float32})
    root = Path(args["checkpoint_dir"])
    if not root.is_absolute():
        raise ValueError("absolute checkpoint_dir required")
    for field in ("bank", "reference", "output"):
        if not Path(alternative[field]).is_absolute():
            raise ValueError(f"absolute {field} required")
    if (root / "ALTERNATIVE_DONE.json").exists():
        raise ValueError("alternative run already completed")
    set_matmul_precision()
    configure_compilation_cache()
    contract = probe.precision_contract(teacher)
    identity = {
        "mode": "sequential_alternative_only_v1",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "alternative": alternative,
        "execution": contract,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip(),
        "implementation": {
            p.name: probe.digest(p.read_bytes())
            for p in (
                Path(__file__),
                Path(probe.__file__),
                Path(__file__).with_name("sequential_hooks.py"),
            )
        },
        "runtime_source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): probe.digest(
                p.read_bytes()
            )
            for directory in ("experiments", "scale_rl", "utils", "diagnostics")
            for p in sorted(
                (Path(__file__).resolve().parents[1] / directory).rglob("*.py")
            )
        },
    }
    identity_dir = root / "alternative_identity"
    if identity_dir.exists():
        if probe.verify(identity_dir)["identity"] != identity:
            raise ValueError("alternative resume identity differs")
    else:
        if latest_state_dir(root / "state") is not None:
            raise ValueError("cannot adopt an existing non-alternative checkpoint")
        probe.commit_directory(
            identity_dir, lambda tmp: probe.seal(tmp, {"identity": identity})
        )
    observer = Observer(**{**alternative, "teacher": teacher})
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    trainer = Exp12Trainer(cfg, str(root))
    state_root = root / "state"
    latest = latest_state_dir(state_root)
    fresh_path = Path(alternative["reference"]) / "fresh.msgpack"

    def capture(t):
        # Exact current parameters, called by the real trainer before its first
        # replay sampling/update. No training RNG split or RunProbes object.
        if observer.fresh is None:
            fresh = jax.tree_util.tree_map(
                lambda x: x.copy(), t._sac_agent.critic.params
            )
            observer.initialize(t, fresh)
        else:
            observer.observe(t)  # verified phase resume after warmup replay

    def after_step(t):
        due = t.interaction_step % int(cfg.evaluation_per_interaction_step) == 0
        periodic = t.interaction_step % int(args["checkpoint_interval"]) == 0
        # Save the last pre-update warmup boundary. If capture/fit is interrupted,
        # restore the preceding complete training step and replay it exactly.
        pre_capture = t.update_step == 0 and (
            t.buffer._num_in_buffer
            < cfg.buffer.min_length
            <= t.buffer._num_in_buffer + cfg.num_train_envs
        )
        if pre_capture or due or periodic:
            t.save(state_root)
        if due and observer.fresh is not None:
            observer.observe(t)

    succeeded = False
    try:
        if latest is None:
            if fresh_path.exists():
                raise ValueError(
                    "reference exists without recoverable training checkpoint"
                )
            trainer.start()
            trainer.save(state_root)
        else:
            trainer.restore(latest)
            if fresh_path.exists():
                probe.verify(alternative["reference"])
                fresh = serialization.from_bytes(
                    trainer._sac_agent.critic.params, fresh_path.read_bytes()
                )
                observer.restore(trainer, fresh)
            elif trainer.update_step:
                raise ValueError(
                    "trained checkpoint missing exact saved fresh reference"
                )
            if (
                trainer.update_step
                and trainer.interaction_step % int(cfg.evaluation_per_interaction_step)
                == 0
            ):
                # The checkpoint was saved before the diagnostic. A committed
                # phase is verified/reused; an unfinished phase is recomputed.
                observer.observe(trainer)
        trainer.train(
            int(cfg.num_interaction_steps),
            after_step=after_step,
            before_first_update=capture,
        )
        if observer.fresh is None:
            raise ValueError("run ended before fresh capture; alternative incomplete")
        trainer.save(state_root)
        probe.write_json(
            root / "ALTERNATIVE_DONE.json",
            {
                "interaction_step": trainer.interaction_step,
                "update_step": trainer.update_step,
                "mode": "alternative_only",
                "scientific_qualification": False,
                "GPU_qualified": False,
            },
        )
        succeeded = True
        return root
    finally:
        trainer.close()
        # The trainer's logger is still the ordinary training logger.
        import wandb

        if wandb.run is not None:
            wandb.finish(exit_code=0 if succeeded else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request",
        required=True,
        help="Explicit args/alternative JSON; no pilot defaults",
    )
    request = json.loads(Path(parser.parse_args().request).read_text())
    run(request["args"], request.get("alternative"))


if __name__ == "__main__":
    main()
