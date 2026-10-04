"""Experiment 1 entry point (.claude/methodology-exp1-exp2.md).

One run = (critic architecture, environment, seed). Resumes bit-exactly from
the latest complete state in `<checkpoint_dir>/state`.
"""

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import omegaconf
from dotmap import DotMap
from hydra.core.global_hydra import GlobalHydra

from analysis.metrics_store import RunIdentity
from experiments.angle_1 import DONE_MARKER
from experiments.exp12 import ledger
from experiments.exp12.run_probes import RunProbes
from experiments.exp12.state import latest_state_dir
from experiments.exp12.trainer import Exp12Trainer
from experiments.registry import register_experiment
from scale_rl.common.logger import get_architecture_id
from utils.atomic_io import atomic_write_text
from utils.paths import require_absolute
from utils.run_metadata import (
    RUN_METADATA_FILENAME,
    build_run_metadata,
    code_version,
    check_resume_matches,
    load_run_metadata,
    record_launch,
    save_run_metadata,
)

EXPERIMENT = "exp1"


def compose_config(config_path: str, config_name: str, overrides):
    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_path))
    cfg = hydra.compose(config_name=config_name, overrides=list(overrides))
    omegaconf.OmegaConf.register_new_resolver("eval", eval, replace=True)
    omegaconf.OmegaConf.resolve(cfg)
    if "run_role" not in cfg or cfg.run_role not in ("confirmatory", "dev"):
        raise ValueError("Exp 1/2 runs need configs/base_exp12.yaml (run_role confirmatory|dev).")
    return cfg


def run_identity(cfg) -> RunIdentity:
    return RunIdentity(
        experiment=EXPERIMENT, architecture=get_architecture_id(cfg), environment=cfg.env_name, seed=cfg.seed
    )


def record_metadata(cfg, run_dir: Path, resumed: bool) -> None:
    identity = run_identity(cfg)
    metadata = build_run_metadata(
        omegaconf.OmegaConf.to_container(cfg, resolve=True),
        identity={**vars(identity), "run_key": identity.run_key, "run_role": cfg.run_role},
        launch={"started_at": datetime.now(timezone.utc).isoformat(), "resumed": resumed},
    )
    path = run_dir / RUN_METADATA_FILENAME
    stored = load_run_metadata(path)
    if stored is not None:
        check_resume_matches(stored, metadata, where=str(run_dir))
        metadata = record_launch(stored, metadata)
    save_run_metadata(path, metadata)


@register_experiment(EXPERIMENT)
def run(args: dict) -> None:
    args = DotMap(args)
    run_dir = Path(require_absolute(args.checkpoint_dir or "", "checkpoint_dir"))
    if (run_dir / DONE_MARKER).exists():
        raise ValueError(f"{run_dir} already holds a completed run ({DONE_MARKER} exists).")
    cfg = compose_config(args.config_path, args.config_name, args.overrides)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    state_root = run_dir / "state"
    latest = latest_state_dir(state_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    record_metadata(cfg, run_dir, resumed=latest is not None)

    trainer = Exp12Trainer(cfg, str(run_dir))
    if latest is not None:
        trainer.restore(latest)
        print(f"[exp1] resumed from interaction_step {trainer.interaction_step}")
    else:
        trainer.start()

    num_steps = int(cfg.num_interaction_steps)
    checkpoint_start = int(args.checkpoint_start_frac * num_steps)
    probes = RunProbes(trainer, str(run_dir)) if cfg.probe.enabled else None

    identity = run_identity(cfg)
    ledger_identity = {
        "run_key": identity.run_key, "run_role": cfg.run_role, "architecture": identity.architecture,
        "environment": identity.environment, "seed": identity.seed, "budget_env_steps": int(cfg.num_env_steps),
        "num_interaction_steps": num_steps, "num_checks": int(cfg.probe.checks),
        "code_commit": code_version()["commit"],
    }

    def save(t: Exp12Trainer, status: str = "running") -> None:
        t.save(state_root)
        if probes is not None:
            probes.write_csv()
            ledger.write_run(ledger_identity, probes.records, probes.f_star, status, probes.dir,
                             results_root=cfg.results_root)

    def after_step(t: Exp12Trainer) -> None:
        step = t.interaction_step
        if probes is not None:
            probes.maybe_check(t)
        if step >= checkpoint_start and step % args.checkpoint_interval == 0 and step < num_steps:
            save(t)

    trainer.train(
        num_steps, after_step=after_step, before_first_update=probes.capture_fresh if probes is not None else None
    )
    save(trainer, status="complete")
    atomic_write_text(run_dir / DONE_MARKER, json.dumps({
        "interaction_step": num_steps,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }))
    trainer.close()
