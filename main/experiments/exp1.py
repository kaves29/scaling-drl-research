"""Experiment 1 entry point (.claude/methodology-exp1-exp2.md).

One run = (critic architecture, environment, seed). Resumes bit-exactly from
the latest complete state in `<checkpoint_dir>/state`.
"""

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import omegaconf
from dotmap import DotMap
from hydra.core.global_hydra import GlobalHydra

from analysis.metrics_store import RunIdentity
from experiments.angle_1 import DONE_MARKER
from experiments.exp12 import exp2_ledger, fork, ledger
from experiments.exp12.precision import configure_compilation_cache, runtime_info, set_matmul_precision
from experiments.exp12.run_probes import RunProbes
from experiments.exp12.state import latest_state_dir, load_meta
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


class ForkNow(Exception):
    """Raised out of the training loop at f*_run once the fork state is on disk."""

    def __init__(self, plan):
        super().__init__(f"fork at interaction step {plan['fork_step']}")
        self.plan = plan


def compose_config(config_path: str, config_name: str, overrides):
    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_path))
    cfg = hydra.compose(config_name=config_name, overrides=list(overrides))
    omegaconf.OmegaConf.register_new_resolver("eval", eval, replace=True)
    omegaconf.OmegaConf.resolve(cfg)
    if "run_role" not in cfg or cfg.run_role not in ("confirmatory", "dev"):
        raise ValueError("Exp 1/2 runs need configs/base_exp12.yaml (run_role confirmatory|dev).")
    return cfg


def run_identity(cfg, experiment: str = EXPERIMENT) -> RunIdentity:
    return RunIdentity(
        experiment=experiment, architecture=get_architecture_id(cfg), environment=cfg.env_name, seed=cfg.seed
    )


def record_metadata(cfg, run_dir: Path, resumed: bool, experiment: str = EXPERIMENT) -> None:
    identity = run_identity(cfg, experiment)
    metadata = build_run_metadata(
        omegaconf.OmegaConf.to_container(cfg, resolve=True),
        identity={**vars(identity), "run_key": identity.run_key, "run_role": cfg.run_role},
        launch={"started_at": datetime.now(timezone.utc).isoformat(), "resumed": resumed,
                "device": fork.device_info(), "runtime": runtime_info()},
    )
    path = run_dir / RUN_METADATA_FILENAME
    stored = load_run_metadata(path)
    if stored is not None:
        check_resume_matches(stored, metadata, where=str(run_dir))
        metadata = record_launch(stored, metadata)
    save_run_metadata(path, metadata)


@register_experiment(EXPERIMENT)
def run(args: dict) -> None:
    set_matmul_precision()
    configure_compilation_cache()
    args = DotMap(args)
    run_dir = Path(require_absolute(args.checkpoint_dir or "", "checkpoint_dir"))
    if (run_dir / DONE_MARKER).exists():
        raise ValueError(f"{run_dir} already holds a completed run ({DONE_MARKER} exists).")
    cfg = compose_config(args.config_path, args.config_name, args.overrides)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    state_root = run_dir / "state"
    latest = latest_state_dir(state_root)
    plan = fork.read_fork(run_dir) if fork.is_ready(run_dir) else None
    if plan is not None:
        fork.check_same_device(run_dir)  # the control continues on the fork's device model
    enter_control_on_restore = False
    if plan is not None and (latest is None or load_meta(latest)["interaction_step"] < plan["fork_step"]):
        latest = latest_state_dir(fork.fork_dir(run_dir) / "state")  # control restarts from the fork state
        enter_control_on_restore = True
    run_dir.mkdir(parents=True, exist_ok=True)
    record_metadata(cfg, run_dir, resumed=latest is not None)

    num_steps = int(cfg.num_interaction_steps)
    checkpoint_start = int(args.checkpoint_start_frac * num_steps)
    identity = run_identity(cfg)
    ledger_identity = {
        "run_key": identity.run_key, "run_role": cfg.run_role, "architecture": identity.architecture,
        "environment": identity.environment, "seed": identity.seed, "budget_env_steps": int(cfg.num_env_steps),
        "num_interaction_steps": num_steps, "num_checks": int(cfg.probe.checks),
        "code_commit": code_version()["commit"],
    }
    forks_here = bool(cfg.fork.enabled) and identity.architecture in list(cfg.fork.architectures)
    fork.check_validation_flags(cfg)
    if forks_here and not cfg.probe.enabled:
        raise ValueError("forking needs the probes (f*_run comes from them)")
    live = {}

    def build(state_dir=None):
        trainer = Exp12Trainer(cfg, str(run_dir))
        if state_dir is not None:
            t0 = time.perf_counter()
            trainer.restore(state_dir)
            print(f"[exp1] restored interaction_step {trainer.interaction_step} from {state_dir} "
                  f"in {time.perf_counter() - t0:.1f} s", flush=True)
        else:
            trainer.start()
        live["trainer"] = trainer
        live["probes"] = RunProbes(trainer, str(run_dir)) if cfg.probe.enabled else None

    def enter_control(plan):
        """The restored process continues as the CONTROL arm of Experiment 2."""
        t, d = live["trainer"], fork.fork_dir(run_dir)
        t.extra_state["fork"] = plan
        fork.save_npz(d / "check1_control.npz",
                      fork.panel_q_and_grad(t._sac_agent.critic, fork.load_npz(d / "panel.npz")))
        t.extra_state["post_fork_evals"] = fork.post_fork_eval(t, plan, 0, "control")
        live["probes"].extend_to(plan["control_end_step"])
        atomic_write_text(d / fork.READY, plan["run_key"])

    def save(t: Exp12Trainer, status: str = "running") -> None:
        t.save(state_root)
        probes = live["probes"]
        if probes is None:
            return
        probes.write_csv()
        plan = t.extra_state.get("fork")
        ledger.write_run(ledger_identity, probes.records, probes.f_star, status, probes.dir,
                         results_root=cfg.results_root, fork_step=plan["fork_step"] if plan else None,
                         metrics_rows=t.metrics_rows, action_repeat=int(cfg.action_repeat))
        if plan is not None:
            exp2_ledger.write_json(identity.run_key, "fork.json", plan, cfg.results_root)
            exp2_ledger.write_arm(identity.run_key, "control", plan, probes.records,
                                  t.extra_state.get("post_fork_evals", []), t.metrics_rows,
                                  int(cfg.action_repeat), cfg.results_root)

    def after_step(t: Exp12Trainer) -> None:
        step = t.interaction_step
        probes = live["probes"]
        if probes is not None:
            probes.maybe_check(t)
        plan = t.extra_state.get("fork")
        if plan is not None:
            k = fork.eval_due(plan, step)
            if k:
                t.extra_state["post_fork_evals"] += fork.post_fork_eval(t, plan, k, "control")
            snapshot = cfg.fork.identity_snapshot_steps
            if snapshot is not None and step == plan["fork_step"] + int(snapshot):
                t.save(fork.fork_dir(run_dir) / "identity_control")  # validation only
                if cfg.testing.stop_after_identity_snapshot:
                    raise fork.ValidationDone
        elif forks_here and probes.f_star is not None and probes.f_star["interaction_step"] == step:
            plan = fork.fork_plan(cfg, step, probes.f_star["check_index"])
            t0 = time.perf_counter()
            fork.write_fork(t, run_dir, {**plan, "run_key": identity.run_key}, identity.run_key,
                            str(probes.fresh_path))
            print(f"[exp1] fork state saved at interaction_step {step} in {time.perf_counter() - t0:.1f} s", flush=True)
            raise ForkNow(plan)
        last = t.extra_state["fork"]["control_end_step"] if t.extra_state.get("fork") else num_steps
        if step >= checkpoint_start and step % args.checkpoint_interval == 0 and step < last:
            save(t)

    build(latest)
    if enter_control_on_restore:
        enter_control({**plan})
    while True:
        trainer = live["trainer"]
        plan = trainer.extra_state.get("fork")
        last_step = plan["control_end_step"] if plan else num_steps
        try:
            trainer.train(last_step, after_step=after_step,
                          before_first_update=live["probes"].capture_fresh if live["probes"] is not None else None)
            break
        except ForkNow as signal:
            trainer.close()
            build(latest_state_dir(fork.fork_dir(run_dir) / "state"))
            enter_control({**signal.plan, "run_key": identity.run_key})
        except fork.ValidationDone:
            print("[exp1] identity snapshot saved; stopping (testing.stop_after_identity_snapshot)")
            trainer.close()
            return
    trainer = live["trainer"]
    save(trainer, status="complete")
    atomic_write_text(run_dir / DONE_MARKER, json.dumps({
        "interaction_step": trainer.interaction_step,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }))
    trainer.close()
