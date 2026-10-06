"""Experiment 2 arm job: restores the fork state of an Exp 1 run and continues as one arm.

    python run.py --experiment exp2_arm --config_name base_exp12 --overrides <the parent's overrides> \\
        --overrides fork.source=/abs/parent_run_dir --overrides fork.arm=injected \\
        --overrides injection.m=<frozen m> --checkpoint_dir /abs/arm_dir

`injected` applies plasticity injection at the fork, `identity` does not (fork-fidelity
validation only). The CONTROL arm is the parent process itself (experiments/exp1.py).
Both go through the same restore path. Resumable like exp1, and refuses to start
unless the parent's fork is complete and its config equals the parent's.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotmap import DotMap

from experiments.angle_1 import DONE_MARKER
from experiments.exp1 import compose_config, record_metadata, run_identity
from experiments.exp12 import exp2_ledger, fork
from experiments.exp12.fork import Check1Failed
from experiments.exp12.injection import M_LABELS
from experiments.exp12.precision import configure_compilation_cache, set_matmul_precision
from experiments.exp12.probe import iqm
from experiments.exp12.run_probes import RunProbes, run_probe_networks
from experiments.exp12.state import latest_state_dir
from experiments.exp12.trainer import Exp12Trainer
from experiments.exp12.trigger import bootstrap_interval, trigger_config
from experiments.registry import register_experiment
from utils.atomic_io import atomic_write_text
from utils.paths import require_absolute
from utils.run_metadata import RUN_METADATA_FILENAME, differing_keys, load_run_metadata
from utils.run_metadata import _strip_locations as strip_locations

ARM_KEYS = {"fork.source", "fork.arm", "injection.m"}


def check_matches_parent(cfg, source: Path) -> None:
    parent = load_run_metadata(source / RUN_METADATA_FILENAME)
    if parent is None:
        raise ValueError(f"{source} has no {RUN_METADATA_FILENAME}")
    import omegaconf

    mine = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    diff = set(differing_keys(strip_locations(parent["resolved_config"]), strip_locations(mine))) - ARM_KEYS
    if diff:
        raise ValueError(f"arm config differs from the parent run's: {sorted(diff)}")


def check2(trainer, probes, plan, pre_params) -> dict:
    """Probe copies of the injected and the control critic at the fork, paired with the fresh reference."""
    critics = {"injected": probes.current_critic(trainer), "control": (probes.critic_def, pre_params),
               "fresh": (probes.critic_def, probes.fresh)}
    k = plan["fork_check_index"]
    result = run_probe_networks(trainer.agent, trainer.buffer, probes.critic_def, critics, probes.tx,
                                probes.injected_tx, int(trainer.cfg.seed), k, probes.cfg)
    tc = trigger_config(trainer.cfg)
    diff = result["injected"]["score"] - result["control"]["score"]
    low, high = bootstrap_interval(diff, int(trainer.cfg.seed), k, tc.resamples, tc.confidence)
    out = {
        "check_index": k,
        **{f"score_{n}_rounds": result[n]["score"].tolist() for n in critics},
        **{f"score_{n}_iqm": iqm(result[n]["score"]) for n in critics},
        "loss_injected_rounds": (result["fresh"]["score"] - result["injected"]["score"]).tolist(),
        "loss_control_rounds": (result["fresh"]["score"] - result["control"]["score"]).tolist(),
        # Paired difference P(injected) - P(control) per round, its IQM and percentile bootstrap
        # interval (over 5 rounds only; see the Methodology limitations).
        "paired_difference_rounds": diff.tolist(), "paired_difference_iqm": iqm(diff),
        "paired_difference_ci_low": low, "paired_difference_ci_high": high,
        "confidence": tc.confidence, "bootstrap_resamples": tc.resamples,
        # Check 2 passes when the interval lies above 0 (approved 2026-10-04). Reported only.
        "pass": bool(np.isfinite(low) and low > 0),
        # Twin critics (amendment (z)): each network's scores; the paired difference above uses their mean.
        **{f"score_{n}_rounds": result[n]["score"].tolist() for n in result if n not in critics},
    }
    probes.dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(probes.dir / "check2_curves.npz",
                        **{f"{n}_{f}": v for n, d in result.items() for f, v in d.items()})
    return out


@register_experiment("exp2_arm")
def run(args: dict) -> None:
    set_matmul_precision()
    configure_compilation_cache()
    args = DotMap(args)
    arm_dir = Path(require_absolute(args.checkpoint_dir or "", "checkpoint_dir"))
    if (arm_dir / DONE_MARKER).exists():
        raise ValueError(f"{arm_dir} already holds a completed arm ({DONE_MARKER} exists).")
    cfg = compose_config(args.config_path, args.config_name, args.overrides)
    arm = cfg.fork.arm
    if arm not in ("injected", "identity"):
        raise ValueError("fork.arm must be 'injected' or 'identity' (the control arm is the Exp 1 process)")
    if arm == "injected" and cfg.injection.m not in M_LABELS:
        raise ValueError(f"injection.m must be one of {M_LABELS} for the injected arm")
    source = Path(require_absolute(str(cfg.fork.source), "fork.source"))
    if not fork.is_ready(source):
        raise ValueError(f"{source} has no complete fork ({fork.READY} missing)")
    check_matches_parent(cfg, source)
    fork.check_validation_flags(cfg)
    fork.check_same_device(source)
    plan = fork.read_fork(source)
    fork.validate_control_restore(source, cfg.results_root)
    run_key = run_identity(cfg).run_key
    arm_dir.mkdir(parents=True, exist_ok=True)
    state_root = arm_dir / "state"
    latest = latest_state_dir(state_root)
    record_metadata(cfg, arm_dir, resumed=latest is not None, experiment="exp2_arm")

    trainer = Exp12Trainer(cfg, str(arm_dir))
    fresh_dir = str(source / "fresh_critic")
    if latest is not None:
        trainer.restore(latest)
        probes = RunProbes(trainer, str(arm_dir), fresh_dir=fresh_dir)
    else:
        t0 = time.perf_counter()
        trainer.restore(latest_state_dir(fork.fork_dir(source) / "state"), new_wandb_run=True)
        print(f"[exp2_arm] fork state restored in {time.perf_counter() - t0:.1f} s", flush=True)
        probes = RunProbes(trainer, str(arm_dir), fresh_dir=fresh_dir)
        panel = fork.load_npz(fork.fork_dir(source) / "panel.npz")
        pre_params = trainer._sac_agent.critic.params
        pre = fork.panel_q_and_grad(trainer._sac_agent.critic, panel)
        if arm == "injected":
            trainer.inject(cfg.injection.m, int(cfg.seed))
        after = fork.panel_q_and_grad(trainer._sac_agent.critic, panel)
        control = fork.load_npz(fork.fork_dir(source) / "check1_control.npz")
        result = fork.check1(pre, after, control, float(cfg.checks.check1_tolerance_eps),
                             injected=arm == "injected")
        fork.save_npz(arm_dir / "check1_after.npz", after)
        exp2_ledger.write_json(run_key, f"check1_{arm}.json", result, cfg.results_root)
        if not result["pass"]:
            raise Check1Failed(f"Check 1 failed for the {arm} arm of {run_key} (max {result['max_eps_units']:.3g} "
                               "eps units); stop and ask the "
                               "project lead. Details in check1_" + arm + ".json")
        if arm == "injected":
            exp2_ledger.write_json(run_key, "check2.json", check2(trainer, probes, plan, pre_params),
                                   cfg.results_root)
        trainer.extra_state["fork"] = plan
        trainer.extra_state["arm"] = arm
        trainer.extra_state["post_fork_evals"] = fork.post_fork_eval(trainer, plan, 0, arm)
    probes.extend_to(plan["arm_end_step"])

    def save(t, status="running"):
        t.save(state_root)
        probes.write_csv()
        exp2_ledger.write_arm(run_key, arm, plan, probes.records, t.extra_state["post_fork_evals"],
                              t.metrics_rows, int(cfg.action_repeat), cfg.results_root)

    def after_step(t):
        probes.maybe_check(t)
        k = fork.eval_due(plan, t.interaction_step)
        if k:
            t.extra_state["post_fork_evals"] += fork.post_fork_eval(t, plan, k, arm)
        snapshot = cfg.fork.identity_snapshot_steps
        if snapshot is not None and t.interaction_step == plan["fork_step"] + int(snapshot):
            t.save(arm_dir / f"identity_{arm}")  # validation only
            if cfg.testing.stop_after_identity_snapshot:
                raise fork.ValidationDone
        if t.interaction_step % args.checkpoint_interval == 0 and t.interaction_step < plan["arm_end_step"]:
            save(t)

    if latest is None:
        save(trainer)
    try:
        trainer.train(plan["arm_end_step"], after_step=after_step)
    except fork.ValidationDone:
        print("[exp2_arm] identity snapshot saved; stopping (testing.stop_after_identity_snapshot)")
        trainer.close()
        return
    save(trainer, status="complete")
    atomic_write_text(arm_dir / DONE_MARKER, json.dumps({
        "arm": arm, "interaction_step": trainer.interaction_step,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }))
    trainer.close()
