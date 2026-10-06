#!/usr/bin/env python3
"""Positive control and m-selection (Methodology "Positive control", amendments (c)-(e), (q)).

Reads a finished-or-forked DEVELOPMENT Exp 1 run (run_role=dev, seed outside 1-5,
D6W1536 on dog-run) and never trains. At the run's own fork state (its f*_run,
taken with the unchanged Exp 1 trigger) it probes, on one shared pool, target and
minibatch order (the trigger check's own probe streams):
    fresh       the run's stored untrained critic (steps 1-2: the healthy reference, L = 0)
    degraded    the critic at f*_run (step 3)
    injected_m  the degraded critic after plasticity injection, m in last/half/all (step 4)
recovery(m) = (L_trigger - L_injected(m)) / L_trigger and the m rule of amendment (d).
The probe noise is the pooled SD of the per-round L of these four critics, divided by
L_trigger. The one-time shared-offset check (amendment (c)) repeats the fresh/degraded
probe with one offset (the degraded critic's mean, the fresh critic's, their average);
it is reported separately and plays no part in the noise or the stop.

Writes <out_dir>/positive_control.json and positive_control_curves.npz. Exit codes:
0 = m chosen (still to be frozen by the project lead in injection.m), 3 = STOP and
consult (no trigger, L_trigger <= 0, or noise >= 0.10 in recovery units). This run
and its results stay out of the confirmatory analyses (ledger.load excludes dev runs).

    python scripts/positive_control.py --run_dir /abs/dev_run --out_dir /abs/positive_control
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.hardware import configure_hardware_env  # noqa: E402

configure_hardware_env()

from experiments.exp12.precision import configure_compilation_cache, set_matmul_precision  # noqa: E402

set_matmul_precision()
configure_compilation_cache()

import numpy as np  # noqa: E402

STOP_EXIT = 3
SHARED_OFFSETS = ("degraded", "fresh", "mean")


def _stub_wandb():
    mock.patch("wandb.init").start()
    mock.patch("wandb.log").start()


def _setting_errors(cfg, allow_any_setting: bool):
    from scale_rl.common.logger import get_architecture_id

    errors = []
    if cfg.run_role != "dev":
        errors.append(f"run_role is {cfg.run_role!r}, not 'dev'")
    if 1 <= int(cfg.seed) <= 5:
        errors.append(f"seed {cfg.seed} is a main-run seed (1-5)")
    pc = cfg.positive_control
    if not allow_any_setting:
        if get_architecture_id(cfg) != pc.architecture:
            errors.append(f"architecture {get_architecture_id(cfg)} is not {pc.architecture}")
        if cfg.env_name != pc.env_name:
            errors.append(f"environment {cfg.env_name} is not {pc.env_name}")
    return errors


def load_fork(cfg, run_dir: Path):
    """Agent, buffer and probe records of the fork state; envs and loggers are not needed."""
    from experiments.exp12 import fork
    from experiments.exp12.state import latest_state_dir, load_buffer, load_meta
    from experiments.exp12.trainer import Exp12Trainer

    state = latest_state_dir(fork.fork_dir(run_dir) / "state")
    trainer = Exp12Trainer(cfg, tempfile.mkdtemp(prefix="positive_control_"))
    trainer.agent.load_checkpoint(str(state))
    load_buffer(trainer.buffer, state)
    return trainer, load_meta(state)["extra_state"]["probe_records"]


def probe_critics(trainer, fresh_params):
    from experiments.exp12.injection import M_LABELS, inject, injection_key
    from experiments.exp12.run_probes import original_critic_def

    a, cfg = trainer._sac_agent, trainer.cfg
    critic_def = original_critic_def(cfg)
    critics = {"fresh": (critic_def, fresh_params), "degraded": (critic_def, a.critic.params)}
    for m in M_LABELS:
        injected, _ = inject(a.critic, a._target_critic, m, injection_key(int(cfg.seed)),
                             float(cfg.agent.critic_learning_rate), float(cfg.agent.critic_weight_decay))
        critics[f"injected_{m}"] = (injected.network_def, injected.params, injected.tx)
    return critic_def, critics


def run(args) -> int:
    import omegaconf
    import orbax.checkpoint

    from experiments.angle_1 import DONE_MARKER
    from experiments.exp12 import fork
    from experiments.exp12.m_selection import evaluate, loss_rounds
    from experiments.exp12.probe import critic_optimizer, iqm, probe_config, run_probe
    from experiments.exp12.run_probes import FRESH_CRITIC_DIR
    from utils.run_metadata import RUN_METADATA_FILENAME, load_run_metadata

    run_dir, out_dir = Path(args.run_dir), Path(args.out_dir)
    if not run_dir.is_absolute() or not out_dir.is_absolute():
        raise SystemExit("--run_dir and --out_dir must be absolute")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_run_metadata(run_dir / RUN_METADATA_FILENAME)
    if meta is None:
        raise SystemExit(f"{run_dir} has no {RUN_METADATA_FILENAME}")
    cfg = omegaconf.OmegaConf.create(meta["resolved_config"])
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir),
        "run_identity": meta.get("identity"), "healthy_reference": "fresh critic (check 0), L_healthy = 0",
        "noise_statistic": "pooled SD of per-round L / L_trigger", "noise_threshold": float(cfg.positive_control.noise_threshold),
        "similar_within": float(cfg.positive_control.similar_within), "allow_any_setting": args.allow_any_setting,
        "status": None, "stop": None, "chosen_m": None,
    }

    def finish(status, stop=None):
        report.update(status=status, stop=stop)
        with open(out_dir / "positive_control.json", "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(json.dumps({k: report[k] for k in ("status", "stop", "chosen_m")}), flush=True)
        return 0 if status == "m_chosen" else STOP_EXIT

    errors = _setting_errors(cfg, args.allow_any_setting)
    if errors:
        raise SystemExit("not a valid positive-control run: " + "; ".join(errors))
    if not fork.is_ready(run_dir):
        finished = (run_dir / DONE_MARKER).exists()
        return finish("stop", "the development run has no fork (it never triggered" +
                      (")" if finished else ", or it is still running)") + "; consult the project lead, the "
                      "trigger is never loosened")

    plan = fork.read_fork(run_dir)
    k = plan["fork_check_index"]
    report["fork"] = plan
    trainer, records = load_fork(cfg, run_dir)
    trigger_record = next(r for r in records if r["check_index"] == k)
    report["trigger_check"] = {"check_index": k, "interaction_step": trigger_record["interaction_step"],
                               "loss_rounds_recorded": loss_rounds(trigger_record).tolist(),
                               "loss_iqm_recorded": trigger_record["loss_iqm"],
                               "forced": bool(cfg.testing.force_trigger_check is not None)}

    fresh = orbax.checkpoint.PyTreeCheckpointer().restore(str(run_dir / FRESH_CRITIC_DIR))["params"]
    critic_def, critics = probe_critics(trainer, fresh)
    tx, pcfg, seed = critic_optimizer(cfg.agent), probe_config(cfg), int(cfg.seed)
    result = run_probe(trainer.agent, trainer.buffer, critic_def, critics, tx, seed, k, pcfg)
    loss = {n: result["fresh"]["score"] - result[n]["score"] for n in critics if n != "fresh"}
    report["probe"] = {n: {"score_rounds": result[n]["score"].tolist(), "score_iqm": iqm(result[n]["score"]),
                           "b_rounds": result[n]["b"].tolist(), "offset_rounds": result[n]["offset"].tolist()}
                       for n in critics}
    report["loss_rounds"] = {n: v.tolist() for n, v in loss.items()}
    report["loss_iqm"] = {n: iqm(v) for n, v in loss.items()}
    # The degraded critic on the trigger check's own streams must reproduce the recorded trigger probe.
    report["trigger_reproduction_max_abs_diff"] = float(np.abs(loss["degraded"] - loss_rounds(trigger_record)).max())
    np.savez_compressed(out_dir / "positive_control_curves.npz",
                        **{f"{n}_{f}": v for n, d in result.items() for f, v in d.items()})

    sensitivity = {}
    for mode in SHARED_OFFSETS:
        shared = run_probe(trainer.agent, trainer.buffer, critic_def,
                           {"fresh": critics["fresh"], "degraded": critics["degraded"]}, tx, seed, k, pcfg,
                           shared_offset=mode)
        l_shared = shared["fresh"]["score"] - shared["degraded"]["score"]
        sensitivity[mode] = {"loss_rounds": l_shared.tolist(), "loss_iqm": iqm(l_shared),
                             "offset_rounds": shared["degraded"]["offset"].tolist(),
                             "sign_agrees": bool(np.sign(iqm(l_shared)) == np.sign(report["loss_iqm"]["degraded"]))}
    report["shared_offset_sensitivity"] = sensitivity

    pc = cfg.positive_control
    selection = evaluate(loss, float(pc.noise_threshold), float(pc.similar_within))
    report["selection"] = selection
    report["chosen_m"] = selection["chosen_m"]
    trainer.close()
    if selection["stop"] is not None:
        return finish("stop", selection["stop"])
    return finish("m_chosen")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--allow_any_setting", action="store_true",
                        help="TEST-ONLY: skip the D6W1536 / dog-run requirement (recorded in the output)")
    args = parser.parse_args(argv)
    _stub_wandb()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
