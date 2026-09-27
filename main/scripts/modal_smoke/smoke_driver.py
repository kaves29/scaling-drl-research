"""Minimal real end-to-end smoke test of Angle 2A/2B/2C (runs inside the Modal container).

Every stage calls the angle's real, unmodified entry point. The only
substitutions are the ones already used by this repo's own real-entry-point
tests: the 10-agent baseline-calibration pool lookup is patched to a 4-agent
list (C(4,2)=6 pairs), and 2A/2C read synthetic onset rows from
results/smoke_ledgers (never results/ledgers).

Usage: python scripts/modal_smoke/smoke_driver.py <stage> [--scale smoke|dryrun]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

from hydra.core.global_hydra import GlobalHydra

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SMOKE = RESULTS / "smoke"
SMOKE_LEDGER_ROOT = "results/smoke_ledgers"
REAL_LEDGER_ROOT = RESULTS / "ledgers"

ENVIRONMENT = "cheetah-run"
ENV_GROUP = "dmc_medium"
SEED = 1
POOL_SEEDS = (1, 2, 3, 4)
PROJECT = "EchoCritic-smoke"
REFERENCE = (2, 512)
SCALED_A = (3, 512)
SCALED_B = (5, 768)

SCALES = {
    # onset/pool steps are interaction steps (action_repeat=2 -> env steps = 2x).
    "smoke": dict(onset_step=6000, pool_steps=6000, min_length=1000, probes=10, mc_rollouts=5, states_2b=40,
                  ckpt_interval=1000),
    "dryrun": dict(onset_step=60, pool_steps=60, min_length=20, probes=3, mc_rollouts=2, states_2b=5,
                   ckpt_interval=20),
}


def _label(arch):
    return f"D{arch[0]}W{arch[1]}"


def _chdir_repo():
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def _stamp(stage, **extra):
    SMOKE.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, "finished_at": time.time(), **extra}
    with open(SMOKE / "stages.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def _pool_identities():
    return [type("Ident", (), {"seed": s, "architecture": _label(REFERENCE)})() for s in POOL_SEEDS]


def stage_preflight(scale):
    ckpt = SMOKE / "preflight_ckpt"
    cmd = [
        sys.executable, "scripts/preflight_checkpoint_check.py", "--checkpoint_dir", str(ckpt),
        "--override", f"critic_num_blocks={REFERENCE[0]}", "--override", f"critic_hidden_dim={REFERENCE[1]}",
        "--override", "updates_per_interaction_step=5",
        "--override", f"env_name={ENVIRONMENT}", "--override", f"env={ENV_GROUP}",
        "--override", "seed=999", "--override", f"project_name={PROJECT}",
    ]
    if scale == "dryrun":
        cmd += ["--num_env_steps", "200", "--checkpoint_interval", "40"]
    rc = subprocess.call(cmd, cwd=REPO)
    _stamp("preflight", returncode=rc)
    if rc != 0:
        raise SystemExit(f"preflight FAILED (rc={rc}); stopping before anything else runs.")


def stage_pool(scale):
    """4 real D2W512 pool agents via Angle 1's own save_probe_capture_snapshot path."""
    cfg = SCALES[scale]
    for seed in POOL_SEEDS:
        ckpt = SMOKE / "pool_ckpt" / f"seed{seed}"
        cmd = [
            sys.executable, "-u", "run.py", "--experiment", "angle_1", "--config_name", "base_sac",
            "--overrides", f"env={ENV_GROUP}", "--overrides", f"env_name={ENVIRONMENT}",
            "--overrides", f"seed={seed}", "--overrides", f"project_name={PROJECT}",
            "--overrides", f"critic_num_blocks={REFERENCE[0]}", "--overrides", f"critic_hidden_dim={REFERENCE[1]}",
            "--overrides", f"num_env_steps={cfg['pool_steps'] * 2}",
            "--overrides", f"buffer.min_length={cfg['min_length']}",
            "--overrides", "+save_probe_capture_snapshot=true",
            "--checkpoint_dir", str(ckpt),
            "--checkpoint_interval", str(cfg["ckpt_interval"]),
        ]
        print(f"[smoke] pool seed={seed}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=REPO)
        if rc != 0:
            raise SystemExit(f"pool agent seed={seed} FAILED (rc={rc})")
    _stamp("pool")


def stage_ledger(scale):
    """Synthetic onset rows, written through the real ledger writer, into smoke_ledgers/ only."""
    from utils.onset_ledger import WandbIdentity, log_onset_event

    onset_step = SCALES[scale]["onset_step"]
    for arch in (SCALED_A, SCALED_B):
        label = _label(arch)
        run_obj = type("SmokeRun", (), {"name": f"SMOKE-SYNTHETIC-{label}-{SEED}", "id": "smoke-synthetic"})()
        row = {
            "run_key": f"angle1_{label}_{ENVIRONMENT}_seed{SEED}",
            "exact_run_name": run_obj.name,
            "wandb_run_id": run_obj.id,
            "architecture": label,
            "environment": ENVIRONMENT,
            "seed": SEED,
            "critic_degradation_onset_step": onset_step,
            "critic_degradation_method": "td_variance_p95_sustained_v1",
            "propagation_onset_step": None,
            "propagation_method": None,
            "propagation_lag": None,
            "status": "success",
            "detection_notes": "SYNTHETIC SMOKE-TEST FIXTURE - not a detected onset; never use for results",
        }
        log_onset_event(
            exp_name="angle_1", architecture=label, row=row,
            identity=WandbIdentity(run_obj=run_obj), root=SMOKE_LEDGER_ROOT, mirror_to_wandb=False,
        )
    if REAL_LEDGER_ROOT.exists():
        raise SystemExit(f"{REAL_LEDGER_ROOT} exists on this volume - synthetic rows must never go there.")
    _stamp("ledger", onset_step=onset_step)


def stage_angle2a(scale):
    from analysis.baseline_calibration_pool import POOL_STORAGE_ROOT
    from experiments.angle_2_a import run

    cfg = SCALES[scale]
    assert Path(POOL_STORAGE_ROOT).resolve().is_relative_to(RESULTS.resolve()), POOL_STORAGE_ROOT
    with mock.patch(
        "experiments.angle_2a.pool_null_baseline.get_baseline_calibration_pool", return_value=_pool_identities(),
    ):
        run({
            "config_path": str(REPO / "configs"),
            "config_name": "base_angle2a",
            "overrides": [
                f"env={ENV_GROUP}", f"env_name={ENVIRONMENT}", f"seed={SEED}", f"project_name={PROJECT}",
                f"angle_2_a.scaled_a.critic_num_blocks={SCALED_A[0]}",
                f"angle_2_a.scaled_a.critic_hidden_dim={SCALED_A[1]}",
                f"angle_2_a.scaled_b.critic_num_blocks={SCALED_B[0]}",
                f"angle_2_a.scaled_b.critic_hidden_dim={SCALED_B[1]}",
                f"angle_2_a.reference.critic_num_blocks={REFERENCE[0]}",
                f"angle_2_a.reference.critic_hidden_dim={REFERENCE[1]}",
                f"buffer.min_length={cfg['min_length']}",
                f"angle_2_a.num_probes_per_source={cfg['probes']}",
                "angle_2_a.r_calibration.enabled=false",
                f"angle_2_a.num_mc_rollouts={cfg['mc_rollouts']}",
                "angle_2_a.prereq_check.enabled=false",
                "angle_2_a.run_null_baseline=true",
                f"angle_2_a.onset_ledger_root={SMOKE_LEDGER_ROOT}",
            ],
            "checkpoint_dir": str(SMOKE / "angle2a_ckpt"),
            "checkpoint_interval": cfg["ckpt_interval"],
        })
    _stamp("angle2a")


def stage_angle2b(scale):
    from experiments.angle_2_b import run

    with mock.patch(
        "experiments.angle_2b.null_baseline.get_baseline_calibration_pool", return_value=_pool_identities(),
    ):
        run({
            "config_path": str(REPO / "configs"),
            "config_name": "base_angle2b",
            "overrides": [
                f"env_name={ENVIRONMENT}", f"seed={SEED}", f"project_name={PROJECT}",
                "angle_2_b.matchup_names=[matchup_1,matchup_2]",
                f"angle_2_b.num_states_per_source={SCALES[scale]['states_2b']}",
            ],
        })
    _stamp("angle2b")


def stage_angle2c(scale):
    from experiments.angle_2_c import run

    run({
        "config_path": str(REPO / "configs"),
        "config_name": "base_angle2c",
        "overrides": [
            f"env_name={ENVIRONMENT}", f"seed={SEED}", f"project_name={PROJECT}",
            "angle_2_c.matchup_names=[matchup_1,matchup_2]",
            f"angle_2_c.onset_ledger_root={SMOKE_LEDGER_ROOT}",
        ],
    })
    _stamp("angle2c")


def stage_inspect(scale):
    from scripts.modal_smoke.inspect_outputs import inspect_all

    report = inspect_all(RESULTS, ENVIRONMENT, SEED)
    out = SMOKE / "inspection_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"[smoke] inspection report -> {out}")


STAGES = {
    "preflight": stage_preflight,
    "pool": stage_pool,
    "ledger": stage_ledger,
    "angle2a": stage_angle2a,
    "angle2b": stage_angle2b,
    "angle2c": stage_angle2c,
    "inspect": stage_inspect,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=list(STAGES))
    parser.add_argument("--scale", default="smoke", choices=list(SCALES))
    args = parser.parse_args()
    if args.scale == "dryrun":
        os.environ.setdefault("WANDB_MODE", "disabled")
    _chdir_repo()
    for stage in args.stages:
        t0 = time.time()
        print(f"\n[smoke] ===== stage {stage} (scale={args.scale}) =====", flush=True)
        GlobalHydra.instance().clear()
        STAGES[stage](args.scale)
        print(f"[smoke] ===== stage {stage} done in {time.time() - t0:.1f}s =====", flush=True)


if __name__ == "__main__":
    main()
