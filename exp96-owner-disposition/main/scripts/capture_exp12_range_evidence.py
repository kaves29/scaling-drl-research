"""Retain existing full-setting hopper range curves and replay provenance; no jobs."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))

import numpy as np


def capture_result(out_dir, arch, original, *args, **kwargs):
    """Delegate once, preserve result identity, and retain already-returned host arrays."""
    path = out_dir / f"range_{arch}_pool_25600_curves.npz"
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    result = original(*args, **kwargs)
    with open(path, "xb") as stream:
        np.savez_compressed(
            stream,
            **{
                f"{name}_{field}": value
                for name, fields in result.items()
                for field, value in fields.items()
            },
        )
    return result


def main(argv=None):
    """Capture the already-approved case without refitting or changing any setting.

    The seed, environment, configured pool, architecture choices and source
    range implementation are the original Block B case. This driver adds
    evidence retention only; a single-size result is not full qualification.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--arch", choices=("D2W512", "D4W1024", "D4W1536"), required=True
    )
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    from scripts import probe_fresh_checks as checks
    from scripts import exp12_reports
    from experiments.exp12 import probe
    from experiments.exp12.precision import runtime_info
    from omegaconf import OmegaConf
    from utils.run_metadata import code_version

    code, runtime = code_version(), runtime_info()
    if code["commit"] != args.expected_commit or code["dirty"] is not False:
        parser.error("require the exact expected clean source revision")
    if runtime["platform"] != "gpu" or "A100" not in runtime["device_kind"]:
        parser.error(
            "full-setting capture requires an A100; CPU tests verify the observer only"
        )
    if not args.out_dir.is_absolute():
        parser.error("--out-dir must be absolute")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "code": code,
        "runtime": runtime,
        "seed": 990,
        "arch": args.arch,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "scope": "one-size existing hopper configured-pool range case; not full campaign qualification",
    }
    original_trainer = checks._fresh_trainer

    def retain_replay(*a, **kw):
        trainer = original_trainer(*a, **kw)
        if probe.probe_config(trainer.cfg) != probe.ProbeConfig(
            5, 1000, 20, 25600, 256, 1e5, 2560
        ):
            raise ValueError("capture requires unchanged approved full probe settings")
        metadata["resolved_config"] = OmegaConf.to_container(trainer.cfg, resolve=True)
        n = trainer.buffer._num_in_buffer
        arrays = {
            "observations": trainer.buffer._observations[:n],
            "actions": trainer.buffer._actions[:n],
        }
        if hasattr(trainer.agent, "obs_rms"):
            arrays.update(
                {
                    f"obs_rms_{field}": np.asarray(
                        getattr(trainer.agent.obs_rms, field)
                    )
                    for field in ("mean", "var", "count")
                }
            )
            arrays["normalization_epsilon"] = np.asarray(trainer.agent.epsilon)
        with open(args.out_dir / "warmup_replay.npz", "xb") as stream:
            np.savez_compressed(stream, **arrays)
        return trainer

    original_probe, original_base = probe.run_probe, probe._base_targets
    teacher_pools = []

    def retain_base(network, key, obs, act, chunk, scale):
        base = original_base(network, key, obs, act, chunk, scale)
        teacher_pools.append(
            {
                "teacher_key": key,
                "observations": obs,
                "actions": act,
                "base_targets": base,
            }
        )
        return base

    checks._stub_wandb()
    case = SimpleNamespace(
        env="hopper-hop", env_group="dmc_medium", seed=990, override=[], pools=[25600]
    )
    with mock.patch.object(
        checks, "_fresh_trainer", side_effect=retain_replay
    ), mock.patch.object(
        probe,
        "run_probe",
        side_effect=lambda *a, **kw: capture_result(
            args.out_dir, args.arch, original_probe, *a, **kw
        ),
    ), mock.patch.object(
        probe, "_base_targets", side_effect=retain_base
    ):
        rows = checks.range_mode(
            case, args.arch, *checks.ARCHS[args.arch], args.out_dir
        )
    if len(teacher_pools) != 5:
        raise ValueError(
            "require exactly five observed teacher pools; partial evidence cannot qualify"
        )
    # Read computed values after fitting; never regenerate a target or consume an RNG.
    for round_index, arrays in enumerate(teacher_pools):
        with open(
            args.out_dir / f"round_{round_index}_teacher_pool.npz", "xb"
        ) as stream:
            np.savez_compressed(stream, **arrays)
    metadata["files_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in args.out_dir.iterdir()
        if path.is_file()
    }
    metadata["recorded_ratio"] = rows[0]["score_over_b"]
    metadata["approved_range_minimum"] = exp12_reports.RANGE_MIN_P_OVER_B
    metadata["range_verdict"] = exp12_reports.range_criterion_rows(rows, [args.arch])
    metadata["range_rule_pass_for_this_size_only"] = metadata["range_verdict"]["pass"]
    with open(args.out_dir / "capture_metadata.json", "x") as stream:
        json.dump(metadata, stream, indent=2)
    return 0 if metadata["range_rule_pass_for_this_size_only"] else 3


if __name__ == "__main__":
    sys.exit(main())
