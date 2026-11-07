#!/usr/bin/env python3
"""Freeze independent expected configurations and explicit provenance before the grid."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp1 import compose_config
from generate_manifest import EXP12_ARCHS, EXP12_ENVS, exp12_overrides
from utils.run_metadata import code_version


def build(checkpoint_root, commit, gpu_model, injection_m, runtime_by_suite):
    import omegaconf

    root = Path(checkpoint_root)
    if not root.is_absolute():
        raise ValueError("checkpoint_root must be absolute")
    configs = {}
    for architecture in EXP12_ARCHS:
        for environment, group in EXP12_ENVS:
            cfg = compose_config(
                str(Path(__file__).resolve().parents[1] / "configs"),
                "base_exp12",
                exp12_overrides(architecture, environment, group, 1, None),
            )
            configs[f"{architecture[0]}/{environment}"] = (
                omegaconf.OmegaConf.to_container(cfg, resolve=True)
            )
    return {
        "schema_version": 1,
        "code_commit": commit,
        "gpu_model": gpu_model,
        "checkpoint_root": str(root),
        "injection_m": injection_m,
        "runtime_by_suite": runtime_by_suite,
        "configs": configs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--gpu-model", required=True)
    parser.add_argument("--injection-m", choices=("last", "half", "all"), required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    version = code_version()
    if version["commit"] != args.expected_commit or version["dirty"]:
        raise ValueError("freeze from the expected clean source revision")
    manifest = build(
        args.checkpoint_root,
        args.expected_commit,
        args.gpu_model,
        args.injection_m,
        json.loads(args.runtime_json.read_text()),
    )
    with args.out.open("x") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
