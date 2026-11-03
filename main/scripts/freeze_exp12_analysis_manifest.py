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


def build(
    checkpoint_root,
    commit,
    gpu_model,
    injection_m,
    runtime_by_suite,
    *,
    protocol_version=None,
):
    import omegaconf

    root = Path(checkpoint_root)
    if not root.is_absolute():
        raise ValueError("checkpoint_root must be absolute")
    if protocol_version == "no-hb-v1":
        from experiments.exp12.official_exp1 import ENVIRONMENTS, ARCHITECTURES

        environments, architectures = ENVIRONMENTS, ARCHITECTURES
        config_name = "base_exp1_official"
    elif protocol_version is None:
        if injection_m not in ("last", "half", "all"):
            raise ValueError(
                "legacy combined study requires a frozen injection magnitude"
            )
        environments, architectures, config_name = EXP12_ENVS, EXP12_ARCHS, "base_exp12"
    else:
        raise ValueError("unknown protocol version")
    configs = {}
    for architecture in architectures:
        for environment, group in environments:
            cfg = compose_config(
                str(Path(__file__).resolve().parents[1] / "configs"),
                config_name,
                exp12_overrides(architecture, environment, group, 1, None),
            )
            configs[f"{architecture[0]}/{environment}"] = (
                omegaconf.OmegaConf.to_container(cfg, resolve=True)
            )
    return {
        **({"protocol_version": protocol_version} if protocol_version else {}),
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
    parser.add_argument("--injection-m", choices=("last", "half", "all"))
    parser.add_argument("--protocol-version", choices=("no-hb-v1",))
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
        protocol_version=args.protocol_version,
    )
    with args.out.open("x") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
