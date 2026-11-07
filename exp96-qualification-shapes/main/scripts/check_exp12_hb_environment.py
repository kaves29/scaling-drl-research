#!/usr/bin/env python3
"""Capture and check the explicitly pinned Block HB environment contract."""

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HB_COMMIT = "cb1189039151c8aadaaa987b442da54383c87fab"


def validate(current, expected_gpu, main=None):
    if not current["devices"] or any(
        d["platform"] != "gpu" or d["device_kind"] != expected_gpu
        for d in current["devices"]
    ):
        raise ValueError(
            "CUDA backend/GPU model differs from the explicit job contract"
        )
    if main is not None:
        if current["prefix"] == main["prefix"]:
            raise ValueError("HB must use a separate cloned environment")
        if any(
            current["packages"].get(k) != version
            for k, version in main["packages"].items()
        ):
            raise ValueError(
                "HB clone differs from the pinned main environment packages"
            )
        if current["hb_commit"] != HB_COMMIT or current["hb_dirty"]:
            raise ValueError("HumanoidBench checkout is not the pinned clean revision")
    return current


def capture(hb=False):
    import jax

    packages = {
        d.metadata["Name"].lower().replace("_", "-"): d.version
        for d in importlib.metadata.distributions()
    }
    result = {
        "executable": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
        "packages": packages,
        "devices": [
            {"platform": d.platform, "device_kind": d.device_kind}
            for d in jax.devices()
        ],
    }
    if hb:
        spec = importlib.util.find_spec("humanoid_bench")
        if spec is None or spec.origin is None:
            raise ValueError("HumanoidBench is not installed")
        root = Path(spec.origin).resolve().parent.parent

        def git(*args):
            return subprocess.run(
                ["git", "-C", str(root), *args],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()

        result.update(
            hb_commit=git("rev-parse", "HEAD"),
            hb_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
            hb_root=str(root),
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-model", required=True)
    parser.add_argument("--main", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    current = capture(hb=args.main is not None)
    validate(
        current,
        args.gpu_model,
        None if args.main is None else json.loads(args.main.read_text()),
    )
    args.out.write_text(json.dumps(current, indent=2))
