"""Compare all resolved parent configurations in isolated interpreters; CPU only."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def fingerprints(main_root):
    """Import the requested checkout without cross-checkout module reuse."""
    sys.path.insert(0, str(main_root))
    import generate_manifest
    from experiments.exp1 import compose_config
    from omegaconf import OmegaConf
    from utils.run_metadata import _strip_locations

    rows = {}
    with tempfile.TemporaryDirectory() as directory:
        manifests = {}
        generate_manifest.add_exp12_grid(
            manifests, {}, directory + "/ckpt", directory + "/results"
        )
        for command in manifests["exp12_exp1_jobs.txt"]:
            words = shlex.split(command.split(" > ")[0])
            overrides = [
                words[i + 1] for i, word in enumerate(words) if word == "--overrides"
            ]
            cfg = compose_config(str(main_root / "configs"), "base_exp12", overrides)
            key = f"D{cfg.critic_num_blocks}W{cfg.critic_hidden_dim}/{cfg.env_name}/seed_{cfg.seed}"
            if key in rows:
                raise ValueError(f"duplicate parent: {key}")
            payload = {
                "config": _strip_locations(OmegaConf.to_container(cfg, resolve=True)),
                "interval": words[words.index("--checkpoint_interval") + 1],
                "experiment": words[words.index("--experiment") + 1],
            }
            rows[key] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
    if len(rows) != 195:
        raise ValueError(f"expected 195 parents, got {len(rows)}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-main", type=Path)
    parser.add_argument("--emit-main", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.emit_main:
        print(json.dumps(fingerprints(args.emit_main.resolve()), sort_keys=True))
        return 0
    if not args.reference_main:
        parser.error("--reference-main is required")
    candidate = Path(__file__).resolve().parents[1]
    environments = [args.reference_main.resolve(), candidate]
    rows = [
        json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--emit-main",
                    str(root),
                ],
                cwd=root,
                text=True,
            )
        )
        for root in environments
    ]
    if rows[0] != rows[1]:
        differing = sorted(
            k
            for k in rows[0].keys() | rows[1].keys()
            if rows[0].get(k) != rows[1].get(k)
        )
        raise SystemExit(f"configuration fingerprint mismatch: {differing}")
    aggregate = hashlib.sha256(
        json.dumps(rows[1], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    print(
        json.dumps(
            {
                "reference_main": str(environments[0]),
                "candidate_main": str(candidate),
                "matching_parents": len(rows[1]),
                "sha256": aggregate,
                "fingerprints": rows[1],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
