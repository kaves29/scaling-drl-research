"""Write the owner-selected population and commands; never launch or approve it."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from experiments.exp1 import compose_config
from experiments.exp12.official_exp1 import CONFIG_NAME, population
from generate_manifest import exp12_overrides
from utils.run_metadata import code_version, _hash, _strip_locations


def build(root):
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("absolute campaign output root required")
    entries = []
    for row in population():
        arch = (row["architecture"], row["critic_num_blocks"], row["critic_hidden_dim"])
        overrides = exp12_overrides(
            arch, row["environment"], row["group"], row["seed"], str(root / "results")
        )
        cfg = compose_config(
            str(Path(__file__).resolve().parents[1] / "configs"), CONFIG_NAME, overrides
        )
        if (cfg.probe.rounds, cfg.probe.steps, cfg.probe.checks) != (10, 2000, 50):
            raise ValueError("official probe contract mismatch")
        if int(cfg.num_interaction_steps) // 50 < cfg.buffer.min_length:
            raise ValueError("first scheduled check precedes fresh capture")
        checkpoint = (
            root
            / "checkpoints"
            / "exp1"
            / row["architecture"]
            / row["environment"]
            / f"seed_{row['seed']}"
        )
        command = [
            "python",
            "run.py",
            "--experiment",
            "exp1",
            "--config_name",
            CONFIG_NAME,
        ]
        for override in overrides:
            command += ["--overrides", override]
        command += ["--checkpoint_dir", str(checkpoint)]
        command += [
            "--checkpoint_interval",
            str(int(cfg.num_interaction_steps) // 50),
            "--checkpoint_start_frac",
            "0.0",
        ]
        entries.append(
            {
                **row,
                "config_hash": _hash(
                    _strip_locations(OmegaConf.to_container(cfg, resolve=True))
                ),
                "raw_steps": int(cfg.num_env_steps),
                "interaction_steps": int(cfg.num_interaction_steps),
                "command": command,
            }
        )
    return {
        "protocol_version": "no-hb-v1",
        "source": code_version(),
        "approved": False,
        "executable": False,
        "submitted": False,
        "entries": entries,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    with args.out.open("x") as stream:
        json.dump(build(args.root), stream, indent=2)
