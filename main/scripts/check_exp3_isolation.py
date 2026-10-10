"""Read-only base-tree byte audit and 195 resolved Exp1 configuration comparisons."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from experiments.exp1 import compose_config
    from generate_manifest import EXP12_ARCHS, EXP12_ENVS, EXP12_SEEDS, exp12_overrides
    from omegaconf import OmegaConf
    from utils.run_metadata import _hash, _strip_locations

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    git = lambda *cmd: subprocess.check_output(["git", "-C", str(root), *cmd])
    base = git("rev-parse", args.base + "^{commit}").decode().strip()
    if base != args.base:
        raise ValueError("full pinned base SHA required")
    changed = []
    compared = 0
    with tempfile.TemporaryDirectory() as d:
        old_config = Path(d) / "configs"
        for row in git("ls-tree", "-r", base).decode().splitlines():
            prefix, name = row.split("\t", 1)
            _, kind, oid = prefix.split()
            if kind != "blob":
                continue
            contents = git("cat-file", "blob", oid)
            if not (root / name).is_file() or (root / name).read_bytes() != contents:
                changed.append(name)
            compared += 1
            if name.startswith("main/configs/"):
                dest = old_config / name[len("main/configs/") :]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(contents)
        matches, mismatches = {}, []
        for arch in EXP12_ARCHS:
            for env, group in EXP12_ENVS:
                for seed in EXP12_SEEDS:
                    overrides = exp12_overrides(
                        arch, env, group, seed, "/tmp/exp3-isolation-unused-results"
                    )
                    expected = OmegaConf.to_container(
                        compose_config(str(old_config), "base_exp12", overrides),
                        resolve=True,
                    )
                    actual = OmegaConf.to_container(
                        compose_config(
                            str(root / "main/configs"), "base_exp12", overrides
                        ),
                        resolve=True,
                    )
                    key = f"{arch[0]}:{env}:{seed}"
                    if _hash(_strip_locations(expected)) != _hash(
                        _strip_locations(actual)
                    ):
                        mismatches.append(key)
                    else:
                        matches[key] = _hash(_strip_locations(actual))
    output = {
        "base": base,
        "base_files_compared": compared,
        "changed_base_files": changed,
        "configuration_matches": len(matches),
        "mismatches": mismatches,
        "fingerprints": matches,
    }
    with open(args.out, "x") as f:
        json.dump(output, f, indent=2)
    print(json.dumps({k: v for k, v in output.items() if k != "fingerprints"}))
    return 0 if not changed and not mismatches and len(matches) == 195 else 1


if __name__ == "__main__":
    sys.exit(main())
