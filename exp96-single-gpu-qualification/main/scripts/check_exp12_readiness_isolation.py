"""Check production bytes and all195 resolved parent fingerprints against approved base."""

import argparse
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))


def check(base):
    import omegaconf
    from experiments.exp1 import compose_config
    from generate_manifest import EXP12_ARCHS, EXP12_ENVS, EXP12_SEEDS, exp12_overrides
    from utils.run_metadata import _hash, _strip_locations

    files = (
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", base], cwd=MAIN.parent
        )
        .decode()
        .splitlines()
    )
    protected = [
        f
        for f in files
        if not f.endswith(".md")
        or "/.claude/" in f
        or f == "main/docs/exp12_decisions.md"
    ]
    changed = []
    for name in protected:
        old = subprocess.check_output(
            ["git", "show", f"{base}:{name}"], cwd=MAIN.parent
        )
        path = MAIN.parent / name
        if not path.is_file() or path.read_bytes() != old:
            changed.append(name)
    fingerprints = []
    archive = subprocess.check_output(
        ["git", "archive", base, "main/configs"], cwd=MAIN.parent
    )
    with tempfile.TemporaryDirectory() as directory:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(directory, filter="data")
        for architecture in EXP12_ARCHS:
            for env, group in EXP12_ENVS:
                for seed in EXP12_SEEDS:
                    overrides = exp12_overrides(
                        architecture, env, group, seed, "/external/fingerprint/results"
                    )
                    hashes = []
                    for configs in (MAIN / "configs", Path(directory) / "main/configs"):
                        cfg = compose_config(str(configs), "base_exp12", overrides)
                        hashes.append(
                            _hash(
                                _strip_locations(
                                    omegaconf.OmegaConf.to_container(cfg, resolve=True)
                                )
                            )
                        )
                    if hashes[0] != hashes[1]:
                        fingerprints.append(f"{architecture[0]}/{env}/{seed}")
    return dict(
        base=base,
        protected_files=len(protected),
        changed=changed,
        parent_count=195,
        config_mismatches=fingerprints,
        match=not changed and not fingerprints,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = check(args.base)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    sys.exit(0 if result["match"] else 1)
