"""Upload only the twenty selected, immutable Exp1 states from Delta."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCE = "f597ee9898e4d401912092c336c3e5330cf2fda1"
POPULATION = {"dog-run": (1, 2, 3), "humanoid-run": (1, 2, 3),
              "myo-key-turn": (1, 2), "myo-pen-twirl": (1, 2)}


def inspect(base):
    runs, files = [], []
    for env, seeds in POPULATION.items():
        for seed in seeds:
            for architecture in ("D2W512", "D4W1536"):
                rel = Path(architecture) / env / f"seed_{seed}"
                run = base / rel
                metadata = json.loads((run / "run_metadata.json").read_text())
                ident = metadata["identity"]
                if (ident["architecture"], ident["environment"], ident["seed"]) != (architecture, env, seed):
                    raise ValueError(f"identity mismatch: {run}")
                if metadata["code"]["commit"] != SOURCE:
                    raise ValueError(f"source mismatch: {run}")
                done = json.loads((run / "DONE").read_text())
                if done["interaction_step"] != 500000:
                    raise ValueError(f"incomplete parent: {run}")
                state = (run / "state/LATEST").read_text().strip()
                if not state.startswith("step_") or Path(state).name != state:
                    raise ValueError(f"invalid state pointer: {run}")
                directory = run / "state" / state
                if not directory.is_dir() or directory.is_symlink():
                    raise ValueError(f"missing state: {directory}")
                for name in ("meta.pkl", "buffer.npz", "buffer_meta.pkl", "obs_rms.pkl"):
                    if not (directory / name).is_file():
                        raise ValueError(f"missing {name}: {directory}")
                if not (directory / "agent_ckpt").is_dir():
                    raise ValueError(f"missing agent checkpoint: {directory}")
                selected = [run / "DONE", run / "run_metadata.json", run / "state/LATEST"]
                selected += sorted(p for p in directory.rglob("*") if p.is_file())
                for path in selected:
                    if path.is_symlink() or not path.resolve().is_relative_to(run.resolve()):
                        raise ValueError(f"unsafe file: {path}")
                    h = hashlib.sha256()
                    before = path.stat()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                            h.update(block)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError(f"file changed during hash: {path}")
                    files.append((path, (rel / path.relative_to(run)).as_posix(), h.hexdigest()))
                runs.append(dict(architecture=architecture, environment=env, seed=seed,
                                 state=state, source_commit=SOURCE,
                                 config_hash=metadata["config_hash"], done_step=500000))
    return runs, files


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--volume", default="exp1-pilot-states")
    parser.add_argument("--prefix", default="exp1-pilot-parents")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    runs, files = inspect(args.base)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest_dir / "manifest.json"
    checksums = args.manifest_dir / "sha256.txt"
    manifest.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    checksums.write_text("".join(f"{digest}  {rel}\n" for _, rel, digest in files))
    print(f"Validated {len(runs)} parents, {len(files)} files; no source data modified.")
    if args.upload:
        import modal
        volume = modal.Volume.from_name(args.volume)
        with volume.batch_upload() as batch:
            for path, rel, _ in files:
                batch.put_file(path, f"/{args.prefix}/{rel}")
            batch.put_file(manifest, f"/{args.prefix}/manifest.json")
            batch.put_file(checksums, f"/{args.prefix}/sha256.txt")
        print("Upload complete; verify destination checksums before training.")


if __name__ == "__main__":
    main()
