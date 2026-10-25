"""Inventory accessible evidence without unpickling or extracting archives."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

ORIGINAL_NPZ = "66a223890220be7aa9d022e9cdfbcd90909d6580132e20315492a0ff307839ca"


def inventory(roots):
    files, containers = [], []
    names = subprocess.check_output(["rg", "--files", *roots], text=True).splitlines()
    for name in sorted(set(names)):
        p = Path(name)
        if p.suffix not in (".npz", ".pkl", ".ckpt", ".zip", ".gz") and p.name not in (
            "LATEST",
            "_METADATA",
        ):
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        files.append(
            {
                "path": str(p),
                "bytes": p.stat().st_size,
                "sha256": digest,
                "classification": (
                    "provenance-bound original NPZ"
                    if digest == ORIGINAL_NPZ
                    else "not independently authenticated as original campaign state"
                ),
            }
        )
        entries = []
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                for item in z.infolist():
                    if item.filename.endswith(
                        (".npz", ".pkl", "LATEST", "_METADATA", ".ckpt")
                    ):
                        entries.append(
                            {
                                "name": item.filename,
                                "bytes": item.file_size,
                                "npz_sha256": (
                                    hashlib.sha256(z.read(item)).hexdigest()
                                    if item.filename.endswith(".npz")
                                    and item.file_size <= 32 * 1024**2
                                    else None
                                ),
                            }
                        )
                count = len(z.infolist())
        elif p.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(p, "r:gz") as t:
                members = t.getmembers()
                for item in members:
                    if item.name.endswith(
                        (".npz", ".pkl", "LATEST", "_METADATA", ".ckpt")
                    ):
                        entries.append(
                            {"name": item.name, "bytes": item.size, "npz_sha256": None}
                        )
                count = len(members)
        else:
            continue
        containers.append(
            {
                "path": str(p),
                "member_count": count,
                "candidates": entries,
                "safety": "member listing only; small ZIP NPZ bytes hashed, never deserialized; no pickle executed",
            }
        )
    return {
        "roots": roots,
        "candidate_files": files,
        "containers": containers,
        "remote_inventory": False,
        "interpretation": "unbound/synthetic arrays are not original campaign evidence; repeated hashes are copies, not new checkpoints",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", action="append", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = inventory(args.root)
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "candidate_files": len(result["candidate_files"]),
                "containers": len(result["containers"]),
                "original_npz_copies": sum(
                    x["sha256"] == ORIGINAL_NPZ for x in result["candidate_files"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
