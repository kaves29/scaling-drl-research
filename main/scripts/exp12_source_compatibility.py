#!/usr/bin/env python3
"""Certify that two commits share byte-identical scientific source (CPU, git only).

Used when a development fork (e.g. the positive-control pilot) was produced at an
earlier commit than the campaign source. The certificate compares git blob ids of
every tracked file in the scientific scope. Any in-scope difference makes it
INCOMPATIBLE unless the path is listed explicitly with --allow-path; the exact
diff of an allowed path is hashed into the certificate for owner review.
A certificate never relabels a checkpoint's source; it only records evidence.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
REPO = MAIN.parent
SCOPE = (
    "main/experiments/",
    "main/scale_rl/",
    "main/utils/",
    "main/configs/",
    "main/run.py",
    "main/generate_manifest.py",
    "main/requirements.txt",
    "main/scripts/positive_control.py",
)
EXCLUDE = ("main/configs/launch_readiness/",)


def _git(*args):
    return subprocess.check_output(("git", "-C", str(REPO)) + args, text=True)


def _resolve(commit):
    full = _git("rev-parse", "--verify", commit + "^{commit}").strip()
    if not re.fullmatch("[0-9a-f]{40}", full):
        raise ValueError(f"cannot resolve commit {commit}")
    return full


def _in_scope(path):
    return path.startswith(SCOPE) and not path.startswith(EXCLUDE)


def _blobs(commit):
    out = {}
    for line in _git("ls-tree", "-r", "--full-tree", commit).splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, blob = meta.split()
        if kind == "blob" and _in_scope(path):
            out[path] = (mode, blob)
    return out


def certify(source, target, allow_paths=()):
    source, target = _resolve(source), _resolve(target)
    a, b = _blobs(source), _blobs(target)
    differing = sorted(p for p in set(a) & set(b) if a[p] != b[p])
    added, removed = sorted(set(b) - set(a)), sorted(set(a) - set(b))
    allowed = sorted(set(allow_paths))
    for path in allowed:
        if not _in_scope(path):
            raise ValueError(f"allowed path is outside the scientific scope: {path}")
    exceptions = []
    for path in allowed:
        if path not in differing + added + removed:
            raise ValueError(f"allowed path does not differ; remove it: {path}")
        diff = _git("diff", "--no-color", "--full-index", source, target, "--", path)
        exceptions.append(
            {
                "path": path,
                "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "diff_lines": diff.count("\n"),
            }
        )
    unexplained = sorted(set(differing + added + removed) - set(allowed))
    return {
        "schema_version": 1,
        "qualification": "science_source_compatibility",
        "source_commit": source,
        "target_commit": target,
        "scope": list(SCOPE),
        "excluded": list(EXCLUDE),
        "files_compared": len(set(a) | set(b)),
        "identical_files": len([p for p in set(a) & set(b) if a[p] == b[p]]),
        "differing": differing,
        "added": added,
        "removed": removed,
        "allowed_exceptions": exceptions,
        "unexplained_differences": unexplained,
        "verdict": "COMPATIBLE" if not unexplained else "INCOMPATIBLE",
    }


def verify(path, expected_sha256, source, target):
    """Recompute from git; the stored certificate must equal the recomputation."""
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("source-compatibility certificate changed")
    stored = json.loads(path.read_text())
    if stored.get("verdict") != "COMPATIBLE":
        raise ValueError("source-compatibility certificate is not COMPATIBLE")
    if (stored.get("source_commit"), stored.get("target_commit")) != (source, target):
        raise ValueError("source-compatibility certificate covers other commits")
    allowed = [e["path"] for e in stored.get("allowed_exceptions", [])]
    if certify(source, target, allowed) != stored:
        raise ValueError("source-compatibility certificate does not match git")
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("certify")
    make.add_argument("--source", required=True)
    make.add_argument("--target", required=True)
    make.add_argument("--allow-path", action="append", default=[])
    make.add_argument("--out", required=True)
    check = sub.add_parser("verify")
    for name in ("certificate", "sha256", "source", "target"):
        check.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.action == "certify":
        report = certify(args.source, args.target, args.allow_path)
        with Path(args.out).open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({k: report[k] for k in ("verdict", "differing", "unexplained_differences")}))
        return 0 if report["verdict"] == "COMPATIBLE" else 1
    verify(args.certificate, args.sha256, args.source, args.target)
    print("COMPATIBLE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
