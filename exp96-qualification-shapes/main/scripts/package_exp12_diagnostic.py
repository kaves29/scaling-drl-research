#!/usr/bin/env python3
"""Package local diagnostic evidence without importing JAX or contacting Slurm."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

REQUIRED = (
    "commit.txt",
    "backend.json",
    "trace.jsonl",
    "profile.log",
    "exit_status.txt",
)
OPTIONAL = (
    "profile.json",
    "validation.json",
    "identity.json",
    "parent_trace.jsonl",
    "identity_trace.jsonl",
    "trace.stacks.log",
    "parent_trace.stacks.log",
    "identity_trace.stacks.log",
    "resolved_config.json",
    "command.json",
)
SOURCE_PATHS = (
    "main/scripts/sbatch_exp12_runtime_diagnostic.sh",
    "main/scripts/trace_exp12_runtime.py",
    "main/scripts/profile_exp12.py",
    "main/experiments/exp12/runtime_trace.py",
)


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], stderr=subprocess.PIPE
    )


def trace_details(content):
    """Inspect every trace line while retaining malformed or partial raw bytes."""
    valid, invalid, commands, jobs = 0, [], [], []
    last = None
    for number, line in enumerate(content.splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("record is not an object")
        except (ValueError, UnicodeDecodeError) as exc:
            invalid.append({"line": number, "error": str(exc)})
            continue
        valid += 1
        last = row
        if row.get("event") == "wrapper_start":
            if row.get("command") is not None:
                commands.append(row["command"])
            if row.get("slurm_job_id") is not None:
                jobs.append(str(row["slurm_job_id"]))
    return dict(
        valid_records=valid,
        invalid_records=invalid,
        last_record=last,
        recorded_commands=commands,
        recorded_job_ids=jobs,
    )


def package(run_dir, job_id, expected_commit, repo, out, stdout=None, stderr=None):
    """Create a non-overwriting ZIP, provenance manifest and SHA-256 sidecar."""
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
        raise ValueError("job ID must be numeric, optionally with an array-task suffix")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("expected commit must be a full lowercase 40-character hash")
    run_dir, repo, out = Path(run_dir), Path(repo), Path(out)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("run directory must be a real local directory")
    run_dir, out = run_dir.resolve(), out.absolute()
    if out.resolve().is_relative_to(run_dir):
        raise ValueError("output must be outside the input run directory")
    sidecar = Path(str(out) + ".sha256")
    if out.exists() or out.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise ValueError("output or checksum sidecar already exists")
    if not out.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    resolved = (
        git(repo, "rev-parse", "--verify", expected_commit + "^{commit}")
        .decode()
        .strip()
    )
    if resolved != expected_commit:
        raise ValueError("expected revision did not resolve exactly")

    payload, files, missing = {}, [], []

    def add(name, content, source):
        payload[name] = content
        files.append(
            dict(
                path=name,
                source=str(source),
                bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )

    def add_file(path, name, required=False):
        path = Path(path)
        if path.is_symlink():
            raise ValueError(f"symlink artifact refused: {path}")
        if not path.exists():
            if required:
                missing.append(name)
            return
        if not path.is_file():
            raise ValueError(f"artifact is not a regular file: {path}")
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ValueError(f"artifact changed while reading: {path}")
        add(name, content, path)

    for name in (*REQUIRED, *OPTIONAL):
        add_file(run_dir / name, "artifacts/" + name, name in REQUIRED)
    for stream, supplied in (("stdout", stdout), ("stderr", stderr)):
        path = (
            Path(supplied)
            if supplied
            else Path(
                f"{run_dir}.slurm-{job_id}.{'out' if stream == 'stdout' else 'err'}"
            )
        )
        match = re.search(r"\.slurm-([0-9]+(?:_[0-9]+)?)\.(out|err)$", path.name)
        if match and match.group(1) != job_id:
            raise ValueError(f"Slurm log job ID mismatch: {path}")
        add_file(path, f"artifacts/slurm.{stream}", True)

    commits = {}
    raw = payload.get("artifacts/commit.txt")
    if raw is not None:
        commits["commit.txt"] = raw.decode().strip()
    raw = payload.get("artifacts/backend.json")
    backend = json.loads(raw) if raw is not None else {}
    if not isinstance(backend, dict):
        raise ValueError("backend.json must contain an object")
    if "commit" in backend:
        commits["backend.json"] = backend["commit"]
    for source, commit in commits.items():
        if commit != expected_commit:
            raise ValueError(
                f"{source}: recorded commit {commit!r} does not match expected revision"
            )
    traces = {}
    for name, content in payload.items():
        if name.endswith("trace.jsonl"):
            traces[name] = trace_details(content)
            if any(j != job_id for j in traces[name]["recorded_job_ids"]):
                raise ValueError(f"{name}: recorded job ID mismatch")
    # Immutable Git objects supply source context, never proof of job-time overrides.
    paths = (
        git(repo, "ls-tree", "-r", "--name-only", expected_commit, "main/configs")
        .decode()
        .splitlines()
    )
    paths = [p for p in paths if p.endswith((".yaml", ".yml"))]
    for path in (*paths, *SOURCE_PATHS):
        add(
            "source/" + path,
            git(repo, "show", expected_commit + ":" + path),
            f"git:{expected_commit}:{path}",
        )
    argv_available = (
        any(t["recorded_commands"] for t in traces.values())
        or "artifacts/command.json" in payload
    )
    raw_status = payload.get("artifacts/exit_status.txt")
    exit_status = None
    if raw_status is not None:
        try:
            exit_status = int(raw_status.decode().strip())
        except (ValueError, UnicodeDecodeError):
            exit_status = "malformed; inspect the preserved raw artifact"
    manifest = dict(
        schema_version=1,
        job_id=job_id,
        expected_commit=expected_commit,
        exit_status=exit_status,
        job_id_provenance="Caller supplied; checked against recorded trace IDs and recognized Slurm filenames when available.",
        recorded_commits=commits,
        source_revision_recorded=bool(commits),
        run_directory=str(run_dir),
        files=files,
        missing_required=missing,
        traces=traces,
        configuration=dict(
            frozen_source_context=True,
            runtime_command_available=argv_available,
            resolved_config_available="artifacts/resolved_config.json" in payload,
            warning="Frozen YAML/launcher source is context; it does not establish actual job-time overrides or a resolved training configuration.",
        ),
        qualification_status="not_assessed",
        excluded="Caches, temporary files, checkpoints and unlisted large artifacts are excluded.",
    )
    payload["MANIFEST.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    sums = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in sorted(payload.items())
    )
    payload["SHA256SUMS"] = sums.encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=out.parent, prefix=".exp12-package-", delete=False
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name, content in sorted(payload.items()):
                archive.writestr(name, content)
        archive_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
        # Atomic, non-overwriting publication even if a competing writer appears.
        os.link(temporary, out)
        with sidecar.open("x") as handle:
            handle.write(f"{archive_hash}  {out.name}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out", required=True)
    parser.add_argument("--slurm-stdout")
    parser.add_argument("--slurm-stderr")
    args = parser.parse_args()
    try:
        report = package(
            args.run_dir,
            args.job_id,
            args.expected_commit,
            args.repo,
            args.out,
            args.slurm_stdout,
            args.slurm_stderr,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Packaging refused: {exc}\n")
    print(
        json.dumps(
            dict(
                out=args.out,
                missing_required=report["missing_required"],
                qualification_status=report["qualification_status"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
