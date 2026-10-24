"""Fail explicitly if a mutation run spans multiple on-disk source revisions."""

import hashlib
from pathlib import Path
import subprocess


def source_fingerprint(repository):
    repository = Path(repository)
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"])
    files = subprocess.check_output(["git", "-C", str(repository), "ls-files", "-z"])
    digest = hashlib.sha256(head)
    for name in files.split(b"\0"):
        if name:
            path = repository / name.decode()
            digest.update(name)
            digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


class StableSource:
    def __init__(self, repository):
        self.repository = repository
        self.initial = source_fingerprint(repository)

    def check(self):
        if source_fingerprint(self.repository) != self.initial:
            raise RuntimeError(
                "Source changed during mutation suite; rerun on a stable checkout"
            )
