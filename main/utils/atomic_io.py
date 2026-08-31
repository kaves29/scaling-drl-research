"""Small shared helper for crash-safe file writes.

Used by the onset ledger (and anything else that must never leave a
partially-written file in place of a previously-good one).
"""

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: "str | Path", text: str, encoding: str = "utf-8") -> None:
    """Writes `text` to `path` atomically.

    Writes to a temporary file in the same directory as `path` (so the final
    replace is on the same filesystem), flushes + fsyncs it, then does an
    `os.replace`, which is atomic on both POSIX and Windows. If anything
    raises before the replace, the original file at `path` is left untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
