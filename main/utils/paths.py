"""Output locations anchored to the repository, never to the launch directory."""

import os
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")


def results_path(*parts: str, results_root: Optional[str] = None) -> str:
    """`{results_root or RESULTS_ROOT}/{parts...}`, always absolute."""
    root = RESULTS_ROOT if results_root is None else require_absolute(results_root, "results_root")
    return os.path.join(root, *parts)


def require_absolute(path, name: str) -> str:
    path = os.fspath(path)
    if not os.path.isabs(path):
        raise ValueError(
            f"{name} must be an absolute path, got: {path!r}. A relative path would "
            f"resolve against the launch directory."
        )
    return path


def config_root(value, name: str, default: str) -> str:
    """A config-supplied root: null means `default`; anything else must be absolute."""
    if value is None:
        return default
    return require_absolute(str(value), name)


def wandb_dir() -> str:
    """Parent of WandB's local `wandb/` directory; WANDB_DIR wins when set."""
    return os.environ.get("WANDB_DIR") or REPO_ROOT
