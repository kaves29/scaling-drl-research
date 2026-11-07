"""Shared exception types for Angle 2B.

Mirrors experiments/angle_2a/errors.py's convention: one tiny module so
callers can distinguish "this is a configuration/data-availability problem"
from a generic bug without importing the heavier submodules.
"""


class Angle2BConfigError(ValueError):
    """Required Angle 2B configuration is missing/invalid."""


class Angle2BSnapshotError(RuntimeError):
    """A required Angle 2A frozen-agent snapshot (checkpoint, probe-capture
    array, or resolved agent config) could not be found or loaded. Angle 2B
    never retrains or reconstructs a missing snapshot - see CLAUDE.md and
    research-methodology.md's Angle 2B section - so this is always a hard
    failure, never a fallback-to-training path."""
