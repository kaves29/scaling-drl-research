"""Shared exception types for Angle 2C.

Mirrors experiments/angle_2a/errors.py and experiments/angle_2b/errors.py's
convention: one tiny module so callers can distinguish "this is a
configuration/data-availability problem" from a generic bug without
importing the heavier submodules.
"""


class Angle2CConfigError(ValueError):
    """Required Angle 2C configuration is missing/invalid."""


class Angle2CDataError(RuntimeError):
    """A required Angle 2B (or Angle 1 ledger) artifact could not be found
    or loaded. Angle 2C never recomputes/retrains a missing artifact - see
    CLAUDE.md and research-methodology.md's Angle 2C section (it operates
    strictly on Angle 2B's already-computed outputs) - so this is always a
    hard failure, never a fallback-to-recomputation path."""
