"""Shared exception types for Angle 2A.

Kept in one tiny module so callers (run.py's top-level error surface, tests)
can catch/distinguish "this is a configuration/protocol problem" from
generic bugs without importing the heavier submodules.
"""


class Angle2AConfigError(ValueError):
    """Required Angle 2A configuration is missing/invalid (e.g. an
    architecture field was not explicitly supplied)."""


class Angle2AOnsetLookupError(RuntimeError):
    """The Angle 1 onset ledger does not contain a usable onset for the
    requested (architecture, environment, seed)."""


class Angle2AEnvironmentError(RuntimeError):
    """The configured environment does not support the exact-state
    Monte Carlo rollout mechanism Angle 2A's protocol requires."""
