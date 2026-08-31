"""Experiment registry package.

Importing this package registers every known experiment (each module below
calls `@register_experiment(...)` at import time). `run.py` only needs
`from experiments.registry import get_experiment, list_experiments`.
"""

from experiments import angle_1, angle_2_a, angle_2_b  # noqa: F401  (side-effect: registers experiments)
# NOTE: the entry-point modules are angle_2_a.py / angle_2_b.py (WITH the
# underscore before the final letter) deliberately - they must NOT be
# renamed to angle_2a.py / angle_2b.py, because that collides with the
# experiments/angle_2a/ and experiments/angle_2b/ package directories.
# Python gives directory-packages precedence over same-named module files,
# so a same-named angle_2a.py/angle_2b.py becomes permanently unreachable
# via import and its @register_experiment(...) never runs - confirmed by
# direct execution during the 2026-08-23 audit follow-up (angle_2_a) and
# deliberately preserved for angle_2_b.
from experiments.registry import get_experiment, list_experiments  # noqa: F401
