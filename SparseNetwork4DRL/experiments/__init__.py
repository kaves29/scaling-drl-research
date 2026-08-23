"""Experiment registry package.

Importing this package registers every known experiment (each module below
calls `@register_experiment(...)` at import time). `run.py` only needs
`from experiments.registry import get_experiment, list_experiments`.
"""

from experiments import angle_1, angle_2_a  # noqa: F401  (side-effect: registers experiments)
# NOTE: the entry-point module is angle_2_a.py (WITH the underscore between
# "2" and "a") deliberately - it must NOT be renamed to angle_2a.py, because
# that collides with the experiments/angle_2a/ package directory. Python
# gives directory-packages precedence over same-named module files, so a
# same-named angle_2a.py becomes permanently unreachable via import and its
# @register_experiment("angle_2_a") never runs - confirmed by direct
# execution during the 2026-08-23 audit follow-up.
from experiments.registry import get_experiment, list_experiments  # noqa: F401
