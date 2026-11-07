"""Experiment registry package.

Importing this package registers every known experiment (each module below
calls `@register_experiment(...)` at import time). `run.py` only needs
`from experiments.registry import get_experiment, list_experiments`.
"""

from experiments import exp1, exp2_arm  # noqa: F401  (register current experiments)
from experiments.registry import get_experiment, list_experiments  # noqa: F401
