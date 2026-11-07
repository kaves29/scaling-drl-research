"""Minimal experiment registry/dispatcher.

`run.py` should never grow a chain of `if experiment == "...":` branches.
Instead, each experiment module registers itself here with `@register_experiment("name")`,
and `run.py` just does `get_experiment(args["experiment"])(args)`.
"""

from typing import Callable, Dict, List

_REGISTRY: Dict[str, Callable[[dict], None]] = {}


class UnknownExperimentError(KeyError):
    pass


def register_experiment(name: str):
    """Class/function decorator that registers an experiment entry point under `name`."""

    def decorator(fn: Callable[[dict], None]) -> Callable[[dict], None]:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(
                f"Experiment name '{name}' is already registered to "
                f"{_REGISTRY[name]!r}; refusing to silently overwrite it with {fn!r}."
            )
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_experiment(name: str) -> Callable[[dict], None]:
    if name not in _REGISTRY:
        available = ", ".join(list_experiments()) or "<none registered>"
        raise UnknownExperimentError(
            f"Unknown experiment '{name}'. Available experiments: {available}"
        )
    return _REGISTRY[name]


def list_experiments() -> List[str]:
    return sorted(_REGISTRY)
