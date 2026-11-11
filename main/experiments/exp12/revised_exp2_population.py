"""Run population and checkpoint schedule for the revised online pilot."""


def population():
    seeds = {"dog-run": (1, 2, 3), "humanoid-run": (1, 2, 3),
             "myo-key-turn": (1, 2), "myo-pen-twirl": (1, 2)}
    return [{"environment": environment, "seed": seed, "reference_regime": regime}
            for environment, values in seeds.items() for seed in values
            for regime in ("reference_small", "reference_large")]


def checkpoint_interactions():
    """Nearest whole actor interaction to each fifteenth of the raw budget."""
    return [(i * 500_000 + 7) // 15 for i in range(1, 16)]
