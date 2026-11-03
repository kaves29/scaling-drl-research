"""Owner-selected official population, separate from diagnostic pilots."""

ENVIRONMENTS = (
    ("dog-run", "dmc_hard"),
    ("dog-trot", "dmc_hard"),
    ("humanoid-run", "dmc_hard"),
    ("humanoid-walk", "dmc_hard"),
    ("humanoid-stand", "dmc_hard"),
    ("swimmer-swimmer15", "dmc_medium"),
    ("hopper-hop", "dmc_medium"),
    ("myo-key-turn", "myosuite_simba"),
    ("myo-pen-twirl", "myosuite_simba"),
    ("myo-pose-hard", "myosuite_simba"),
    ("myo-reach", "myosuite_simba"),
)
SEEDS = tuple(range(1, 11))
ARCHITECTURES = (("D2W512", 2, 512), ("D4W1024", 4, 1024), ("D4W1536", 4, 1536))
CONFIG_NAME = "base_exp1_official"


def population():
    """Identities only: not a qualified launch manifest or resource plan."""
    return [
        {
            "environment": env,
            "group": group,
            "architecture": name,
            "critic_num_blocks": depth,
            "critic_hidden_dim": width,
            "seed": seed,
        }
        for env, group in ENVIRONMENTS
        for name, depth, width in ARCHITECTURES
        for seed in SEEDS
    ]
