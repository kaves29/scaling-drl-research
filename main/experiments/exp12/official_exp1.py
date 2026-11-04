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
# Owner explicitly included development-exposed seed 9 on 4 November 2026.
SEEDS = tuple(range(1, 11))
ARCHITECTURES = (("D2W512", 2, 512), ("D4W1024", 4, 1024), ("D4W1536", 4, 1536))
CONFIG_NAME = "base_exp1_official"


def validate_config(cfg):
    """Reject accidental pilot/legacy overrides in official confirmatory runs."""
    if cfg.run_role == "dev":
        return
    if cfg.env_name not in {env for env, _ in ENVIRONMENTS} or cfg.seed not in SEEDS:
        raise ValueError("environment/seed outside official Exp1 population")
    allowed = {(depth, width) for _, depth, width in ARCHITECTURES}
    expected_budget = (
        500000 if cfg.env_name in ("hopper-hop", "swimmer-swimmer15") else 1000000
    )
    if (
        (int(cfg.agent.critic_num_blocks), int(cfg.agent.critic_hidden_dim))
        not in allowed
        or (cfg.agent.actor_num_blocks, cfg.agent.actor_hidden_dim) != (1, 128)
        or (cfg.probe.rounds, cfg.probe.steps, cfg.probe.checks) != (10, 2000, 50)
        or cfg.num_env_steps != expected_budget
        or (cfg.action_repeat, cfg.num_train_envs, cfg.updates_per_interaction_step)
        != (2, 1, 2)
    ):
        raise ValueError("official Exp1 configuration differs from frozen profile")


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
