import gymnasium as gym
from dm_control import suite
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium

"""# 20 tasks
DMC_EASY_MEDIUM = [
    "acrobot-swingup",
    "cartpole-balance",
    "cartpole-balance_sparse",
    "cartpole-swingup",
    "cartpole-swingup_sparse",
    "cheetah-run",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "fish-swim",
    "hopper-hop",
    "hopper-stand",
    "pendulum-swingup",
    "quadruped-walk",
    "quadruped-run",
    "reacher-easy",
    "reacher-hard",
    "walker-stand",
    "walker-walk",
    "walker-run",
]

# 8 tasks
DMC_SPARSE = [
    "cartpole-balance_sparse",
    "cartpole-swingup_sparse",
    "ball_in_cup-catch",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "reacher-easy",
    "reacher-hard",
]

# 7 tasks
DMC_HARD = [
    "humanoid-stand",
    "humanoid-walk",
    "humanoid-run",
    "dog-stand",
    "dog-walk",
    "dog-run",
    "dog-trot",
]"""

DMC_MED = [
    "cheetah-run",
    "quadruped-run",
    "manipulator-bring_ball"
     ]

DMC_HARD = [
    "humanoid-run",
    "dog-trot",
    "dog-run"
             ]

# Held out for Angle 3 only; mirrors MYOSUITE_HELDOUT2. Deliberately not
# added to DMC_MED, which is the core (Angle 1/2) Medium-tier set.
DMC_HELDOUT2 = [
    "hopper-hop",
    "fish-swim",
]


def validate_dmc_not_heldout(env_type: str, env_name: str) -> None:
    """Mirrors validate_myosuite_core4() for DMC. No "unrecognized name"
    check needed here: dm_control.suite.load() (via make_dmc_env) already
    raises on an unknown domain/task. No-op for non-dmc env_type."""
    if env_type == "dmc" and env_name in DMC_HELDOUT2:
        raise ValueError(
            f"env_name='{env_name}' is a DMC environment held out for Angle 3 "
            f"only (see scale_rl/envs/dmc.py's DMC_HELDOUT2). Angle 1/2 may "
            f"only train on the core DMC set (DMC_MED + DMC_HARD)."
        )

def make_dmc_env(
    env_name: str,
    seed: int,
    flatten: bool = True,
) -> gym.Env:
    domain_name, task_name = env_name.split("-")
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )
    env = DmControltoGymnasium(env, render_mode=None)
    if flatten and isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)

    return env

