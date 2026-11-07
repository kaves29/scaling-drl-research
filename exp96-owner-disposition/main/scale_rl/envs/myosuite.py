import gymnasium as gym

MYOSUITE_TASKS = [
    "myo-reach",
    "myo-reach-hard",
    "myo-pose",
    "myo-pose-hard",
    "myo-obj-hold",
    "myo-obj-hold-hard",
    "myo-key-turn",
    "myo-key-turn-hard",
    "myo-pen-twirl",
    "myo-pen-twirl-hard",
]


MYOSUITE_TASKS_DICT = {
    "myo-reach": "myoHandReachFixed-v0",
    "myo-reach-hard": "myoHandReachRandom-v0",
    "myo-pose": "myoHandPoseFixed-v0",
    "myo-pose-hard": "myoHandPoseRandom-v0",
    "myo-obj-hold": "myoHandObjHoldFixed-v0",
    "myo-obj-hold-hard": "myoHandObjHoldRandom-v0",
    "myo-key-turn": "myoHandKeyTurnFixed-v0",
    "myo-key-turn-hard": "myoHandKeyTurnRandom-v0",
    "myo-pen-twirl": "myoHandPenTwirlFixed-v0",
    "myo-pen-twirl-hard": "myoHandPenTwirlRandom-v0",
    # "-random" is this file's Fixed/Random suffix convention, unrelated to
    # the Hard/Medium tier used elsewhere.
    "myo-elbow-pose-random": "myoElbowPose1D6MRandom-v0",
    # No Fixed/Random variant exists for this task in myosuite's registry.
    "myo-leg-walk": "myoLegWalk-v0",
    # Only registered match is under myosuite's "Challenge" naming, "-v1" not
    # "-v0" - a real registry naming gap, not a typo.
    "myo-baoding-p1": "myoChallengeBaodingP1-v1",
}

# Angle 1/2 core-4 MyoSuite set (research-methodology.md's environment
# table); mirrors DMC_MED/DMC_HARD in scale_rl/envs/dmc.py.
MYOSUITE_CORE4 = [
    "myo-elbow-pose-random",  # MyoElbowPose1D6MRandom, Medium tier
    "myo-reach",              # MyoHandReachFixed, Hard tier
    "myo-key-turn",           # MyoHandKeyTurnFixed, Hard tier
    "myo-leg-walk",           # MyoLegWalk, Hard tier
]

# Held out for Angle 3 only.
MYOSUITE_HELDOUT2 = [
    "myo-pen-twirl",   # MyoHandPenTwirlFixed, Hard tier
    "myo-baoding-p1",  # MyoHandBaodingBallsP1, Hard tier
]


def validate_myosuite_core4(env_type: str, env_name: str) -> None:
    """Rejects a held-out-2 MyoSuite env_name for Angle 1/2. No-op for
    non-myosuite env_type - see dmc.py's validate_dmc_not_heldout for that
    side. Angle 2A supports myosuite as of 2026-09-07 (see
    experiments/angle_2a/env_state.py); 2B/2C still never instantiate a live
    env (they only ever read Angle 2A's already-persisted snapshots)."""
    if env_type != "myosuite" or env_name in MYOSUITE_CORE4:
        return
    if env_name in MYOSUITE_HELDOUT2:
        raise ValueError(
            f"env_name='{env_name}' is a MyoSuite environment held out for "
            f"Angle 3 only (see scale_rl/envs/myosuite.py). Angle 1/2 may "
            f"only train on the core-4 set: {MYOSUITE_CORE4}."
        )
    raise ValueError(
        f"env_name='{env_name}' is not one of the core-4 MyoSuite "
        f"environments Angle 1/2 may train on ({MYOSUITE_CORE4}), and is not "
        f"in the held-out-2 set either - check for a typo."
    )


class MyosuiteGymnasiumVersionWrapper(gym.Wrapper):
    """
    myosuite originally requires gymnasium==0.15
    however, we are currently using  gymnasium==1.0.0a2,
    hence requiring some minor fix to the
      - fix a.
      - fix b.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["success"] = info["solved"]
        return obs, reward, terminated, truncated, info

    def render(
        self, width: int = 192, height: int = 192, camera_id: str = "hand_side_inter"
    ):
        return self.unwrapped_env.sim.renderer.render_offscreen(
            width=width,
            height=height,
            camera_id=camera_id,
        )


def make_myosuite_env(
    env_name: str,
    seed: int,
    **kwargs,
) -> gym.Env:
    from myosuite.utils import gym as myo_gym

    env = myo_gym.make(MYOSUITE_TASKS_DICT[env_name])
    env = MyosuiteGymnasiumVersionWrapper(env)

    return env