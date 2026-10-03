"""Subprocess entry for Exp 1/2 tests that must cross a process boundary."""

import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

ENV_KW = dict(num_train_envs=1, num_eval_envs=1, rescale_action=True, no_termination=False,
              action_repeat=2, reward_scale=1.0, max_episode_steps=1000)
RECORD_STEPS, CONTINUE_STEPS = 730, 400


def _envs(env_type, env_name, seed=3):
    from experiments.exp12.envs import create_envs

    return create_envs(env_type=env_type, seed=seed, env_name=env_name, **ENV_KW)[0]


def _rollout(env, actions):
    out = []
    for a in actions:
        obs, rew, term, trunc, _ = env.step(a)
        out.append((obs.copy(), rew.copy(), term.copy(), trunc.copy()))
    return out


def env_record(env_list, out_path):
    from experiments.exp12.envs import env_restore_state

    results = {}
    for env_type, env_name in env_list:
        env = _envs(env_type, env_name)
        env.reset()
        rng = np.random.default_rng(0)
        shape = (1,) + env.single_action_space.shape
        for a in rng.uniform(-1, 1, (RECORD_STEPS,) + shape):
            env.step(a)
        state = env_restore_state(env)
        actions = rng.uniform(-1, 1, (CONTINUE_STEPS,) + shape)
        results[(env_type, env_name)] = {"state": state, "actions": actions, "outputs": _rollout(env, actions)}
    with open(out_path, "wb") as f:
        pickle.dump(results, f)


def env_replay(in_path, out_path):
    from experiments.exp12.envs import restore_env

    with open(in_path, "rb") as f:
        recorded = pickle.load(f)
    results = {}
    for (env_type, env_name), rec in recorded.items():
        env = _envs(env_type, env_name)
        env.reset()
        restore_env(env, rec["state"])
        results[(env_type, env_name)] = _rollout(env, rec["actions"])
    with open(out_path, "wb") as f:
        pickle.dump(results, f)


def exp1_run(spec_path):
    from exp12_helpers import CONFIG_PATH, patch_wandb

    with open(spec_path) as f:
        spec = json.load(f)
    patch_wandb()
    from experiments import exp1
    from experiments.exp12.trainer import Exp12Trainer

    crash_step = spec.get("crash_step")
    if crash_step is not None:
        original_train = Exp12Trainer.train

        def train(self, last_step, after_step=None, **kwargs):
            def hook(t):
                if after_step is not None:
                    after_step(t)
                if t.interaction_step == crash_step:
                    os._exit(3)

            return original_train(self, last_step, after_step=hook, **kwargs)

        Exp12Trainer.train = train
    exp1.run({
        "experiment": "exp1", "config_path": CONFIG_PATH, "config_name": "base_exp12",
        "overrides": spec["overrides"], "checkpoint_dir": spec["checkpoint_dir"],
        "checkpoint_interval": spec["checkpoint_interval"], "checkpoint_start_frac": 0.0,
    })


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "env_record":
        env_record([tuple(x.split(":")) for x in sys.argv[3:]], sys.argv[2])
    elif mode == "env_replay":
        env_replay(sys.argv[2], sys.argv[3])
    elif mode == "exp1":
        exp1_run(sys.argv[2])
