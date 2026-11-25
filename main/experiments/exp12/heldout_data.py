"""Collect a reference-policy dataset that never enters any training replay."""
import hashlib
import io
import json
import random

import jax
import jax.numpy as jnp
import numpy as np

from experiments.exp12.heldout_returns import discounted_episode_returns
from experiments.exp12.online_exp2_pilot import _branch_env, _seed
from scale_rl.envs.wrappers.repeat_action import executed_raw_steps, limit_raw_steps
from utils.atomic_io import atomic_write_bytes


def collect_reference_test(cfg, trainer, parent, path, raw_steps=5000):
    """Persist a deterministic test trajectory with a provenance-bound identity."""
    identity = {"protocol": "exp2-heldout-v1", "parent": parent, "raw_steps": raw_steps}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as prior:
            if str(prior['identity_sha256']) != identity_hash:
                raise ValueError('held-out dataset identity mismatch')
        return hashlib.sha256(path.read_bytes()).hexdigest()
    seed = _seed('exp2-heldout-reference-test', cfg.env_name, cfg.seed,
                 cfg.online_exp2.reference_regime)
    np_state, py_state = np.random.get_state(), random.getstate()
    env = _branch_env(cfg, seed)
    rows = {k: [] for k in ('observation', 'action', 'reward', 'next_observation',
                            'terminated', 'truncated', 'episode', 'raw_steps')}
    key = jax.random.PRNGKey(seed)
    episode, elapsed = 0, 0
    try:
        np.random.seed(seed)
        random.seed(seed)
        obs, _ = env.reset()
        while elapsed < raw_steps:
            limit_raw_steps(env, raw_steps - elapsed)
            normalized = trainer.agent._normalize(obs) if hasattr(trainer.agent, '_normalize') else obs
            key, action_key = jax.random.split(key)
            action = np.asarray(trainer._sac_agent.actor(
                observations=jnp.asarray(normalized), temperature=1.0).sample(seed=action_key))
            next_obs, reward, term, trunc, info = env.step(action)
            raw = executed_raw_steps(env)
            if raw < 1 or elapsed + raw > raw_steps:
                raise ValueError('invalid held-out step accounting')
            final = next_obs
            if bool(term[0] or trunc[0]) and 'final_observation' in info:
                final = np.asarray([info['final_observation'][0]])
            for name, val in zip(rows, (obs[0], action[0], reward[0], final[0], term[0], trunc[0], episode, raw)):
                rows[name].append(np.array(val, copy=True))
            elapsed += raw
            obs = next_obs
            if bool(term[0] or trunc[0]):
                episode += 1
                if 'final_observation' not in info:
                    obs, _ = env.reset()
        arrays = {k: np.asarray(v) for k, v in rows.items()}
        returns, complete, censored = discounted_episode_returns(
            arrays['reward'], arrays['terminated'], arrays['truncated'], float(cfg.gamma))
        arrays.update(reward_to_go=returns, complete=complete, censored=censored,
                      identity_sha256=np.asarray(identity_hash), seed=np.asarray(seed),
                      identity_json=np.asarray(json.dumps(identity, sort_keys=True)))
        data = io.BytesIO()
        np.savez_compressed(data, **arrays)
        atomic_write_bytes(path, data.getvalue())
        return hashlib.sha256(path.read_bytes()).hexdigest()
    finally:
        env.close()
        np.random.set_state(np_state)
        random.setstate(py_state)
