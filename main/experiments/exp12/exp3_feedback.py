"""Cross critic size with shared or endogenous new experience on common replay."""

from collections import deque

import jax
import numpy as np

from experiments.exp12.exp3_pilot import normalize, sample
from scale_rl.agents.sac.sac_agent import _update_sac_networks

FIELDS = (
    "observation",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_observation",
)


def interaction(env, actor, normalizer, obs, key):
    action = np.asarray(sample(actor, normalize(normalizer, obs), key)[0])
    next_obs, reward, term, trunc, info = env.step(action)
    successor = next_obs.copy()
    if term[0] or trunc[0]:
        if "final_observation" in info:
            successor[0] = info["final_observation"][0]
        else:
            next_obs, _ = env.reset()
    row = (obs[0].copy(), action[0], reward[0], term[0], trunc[0], successor[0])
    return next_obs, row


def n_step_row(window, gamma):
    reward = 0.0
    for i, row in enumerate(window):
        reward += gamma**i * float(row[2])
        if row[3] or row[4]:
            break
    obs, action = window[0][:2]
    return obs, action, reward, *row[3:]


def run_feedback(ref, shadows, actor, normalizer, env, seeds, horizon, steps, evaluate):
    cfg = ref.cfg.agent
    start_temperature = ref.trainer._sac_agent.temperature
    buffer = ref.trainer.buffer
    gamma, n = float(cfg.gamma), int(cfg.n_step)
    cdq, tau = bool(cfg.critic_use_cdq), float(cfg.target_tau)
    target_entropy = float(ref.trainer._sac_agent._cfg.temp_target_entropy)
    batch_size = int(ref.cfg.buffer.sample_batch_size)
    key = jax.random.PRNGKey(seeds[0])
    obs, _ = env.reset(seed=seeds[0])
    stream = []
    for i in range(steps):
        obs, row = interaction(env, actor, normalizer, obs, jax.random.fold_in(key, i))
        stream.append(row)

    outcomes = []
    for name, shadow in shadows.items():
        for mode in ("shared", "own"):
            obs, _ = env.reset(seed=seeds[0])
            model, critic, target = actor, shadow["critic"], shadow["target"]
            temperature = start_temperature
            updates_key = jax.random.fold_in(key, 1929)
            window = deque(maxlen=n)
            experience = []
            rng = np.random.default_rng(seeds[0])
            churn_batch = None
            for i in range(steps):
                if mode == "own":
                    obs, row = interaction(
                        env, model, normalizer, obs, jax.random.fold_in(key, i)
                    )
                else:
                    row = stream[i]
                window.append(row)
                if len(window) < n:
                    continue
                experience.append(n_step_row(window, gamma))
                old = rng.integers(0, len(buffer), size=batch_size // 2)
                new = rng.integers(0, len(experience), size=batch_size - len(old))
                additions = zip(*(experience[j] for j in new))
                batch = {}
                for field, fresh in zip(FIELDS, additions):
                    values = getattr(buffer, "_" + field + "s")
                    combined = np.concatenate(
                        (values[old], np.asarray(fresh, dtype=values.dtype))
                    )
                    batch[field] = (
                        normalize(normalizer, combined)
                        if field in ("observation", "next_observation")
                        else combined
                    )
                if churn_batch is None:
                    churn_batch = batch
                updates_key, model, critic, target, temperature, _ = (
                    _update_sac_networks(
                        updates_key,
                        model,
                        critic,
                        target,
                        temperature,
                        batch,
                        gamma,
                        n,
                        cdq,
                        tau,
                        target_entropy,
                        churn_batch,
                        compute_actor_grad_cosine=False,
                    )
                )
            outcomes.append(
                dict(
                    critic=name,
                    arm=mode,
                    updates=len(experience),
                    returns=evaluate(env, model, normalizer, seeds, horizon),
                )
            )
    return outcomes
