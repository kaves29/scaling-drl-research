"""Regression coverage for throughput Step 3 (2026-09-24): fusing
updates_per_interaction_step calls to SACAgent.update() into one
jax.lax.scan via SACAgent.update_scanned().

No GPU available for this session's work - runs the real (tiny) SACAgent
on CPU, comparing two agents built from the identical seed/config, fed the
identical sequence of batches: one driven by repeated update() calls (the
original per-update path, still used unchanged by Angle 2A), the other by
update_scanned() calls (the new fused path Angle 1 now uses). Reports the
actual measured max absolute deviation in losses rather than assuming
exact equivalence.
"""

import unittest

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

from scale_rl.agents import create_agent

OBS_DIM = 4
ACT_DIM = 2


def _make_cfg():
    return OmegaConf.create(
        {
            "agent_type": "sac",
            "seed": 0,
            "num_train_envs": 1,
            "max_episode_steps": 100,
            "normalize_observation": False,
            "actor_block_type": "residual",
            "actor_num_blocks": 1,
            "actor_hidden_dim": 8,
            "actor_learning_rate": 1e-4,
            "actor_weight_decay": 1e-2,
            "critic_block_type": "residual",
            "critic_num_blocks": 1,
            "critic_hidden_dim": 8,
            "critic_learning_rate": 1e-4,
            "critic_weight_decay": 1e-2,
            "critic_use_cdq": False,
            "temp_target_entropy": None,
            "temp_target_entropy_coef": -0.5,
            "temp_initial_value": 0.01,
            "temp_learning_rate": 1e-4,
            "temp_weight_decay": 0.0,
            "target_tau": 0.005,
            "gamma": 0.99,
            "n_step": 1,
            "mixed_precision": False,
            "actor_sparsity": 0.0,
            "critic_sparsity": 0.0,
            # Disabled for this test: isolates Step 3's fusion from Step 1's
            # cadence gating (covered separately in test_throughput_steps_1_2.py).
            # A value larger than every update_step used here means it never
            # fires, so both paths' losses aren't perturbed by an extra
            # rng.split() on some updates but not others - fusion alone is
            # what's under test here.
            "actor_grad_cosine_every": 10_000,
        }
    )


def _make_spaces():
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
    return observation_space, action_space


def _make_random_batch(rng, batch_size=8):
    return {
        "observation": rng.standard_normal((batch_size, OBS_DIM)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(batch_size, ACT_DIM)).astype(np.float32),
        "reward": rng.standard_normal(batch_size).astype(np.float32),
        "terminated": np.zeros(batch_size, dtype=np.float32),
        "truncated": np.zeros(batch_size, dtype=np.float32),
        "next_observation": rng.standard_normal((batch_size, OBS_DIM)).astype(np.float32),
    }


class TestStep3ScanFusionNumericalEquivalence(unittest.TestCase):
    def test_unfused_vs_fused_loss_trajectories(self):
        cfg = _make_cfg()
        obs_space, act_space = _make_spaces()

        agent_unfused = create_agent(observation_space=obs_space, action_space=act_space, cfg=cfg)
        agent_fused = create_agent(observation_space=obs_space, action_space=act_space, cfg=cfg)

        # Same seed -> same init: sanity-check before running anything.
        import jax.numpy as jnp
        np.testing.assert_allclose(
            np.asarray(jnp.ravel(jnp.concatenate([jnp.ravel(x) for x in jax_tree_leaves(agent_unfused._actor.params)]))),
            np.asarray(jnp.ravel(jnp.concatenate([jnp.ravel(x) for x in jax_tree_leaves(agent_fused._actor.params)]))),
        )

        rng = np.random.default_rng(42)
        NUM_INTERACTION_STEPS = 40
        UPDATES_PER_STEP = 5  # matches this study's locked UTD=5

        # Pre-generate every batch upfront so both paths see byte-identical inputs.
        all_batches = [
            [_make_random_batch(rng) for _ in range(UPDATES_PER_STEP)]
            for _ in range(NUM_INTERACTION_STEPS)
        ]

        unfused_actor_losses = []
        unfused_critic_losses = []
        update_step = 1
        for step_batches in all_batches:
            for batch in step_batches:
                info = agent_unfused.update(update_step, dict(batch))
                unfused_actor_losses.append(float(info["train/actor_loss"]))
                unfused_critic_losses.append(float(info["train/critic_loss"]))
                update_step += 1

        fused_actor_losses = []
        fused_critic_losses = []
        update_step = 1
        for step_batches in all_batches:
            batch_sequence = {
                key: np.stack([b[key] for b in step_batches]) for key in step_batches[0].keys()
            }
            infos = agent_fused.update_scanned(update_step, batch_sequence, UPDATES_PER_STEP)
            for info in infos:
                fused_actor_losses.append(float(info["train/actor_loss"]))
                fused_critic_losses.append(float(info["train/critic_loss"]))
            update_step += UPDATES_PER_STEP

        self.assertEqual(len(unfused_actor_losses), len(fused_actor_losses))
        self.assertEqual(len(unfused_actor_losses), NUM_INTERACTION_STEPS * UPDATES_PER_STEP)

        actor_diffs = [abs(a - b) for a, b in zip(unfused_actor_losses, fused_actor_losses)]
        critic_diffs = [abs(a - b) for a, b in zip(unfused_critic_losses, fused_critic_losses)]

        print(f"[step3-equivalence] actor_loss max abs deviation over {len(actor_diffs)} updates: {max(actor_diffs)!r}")
        print(f"[step3-equivalence] critic_loss max abs deviation over {len(critic_diffs)} updates: {max(critic_diffs)!r}")
        print(f"[step3-equivalence] actor_loss first 5 unfused: {unfused_actor_losses[:5]}")
        print(f"[step3-equivalence] actor_loss first 5 fused:   {fused_actor_losses[:5]}")
        print(f"[step3-equivalence] actor_loss last 5 unfused: {unfused_actor_losses[-5:]}")
        print(f"[step3-equivalence] actor_loss last 5 fused:   {fused_actor_losses[-5:]}")

        # Not asserting exact bit-identity, same reasoning as Step 2's test:
        # report the real number, and only fail on genuine drift (not
        # float32-reassociation-level noise).
        self.assertLess(max(actor_diffs), 1e-3, "Unexpected actor_loss divergence between unfused and fused paths")
        self.assertLess(max(critic_diffs), 1e-3, "Unexpected critic_loss divergence between unfused and fused paths")


def jax_tree_leaves(pytree):
    import jax
    return jax.tree_util.tree_leaves(pytree)


if __name__ == "__main__":
    unittest.main()
