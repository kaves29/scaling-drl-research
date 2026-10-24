"""Static sparse-mask checkpoint fidelity; dense seed portability is retained."""

import copy
import tempfile
import unittest

import numpy as np

from experiments.exp12.fixed_task import tree_sha256
from tests.test_batch1_reproducibility import agent
from tests.test_sac_agent_checkpoint import _make_random_batch


class SparseCheckpointTest(unittest.TestCase):
    def test_changed_critic_mask_seed_is_rejected(self):
        self.check_seed_rejected("critic_sparsity")

    def test_changed_actor_mask_seed_is_rejected(self):
        self.check_seed_rejected("actor_sparsity")

    def check_seed_rejected(self, field):
        original = agent(**{field: 0.5})
        target = agent(seed=1, **{field: 0.5})
        name = "critic" if field == "critic_sparsity" else "actor"
        self.assertNotEqual(
            tree_sha256(getattr(original, name).network_mask),
            tree_sha256(getattr(target, name).network_mask),
        )
        with tempfile.TemporaryDirectory() as tmp:
            original.save_checkpoint(tmp)
            with self.assertRaisesRegex(ValueError, "configuration"):
                target.load_checkpoint(tmp)

    def test_same_seed_sparse_masks_restore_and_next_update_is_exact(self):
        self.check_next_update(actor_sparsity=0.5, critic_sparsity=0.5)

    def test_dense_restore_can_replace_a_different_constructor_seed(self):
        self.check_next_update(destination_seed=1)

    def check_next_update(self, destination_seed=0, **cfg):
        original = agent(**cfg)
        batch = _make_random_batch(np.random.default_rng(828))
        original.update(0, copy.deepcopy(batch), compute_actor_grad_cosine=False)
        with tempfile.TemporaryDirectory() as tmp:
            original.save_checkpoint(tmp)
            restored = agent(seed=destination_seed, **cfg)
            restored.load_checkpoint(tmp)
            for name in ("actor", "critic"):
                self.assertEqual(
                    tree_sha256(getattr(original, name).network_mask),
                    tree_sha256(getattr(restored, name).network_mask),
                )
            original.update(1, copy.deepcopy(batch), compute_actor_grad_cosine=False)
            restored.update(1, copy.deepcopy(batch), compute_actor_grad_cosine=False)
            for name in ("_rng", "_actor", "_critic", "_target_critic", "_temperature"):
                self.assertEqual(
                    tree_sha256(getattr(original, name)),
                    tree_sha256(getattr(restored, name)),
                )


if __name__ == "__main__":
    unittest.main()
