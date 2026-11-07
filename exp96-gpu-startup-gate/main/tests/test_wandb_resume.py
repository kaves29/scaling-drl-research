"""Covers WandbTrainerLogger's run_id param (scale_rl/common/logger.py),
which lets a resumed run reattach to its original WandB run instead of
starting a new one. Mocks wandb.init directly rather than using
WANDB_MODE=offline, which silently ignores resume/id and can't verify this.
"""

import unittest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from scale_rl.common.logger import WandbTrainerLogger


def _make_cfg():
    return OmegaConf.create(
        {
            "project_name": "test-project",
            "updates_per_interaction_step": 5,
            "seed": 0,
            "env": {"env_name": "cheetah-run"},
            "agent": {
                "critic_num_blocks": 2,
                "critic_hidden_dim": 512,
                "actor_num_blocks": 1,
                "actor_hidden_dim": 128,
            },
        }
    )


class TestWandbTrainerLoggerResume(unittest.TestCase):
    @patch("scale_rl.common.logger.wandb")
    def test_fresh_run_passes_no_id_and_no_resume(self, mock_wandb):
        mock_wandb.init.return_value = MagicMock(id="new-run-id")
        WandbTrainerLogger(_make_cfg())

        _, kwargs = mock_wandb.init.call_args
        self.assertIsNone(kwargs["id"])
        self.assertIsNone(kwargs["resume"])

    @patch("scale_rl.common.logger.wandb")
    def test_resumed_run_passes_saved_id_with_resume_allow(self, mock_wandb):
        mock_wandb.init.return_value = MagicMock(id="saved-run-id-123")
        WandbTrainerLogger(_make_cfg(), run_id="saved-run-id-123")

        _, kwargs = mock_wandb.init.call_args
        self.assertEqual(kwargs["id"], "saved-run-id-123")
        self.assertEqual(kwargs["resume"], "allow")


if __name__ == "__main__":
    unittest.main()
