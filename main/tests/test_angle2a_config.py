import unittest

import hydra
import omegaconf
from hydra.core.global_hydra import GlobalHydra

from experiments.angle_2a.config import (
    architecture_label,
    build_role_agent_cfg,
    validate_angle2a_config,
)
from experiments.angle_2a.errors import Angle2AConfigError

FULL_OVERRIDES = [
    "angle_2_a.scaled_a.critic_num_blocks=5",
    "angle_2_a.scaled_a.critic_hidden_dim=768",
    "angle_2_a.scaled_b.critic_num_blocks=7",
    "angle_2_a.scaled_b.critic_hidden_dim=1024",
    "angle_2_a.reference.critic_num_blocks=2",
    "angle_2_a.reference.critic_hidden_dim=512",
]


def _compose(overrides):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path="../configs")
    cfg = hydra.compose(config_name="base_angle2a", overrides=overrides)

    def eval_resolver(s):
        return eval(s)

    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver, replace=True)
    return cfg


class TestAngle2AConfigValidation(unittest.TestCase):
    def test_missing_all_architecture_fields_raises_clear_error(self):
        cfg = _compose([])
        with self.assertRaises(Angle2AConfigError) as ctx:
            validate_angle2a_config(cfg)
        message = str(ctx.exception)
        for field in [
            "angle_2_a.scaled_a.critic_num_blocks",
            "angle_2_a.scaled_a.critic_hidden_dim",
            "angle_2_a.scaled_b.critic_num_blocks",
            "angle_2_a.scaled_b.critic_hidden_dim",
            "angle_2_a.reference.critic_num_blocks",
            "angle_2_a.reference.critic_hidden_dim",
        ]:
            self.assertIn(field, message)

    def test_partially_missing_fields_are_all_reported_together(self):
        cfg = _compose(
            [
                "angle_2_a.scaled_a.critic_num_blocks=5",
                "angle_2_a.scaled_a.critic_hidden_dim=768",
                "angle_2_a.reference.critic_num_blocks=2",
                "angle_2_a.reference.critic_hidden_dim=512",
            ]
        )
        with self.assertRaises(Angle2AConfigError) as ctx:
            validate_angle2a_config(cfg)
        # inspect the actual missing-fields list the code computed, not the
        # rendered message (which also contains a full example command that
        # legitimately mentions every field name).
        message = str(ctx.exception)
        missing_list_str = message.split("required field(s): ")[1].split(". Example:")[0]
        missing_fields = eval(missing_list_str)
        self.assertCountEqual(
            missing_fields,
            ["angle_2_a.scaled_b.critic_num_blocks", "angle_2_a.scaled_b.critic_hidden_dim"],
        )

    def test_fully_specified_config_validates_and_returns_architectures(self):
        cfg = _compose(FULL_OVERRIDES)
        architectures = validate_angle2a_config(cfg)

        self.assertEqual(architectures["scaled_a"].critic_num_blocks, 5)
        self.assertEqual(architectures["scaled_a"].critic_hidden_dim, 768)
        self.assertEqual(architectures["scaled_b"].critic_num_blocks, 7)
        self.assertEqual(architectures["scaled_b"].critic_hidden_dim, 1024)
        self.assertEqual(architectures["reference"].critic_num_blocks, 2)
        self.assertEqual(architectures["reference"].critic_hidden_dim, 512)

    def test_no_hidden_default_leaks_through_when_only_some_roles_given(self):
        # Deliberately valid scaled_a/scaled_b but missing reference: the
        # implementation must not quietly substitute the top-level
        # critic_num_blocks/critic_hidden_dim (which are themselves `???` in
        # configs/base_angle2a.yaml specifically to prevent this).
        cfg = _compose(
            [
                "angle_2_a.scaled_a.critic_num_blocks=5",
                "angle_2_a.scaled_a.critic_hidden_dim=768",
                "angle_2_a.scaled_b.critic_num_blocks=7",
                "angle_2_a.scaled_b.critic_hidden_dim=1024",
            ]
        )
        with self.assertRaises(Angle2AConfigError):
            validate_angle2a_config(cfg)

    def test_architecture_label_matches_angle1_naming_convention(self):
        cfg = _compose(FULL_OVERRIDES)
        architectures = validate_angle2a_config(cfg)

        self.assertEqual(architecture_label(architectures["scaled_a"]), "D5W768")
        self.assertEqual(architecture_label(architectures["scaled_b"]), "D7W1024")
        self.assertEqual(architecture_label(architectures["reference"]), "D2W512")

    def test_build_role_agent_cfg_uses_role_architecture_not_top_level_fields(self):
        cfg = _compose(FULL_OVERRIDES)
        architectures = validate_angle2a_config(cfg)

        agent_cfg = build_role_agent_cfg(cfg.agent, architectures["scaled_b"])

        self.assertEqual(agent_cfg["critic_num_blocks"], 7)
        self.assertEqual(agent_cfg["critic_hidden_dim"], 1024)
        # actor architecture must be untouched/shared across all roles
        self.assertEqual(agent_cfg["actor_num_blocks"], cfg.actor_num_blocks)
        self.assertEqual(agent_cfg["actor_hidden_dim"], cfg.actor_hidden_dim)

    def test_build_role_agent_cfg_is_independent_per_role(self):
        cfg = _compose(FULL_OVERRIDES)
        architectures = validate_angle2a_config(cfg)

        scaled_agent_cfg = build_role_agent_cfg(cfg.agent, architectures["scaled_a"])
        reference_agent_cfg = build_role_agent_cfg(cfg.agent, architectures["reference"])

        self.assertNotEqual(scaled_agent_cfg["critic_num_blocks"], reference_agent_cfg["critic_num_blocks"])
        self.assertNotEqual(scaled_agent_cfg["critic_hidden_dim"], reference_agent_cfg["critic_hidden_dim"])


if __name__ == "__main__":
    unittest.main()
