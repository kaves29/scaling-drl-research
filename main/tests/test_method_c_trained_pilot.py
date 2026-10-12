"""Focused guard tests for the bounded trained-critic development pilot."""

import unittest
from types import SimpleNamespace

from experiments.exp12.trigger import bootstrap_interval
from scripts.sci_investigation.method_c_trained_pilot import validation_steps, verify_recorded_decisions


def config():
    return SimpleNamespace(
        env_name="hopper-hop", seed=990, critic_num_blocks=4, critic_hidden_dim=1536,
        num_interaction_steps=250_000, run_role="dev",
        probe=SimpleNamespace(checks=20, rounds=5, steps=2000, pool_size=25_600, batch_size=256),
        trigger=SimpleNamespace(resamples=10_000, confidence=.95, consecutive_checks=2,
                                null_threshold=0.0, eligible_fraction=.95),
        testing=SimpleNamespace(force_trigger_check=None), buffer=SimpleNamespace(min_length=5000),
    )


class MethodCTrainedPilotTest(unittest.TestCase):
    def test_full_budget_sets_first_two_natural_checks(self):
        self.assertEqual(validation_steps(config()), (12_500, 25_000))

    def test_rejects_changed_scientific_settings_and_forced_trigger(self):
        for name, value in (("num_interaction_steps", 25_000), ("critic_hidden_dim", 1024),
                            ("run_role", "confirmatory")):
            cfg = config()
            setattr(cfg, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validation_steps(cfg)
        for group, name, value in (("probe", "rounds", 1), ("probe", "steps", 1000),
                                   ("trigger", "consecutive_checks", 1),
                                   ("trigger", "null_threshold", .001),
                                   ("testing", "force_trigger_check", 2)):
            cfg = config()
            setattr(getattr(cfg, group), name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validation_steps(cfg)

    def test_saved_intervals_and_two_consecutive_decisions_are_recomputed(self):
        cfg = config()
        records = [{"check_index": 0}]
        for k in (1, 2):
            loss = [.01, .02, .03, .04, .05]
            low, high = bootstrap_interval(loss, 990, k, 10_000, .95)
            records.append({"check_index": k, "interaction_step": k * 12_500,
                            **{f"loss_r{i}": v for i, v in enumerate(loss)},
                            "ci_low": low, "ci_high": high, "triggered": True, "valid": True})
        self.assertEqual(verify_recorded_decisions(records, cfg),
                         {"check_index": 2, "interaction_step": 25_000})
        records[2]["ci_low"] = 0.0
        with self.assertRaisesRegex(ValueError, "bootstrap differs"):
            verify_recorded_decisions(records, cfg)


if __name__ == "__main__":
    unittest.main()
