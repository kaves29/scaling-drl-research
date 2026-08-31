"""Orchestration-level proof that experiments.angle_2_a.run() no longer
shares one reference trajectory across Matchup 1 and Matchup 2.

Supersedes tests/test_angle2a_reference_sharing.py (deleted): that file
tested train_reference_agent_with_snapshots()/ReferenceTrajectory/
ProbeCapture.snapshot(), which existed solely to support the shared-
trajectory design research-methodology.md's Angle 2A section forbids
("Each scaled critic gets its own independently-trained reference critic...
never shared across two scaled architectures") and which has been removed -
see the 2026-08-28 audit notes in the End-of-Task Summary.

This test exercises the REAL entry point (experiments.angle_2_a.run(), with
real Hydra config resolution against the actual configs/base_angle2a.yaml),
mocking only what a local checkout genuinely can't provide (a real Angle 1
onset ledger) or shouldn't run in a test (real SAC training via
run_matchup) - see tests/test_angle2a_matchup_independence.py for the
lower-level proof that run_matchup() itself trains R fresh whenever no
reference_handle is supplied.
"""

import unittest
from unittest import mock

import hydra
from hydra.core.global_hydra import GlobalHydra

import experiments.angle_2_a as angle_2_a_module
from experiments.angle_2a.onset_lookup import OnsetLookupResult

FULL_OVERRIDES = [
    "angle_2_a.run_null_baseline=false",  # isolate the two real matchups only
    "angle_2_a.scaled_a.critic_num_blocks=5",
    "angle_2_a.scaled_a.critic_hidden_dim=768",
    "angle_2_a.scaled_b.critic_num_blocks=7",
    "angle_2_a.scaled_b.critic_hidden_dim=1024",
    "angle_2_a.reference.critic_num_blocks=2",
    "angle_2_a.reference.critic_hidden_dim=512",
]


def _fake_onset(onset_step: int, architecture: str) -> OnsetLookupResult:
    return OnsetLookupResult(
        onset_step=onset_step,
        architecture=architecture,
        environment="reacher-hard",
        seed=108,
        run_key=f"fake_run_key_{architecture}",
        exact_run_name=None,
        wandb_run_id=None,
        critic_degradation_method=None,
    )


class ReferenceIndependenceOrchestrationTest(unittest.TestCase):
    def setUp(self):
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        self.addCleanup(lambda: GlobalHydra.instance().clear() if GlobalHydra.instance().is_initialized() else None)

    def test_both_real_matchups_get_independent_reference_training_no_shared_handle(self):
        calls = []

        def fake_run_matchup(**kwargs):
            calls.append(kwargs)
            result = mock.Mock()
            result.run_metadata = {}
            return result

        # Different onset steps per matchup (120k vs 180k), matching the
        # real methodology (each scaled critic reaches its OWN t*) - if the
        # orchestration still tried to share one reference trajectory, this
        # is exactly the scenario that mechanism existed to serve.
        onsets_by_architecture = {
            "D5W768": _fake_onset(120_000, "D5W768"),
            "D7W1024": _fake_onset(180_000, "D7W1024"),
        }

        def fake_lookup(architecture, **kwargs):
            return onsets_by_architecture[architecture]

        with mock.patch.object(angle_2_a_module, "lookup_critic_degradation_onset", side_effect=fake_lookup), \
             mock.patch.object(angle_2_a_module, "run_matchup", side_effect=fake_run_matchup):
            angle_2_a_module.run(
                {
                    "experiment": "angle_2_a",
                    "config_path": "./configs",
                    "config_name": "base_angle2a",
                    "overrides": FULL_OVERRIDES,
                }
            )

        self.assertEqual(len(calls), 2, "expected exactly one run_matchup call per real matchup")
        names = {c["matchup_name"] for c in calls}
        self.assertEqual(names, {"matchup_1", "matchup_2"})

        for c in calls:
            # The old shared-trajectory design passed reference_handle= a
            # snapshot of the one shared trajectory, and reference_run_key=
            # its shared identifier, to both real matchups. Neither may be
            # passed (as a non-None value) any more - each call must let
            # run_matchup train R fresh and independently, exactly as
            # kwargs.get(...) with no override would.
            self.assertIsNone(c.get("reference_handle"), f"{c['matchup_name']} was passed a reference_handle")
            self.assertIsNone(c.get("reference_run_key"), f"{c['matchup_name']} was passed a reference_run_key")

        # Each matchup must still use its own onset_step (never averaged,
        # never the other matchup's).
        onset_steps_by_matchup = {c["matchup_name"]: c["onset_step"] for c in calls}
        self.assertEqual(onset_steps_by_matchup["matchup_1"], 120_000)
        self.assertEqual(onset_steps_by_matchup["matchup_2"], 180_000)


if __name__ == "__main__":
    unittest.main()
