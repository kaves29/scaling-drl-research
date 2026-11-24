"""Exercise real SAC proposals and simulator branching on CPU fixtures."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import flax.serialization
import numpy as np

from experiments.exp12 import exp3_pilot as pilot
from experiments.exp12.exp3_feedback import n_step_row, run_feedback
from experiments.online_exp2 import OnlineExp2
from tests.test_revised_exp2_runner import patches, revised_cfg, wrapped_branch_env


class PilotIntegration(unittest.TestCase):
    def test_all_six_pilots_emit_results_without_changing_parent(self):
        with tempfile.TemporaryDirectory() as tmp, patches():
            ref = OnlineExp2(revised_cfg(), str(Path(tmp) / "ref"))
            ref.parent = {"test": True}
            other = OnlineExp2(revised_cfg(), str(Path(tmp) / "other"))
            other.parent = {"test": True}
            try:
                for i in range(520):
                    obs = np.full((1, 3), i / 520, dtype=np.float32)
                    ref.trainer.buffer.add(
                        dict(
                            observation=obs,
                            next_observation=obs + 0.01,
                            action=np.zeros((1, 2), np.float32),
                            reward=np.ones(1, np.float32),
                            terminated=np.zeros(1, bool),
                            truncated=np.zeros(1, bool),
                        )
                    )
                args = SimpleNamespace(
                    seed=1,
                    env="dog-run",
                    reference="small",
                    manifest=Path(tmp),
                    exp2_root=None,
                    exp2_commit=None,
                    episodes=1,
                    horizon=2,
                    pilots=[1, 2, 3, 5, 6],
                    feedback_steps=3,
                )
                with patch.object(
                    pilot, "ReplayExp2Pilot", return_value=ref
                ), patch.object(
                    pilot,
                    "load_exp1_agent",
                    return_value=(other.trainer.agent, other.parent),
                ), patch.object(
                    pilot, "compose_replay_config", return_value=revised_cfg()
                ), patch.object(
                    pilot, "_branch_env", wrapped_branch_env
                ), patch.object(
                    pilot, "probe_ranking", wraps=pilot.probe_ranking
                ):
                    report = pilot.run(args)
                self.assertEqual(
                    {row["pilot"] for row in report["actor_updates"]}, {1, 2, 3, 5, 6}
                )
                self.assertEqual(len(report["baseline_returns"]), 1)
                self.assertEqual(int(ref.trainer._sac_agent.actor.update_step), 0)
                shadows = {
                    name: dict(critic=s.critic, target=s.target)
                    for name, s in ref.shadows.items()
                }
                env = wrapped_branch_env()
                try:
                    rows = run_feedback(
                        ref,
                        shadows,
                        ref.trainer._sac_agent.actor,
                        None,
                        env,
                        [1],
                        2,
                        4,
                        pilot.evaluate,
                    )
                finally:
                    env.close()
                self.assertEqual(
                    {(r["critic"], r["arm"]) for r in rows},
                    {(c, m) for c in ("small", "large") for m in ("shared", "own")},
                )
                self.assertTrue(all(r["updates"] > 0 for r in rows))
            finally:
                ref.trainer.close()
                other.trainer.close()

    def test_exp2_loader_binds_critic_state_to_parent_and_source(self):
        with tempfile.TemporaryDirectory() as tmp, patches():
            ref = OnlineExp2(revised_cfg(), tmp)
            ref.parent = {"seed": 1}
            try:
                root = Path(tmp)
                checkpoint = root / "state" / "step_fixture"
                checkpoint.mkdir(parents=True)
                (checkpoint.parent / "LATEST").write_text(checkpoint.name)
                payload = flax.serialization.to_bytes(
                    {
                        "shadows": {
                            name: dict(
                                critic=shadow.critic,
                                target=shadow.target,
                                fresh_params=shadow.fresh_params,
                                updates=9,
                            )
                            for name, shadow in ref.shadows.items()
                        },
                        "key": ref.key,
                        "diagnostic_rng": json.dumps(
                            ref.diagnostic_rng.bit_generator.state
                        ),
                    }
                )
                (checkpoint / "online_exp2.msgpack").write_bytes(payload)
                meta = {
                    "protocol": "exp1-final-replay-v1",
                    "source_commit": "test-writer",
                    "parent": ref.parent,
                    "config_sha256": ref.config_hash,
                    "source_sha256": ref.source_hash,
                    "offline_critic_updates": 9,
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                }
                (checkpoint / "online_exp2_manifest.json").write_text(json.dumps(meta))
                models, steps = pilot.load_shadows(ref, root, "test-writer")
                self.assertEqual(steps, 9)
                self.assertEqual(set(models), {"small", "large"})
                with self.assertRaisesRegex(ValueError, "pinned source"):
                    pilot.load_shadows(ref, root, "wrong-writer")
                (checkpoint / "online_exp2.msgpack").write_bytes(payload + b"corrupt")
                with self.assertRaisesRegex(ValueError, "pinned source"):
                    pilot.load_shadows(ref, root, "test-writer")
            finally:
                ref.trainer.close()

    def test_nstep_stops_at_episode_boundary(self):
        rows = [(0, 1, 2, False, True, 3), (3, 4, 100, False, False, 5)]
        self.assertEqual(n_step_row(rows, 0.9), (0, 1, 2.0, False, True, 3))

    def test_ranking_restores_simulator_and_compares_both_critics(self):
        with tempfile.TemporaryDirectory() as tmp, patches():
            ref = OnlineExp2(revised_cfg(), tmp)
            from gymnasium.vector import SyncVectorEnv
            from tests.test_revised_exp2_runner import RestorableRepeat, TinyEnv

            env = SyncVectorEnv([lambda: RestorableRepeat(TinyEnv(), 2)])
            try:
                # Exercise the ranker with a short warmup (the fixture truncates early).
                rows = pilot.probe_ranking(
                    env,
                    ref.trainer._sac_agent.actor,
                    None,
                    dict.fromkeys(("small", "large"), ref.shadows["small"].critic),
                    bool(ref.cfg.agent.critic_use_cdq),
                    [1],
                    2,
                    0.99,
                    0.2,
                    warmup=1,
                )
                self.assertEqual(len(rows), 2)
                self.assertEqual(
                    rows[0]["soft_return_differences"],
                    rows[1]["soft_return_differences"],
                )
                self.assertTrue(
                    all(len(row["soft_return_differences"]) == 4 for row in rows)
                )
            finally:
                env.close()
                ref.trainer.close()
