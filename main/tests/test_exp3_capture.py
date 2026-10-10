"""Prospective Exp3 capture through the REAL Exp1 and Exp2 entry points (CPU, tiny fixture).

Exp1 parent with a forced dev trigger (fork at 100, arm end 200, control end 400,
saves every 20 steps as in the grid's one-save-per-check cadence), recorded with
retention; the injected Exp2 arm likewise; an unrecorded parent as the parity reference.
Run from main/: python -m unittest tests.test_exp3_capture -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import jax

from experiments.exp12 import fork
from experiments.exp12.state import latest_state_dir, state_differences
from experiments.exp3.artifacts import Artifact, core, require_matched, tree_equal
from experiments.exp3.passive import PassivePair
from experiments.exp3.recording import record_exp2_scope
from experiments.exp3.retention import read_snapshot
from experiments.exp3.runner import make_passive, run
from experiments.exp3.streams import StreamReader
from tests.exp12_helpers import patch_wandb
from tests.test_exp12_fork import fork_overrides, run_arm, run_exp1

INTERVAL, FORK, ARM_END, M = 20, 100, 200, "half"


class RealEntryPointCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patchers = patch_wandb()
        cls.tmp = Path(tempfile.mkdtemp())
        t = cls.tmp
        cls.run_dir, cls.arm_dir, cls.ref_dir = t / "parent", t / "arm", t / "reference"
        overrides = fork_overrides(str(t / "results"))
        with record_exp2_scope(
            t / "stream_u",
            16,
            retain={
                "root": str(t / "snap_u"),
                "until_step": "arm_end",
                "replay": "omit",
            },
        ):
            run_exp1(str(cls.run_dir), overrides, interval=INTERVAL)
        with record_exp2_scope(
            t / "stream_i",
            16,
            retain={
                "root": str(t / "snap_i"),
                "until_step": "arm_end",
                "replay": "omit",
            },
        ):
            run_arm(
                str(cls.arm_dir),
                overrides,
                str(cls.run_dir),
                "injected",
                m=M,
                interval=INTERVAL,
            )
        run_exp1(
            str(cls.ref_dir),
            fork_overrides(str(t / "results_reference")),
            interval=INTERVAL,
        )
        cls.fork_state = latest_state_dir(fork.fork_dir(cls.run_dir) / "state")
        cls.parent_meta = cls.run_dir / "run_metadata.json"
        cls.arm_meta = cls.arm_dir / "run_metadata.json"
        cls.seed = json.loads(cls.parent_meta.read_text())["resolved_config"]["seed"]

    @classmethod
    def tearDownClass(cls):
        for p in cls.patchers:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def spec2(self, **change):
        return {
            "artifacts": {
                "fork": {
                    "state": str(self.fork_state),
                    "metadata": str(self.parent_meta),
                }
            },
            "normalization": "source_statistics",
            "injection_m": M,
            "injection_seed": self.seed,
            "fork_panel": str(fork.fork_dir(self.run_dir) / "panel.npz"),
            **change,
        }

    def snap(self, arm, step):
        return self.tmp / f"snap_{arm}" / f"step_{step:09d}"

    def test_c1_streams_carry_actual_arm_injection_and_passive_must_match(self):
        u, i = StreamReader(self.tmp / "stream_u"), StreamReader(self.tmp / "stream_i")
        self.assertEqual(
            (u.manifest["provenance"]["arm"], u.manifest["provenance"]["injection"]),
            ("control", None),
        )
        self.assertEqual(
            (i.manifest["provenance"]["arm"], i.manifest["provenance"]["injection"]),
            ("injected", {"m": M, "seed": self.seed}),
        )
        self.assertEqual(
            [r["step"] for r in u][:1] + [u.manifest["count"]], [FORK + 1, 400 - FORK]
        )
        self.assertEqual(i.manifest["count"], ARM_END - FORK)
        make_passive(self.spec2(), i, source="i")
        make_passive(self.spec2(), u, source="u")
        for change in ({"injection_seed": self.seed + 1}, {"injection_m": "all"}):
            with self.assertRaisesRegex(ValueError, "injection differs"):
                make_passive(self.spec2(**change), i, source="i")

    def test_retained_snapshots_are_the_already_written_matched_saves(self):
        steps = lambda arm: sorted(
            int(p.name[5:]) for p in (self.tmp / f"snap_{arm}").glob("step_*")
        )
        matched = list(range(FORK + INTERVAL, ARM_END + 1, INTERVAL))
        self.assertEqual(
            steps("u"), matched
        )  # U at the fork is the retained fork state itself
        self.assertEqual(
            steps("i"), [FORK] + matched
        )  # I at the fork: the arm's post-injection save
        self.assertFalse(list(self.tmp.glob("snap_*/CONFLICT_*")))
        # The routine state roots still keep only LATEST: retention changed no source behaviour.
        self.assertEqual(len(list((self.run_dir / "state").glob("step_*"))), 1)
        final_arm = latest_state_dir(self.arm_dir / "state")
        record = read_snapshot(self.snap("i", ARM_END))
        self.assertEqual(record["replay"], "omitted")
        for name, sha in record["files"].items():
            from experiments.exp3.artifacts import digest

            self.assertEqual(digest(final_arm / name), sha, name)

    def test_lean_snapshots_pair_and_refuse_replay_use(self):
        f = Artifact(self.fork_state, self.parent_meta)
        for step in range(FORK + INTERVAL, ARM_END + 1, INTERVAL):
            u = Artifact(self.snap("u", step), self.parent_meta, load_replay=False)
            i = Artifact(self.snap("i", step), self.arm_meta, load_replay=False)
            require_matched(u, i, f)
        with self.assertRaisesRegex(ValueError, "omitted"):
            Artifact(self.snap("i", ARM_END), self.arm_meta, load_replay=True)

    def test_pilot1_runs_at_fork_time_and_post_fork_on_real_artifacts(self):
        base = {
            "schema_version": 1,
            "run_role": "dev",
            "approved": True,
            "pilot": 1,
            "chunk_size": 2,
            "rng_seed": 9,
            "panel_size": 3,
            "normalization": "common_fork",
            "alpha_source": "fork",
            "intervention_space": "action",
            "interventions": ["sanity", "full"],
            "zero_signal_policy": "error",
            "outcome_eval_episodes": 1,
        }
        fork_art = {"state": str(self.fork_state), "metadata": str(self.parent_meta)}
        for label, untreated, injected in (
            (
                "fork_time",
                fork_art,
                {"state": str(self.snap("i", FORK)), "metadata": str(self.arm_meta)},
            ),
            (
                "post_fork",
                {
                    "state": str(self.snap("u", ARM_END)),
                    "metadata": str(self.parent_meta),
                },
                {"state": str(self.snap("i", ARM_END)), "metadata": str(self.arm_meta)},
            ),
        ):
            with self.subTest(label):
                spec = {
                    **base,
                    "artifacts": {
                        "fork": fork_art,
                        "untreated": untreated,
                        "injected": injected,
                    },
                }
                report = run(spec, self.tmp / f"pilot1_{label}")
                if label == "fork_time":
                    self.assertTrue(report["fork_equivalence"]["pass"])
                self.assertEqual(set(report["actors"]), {"u", "i", "f"})

    def test_recording_and_retention_do_not_change_the_source_trajectory(self):
        recorded = latest_state_dir(self.run_dir / "state")
        reference = latest_state_dir(self.ref_dir / "state")
        self.assertEqual(
            state_differences(
                recorded, reference, ignore_meta=("wandb_run_id", "extra_state")
            ),
            [],
        )
        with open(recorded / "meta.pkl", "rb") as a, open(
            reference / "meta.pkl", "rb"
        ) as b:
            import pickle

            ea, eb = pickle.load(a)["extra_state"], pickle.load(b)["extra_state"]
        from experiments.exp12.state import _deep_equal

        self.assertTrue(
            _deep_equal(ea, eb),
            "complete probe/trigger/fork/evaluation extra_state parity",
        )
        self.assertEqual(ea["fork"]["fork_step"], eb["fork"]["fork_step"])
        self.assertEqual(ea["post_fork_evals"], eb["post_fork_evals"])

    def test_pilot3_runs_on_retained_lean_matched_states(self):
        spec = {
            "schema_version": 1,
            "run_role": "dev",
            "approved": True,
            "pilot": 3,
            "enabled": True,
            "chunk_size": 2,
            "rng_seed": 9,
            "updates": 2,
            "batch_size": 3,
            "panel_size": 3,
            "checkpoint_every": 1,
            "alpha_source": "fork",
            "critic_source": "untreated",
            "evaluator_source": "fork",
            "alternative_evaluator": "injected",
            "normalization": "common_fork",
            "artifacts": {
                "fork": {
                    "state": str(self.fork_state),
                    "metadata": str(self.parent_meta),
                },
                "untreated": {
                    "state": str(self.snap("u", ARM_END)),
                    "metadata": str(self.parent_meta),
                },
                "injected": {
                    "state": str(self.snap("i", ARM_END)),
                    "metadata": str(self.arm_meta),
                },
            },
        }
        report = run(spec, self.tmp / "pilot3_lean")
        self.assertTrue(report)
        self.assertTrue((self.tmp / "pilot3_lean/DONE").exists())

    def test_source_checkpoints_preserve_selected_backend_and_fp32_state(self):
        import base64
        import numpy as np

        expected = jax.default_backend()
        for arm, metadata in (("u", self.parent_meta), ("i", self.arm_meta)):
            state = self.snap(arm, ARM_END)
            agent = core(Artifact(state, metadata, load_replay=False).agent)
            for name in ("_actor", "_critic", "_target_critic", "_temperature"):
                network = getattr(agent, name)
                for field in ("params", "opt_state"):
                    leaves = [
                        x
                        for x in jax.tree_util.tree_leaves(getattr(network, field))
                        if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating)
                    ]
                    if not leaves:
                        self.assertEqual((name, field), ("_target_critic", "opt_state"))
                        continue
                    self.assertEqual(
                        {d.platform for x in leaves for d in x.devices()}, {expected}
                    )
                    self.assertEqual({str(x.dtype) for x in leaves}, {"float32"})
            sharding = json.loads((state / "agent_ckpt/_sharding").read_text())
            records = {
                base64.b64decode(k).decode(): json.loads(v) for k, v in sharding.items()
            }
            for network in ("actor", "critic", "target_critic", "temperature"):
                params = [
                    v for k, v in records.items() if k.startswith(network + ".params.")
                ]
                self.assertTrue(params)
                for value in params:
                    if expected == "gpu":
                        self.assertRegex(value["device_str"], r"^(cuda|gpu):[0-9]+$")
                    else:
                        self.assertIn("cpu", value["device_str"].lower())

    def test_captured_keys_make_passive_u_reproduce_active_u(self):
        """Positive control on real entry points: a passive U learner given each arrival's captured
        source key reproduces the retained ACTIVE U snapshot bit for bit; without the keys it does not.
        Everything else (arrival order, normalization, replay, sampling, update kernel) is unmodified.
        """
        reader = StreamReader(self.tmp / "stream_u")
        active = core(
            Artifact(self.snap("u", ARM_END), self.parent_meta, load_replay=False).agent
        )
        for use_keys in (True, False):
            a, b = Artifact(self.fork_state, self.parent_meta), Artifact(
                self.fork_state, self.parent_meta
            )
            pair = PassivePair(
                a, b, a.meta["numpy_rng_state"], "source_statistics", reader.fingerprint
            )
            for record in reader:
                if record["step"] > ARM_END:
                    break
                if use_keys:
                    for v in pair.learners.values():
                        core(v.agent)._rng = jax.numpy.asarray(record["agent_key"])
                pair.deliver(record, int(record["updates"]))
            learner = core(pair.learners["u"].agent)
            same = tree_equal(
                (
                    learner._actor,
                    learner._critic,
                    learner._target_critic,
                    learner._temperature,
                ),
                (
                    active._actor,
                    active._critic,
                    active._target_critic,
                    active._temperature,
                ),
            )
            self.assertEqual(same, use_keys, f"use_keys={use_keys}")


if __name__ == "__main__":
    unittest.main()
