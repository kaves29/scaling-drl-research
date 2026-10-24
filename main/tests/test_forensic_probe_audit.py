"""Audit-only known answers and failing-before/passing-proposed-copy regressions."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile

import jax.numpy as jnp
import numpy as np
import optax

from audits.echocritic_forensic.controls import ScalarCritic
from audits.echocritic_forensic.evidence import source_iqm
from audits.echocritic_forensic.proposals import (
    centered_residual_source,
    seed_grouping_source,
    replicate_configuration,
)

REPOSITORY = Path(__file__).resolve().parents[2]


def copied_module(name, source, package=None):
    module = types.ModuleType(name)
    module.__package__ = package
    sys.modules[name] = module
    try:
        exec(compile(source, "<audit-only-copy>", "exec"), module.__dict__)
    finally:
        del sys.modules[name]
    return module


class ForensicKnownAnswerTest(unittest.TestCase):
    def test_historical_five_and_fractional_ten_are_distinct(self):
        self.assertEqual(source_iqm([0, 1, 4, 10, 100], 5), 5.0)
        values = [0, 1, 2, 4, 8, 16, 32, 64, 128, 1000]
        self.assertEqual(source_iqm(values, 10), 18.6)
        self.assertNotEqual(source_iqm(values, 10), np.mean(sorted(values)[2:8]))
        for revision, expected in (("f8619dd", 21.0), ("f597ee9", 18.6)):
            import ast
            from scipy.stats import trim_mean

            text = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(REPOSITORY),
                    "show",
                    revision + ":main/experiments/exp12/probe.py",
                ],
                text=True,
            )
            function = next(
                n
                for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == "iqm"
            )
            namespace = {"np": np, "trim_mean": trim_mean}
            exec(
                compile(
                    ast.Module(body=[function], type_ignores=[]), "<frozen-iqm>", "exec"
                ),
                namespace,
            )
            self.assertEqual(namespace["iqm"](values), expected)
            self.assertEqual(namespace["iqm"]([0, 1, 4, 10, 100]), 5.0)

    def test_paired_iqm_does_not_equal_separate_iqm_subtraction(self):
        fresh = -np.arange(10, dtype=np.float32)
        current = np.roll(fresh, 3)
        self.assertEqual(source_iqm(fresh - current, 10), -2.0)
        self.assertEqual(source_iqm(fresh, 10) - source_iqm(current, 10), 0.0)

    def test_incomplete_rounds_are_not_silent_success(self):
        for values in ([1, 2], [1] * 9 + [float("nan")]):
            with self.assertRaises(ValueError):
                source_iqm(values, 10)

    def test_coupled_seed_candidate_is_nondestructive_and_rejects_mismatch(self):
        config = {
            "seed": 1,
            "agent": {"seed": 1, "learning_rate": 0.0001},
            "env": {"seed": 1},
            "probe": {"seed": 99},
        }
        result = replicate_configuration(config)
        self.assertEqual(
            result,
            {"agent": {"learning_rate": 0.0001}, "env": {}, "probe": {"seed": 99}},
        )
        self.assertEqual(config["agent"]["seed"], 1)
        with self.assertRaises(ValueError):
            replicate_configuration({**config, "env": {"seed": 8}})

    def test_task_target_array_corruption_is_detected_by_existing_bank(self):
        from experiments.exp12.probe_tasks import FrozenTask

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "task"
            task = FrozenTask(
                np.zeros((4, 2), np.float32),
                np.zeros((4, 1), np.float32),
                np.arange(4, dtype=np.float32),
                np.array([[0, 1], [2, 3]], np.int32),
                {"kind": "synthetic"},
            )
            task.save(root)
            path = next(root.glob("*.npz"))
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays["targets"][0] += 1
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                FrozenTask.load(root)


class UnappliedProposalRegressionTest(unittest.TestCase):
    def test_extreme_offset_contract_fails_original_passes_copied_candidate(self):
        text = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPOSITORY),
                "show",
                "f597ee9:main/experiments/exp12/probe.py",
            ],
            text=True,
        )
        original = copied_module("audit_original_probe", text)
        proposed = copied_module("audit_proposed_probe", centered_residual_source(text))
        network = ScalarCritic()
        observations, actions = jnp.zeros((4, 1)), jnp.zeros((4, 1))
        params = {"bias": jnp.asarray(2**24, jnp.float32)}
        base = jnp.asarray([-0.25, 0.25, -0.25, 0.25], jnp.float32)
        indices = jnp.asarray([[0, 1, 2, 3]], jnp.int32)
        optimizer = optax.adamw(0.0, weight_decay=0.01)

        def score(module):
            _, final, _ = module._fit(
                network, optimizer, params, observations, actions, base, indices, 2
            )
            return float(jnp.var(base) - final)

        self.assertEqual(score(original), 0.0625)
        self.assertEqual(score(proposed), 0.0)
        # Original production source is never replaced, patched or written.
        self.assertEqual(
            hashlib.sha256(text.encode()).digest(),
            hashlib.sha256(
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(REPOSITORY),
                        "show",
                        "f597ee9:main/experiments/exp12/probe.py",
                    ]
                )
            ).digest(),
        )

    def test_two_nested_seed_replicates_split_original_join_proposed_copy(self):
        from analysis.campaign import pipeline

        text = Path(pipeline.__file__).read_text()
        proposed = copied_module(
            "analysis.campaign.audit_proposed_pipeline",
            seed_grouping_source(text),
            "analysis.campaign",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "synthetic.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for seed in (1, 2):
                    metadata = {
                        "id": str(seed),
                        "name": "same-name",
                        "state": "finished",
                        "campaign": "exp96",
                        "step_key": "interaction_step",
                        "code": {"commit": "a" * 40, "dirty": False},
                        "config": {
                            "env_name": "dog-run",
                            "architecture": "D2W512",
                            "seed": seed,
                            "num_interaction_steps": 2,
                            "agent": {"seed": seed},
                            "env": {"seed": seed},
                        },
                    }
                    z.writestr(f"{seed}/metadata.json", json.dumps(metadata))
                    z.writestr(
                        f"{seed}/history.jsonl",
                        "\n".join(
                            json.dumps(
                                {
                                    "interaction_step": step,
                                    "eval/avg_return": seed * step,
                                }
                            )
                            for step in (0, 1, 2)
                        ),
                    )
            pipeline.run(archive, root / "original", reps=20)
            proposed.run(archive, root / "proposed", reps=20)
            original = json.loads((root / "original/statistics.json").read_text())
            changed = json.loads((root / "proposed/statistics.json").read_text())
            self.assertEqual(len(original["groups"]), 2)
            self.assertEqual([len(g["seed_ids"]) for g in original["groups"]], [1, 1])
            self.assertEqual(len(changed["groups"]), 1)
            self.assertEqual(changed["groups"][0]["seed_ids"], [1, 2])
            self.assertEqual(changed["groups"][0]["last_observed"]["n_seeds"], 2)


class LoggingAndCountsProposalRegressionTest(unittest.TestCase):
    def test_record_check_emits_contemporary_scores_only_in_proposed_copy(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from audits.echocritic_forensic.proposals import contemporary_log_source
        import wandb

        source = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPOSITORY),
                "show",
                "f597ee9:main/experiments/exp12/run_probes.py",
            ],
            text=True,
        )
        result = {
            "fresh": {"score": np.linspace(0, 1, 10, dtype=np.float32)},
            "current": {"score": np.linspace(-1, 0, 10, dtype=np.float32)},
        }
        for name, code, expected in (
            ("audit_original_logging", source, False),
            ("audit_proposed_logging", contemporary_log_source(source), True),
        ):
            module = copied_module(name, code)
            run = module.RunProbes.__new__(module.RunProbes)
            run.trainer = SimpleNamespace(
                interaction_step=20,
                cfg=SimpleNamespace(action_repeat=2),
                extra_state={},
            )
            run.trigger = SimpleNamespace(
                resamples=20,
                confidence=0.95,
                null_threshold=0,
                consecutive_checks=2,
                eligible_fraction=0.8,
            )
            run.cfg = SimpleNamespace(rounds=10, checks=50)
            run.seed, run.forced_check = 1, None
            with patch.object(wandb, "log") as logger:
                row = run.record_check(1, result)
            logged = logger.call_args.args[0]
            self.assertEqual("probe/score_fresh_iqm" in logged, expected)
            self.assertEqual("probe/score_current_iqm" in logged, expected)
            if expected:
                self.assertEqual(
                    logged["probe/score_fresh_iqm"], row["score_fresh_iqm"]
                )
                self.assertEqual(
                    logged["probe/score_current_iqm"], row["score_current_iqm"]
                )
            self.assertEqual(logged["probe/score_fresh_r9"], 1)

    def test_actual_aggregation_excerpt_reports_finite_counts_in_proposed_copy(self):
        import pandas as pd
        from audits.echocritic_forensic.proposals import effective_n_source

        fixture = (
            REPOSITORY
            / "main/audits/echocritic_forensic/metric_aggregation_excerpt.json"
        )
        source = json.loads(fixture.read_text())["excerpt"]
        rows = []
        for seed in range(8):
            row = {"environment": "dog-run", "architecture": "D4W1536"}
            for label in [
                "initial",
                "early",
                "common70",
                "late_observed",
                "change",
                "ratio",
            ]:
                row["x_" + label] = (
                    float("nan") if label == "late_observed" and seed == 7 else seed
                )
            rows.append(row)

        def ci(values):
            mean = float(np.nanmean(values))
            return mean, mean, mean

        results = []
        for code in (source, effective_n_source(source)):
            namespace = {
                "np": np,
                "S": pd.DataFrame(rows),
                "metrics": ["x"],
                "ci": ci,
                "agg": [],
            }
            exec(compile(code, "<isolated-aggregate-excerpt>", "exec"), namespace)
            results.append(namespace["agg"][0])
        original, candidate = results
        self.assertEqual(original["n_seeds"], 8)
        self.assertNotIn("late_observed_n_seeds", original)
        self.assertEqual(candidate["late_observed_n_seeds"], 7)
        self.assertEqual(candidate["common70_n_seeds"], 8)
        for key in original:
            self.assertEqual(original[key], candidate[key])
