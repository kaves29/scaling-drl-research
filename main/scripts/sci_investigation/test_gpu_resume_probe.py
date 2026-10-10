"""Verdict logic of gpu_resume_probe on synthetic reports (no training).

Run from main/: python -m unittest scripts.sci_investigation.test_gpu_resume_probe
"""

import copy
import unittest
import tempfile
import subprocess
import os
from pathlib import Path
from unittest.mock import patch

from scripts.sci_investigation import gpu_resume_probe as g


def child(exit_code=0, done=True, backend="gpu"):
    snap = {
        "interaction_step": 60,
        "update_step": 100,
        "parameter_platforms": {
            n: [backend]
            for n in ("_actor", "_critic", "_target_critic", "_temperature")
        },
    }
    return {
        "exit": exit_code,
        "done": done,
        "backend": backend,
        "device_kinds": [],
        "log": "x.log",
        "seconds": 1.0,
        "state_present": True,
        "state_interaction_step": 300,
        "state_update_step": 582,
        "expected_restore": None,
        "observations": {"training": snap, "restores": [], "saves": []},
    }


def good():
    report = {
        "reference": child(),
        "reference_repeat": child(),
        "reference_repeat_differences": [],
        "cases": {
            str(s): {
                "crash": child(g.CRASH_EXIT, False),
                "resume": child(),
                "differences": [],
                "differences_vs_repeat": [],
            }
            for s in g.CRASH_STEPS
        },
    }
    for s in ("60", "95"):
        case = report["cases"][s]
        source = f"/tmp/crash{s}/state/step_000000060_abc"
        case["resume"]["expected_restore"] = source
        snap = case["resume"]["observations"]["training"]
        case["resume"]["observations"]["restores"] = [
            {
                "source": source,
                "saved_interaction_step": 60,
                "saved_update_step": 100,
                "completed": True,
                **snap,
            }
        ]
        case["crash"]["observations"]["saves"] = [{"path": source, **snap}]
    return report


class ClassifyTest(unittest.TestCase):
    def verdict(self, report, require="gpu"):
        return g.classify(report, require)[0]

    def test_clean_report_passes(self):
        self.assertEqual(g.classify(good(), "gpu"), (g.PASS, []))

    def test_resume_difference_is_a_resume_defect(self):
        r = good()
        r["cases"]["60"]["differences"] = ["agent:critic/params"]
        self.assertEqual(self.verdict(r), g.RESUME_DEFECT)

    def test_failed_relaunch_is_a_resume_defect(self):
        for exit_code, done in ((1, False), (0, False)):
            r = good()
            r["cases"]["5"]["resume"] = child(exit_code, done)
            self.assertEqual(self.verdict(r), g.RESUME_DEFECT)

    def test_differing_references_are_nondeterminism_not_a_resume_defect(self):
        r = good()
        r["reference_repeat_differences"] = ["agent:critic/params"]
        r["cases"]["95"]["differences"] = ["agent:critic/params"]
        self.assertEqual(self.verdict(r), g.NONDETERMINISTIC)

    def test_harness_failures_are_incomplete_and_take_precedence(self):
        mutations = {
            "cpu fallback": lambda r: r["cases"]["60"]["resume"].update(backend="cpu"),
            "no backend record": lambda r: r["reference"].update(backend=None),
            "reference failed": lambda r: r["reference"].update(exit=1, done=False),
            "repeat failed": lambda r: r["reference_repeat"].update(exit=1, done=False),
            "crash did not fire": lambda r: r["cases"]["5"]["crash"].update(
                exit=0, done=True
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name):
                r = good()
                r["reference_repeat_differences"] = [
                    "agent:x"
                ]  # would otherwise read as nondeterminism
                mutate(r)
                self.assertEqual(self.verdict(r), g.INCOMPLETE)

    def test_backend_not_enforced_without_requirement(self):
        r = good()
        for c in [r["reference"], r["reference_repeat"]]:
            c["backend"] = "cpu"
        self.assertEqual(self.verdict(r, None), g.PASS)

    def test_exit_codes_are_distinct(self):
        self.assertEqual(sorted(g.EXIT.values()), [0, 1, 2, 3])

    def test_classify_does_not_mutate(self):
        r = good()
        r["cases"]["5"]["differences"] = ["buffer:obs"]
        before = copy.deepcopy(r)
        g.classify(r, "gpu")
        self.assertEqual(r, before)

    def test_missing_or_unreadable_state_is_not_nondeterminism(self):
        for differences in (
            ["missing state"],
            ["unreadable state: ValueError: corrupt"],
        ):
            r = good()
            r["reference_repeat_differences"] = differences
            self.assertEqual(self.verdict(r), g.INCOMPLETE)
        r = good()
        r["cases"]["60"]["resume"]["state_present"] = False
        self.assertEqual(self.verdict(r), g.INCOMPLETE)

    def test_all_crash_scenarios_required(self):
        for cases in ({}, {"5": good()["cases"]["5"]}):
            r = good()
            r["cases"] = cases
            self.assertEqual(self.verdict(r), g.INCOMPLETE)

    def test_explicit_subset_is_scoped_not_a_full_suite_pass(self):
        r = good()
        r["crash_steps"] = [95]
        r["cases"] = {"95": r["cases"]["95"]}
        self.assertEqual(self.verdict(r), g.PASS)
        for selected in ([], [95, 95], [96], [60, 95]):
            r["crash_steps"] = selected
            self.assertEqual(self.verdict(r), g.INCOMPLETE)

    def test_gpu_startup_does_not_prove_gpu_restore(self):
        for phase in ("training", "restores"):
            r = good()
            obs = r["cases"]["60"]["resume"]["observations"]
            snap = obs[phase] if phase == "training" else obs[phase][0]
            snap["parameter_platforms"]["_critic"] = ["cpu"]
            self.assertEqual(self.verdict(r), g.INCOMPLETE)

    def test_no_op_restore_or_wrong_source_or_counter_is_incomplete(self):
        for mutation in (
            lambda c: c["observations"].update(restores=[]),
            lambda c: c["observations"]["restores"][0].update(source="/wrong"),
            lambda c: c["observations"]["restores"][0].update(update_step=0),
            lambda c: c.update(expected_restore=None),
            lambda c: c["observations"]["restores"][0].pop("completed"),
        ):
            r = good()
            mutation(r["cases"]["60"]["resume"])
            self.assertEqual(self.verdict(r), g.INCOMPLETE)

    def test_identical_premature_done_states_cannot_pass(self):
        for key in ("reference", "reference_repeat"):
            r = good()
            r[key]["state_interaction_step"] = 150
            self.assertEqual(self.verdict(r), g.INCOMPLETE)
        r = good()
        r["cases"]["95"]["resume"]["state_update_step"] = 581
        self.assertEqual(self.verdict(r), g.INCOMPLETE)

    def test_saved_checkpoint_must_also_be_gpu_trained(self):
        r = good()
        r["cases"]["95"]["crash"]["observations"]["saves"][0]["parameter_platforms"][
            "_critic"
        ] = ["cpu"]
        self.assertEqual(self.verdict(r), g.INCOMPLETE)

    def test_repeat_reference_comparison_is_not_ignored(self):
        r = good()
        r["cases"]["95"]["differences_vs_repeat"] = ["buffer:actions"]
        self.assertEqual(self.verdict(r), g.RESUME_DEFECT)

    def test_repetition_difference_keeps_relaunch_error(self):
        r = good()
        r["reference_repeat_differences"] = ["agent:critic/params"]
        r["cases"]["60"]["resume"].update(exit=1, done=False)
        verdict, reasons = g.classify(r, "gpu")
        self.assertEqual(verdict, g.NONDETERMINISTIC)
        self.assertTrue(any("relaunch also failed" in reason for reason in reasons))

    def test_failed_actual_restore_keeps_resume_error_instead_of_missing_state(self):
        r = good()
        resume = r["cases"]["60"]["resume"]
        resume.update(exit=1, done=False, state_present=False)
        restore = resume["observations"]["restores"][0]
        restore.update(completed=False, error="RuntimeError: restore failed")
        del restore["interaction_step"], restore["update_step"]
        r["cases"]["60"].update(
            differences=["missing state"], differences_vs_repeat=["missing state"]
        )
        self.assertEqual(self.verdict(r), g.RESUME_DEFECT)
        r["reference_repeat_differences"] = ["agent:critic/params"]
        verdict, reasons = g.classify(r, "gpu")
        self.assertEqual(verdict, g.NONDETERMINISTIC)
        self.assertTrue(any("relaunch also failed" in reason for reason in reasons))

    def test_invalid_saved_state_becomes_incomplete_evidence(self):
        with patch(
            "experiments.exp12.state.state_differences",
            side_effect=ValueError("corrupt"),
        ):
            self.assertIn("unreadable state:", g.diff(Path("/a"), Path("/b"))[0])

    def test_child_output_and_provenance_exist_even_if_interrupted(self):
        import exp12_helpers  # Import dependencies before mocking the shared subprocess module.

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)

            def interrupted(cmd, *, env, stdout, stderr):
                stdout.write("before hang\n")
                stdout.flush()
                raise KeyboardInterrupt

            with patch.object(g.subprocess, "run", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    g.run(out, "ref")
            self.assertEqual((out / "ref_run.log").read_text(), "before hang\n")
            self.assertTrue((out / "ref_run_receipt.json").is_file())


class CliTest(unittest.TestCase):
    def test_declared_subset_runs_two_references_and_only_one_restart(self):
        report = good()
        case = report["cases"]["95"]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "new"
            runs = [
                (report["reference"], out / "ref"),
                (report["reference_repeat"], out / "ref2"),
                (case["crash"], out / "crash95"),
                (case["resume"], out / "crash95"),
            ]
            with patch.object(g, "provenance", return_value={}), patch.object(
                g, "run", side_effect=runs
            ) as runner, patch.object(
                g, "final_state", return_value=None
            ), patch.object(
                g, "diff", return_value=[]
            ), patch(
                "builtins.print"
            ):
                self.assertEqual(
                    g.main(
                        [
                            "--out",
                            str(out),
                            "--require-backend",
                            "gpu",
                            "--crash-step",
                            "95",
                        ]
                    ),
                    0,
                )
            import json

            saved = json.loads((out / "gpu_resume_probe.json").read_text())
            self.assertEqual(saved["crash_steps"], [95])
            self.assertEqual(list(saved["cases"]), ["95"])
            self.assertEqual(runner.call_count, 4)
            self.assertEqual(runner.call_args_list[2].kwargs, {"crash_step": 95})

    def test_out_must_be_new_absolute(self):
        with self.assertRaises(SystemExit):
            g.main(["--out", "relative/dir"])
        with self.assertRaises(SystemExit):
            g.main(["--out", "/"])


class GatePreflightTest(unittest.TestCase):
    def test_wrong_commit_stops_before_backend_and_keeps_failure_status(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            main = root / "checkout" / "main"
            scripts = main / "scripts"
            scripts.mkdir(parents=True)
            (scripts / "check_exp12_validation_checkout.sh").write_bytes(
                (g.MAIN / "scripts/check_exp12_validation_checkout.sh").read_bytes()
            )
            for args in (
                ["init", "-q"],
                ["add", "."],
                [
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-qm",
                    "fixture",
                ],
            ):
                subprocess.run(
                    ["git", *args],
                    cwd=root / "checkout",
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            env = dict(
                os.environ,
                EXPECTED_COMMIT="0" * 40,
                VALIDATION_ROOT=str(root),
                SLURM_JOB_ID="1",
            )
            result = subprocess.run(
                ["bash", str(g.MAIN / "scripts/sbatch_exp12_gpu_resume_gate.sh")],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("HEAD mismatch", result.stderr)
            out = root / "gpu_resume_95_1"
            self.assertEqual((out / "exit_status.txt").read_text(), "2\n")
            self.assertIn("exit_status.txt", (out / "SHA256SUMS").read_text())
            self.assertFalse((out / "workload.py").exists())


if __name__ == "__main__":
    unittest.main()
